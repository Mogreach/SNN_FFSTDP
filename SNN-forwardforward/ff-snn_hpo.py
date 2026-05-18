"""
Hyperparameter search runner for ff-snn.py.

Usage:
1. Edit SEARCH_SPACE to define the discrete candidate pool.
2. Set MODEL / DATASET / LEARNING_MODE / HIDDEN_LAYER_UPDATE_MODE.
3. Choose SEARCH_STRATEGY from:
   - "grid":
       Exhaustively evaluate every candidate in SEARCH_SPACE.
       Best when the search space is very small and you want full coverage.
   - "random":
       Randomly sample a fixed number of candidates from the discrete pool.
       Best when the search space is moderate or large and you want a cheap baseline.
   - "successive_halving":
       Train many candidates with a small epoch budget first, then keep only the
       better half (or top 1 / factor) for deeper training.
       Best when training is expensive and early validation accuracy is informative.
   - "bayes":
       Use a lightweight Gaussian-process surrogate on the discrete candidate pool.
       It starts with a few random warmup trials, then chooses the next candidate
       according to an acquisition function such as UCB or EI.
       Best when the candidate pool is not huge and each full training run is costly.
4. Adjust the strategy-specific settings below.
5. Run:
   python ff-snn_hpo.py

Outputs:
- A summary CSV under logs/opt/
- A .best.json file containing the current best trial

Notes:
- This script optimizes validation metrics by default, not test accuracy.
- SEARCH_SPACE is treated as a discrete candidate pool. Even Bayesian search
  selects from this pool rather than optimizing a continuous domain directly.
"""
from __future__ import annotations

import csv
import argparse
import itertools
import json
import math
import numbers
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.cnn_models.common import is_cnn_family_model


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "ff-snn.py"
OUT_DIR = ROOT / "logs" / "opt"

# Search-space definition.
# Each key represents one searchable axis.
# The script first expands these axes into a discrete candidate pool, then
# applies the selected search strategy on top of that pool.
SEARCH_SPACE = {
    # Goodness / delta-loss threshold used by hidden layers.
    "loss_threshold": [1.0],
    # Neuron firing threshold for hidden layers.
    "v_threshold": [1.5],
    # Batch size.
    "b": [1024],
    # MLP layer widths. Only used when MODEL == "MLP".
    "dims": [
        [784, 512, 512, 10],
    ],
    # CNN convolution configuration. Only used when MODEL == "CNN".
    # Predefined CNN families such as VGG / ResNet ignore this search axis.
    # Each tuple is: (in_channels, out_channels, kernel_size, stride, padding)
    "cov_cfg": [
        [
            # in_ch, out_ch, k, s, p
            (1, 32, 3, 1, 1),
            (32, 64, 3, 1, 1),
            (64, 128, 3, 1, 1),
            (128, 256, 3, 1, 1),
        ]
    ],
    "T": [16],  # Number of FF-STDP steps per batch.
    "lr": [0.001],
}

# Base training settings shared by every trial.
MODEL = "MLP"  # Model family: "MLP" "CNN" "VGG6" "VGG8" "VGG11" or "ResNet"
DATASET = "MNIST"  # Dataset name. "MNIST", "N-MNIST", "NMNIST", "FashionMNIST", "CIFAR10", "DVS128Gesture"
LEARNING_MODE = "unsupervised"  # "unsupervised" or "supervised"
HIDDEN_LAYER_UPDATE_MODE = "autograd"  # Hidden-layer update: "autograd" or "manual"
NEG_SAMPLE_STRATEGY = "embed_label_onehot"  # Negative-sample generation strategy: "auto, embed_label_onehot, embed_zero_onehot, SCFF."
GOODNESS_STRATEGY = "square_mean"  # Hidden-layer goodness strategy: "auto, square, square_mean, signed_square_mean, membrane_potential_square_mean."
HIDDEN_LOSS_STRATEGY = "supervised_delta"  # Hidden-layer local loss strategy: "auto, pairwise_goodness, supervised_delta, scaled_supervised_delta."
DEVICE = None  # Optional explicit torch device forwarded to ff-snn.py.
DATA_LOADER_WORKERS = 8  # DataLoader workers forwarded to ff-snn.py.

# Profiling is usually better disabled during HPO so search budget is spent on
# more candidate trials instead of extra analysis branches.
CAPTURE_MANUAL_GRAD_METRICS = True  # Whether to collect manual-gradient profiling stats.
CAPTURE_AUTOGRAD_COMPARISON = True  # Whether to run extra autograd comparison branches.

EPOCHS = 20  # Full epoch budget used by a complete trial.
CSV_NOTE = f"{MODEL} {LEARNING_MODE} FF-STDP {HIDDEN_LAYER_UPDATE_MODE}"  # Free-form note written into the summary CSV.

# Search settings shared by all strategies.
SEARCH_STRATEGY = "bayes"  # "grid", "random", "successive_halving", or "bayes"
OPTIMIZE_METRIC = "val_acc_best"  # Main ranking metric, e.g. "val_acc_best" or "test_acc"
RANDOM_SEED = 42  # Seed for random sampling / warmup candidate order.

# Random-search settings.
RANDOM_SEARCH_TRIALS = 16  # Number of randomly sampled candidates to evaluate.

# Bayesian-search settings.
# Bayesian search still works on the discrete candidate pool built from SEARCH_SPACE.
BAYES_CANDIDATE_POOL_SIZE = None  # Optional pre-sampling size for the candidate pool. None -> use the full pool.
BAYES_INIT_RANDOM_TRIALS = 4  # Number of random warmup trials before fitting the surrogate model.
BAYES_MAX_TRIALS = 50  # Total number of trials the Bayesian search is allowed to run.
BAYES_ACQUISITION = "ucb"  # Acquisition function: "ucb" (more exploration) or "ei" (more exploit/explore balance).
BAYES_UCB_BETA = 2.0  # Exploration strength for UCB. Larger -> more willing to try uncertain candidates.
BAYES_EI_XI = 0.01  # Improvement margin for EI. Larger -> asks EI to seek clearer gains over the current best.
BAYES_KERNEL_LENGTH_SCALE = 1.0  # Smoothness scale of the RBF kernel in the Gaussian-process surrogate.
BAYES_NOISE = 1e-6  # Small numerical jitter / observation noise added to stabilize GP solving.

# Successive-halving settings.
SUCCESSIVE_HALVING_INITIAL_CANDIDATES = None  # Optional pre-sampling size for the initial candidate set. None -> use the full set.
SUCCESSIVE_HALVING_INITIAL_EPOCHS = 50  # Epoch budget for the first screening stage.
SUCCESSIVE_HALVING_REDUCTION_FACTOR = 2  # After each stage keep about 1 / factor of the best candidates.

# Console / report settings.
TOP_K_TO_PRINT = 5  # Number of top trials printed to console after the search ends.

if "conv_cfg" not in SEARCH_SPACE and "cov_cfg" in SEARCH_SPACE:
    SEARCH_SPACE["conv_cfg"] = SEARCH_SPACE["cov_cfg"]


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _parse_runtime_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one FF-SNN hyperparameter search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["MLP", "CNN", "VGG6", "VGG8", "VGG11", "ResNet"])
    parser.add_argument("--dataset", choices=["MNIST", "N-MNIST", "NMNIST", "FashionMNIST", "CIFAR10", "DVS128Gesture"])
    parser.add_argument("--learning-mode", choices=["unsupervised", "supervised"])
    parser.add_argument("--hidden-layer-update-mode", choices=["autograd", "manual"])
    parser.add_argument("--neg-sample-strategy")
    parser.add_argument("--goodness-strategy")
    parser.add_argument("--hidden-loss-strategy")
    parser.add_argument("--device", help="Explicit device forwarded to ff-snn.py, e.g. cuda:0 or cpu.")
    parser.add_argument("--workers", type=int, help="DataLoader worker count forwarded to ff-snn.py.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--out-dir", help="HPO output root; training runs are stored below this directory.")
    parser.add_argument("--search-strategy", choices=["grid", "random", "successive_halving", "bayes"])
    parser.add_argument("--optimize-metric")
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--random-search-trials", type=int)
    parser.add_argument("--bayes-candidate-pool-size", type=int)
    parser.add_argument("--bayes-init-random-trials", type=int)
    parser.add_argument("--bayes-max-trials", type=int)
    parser.add_argument("--bayes-acquisition", choices=["ucb", "ei"])
    parser.add_argument("--bayes-ucb-beta", type=float)
    parser.add_argument("--bayes-ei-xi", type=float)
    parser.add_argument("--successive-halving-initial-candidates", type=int)
    parser.add_argument("--successive-halving-initial-epochs", type=int)
    parser.add_argument("--successive-halving-reduction-factor", type=int)
    parser.add_argument("--capture-manual-grad-metrics", type=_parse_bool)
    parser.add_argument("--capture-autograd-comparison", type=_parse_bool)
    parser.add_argument("--csv-note")
    parser.add_argument("--top-k-to-print", type=int)
    parser.add_argument(
        "--search-space-json",
        help="JSON object used to update SEARCH_SPACE for this run.",
    )
    parser.add_argument(
        "--search-space-file",
        help="Path to a JSON object used to update SEARCH_SPACE for this run.",
    )
    return parser.parse_args(argv)


def _load_search_space_override(args: argparse.Namespace) -> dict | None:
    search_space = None
    if args.search_space_file:
        with Path(args.search_space_file).open("r", encoding="utf-8") as sf:
            search_space = json.load(sf)
    if args.search_space_json:
        inline_space = json.loads(args.search_space_json)
        search_space = inline_space if search_space is None else {**search_space, **inline_space}
    return search_space


def _apply_runtime_overrides(args: argparse.Namespace) -> None:
    global MODEL, DATASET, LEARNING_MODE, HIDDEN_LAYER_UPDATE_MODE
    global NEG_SAMPLE_STRATEGY, GOODNESS_STRATEGY, HIDDEN_LOSS_STRATEGY
    global DEVICE, DATA_LOADER_WORKERS, CAPTURE_MANUAL_GRAD_METRICS
    global CAPTURE_AUTOGRAD_COMPARISON, EPOCHS, CSV_NOTE, SEARCH_STRATEGY
    global OPTIMIZE_METRIC, RANDOM_SEED, RANDOM_SEARCH_TRIALS
    global BAYES_CANDIDATE_POOL_SIZE, BAYES_INIT_RANDOM_TRIALS
    global BAYES_MAX_TRIALS, BAYES_ACQUISITION, BAYES_UCB_BETA, BAYES_EI_XI
    global SUCCESSIVE_HALVING_INITIAL_CANDIDATES
    global SUCCESSIVE_HALVING_INITIAL_EPOCHS
    global SUCCESSIVE_HALVING_REDUCTION_FACTOR, TOP_K_TO_PRINT
    global OUT_DIR, SEARCH_SPACE

    if args.model is not None:
        MODEL = args.model
    if args.dataset is not None:
        DATASET = args.dataset
    if args.learning_mode is not None:
        LEARNING_MODE = args.learning_mode
    if args.hidden_layer_update_mode is not None:
        HIDDEN_LAYER_UPDATE_MODE = args.hidden_layer_update_mode
    if args.neg_sample_strategy is not None:
        NEG_SAMPLE_STRATEGY = args.neg_sample_strategy
    if args.goodness_strategy is not None:
        GOODNESS_STRATEGY = args.goodness_strategy
    if args.hidden_loss_strategy is not None:
        HIDDEN_LOSS_STRATEGY = args.hidden_loss_strategy
    if args.device is not None:
        DEVICE = args.device
    if args.workers is not None:
        DATA_LOADER_WORKERS = args.workers
    if args.capture_manual_grad_metrics is not None:
        CAPTURE_MANUAL_GRAD_METRICS = args.capture_manual_grad_metrics
    if args.capture_autograd_comparison is not None:
        CAPTURE_AUTOGRAD_COMPARISON = args.capture_autograd_comparison
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)
    if args.search_strategy is not None:
        SEARCH_STRATEGY = args.search_strategy
    if args.optimize_metric is not None:
        OPTIMIZE_METRIC = args.optimize_metric
    if args.random_seed is not None:
        RANDOM_SEED = args.random_seed
    if args.random_search_trials is not None:
        RANDOM_SEARCH_TRIALS = args.random_search_trials
    if args.bayes_candidate_pool_size is not None:
        BAYES_CANDIDATE_POOL_SIZE = args.bayes_candidate_pool_size
    if args.bayes_init_random_trials is not None:
        BAYES_INIT_RANDOM_TRIALS = args.bayes_init_random_trials
    if args.bayes_max_trials is not None:
        BAYES_MAX_TRIALS = args.bayes_max_trials
    if args.bayes_acquisition is not None:
        BAYES_ACQUISITION = args.bayes_acquisition
    if args.bayes_ucb_beta is not None:
        BAYES_UCB_BETA = args.bayes_ucb_beta
    if args.bayes_ei_xi is not None:
        BAYES_EI_XI = args.bayes_ei_xi
    if args.successive_halving_initial_candidates is not None:
        SUCCESSIVE_HALVING_INITIAL_CANDIDATES = args.successive_halving_initial_candidates
    if args.successive_halving_initial_epochs is not None:
        SUCCESSIVE_HALVING_INITIAL_EPOCHS = args.successive_halving_initial_epochs
    if args.successive_halving_reduction_factor is not None:
        SUCCESSIVE_HALVING_REDUCTION_FACTOR = args.successive_halving_reduction_factor
    if args.top_k_to_print is not None:
        TOP_K_TO_PRINT = args.top_k_to_print

    search_space_override = _load_search_space_override(args)
    if search_space_override:
        SEARCH_SPACE = {**SEARCH_SPACE, **search_space_override}
    if "conv_cfg" not in SEARCH_SPACE and "cov_cfg" in SEARCH_SPACE:
        SEARCH_SPACE["conv_cfg"] = SEARCH_SPACE["cov_cfg"]

    if args.csv_note is not None:
        CSV_NOTE = args.csv_note
    else:
        CSV_NOTE = f"{MODEL} {DATASET} {LEARNING_MODE} FF-STDP {HIDDEN_LAYER_UPDATE_MODE}"


def _dims_to_str(dims: list[int]) -> str:
    return "[" + ",".join(str(d) for d in dims) + "]"


def _uses_dims_search(model_name: str) -> bool:
    return model_name == "MLP"


def _uses_conv_cfg_search(model_name: str) -> bool:
    return model_name == "CNN"


def _is_supported_model(model_name: str) -> bool:
    return model_name == "MLP" or is_cnn_family_model(model_name)


def _search_keys() -> list[str]:
    keys = ["loss_threshold", "v_threshold", "b", "T", "lr"]
    if _uses_dims_search(MODEL):
        keys.append("dims")
    elif _uses_conv_cfg_search(MODEL):
        keys.append("conv_cfg")
    elif not _is_supported_model(MODEL):
        raise ValueError(f"Unsupported MODEL={MODEL}")
    return keys


def _iter_grid(space: dict) -> list[dict]:
    keys = _search_keys()
    values = [space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _sample_candidates(
    candidates: list[dict],
    sample_size: int | None,
    rng: random.Random,
) -> list[dict]:
    if sample_size is None or sample_size >= len(candidates):
        return list(candidates)
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, but got {sample_size}")
    sampled_indices = sorted(rng.sample(range(len(candidates)), k=sample_size))
    return [candidates[idx] for idx in sampled_indices]


def _prepare_candidates() -> list[dict]:
    all_candidates = list(_iter_grid(SEARCH_SPACE))
    rng = random.Random(RANDOM_SEED)

    if SEARCH_STRATEGY == "grid":
        selected = all_candidates
    elif SEARCH_STRATEGY == "random":
        selected = _sample_candidates(all_candidates, RANDOM_SEARCH_TRIALS, rng)
    elif SEARCH_STRATEGY == "successive_halving":
        selected = _sample_candidates(
            all_candidates,
            SUCCESSIVE_HALVING_INITIAL_CANDIDATES,
            rng,
        )
    elif SEARCH_STRATEGY == "bayes":
        selected = _sample_candidates(
            all_candidates,
            BAYES_CANDIDATE_POOL_SIZE,
            rng,
        )
    else:
        raise ValueError(f"Unsupported SEARCH_STRATEGY={SEARCH_STRATEGY}")

    return [
        {
            "candidate_id": f"cand_{idx + 1:04d}",
            "params": params,
        }
        for idx, params in enumerate(selected)
    ]


def _build_cmd(params: dict, *, epoch_budget: int) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "-model",
        MODEL,
        "-epochs",
        str(epoch_budget),
        "-out-dir",
        str(OUT_DIR),
        "-j",
        str(DATA_LOADER_WORKERS),
        "-loss_threshold",
        str(params["loss_threshold"]),
        "-v_threshold",
        str(params["v_threshold"]),
        "-b",
        str(params["b"]),
        "-T",
        str(params["T"]),
        "-lr",
        str(params["lr"]),
    ]
    if DATASET:
        cmd += ["-dataset", DATASET]
    if DEVICE:
        cmd += ["-device", DEVICE]
    cmd += ["-learning_mode", LEARNING_MODE]
    cmd += ["-hidden_layer_update_mode", HIDDEN_LAYER_UPDATE_MODE]
    cmd += ["-neg_sample_strategy", NEG_SAMPLE_STRATEGY]
    cmd += ["-goodness_strategy", GOODNESS_STRATEGY]
    cmd += ["-hidden_loss_strategy", HIDDEN_LOSS_STRATEGY]
    cmd += [
        "-capture_manual_grad_metrics"
        if CAPTURE_MANUAL_GRAD_METRICS
        else "-no-capture_manual_grad_metrics"
    ]
    cmd += [
        "-capture_autograd_comparison"
        if CAPTURE_AUTOGRAD_COMPARISON
        else "-no-capture_autograd_comparison"
    ]
    if _uses_dims_search(MODEL):
        cmd += ["-dims"] + [str(v) for v in params["dims"]]
    elif _uses_conv_cfg_search(MODEL):
        cmd += ["-conv_cfg", str(params["conv_cfg"])]
    return cmd


def _is_numeric_search_axis(options: list) -> bool:
    return all(isinstance(option, numbers.Real) for option in options)


def _encode_axis_value(key: str, value) -> list[float]:
    options = list(SEARCH_SPACE[key])
    if not options:
        return []
    if _is_numeric_search_axis(options):
        if len(options) == 1:
            return [0.0]
        min_value = float(min(options))
        max_value = float(max(options))
        if math.isclose(max_value, min_value):
            return [0.0]
        normalized = (float(value) - min_value) / (max_value - min_value)
        return [normalized]

    encoded = [0.0] * len(options)
    for idx, option in enumerate(options):
        if option == value:
            encoded[idx] = 1.0
            return encoded
    raise ValueError(f"Value {value!r} is not part of SEARCH_SPACE[{key!r}]")


def _candidate_feature_vector(params: dict) -> np.ndarray:
    features: list[float] = []
    for key in _search_keys():
        features.extend(_encode_axis_value(key, params[key]))
    return np.asarray(features, dtype=np.float64)


def _build_feature_matrix(candidates: list[dict]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(
        [_candidate_feature_vector(candidate["params"]) for candidate in candidates],
        axis=0,
    )


def _find_latest_metrics(root: Path, since_ts: float) -> Path | None:
    latest_path = None
    latest_mtime = -1.0
    for path in root.rglob("metrics.json"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= since_ts and mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    return latest_path


def _wait_for_metrics(root: Path, since_ts: float, timeout_s: int = 60) -> Path | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        path = _find_latest_metrics(root, since_ts)
        if path is not None:
            return path
        time.sleep(1)
    return None


def _resolve_summary_path(base_path: Path, fieldnames: list[str]) -> tuple[Path, bool]:
    if not base_path.exists() or base_path.stat().st_size == 0:
        return base_path, True

    with base_path.open("r", newline="", encoding="utf-8") as rf:
        reader = csv.reader(rf)
        existing_header = next(reader, [])

    if existing_header == fieldnames:
        return base_path, False

    stem = base_path.stem
    suffix = base_path.suffix
    idx = 2
    while True:
        candidate = base_path.with_name(f"{stem}.schema{idx}{suffix}")
        if not candidate.exists() or candidate.stat().st_size == 0:
            return candidate, True
        with candidate.open("r", newline="", encoding="utf-8") as rf:
            reader = csv.reader(rf)
            header = next(reader, [])
        if header == fieldnames:
            return candidate, False
        idx += 1


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _resolve_score(metrics: dict) -> tuple[str | None, float | None]:
    candidate_keys = [
        OPTIMIZE_METRIC,
        "val_acc_best",
        "test_acc",
        "train_acc_best",
    ]
    seen = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        value = _safe_float(metrics.get(key))
        if value is not None:
            return key, value
    return None, None


def _trial_sort_key(record: dict) -> tuple[float, float, float, float]:
    row = record["row"]
    score = record["score_value"]
    val_best = _safe_float(row.get("val_acc_best"))
    test_acc = _safe_float(row.get("test_acc"))
    loss_mean = _safe_float(row.get("last_epoch_loss_mean"))
    return (
        score if score is not None else float("-inf"),
        val_best if val_best is not None else float("-inf"),
        test_acc if test_acc is not None else float("-inf"),
        -loss_mean if loss_mean is not None else float("-inf"),
    )


def _build_halving_schedule(max_epochs: int) -> list[int]:
    if max_epochs <= 0:
        raise ValueError(f"EPOCHS must be positive, but got {max_epochs}")
    if SUCCESSIVE_HALVING_REDUCTION_FACTOR <= 1:
        raise ValueError(
            "SUCCESSIVE_HALVING_REDUCTION_FACTOR must be > 1 for halving search."
        )

    current = max(1, min(SUCCESSIVE_HALVING_INITIAL_EPOCHS, max_epochs))
    schedule = []
    while current < max_epochs:
        schedule.append(current)
        next_epoch = min(
            max_epochs,
            max(current + 1, current * SUCCESSIVE_HALVING_REDUCTION_FACTOR),
        )
        if next_epoch == current:
            break
        current = next_epoch

    if not schedule or schedule[-1] != max_epochs:
        schedule.append(max_epochs)
    return schedule


def _rbf_kernel(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    length_scale: float,
) -> np.ndarray:
    if x_a.size == 0 or x_b.size == 0:
        return np.zeros((x_a.shape[0], x_b.shape[0]), dtype=np.float64)
    diff = x_a[:, None, :] - x_b[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    safe_length_scale = max(float(length_scale), 1e-8)
    return np.exp(-0.5 * sq_dist / (safe_length_scale * safe_length_scale))


def _standardize_targets(targets: list[float]) -> tuple[np.ndarray, float, float]:
    y = np.asarray(targets, dtype=np.float64)
    mean = float(y.mean())
    std = float(y.std())
    if std < 1e-8:
        std = 1.0
    return (y - mean) / std, mean, std


def _gp_predict(
    train_x: np.ndarray,
    train_y: list[float],
    query_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if train_x.shape[0] == 0:
        query_count = query_x.shape[0]
        return (
            np.zeros(query_count, dtype=np.float64),
            np.ones(query_count, dtype=np.float64),
        )

    y_standardized, y_mean, y_std = _standardize_targets(train_y)
    kernel_xx = _rbf_kernel(
        train_x,
        train_x,
        length_scale=BAYES_KERNEL_LENGTH_SCALE,
    )
    kernel_xx = kernel_xx + BAYES_NOISE * np.eye(train_x.shape[0], dtype=np.float64)
    kernel_xs = _rbf_kernel(
        train_x,
        query_x,
        length_scale=BAYES_KERNEL_LENGTH_SCALE,
    )
    kernel_ss_diag = np.ones(query_x.shape[0], dtype=np.float64)

    try:
        alpha = np.linalg.solve(kernel_xx, y_standardized)
        solved_kernel = np.linalg.solve(kernel_xx, kernel_xs)
    except np.linalg.LinAlgError:
        kernel_xx = kernel_xx + 1e-4 * np.eye(train_x.shape[0], dtype=np.float64)
        alpha = np.linalg.solve(kernel_xx, y_standardized)
        solved_kernel = np.linalg.solve(kernel_xx, kernel_xs)

    pred_mean_standardized = kernel_xs.T @ alpha
    pred_var_standardized = kernel_ss_diag - np.sum(kernel_xs * solved_kernel, axis=0)
    pred_var_standardized = np.maximum(pred_var_standardized, BAYES_NOISE)

    pred_mean = pred_mean_standardized * y_std + y_mean
    pred_std = np.sqrt(pred_var_standardized) * y_std
    return pred_mean, pred_std


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    erf_vectorized = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vectorized(x / math.sqrt(2.0)))


def _bayes_acquisition(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    best_score: float,
) -> np.ndarray:
    safe_std = np.maximum(pred_std, 1e-8)
    if BAYES_ACQUISITION == "ucb":
        return pred_mean + BAYES_UCB_BETA * safe_std
    if BAYES_ACQUISITION == "ei":
        improvement = pred_mean - best_score - BAYES_EI_XI
        z = improvement / safe_std
        return improvement * _normal_cdf(z) + safe_std * _normal_pdf(z)
    raise ValueError(f"Unsupported BAYES_ACQUISITION={BAYES_ACQUISITION}")


def _base_row(
    *,
    params: dict,
    candidate_id: str,
    candidate_index: int,
    stage_index: int,
    stage_epochs: int,
    stage_candidates: int,
) -> dict:
    row = {
        "search_strategy": SEARCH_STRATEGY,
        "optimize_metric": OPTIMIZE_METRIC,
        "score_metric_used": "",
        "score_value": "",
        "selection_metric": "",
        "acquisition_value": "",
        "surrogate_mean": "",
        "surrogate_std": "",
        "candidate_id": candidate_id,
        "candidate_index": candidate_index,
        "stage_index": stage_index,
        "stage_epochs": stage_epochs,
        "stage_rank": "",
        "stage_candidates": stage_candidates,
        "promoted": False,
        "learning_mode": LEARNING_MODE,
        "hidden_layer_update_mode": HIDDEN_LAYER_UPDATE_MODE,
        "neg_sample_strategy": NEG_SAMPLE_STRATEGY,
        "goodness_strategy": GOODNESS_STRATEGY,
        "hidden_loss_strategy": HIDDEN_LOSS_STRATEGY,
        "capture_manual_grad_metrics": CAPTURE_MANUAL_GRAD_METRICS,
        "capture_autograd_comparison": CAPTURE_AUTOGRAD_COMPARISON,
        "dataset": DATASET,
        "model": MODEL,
        "note": CSV_NOTE,
        "max_epochs": EPOCHS,
        "epoch": stage_epochs,
        "loss_threshold": params["loss_threshold"],
        "v_threshold": params["v_threshold"],
        "dims": "",
        "conv_cfg": "",
        "b": params["b"],
        "T": params["T"],
        "lr": params["lr"],
        "train_acc_last": "",
        "train_acc_best": "",
        "val_acc_last": "",
        "val_acc_best": "",
        "test_acc": "",
        "test_duration_s": "",
        "last_epoch_loss_mean": "",
        "last_epoch_goodness_pos_mean": "",
        "last_epoch_goodness_neg_mean": "",
        "last_epoch_firing_pos_mean": "",
        "last_epoch_firing_neg_mean": "",
        "train_gpu_mem_alloc_mean_mb": "",
        "train_gpu_mem_reserved_mean_mb": "",
        "bp_gpu_mem_peak_alloc_mean_mb": "",
        "bp_gpu_mem_peak_reserved_mean_mb": "",
        "bp_gpu_mem_peak_alloc_max_mb": "",
        "bp_gpu_mem_peak_reserved_max_mb": "",
        "bp_only_gpu_mem_peak_alloc_mean_mb": "",
        "bp_only_gpu_mem_peak_reserved_mean_mb": "",
        "bp_only_gpu_mem_peak_alloc_max_mb": "",
        "bp_only_gpu_mem_peak_reserved_max_mb": "",
        "manual_grad_peak_alloc_mean_mb": "",
        "manual_grad_peak_reserved_mean_mb": "",
        "manual_grad_peak_alloc_max_mb": "",
        "manual_grad_peak_reserved_max_mb": "",
        "manual_grad_time_mean_ms": "",
        "manual_grad_peak_alloc_per_sample_kb": "",
        "manual_grad_peak_reserved_per_sample_kb": "",
        "manual_grad_time_per_sample_us": "",
        "manual_grad_ops_est_total": "",
        "manual_grad_ops_est_gops": "",
        "manual_grad_ops_est_per_sample": "",
        "manual_grad_ops_est_gops_per_s": "",
        "manual_grad_samples_per_s": "",
        "autograd_cmp_peak_alloc_mean_mb": "",
        "autograd_cmp_peak_reserved_mean_mb": "",
        "autograd_cmp_peak_alloc_max_mb": "",
        "autograd_cmp_peak_reserved_max_mb": "",
        "autograd_cmp_time_mean_ms": "",
        "autograd_cmp_peak_alloc_per_sample_kb": "",
        "autograd_cmp_peak_reserved_per_sample_kb": "",
        "autograd_cmp_time_per_sample_us": "",
        "autograd_cmp_samples_per_s": "",
        "manual_vs_autograd_alloc_reduction_pct": "",
        "manual_vs_autograd_time_reduction_pct": "",
        "manual_vs_autograd_alloc_per_sample_reduction_pct": "",
        "manual_vs_autograd_time_per_sample_reduction_pct": "",
        "manual_vs_autograd_throughput_gain_pct": "",
        "run_dir": "",
        "status": "ok",
        "error": "",
    }
    if _uses_dims_search(MODEL):
        row["dims"] = _dims_to_str(params["dims"])
    elif _uses_conv_cfg_search(MODEL):
        row["conv_cfg"] = str(params["conv_cfg"])
    return row


def _run_trial(
    *,
    candidate: dict,
    candidate_index: int,
    stage_index: int,
    stage_epochs: int,
    stage_candidates: int,
    selection_info: dict | None = None,
) -> dict:
    params = candidate["params"]
    row = _base_row(
        params=params,
        candidate_id=candidate["candidate_id"],
        candidate_index=candidate_index,
        stage_index=stage_index,
        stage_epochs=stage_epochs,
        stage_candidates=stage_candidates,
    )
    if selection_info:
        row.update(selection_info)

    started = time.time()
    cmd = _build_cmd(params, epoch_budget=stage_epochs)
    print(
        f"[HPO] stage={stage_index} epochs={stage_epochs} "
        f"candidate={candidate['candidate_id']} params={params}"
    )
    result = subprocess.run(cmd)
    if result.returncode != 0:
        row["status"] = "failed"
        row["error"] = f"train_exit_code={result.returncode}"
        return {
            "candidate": candidate,
            "row": row,
            "score_value": None,
        }

    metrics_path = _wait_for_metrics(OUT_DIR, started)
    if metrics_path is None:
        row["status"] = "failed"
        row["error"] = "metrics_not_found"
        return {
            "candidate": candidate,
            "row": row,
            "score_value": None,
        }

    with metrics_path.open("r", encoding="utf-8") as mf:
        metrics = json.load(mf)

    score_metric_used, score_value = _resolve_score(metrics)
    row.update(metrics)
    row["score_metric_used"] = score_metric_used or ""
    row["score_value"] = score_value if score_value is not None else ""
    row["run_dir"] = str(metrics_path.parent)
    return {
        "candidate": candidate,
        "row": row,
        "score_value": score_value,
    }


def _write_stage_rows(
    writer: csv.DictWriter,
    records: list[dict],
    *,
    promoted_ids: set[str],
) -> None:
    ranked_successes = sorted(
        [
            record
            for record in records
            if record["row"]["status"] == "ok" and record["score_value"] is not None
        ],
        key=_trial_sort_key,
        reverse=True,
    )
    rank_by_id = {
        record["candidate"]["candidate_id"]: rank
        for rank, record in enumerate(ranked_successes, start=1)
    }

    for record in records:
        candidate_id = record["candidate"]["candidate_id"]
        record["row"]["stage_rank"] = rank_by_id.get(candidate_id, "")
        record["row"]["promoted"] = candidate_id in promoted_ids
        writer.writerow(record["row"])


def _flush_csv(csv_file) -> None:
    csv_file.flush()
    os.fsync(csv_file.fileno())


def _write_incremental_record(
    writer: csv.DictWriter,
    csv_file,
    record: dict,
) -> None:
    writer.writerow(record["row"])
    _flush_csv(csv_file)


def _rank_successful_records(records: list[dict]) -> list[dict]:
    return sorted(
        [
            record
            for record in records
            if record["row"]["status"] == "ok" and record["score_value"] is not None
        ],
        key=_trial_sort_key,
        reverse=True,
    )


def _write_best_json(summary_path: Path, ranked_records: list[dict]) -> None:
    if not ranked_records:
        return
    best_path = summary_path.with_suffix(".best.json")
    with best_path.open("w", encoding="utf-8") as bf:
        json.dump(ranked_records[0]["row"], bf, ensure_ascii=False, indent=2)


def _run_single_stage_search(
    writer: csv.DictWriter,
    csv_file,
    summary_path: Path,
) -> list[dict]:
    candidates = _prepare_candidates()
    records = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        record = _run_trial(
            candidate=candidate,
            candidate_index=candidate_index,
            stage_index=1,
            stage_epochs=EPOCHS,
            stage_candidates=len(candidates),
        )
        records.append(record)
        _write_incremental_record(writer, csv_file, record)
        _write_best_json(summary_path, _rank_successful_records(records))

    return _rank_successful_records(records)


def _run_bayesian_search(
    writer: csv.DictWriter,
    csv_file,
    summary_path: Path,
) -> list[dict]:
    if BAYES_INIT_RANDOM_TRIALS <= 0:
        raise ValueError("BAYES_INIT_RANDOM_TRIALS must be positive.")
    if BAYES_MAX_TRIALS <= 0:
        raise ValueError("BAYES_MAX_TRIALS must be positive.")

    rng = random.Random(RANDOM_SEED)
    candidates = _prepare_candidates()
    if not candidates:
        return []

    max_trials = min(BAYES_MAX_TRIALS, len(candidates))
    feature_matrix = _build_feature_matrix(candidates)
    candidate_index_map = {
        candidate["candidate_id"]: idx for idx, candidate in enumerate(candidates)
    }
    candidate_by_id = {
        candidate["candidate_id"]: candidate for candidate in candidates
    }

    evaluated_records: list[dict] = []
    evaluated_ids: list[str] = []
    successful_ids: list[str] = []
    successful_scores: list[float] = []

    while len(evaluated_records) < max_trials:
        unevaluated_ids = [
            candidate["candidate_id"]
            for candidate in candidates
            if candidate["candidate_id"] not in evaluated_ids
        ]
        if not unevaluated_ids:
            break

        use_random_warmup = len(successful_ids) < BAYES_INIT_RANDOM_TRIALS
        if use_random_warmup:
            next_candidate_id = rng.choice(unevaluated_ids)
            selection_info = {
                "selection_metric": "random_warmup",
                "acquisition_value": "",
                "surrogate_mean": "",
                "surrogate_std": "",
            }
        else:
            train_indices = [candidate_index_map[candidate_id] for candidate_id in successful_ids]
            query_indices = [
                candidate_index_map[candidate_id] for candidate_id in unevaluated_ids
            ]
            train_x = feature_matrix[train_indices]
            query_x = feature_matrix[query_indices]
            pred_mean, pred_std = _gp_predict(train_x, successful_scores, query_x)
            best_score = max(successful_scores)
            acquisition_values = _bayes_acquisition(pred_mean, pred_std, best_score)
            best_query_idx = int(np.argmax(acquisition_values))
            next_candidate_id = unevaluated_ids[best_query_idx]
            selection_info = {
                "selection_metric": BAYES_ACQUISITION,
                "acquisition_value": float(acquisition_values[best_query_idx]),
                "surrogate_mean": float(pred_mean[best_query_idx]),
                "surrogate_std": float(pred_std[best_query_idx]),
            }

        next_candidate = candidate_by_id[next_candidate_id]
        trial_index = len(evaluated_records) + 1
        record = _run_trial(
            candidate=next_candidate,
            candidate_index=trial_index,
            stage_index=trial_index,
            stage_epochs=EPOCHS,
            stage_candidates=len(unevaluated_ids),
            selection_info=selection_info,
        )
        evaluated_records.append(record)
        evaluated_ids.append(next_candidate_id)

        if record["row"]["status"] == "ok" and record["score_value"] is not None:
            successful_ids.append(next_candidate_id)
            successful_scores.append(float(record["score_value"]))

        _write_incremental_record(writer, csv_file, record)
        _write_best_json(summary_path, _rank_successful_records(evaluated_records))

    return _rank_successful_records(evaluated_records)


def _run_successive_halving_search(
    writer: csv.DictWriter,
    csv_file,
    summary_path: Path,
) -> list[dict]:
    schedule = _build_halving_schedule(EPOCHS)
    remaining_candidates = _prepare_candidates()
    final_ranked = []
    all_successful_records = []

    for stage_index, stage_epochs in enumerate(schedule, start=1):
        if not remaining_candidates:
            break

        print(
            f"[HPO] successive_halving stage {stage_index}/{len(schedule)} | "
            f"epochs={stage_epochs} | candidates={len(remaining_candidates)}"
        )
        stage_records = []
        for candidate_index, candidate in enumerate(remaining_candidates, start=1):
            record = _run_trial(
                candidate=candidate,
                candidate_index=candidate_index,
                stage_index=stage_index,
                stage_epochs=stage_epochs,
                stage_candidates=len(remaining_candidates),
            )
            stage_records.append(record)
            _write_incremental_record(writer, csv_file, record)
            all_successful_records.extend(
                [record]
                if record["row"]["status"] == "ok" and record["score_value"] is not None
                else []
            )
            _write_best_json(summary_path, _rank_successful_records(all_successful_records))

        ranked_stage_records = _rank_successful_records(stage_records)
        final_ranked = ranked_stage_records

        if stage_index == len(schedule) or not ranked_stage_records:
            break

        keep_count = max(
            1,
            math.ceil(len(ranked_stage_records) / SUCCESSIVE_HALVING_REDUCTION_FACTOR),
        )
        promoted_records = ranked_stage_records[:keep_count]
        promoted_ids = {
            record["candidate"]["candidate_id"] for record in promoted_records
        }
        for record in stage_records:
            record["row"]["promoted"] = record["candidate"]["candidate_id"] in promoted_ids
        remaining_candidates = [
            record["candidate"] for record in promoted_records
        ]

    if final_ranked:
        return final_ranked
    return sorted(
        all_successful_records,
        key=_trial_sort_key,
        reverse=True,
    )


def _write_best_result(summary_path: Path, ranked_records: list[dict]) -> None:
    if not ranked_records:
        print("[HPO] No successful trial found.")
        return

    best_record = ranked_records[0]["row"]
    best_path = summary_path.with_suffix(".best.json")
    with best_path.open("w", encoding="utf-8") as bf:
        json.dump(best_record, bf, ensure_ascii=False, indent=2)

    print(f"[HPO] Best trial saved to: {best_path}")
    print(
        "[HPO] Best result: "
        f"candidate={best_record['candidate_id']} "
        f"score={best_record['score_value']} "
        f"metric={best_record['score_metric_used']} "
        f"run_dir={best_record['run_dir']}"
    )
    print("[HPO] Top trials:")
    for rank, record in enumerate(ranked_records[:TOP_K_TO_PRINT], start=1):
        row = record["row"]
        print(
            f"  #{rank}: candidate={row['candidate_id']} "
            f"score={row['score_value']} "
            f"val_acc_best={row.get('val_acc_best', '')} "
            f"test_acc={row.get('test_acc', '')} "
            f"epochs={row['epoch']}"
        )


def main() -> None:
    runtime_args = _parse_runtime_args()
    _apply_runtime_overrides(runtime_args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_name = (
        f"{LEARNING_MODE}-{HIDDEN_LAYER_UPDATE_MODE}-{DATASET}-{MODEL}-"
        f"{SEARCH_STRATEGY}.csv"
    )
    summary_path = OUT_DIR / summary_name

    fieldnames = [
        "search_strategy",
        "optimize_metric",
        "score_metric_used",
        "score_value",
        "selection_metric",
        "acquisition_value",
        "surrogate_mean",
        "surrogate_std",
        "candidate_id",
        "candidate_index",
        "stage_index",
        "stage_epochs",
        "stage_rank",
        "stage_candidates",
        "promoted",
        "learning_mode",
        "hidden_layer_update_mode",
        "neg_sample_strategy",
        "goodness_strategy",
        "hidden_loss_strategy",
        "capture_manual_grad_metrics",
        "capture_autograd_comparison",
        "dataset",
        "model",
        "note",
        "max_epochs",
        "epoch",
        "loss_threshold",
        "v_threshold",
        "dims",
        "conv_cfg",
        "b",
        "T",
        "lr",
        "train_acc_last",
        "train_acc_best",
        "val_acc_last",
        "val_acc_best",
        "test_acc",
        "test_duration_s",
        "last_epoch_loss_mean",
        "last_epoch_goodness_pos_mean",
        "last_epoch_goodness_neg_mean",
        "last_epoch_firing_pos_mean",
        "last_epoch_firing_neg_mean",
        "train_gpu_mem_alloc_mean_mb",
        "train_gpu_mem_reserved_mean_mb",
        "bp_gpu_mem_peak_alloc_mean_mb",
        "bp_gpu_mem_peak_reserved_mean_mb",
        "bp_gpu_mem_peak_alloc_max_mb",
        "bp_gpu_mem_peak_reserved_max_mb",
        "bp_only_gpu_mem_peak_alloc_mean_mb",
        "bp_only_gpu_mem_peak_reserved_mean_mb",
        "bp_only_gpu_mem_peak_alloc_max_mb",
        "bp_only_gpu_mem_peak_reserved_max_mb",
        "manual_grad_peak_alloc_mean_mb",
        "manual_grad_peak_reserved_mean_mb",
        "manual_grad_peak_alloc_max_mb",
        "manual_grad_peak_reserved_max_mb",
        "manual_grad_time_mean_ms",
        "manual_grad_peak_alloc_per_sample_kb",
        "manual_grad_peak_reserved_per_sample_kb",
        "manual_grad_time_per_sample_us",
        "manual_grad_ops_est_total",
        "manual_grad_ops_est_gops",
        "manual_grad_ops_est_per_sample",
        "manual_grad_ops_est_gops_per_s",
        "manual_grad_samples_per_s",
        "autograd_cmp_peak_alloc_mean_mb",
        "autograd_cmp_peak_reserved_mean_mb",
        "autograd_cmp_peak_alloc_max_mb",
        "autograd_cmp_peak_reserved_max_mb",
        "autograd_cmp_time_mean_ms",
        "autograd_cmp_peak_alloc_per_sample_kb",
        "autograd_cmp_peak_reserved_per_sample_kb",
        "autograd_cmp_time_per_sample_us",
        "autograd_cmp_samples_per_s",
        "manual_vs_autograd_alloc_reduction_pct",
        "manual_vs_autograd_time_reduction_pct",
        "manual_vs_autograd_alloc_per_sample_reduction_pct",
        "manual_vs_autograd_time_per_sample_reduction_pct",
        "manual_vs_autograd_throughput_gain_pct",
        "run_dir",
        "status",
        "error",
    ]

    summary_path, write_header = _resolve_summary_path(summary_path, fieldnames)
    print(f"[HPO] search_strategy={SEARCH_STRATEGY}")
    print(f"[HPO] optimize_metric={OPTIMIZE_METRIC}")
    print(f"[HPO] summary_path={summary_path}")

    with summary_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
            _flush_csv(csv_file)

        if SEARCH_STRATEGY == "successive_halving":
            ranked_records = _run_successive_halving_search(writer, csv_file, summary_path)
        elif SEARCH_STRATEGY == "bayes":
            ranked_records = _run_bayesian_search(writer, csv_file, summary_path)
        else:
            ranked_records = _run_single_stage_search(writer, csv_file, summary_path)

    _write_best_result(summary_path, ranked_records)


if __name__ == "__main__":
    main()
