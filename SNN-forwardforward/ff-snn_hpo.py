"""
Hyperparameter search runner for ff-snn.py.

Edit SEARCH_SPACE below to control the grid.
"""
from __future__ import annotations
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "ff-snn.py"
OUT_DIR = ROOT / "logs" / "opt"

# Edit these lists to define the search grid.
SEARCH_SPACE = {
    "loss_threshold": [1.2],
    "v_threshold": [1.2],
    "b": [512],
    "dims": [
        # [784, 512,512,512,10],
        # [784, 512,512,10],
        # [784, 512, 10],
        [784, 256, 10]
    ],
    # "cov_cfg":[
    #     [
    #         # in_ch, out_ch, k, s, p
    #         (1,  32, 3, 1, 1),
    #         (32, 64, 3, 1, 1),
    #         (64, 128, 3, 1, 1),
    #         (128, 256, 3, 1, 1)
    #     ]
    # ],
    "T": [8],
    "lr": [0.0078125],
}

# Base training settings.
MODEL = "MLP"
DATASET = "MNIST"
LEARNING_MODE = "supervised" # "unsupervised" or "supervised"
HIDDEN_LAYER_UPDATE_MODE = "autograd" # "autograd" or "manual"
CAPTURE_MANUAL_GRAD_METRICS = True
CAPTURE_AUTOGRAD_COMPARISON = True
EPOCHS = 2
# Note written to CSV for experiment traceability.
CSV_NOTE = f"{MODEL} {LEARNING_MODE} FF-STDP {HIDDEN_LAYER_UPDATE_MODE}"

if "conv_cfg" not in SEARCH_SPACE and "cov_cfg" in SEARCH_SPACE:
    SEARCH_SPACE["conv_cfg"] = SEARCH_SPACE["cov_cfg"]


def _dims_to_str(dims: list[int]) -> str:
    return "[" + ",".join(str(d) for d in dims) + "]"


def _iter_grid(space: dict) -> list[dict]:
    keys = ["loss_threshold", "v_threshold", "b", "T", "lr"]
    if MODEL == "MLP":
        keys.append("dims")
    elif MODEL == "CNN":
        keys.append("conv_cfg")
    else:
        raise ValueError(f"Unsupported MODEL={MODEL}")
    values = [space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _build_cmd(params: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "-model",
        MODEL,
        "-epochs",
        str(EPOCHS),
        "-out-dir",
        str(OUT_DIR),
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
    cmd += ["-learning_mode", LEARNING_MODE]
    cmd += ["-hidden_layer_update_mode", HIDDEN_LAYER_UPDATE_MODE]
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
    if MODEL == "MLP":
        cmd += ["-dims"] + [str(v) for v in params["dims"]]
    elif MODEL == "CNN":
        cmd += ["-conv_cfg", str(params["conv_cfg"])]
    return cmd


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_name = f"{LEARNING_MODE}-{HIDDEN_LAYER_UPDATE_MODE}-{DATASET}-{MODEL}.csv"
    summary_path = OUT_DIR / summary_name

    fieldnames = [
        "learning_mode",
        "hidden_layer_update_mode",
        "capture_manual_grad_metrics",
        "capture_autograd_comparison",
        "dataset",
        "model",
        "note",
        "epoch",
        "loss_threshold",
        "v_threshold",
        "dims",
        "conv_cfg",
        "b",
        "T",
        "lr",
        "test_acc",
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
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for params in _iter_grid(SEARCH_SPACE):
            row = {
                "learning_mode": LEARNING_MODE,
                "hidden_layer_update_mode": HIDDEN_LAYER_UPDATE_MODE,
                "capture_manual_grad_metrics": CAPTURE_MANUAL_GRAD_METRICS,
                "capture_autograd_comparison": CAPTURE_AUTOGRAD_COMPARISON,
                "dataset": DATASET,
                "model": MODEL,
                "note": CSV_NOTE,
                "epoch": EPOCHS,
                "loss_threshold": params["loss_threshold"],
                "v_threshold": params["v_threshold"],
                "dims": "",
                "conv_cfg": "",
                "b": params["b"],
                "T": params["T"],
                "lr": params["lr"],
                "status": "ok",
                "error": "",
                "run_dir": "",
            }
            if MODEL == "MLP":
                row["dims"] = _dims_to_str(params["dims"])
            elif MODEL == "CNN":
                row["conv_cfg"] = str(params["conv_cfg"])
            started = time.time()
            cmd = _build_cmd(params)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                row["status"] = "failed"
                row["error"] = f"train_exit_code={result.returncode}"
                writer.writerow(row)
                f.flush()
                continue

            metrics_path = _wait_for_metrics(OUT_DIR, started)
            if metrics_path is None:
                row["status"] = "failed"
                row["error"] = "metrics_not_found"
                writer.writerow(row)
                f.flush()
                continue

            with metrics_path.open("r", encoding="utf-8") as mf:
                metrics = json.load(mf)

            row.update(metrics)
            row["run_dir"] = str(metrics_path.parent)
            writer.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
