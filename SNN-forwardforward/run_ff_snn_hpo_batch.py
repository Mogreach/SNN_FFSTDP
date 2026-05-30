from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
HPO_SCRIPT = ROOT / "ff-snn_hpo.py"
DEFAULT_OUT_DIR = ROOT / "logs" / "opt_batch"


# ---------------------------------------------------------------------------
# Search-space presets
# ---------------------------------------------------------------------------
# Edit the discrete HPO ranges here. The structure is intentionally explicit:
# first choose the model family group, then the dataset, then update the lists.
# This keeps the searchable ranges in one obvious place instead of scattering
# them across helper functions.
SEARCH_SPACE_PRESETS = {
    "MLP": {
        "MNIST": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [1024],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
            "dims": [
                [784, 512, 512, 10],
                [784, 512, 10],
                [784, 256, 10],
            ],
        },
        "FashionMNIST": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [1024],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
            "dims": [
                [784, 512, 512, 10],
                [784, 512, 10],
                [784, 256, 10],
            ],
        },
        "NMNIST": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [512],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
            "dims": [
                [2 * 34 * 34, 512, 512, 10],
                [2 * 34 * 34, 512, 10],
                [2 * 34 * 34, 256, 10],
            ],
        },
        "CIFAR10": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [512],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
            "dims": [
                [3 * 32 * 32, 1024, 512, 10],
                [3 * 32 * 32, 1024, 10],
                [3 * 32 * 32, 512, 10],
            ],
        },
    },
    "CNN": {
        "MNIST": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [1024],
            "T": [8],
            "lr": [0.001],
            "conv_cfg": [
                [[1, 16, 3, 1, 1], [16, 32, 3, 1, 1], [32, 64, 3, 1, 1]],
                [[1, 32, 3, 1, 1], [32, 64, 3, 1, 1], [64, 128, 3, 1, 1]],
                [
                    [1, 32, 3, 1, 1],
                    [32, 64, 3, 1, 1],
                    [64, 128, 3, 1, 1],
                    [128, 256, 3, 1, 1],
                ],
            ],
        },
        "FashionMNIST": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [512],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
            "conv_cfg": [
                [[1, 16, 3, 1, 1], [16, 32, 3, 1, 1], [32, 64, 3, 1, 1]],
                [[1, 32, 3, 1, 1], [32, 64, 3, 1, 1], [64, 128, 3, 1, 1]],
                [
                    [1, 32, 3, 1, 1],
                    [32, 64, 3, 1, 1],
                    [64, 128, 3, 1, 1],
                    [128, 256, 3, 1, 1],
                ],
            ],
        },
        "NMNIST": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [512],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
            "conv_cfg": [
                [[2, 16, 3, 1, 1], [16, 32, 3, 1, 1], [32, 64, 3, 1, 1]],
                [[2, 32, 3, 1, 1], [32, 64, 3, 1, 1], [64, 128, 3, 1, 1]],
                [
                    [2, 32, 3, 1, 1],
                    [32, 64, 3, 1, 1],
                    [64, 128, 3, 1, 1],
                    [128, 256, 3, 1, 1],
                ],
            ],
        },
        "CIFAR10": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [512],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
            "conv_cfg": [
                [[3, 16, 3, 1, 1], [16, 32, 3, 1, 1], [32, 64, 3, 1, 1]],
                [[3, 32, 3, 1, 1], [32, 64, 3, 1, 1], [64, 128, 3, 1, 1]],
                [
                    [3, 32, 3, 1, 1],
                    [32, 64, 3, 1, 1],
                    [64, 128, 3, 1, 1],
                    [128, 256, 3, 1, 1],
                ],
            ],
        },
    },
    "CNN_FAMILY": {
        "MNIST": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [512],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
        },
        "FashionMNIST": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [512],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
        },
        "NMNIST": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [512],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
        },
        "CIFAR10": {
            "loss_threshold": [0.4, 1.2, 2.0],
            "v_threshold": [1.5],
            "b": [512],
            "T": [16],
            "lr": [0.0078125, 0.0009765625],
        },
    },
}


VALID_DATASETS = tuple(SEARCH_SPACE_PRESETS["MLP"].keys())
VALID_MODELS = ("MLP", "CNN", "VGG6", "VGG8", "VGG11", "ResNet")
VALID_LEARNING_MODES = ("unsupervised", "supervised")
VALID_UPDATE_MODES = ("autograd", "manual")
VALID_MANUAL_UPDATE_SCHEDULES = ("separate", "paired")
VALID_NEG_SAMPLE_STRATEGIES = (
    "auto",
    "embed_label_onehot",
    "embed_zero_onehot",
    "SCFF",
    "global_fourier_label",
)
VALID_GOODNESS_STRATEGIES = (
    "auto",
    "spike_square",
    "spike_square_mean",
    "freq_square",
    "freq_square_mean",
    "membrane_potential_square_mean",
    "square",
    "square_mean",
)
VALID_HIDDEN_LOSS_STRATEGIES = (
    "auto",
    "pairwise_goodness",
    "supervised_delta",
    "scaled_supervised_delta",
)

CNN_FAMILY_MODELS = {"VGG6", "VGG8", "VGG11", "ResNet"}
DEFAULT_UPDATE_SCHEDULE = "separate"
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run FF-SNN HPO over dataset/model and experiment "
            "strategy combinations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", default="MNIST,FashionMNIST,NMNIST,CIFAR10")
    parser.add_argument("--exclude-datasets", default="FashionMNIST,NMNIST")
    parser.add_argument("--models", default="MLP,CNN")
    parser.add_argument("--exclude-models", default="")
    parser.add_argument("--learning-modes", default="unsupervised,supervised")
    parser.add_argument("--exclude-learning-modes", default="")
    parser.add_argument("--update-modes", default="autograd,manual")
    parser.add_argument("--exclude-update-modes", default="")
    parser.add_argument("--manual-update-schedules", default="separate,paired")
    parser.add_argument("--exclude-manual-update-schedules", default="")
    parser.add_argument("--neg-sample-strategies", default="embed_label_onehot,SCFF")
    parser.add_argument("--exclude-neg-sample-strategies", default="")
    parser.add_argument("--goodness-strategies", default="spike_square_mean,freq_square_mean")
    parser.add_argument("--exclude-goodness-strategies", default="")
    parser.add_argument("--hidden-loss-strategies", default="pairwise_goodness,supervised_delta")
    parser.add_argument("--exclude-hidden-loss-strategies", default="")

    # Backward-compatible single-value aliases.
    parser.add_argument("--manual-update-schedule", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--neg-sample-strategy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--goodness-strategy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hidden-loss-strategy", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--search-strategy",
        choices=["grid", "random", "successive_halving", "bayes"],
        default="successive_halving",
    )
    parser.add_argument("--optimize-metric", default="val_acc_best")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--bayes-max-trials", type=int, default=10)
    parser.add_argument("--bayes-init-random-trials", type=int, default=4)
    parser.add_argument("--bayes-acquisition", choices=["ucb", "ei"], default="ucb")
    parser.add_argument("--random-search-trials", type=int, default=16)
    parser.add_argument("--successive-halving-initial-epochs", type=int, default=10)
    parser.add_argument("--successive-halving-reduction-factor", type=int, default=2)
    parser.add_argument("--capture-manual-grad-metrics", action="store_true")
    parser.add_argument("--capture-autograd-comparison", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun combinations even if a .best.json already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only rebuild batch_report.md and batch_summary.csv from existing results.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N combinations.")
    parser.add_argument("--top-k-to-print", type=int, default=5)
    args = parser.parse_args(argv)
    _apply_legacy_axis_aliases(args)
    return args

@dataclass(frozen=True)
class Combo:
    dataset: str
    model: str
    learning_mode: str
    hidden_layer_update_mode: str
    manual_update_schedule: str
    neg_sample_strategy: str
    goodness_strategy: str
    hidden_loss_strategy: str

    @property
    def key(self) -> str:
        return "|".join(
            [
                self.dataset,
                self.model,
                self.learning_mode,
                self.hidden_layer_update_mode,
                self.manual_update_schedule,
                self.neg_sample_strategy,
                self.goodness_strategy,
                self.hidden_loss_strategy,
            ]
        )

    def summary_stem(self, search_strategy: str) -> str:
        return "-".join(
            [
                self.learning_mode,
                self.hidden_layer_update_mode,
                self.manual_update_schedule,
                self.neg_sample_strategy,
                self.goodness_strategy,
                self.hidden_loss_strategy,
                self.dataset,
                self.model,
                search_strategy,
            ]
        )


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _apply_legacy_axis_aliases(args: argparse.Namespace) -> None:
    # Keep older one-value flags working while the batch runner moves to
    # include/exclude style plural flags.
    alias_pairs = [
        ("manual_update_schedules", "manual_update_schedule"),
        ("neg_sample_strategies", "neg_sample_strategy"),
        ("goodness_strategies", "goodness_strategy"),
        ("hidden_loss_strategies", "hidden_loss_strategy"),
    ]
    for plural_attr, legacy_attr in alias_pairs:
        legacy_value = getattr(args, legacy_attr, None)
        if legacy_value:
            setattr(args, plural_attr, legacy_value)


def _select_axis_values(
    include_csv: str,
    exclude_csv: str,
    valid_values: tuple[str, ...],
    axis_name: str,
) -> list[str]:
    selected = _split_csv(include_csv) if include_csv else list(valid_values)
    excluded = set(_split_csv(exclude_csv))
    unknown_selected = sorted(set(selected) - set(valid_values))
    unknown_excluded = sorted(excluded - set(valid_values))
    if unknown_selected:
        raise ValueError(
            f"Unsupported {axis_name} values: {unknown_selected}. "
            f"Valid choices: {list(valid_values)}"
        )
    if unknown_excluded:
        raise ValueError(
            f"Unsupported excluded {axis_name} values: {unknown_excluded}. "
            f"Valid choices: {list(valid_values)}"
        )
    filtered = [value for value in selected if value not in excluded]
    if not filtered:
        raise ValueError(f"No values left for axis {axis_name} after exclusion.")
    return filtered


def _search_space_group(model: str) -> str:
    if model == "MLP":
        return "MLP"
    if model == "CNN":
        return "CNN"
    if model in CNN_FAMILY_MODELS:
        return "CNN_FAMILY"
    raise ValueError(f"Unsupported model in batch HPO: {model}")


def build_search_space(dataset: str, model: str) -> dict:
    group = _search_space_group(model)
    try:
        return copy.deepcopy(SEARCH_SPACE_PRESETS[group][dataset])
    except KeyError as exc:
        raise ValueError(
            f"Missing search-space preset for group={group} dataset={dataset}"
        ) from exc


def _expected_best_path(out_dir: Path, combo: Combo, search_strategy: str) -> Path:
    return out_dir / f"{combo.summary_stem(search_strategy)}.best.json"


def _runtime_env() -> dict[str, str]:
    # Keep the batch runner side-effect free with respect to project-local
    # cache directories. If callers want custom cache paths they can still set
    # MPLCONFIGDIR / XDG_CACHE_HOME in the shell before launching the script.
    return os.environ.copy()


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _load_run_records(path: Path) -> dict[str, dict]:
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["combo_key"]] = record
    return records


def _append_run_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(record, ensure_ascii=False))
        wf.write("\n")


def _score_of_row(row: dict) -> float:
    for key in ("score_value", "val_acc_best", "test_acc"):
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
    return float("-inf")


def _combo_from_row(row: dict) -> Combo | None:
    dataset = row.get("dataset", "")
    model = row.get("model", "")
    learning_mode = row.get("learning_mode", "")
    hidden_layer_update_mode = row.get("hidden_layer_update_mode", "")
    if not all([dataset, model, learning_mode, hidden_layer_update_mode]):
        return None
    return Combo(
        dataset=dataset,
        model=model,
        learning_mode=learning_mode,
        hidden_layer_update_mode=hidden_layer_update_mode,
        manual_update_schedule=row.get(
            "manual_update_schedule",
            DEFAULT_UPDATE_SCHEDULE,
        ),
        neg_sample_strategy=row.get("neg_sample_strategy", "auto"),
        goodness_strategy=row.get("goodness_strategy", "auto"),
        hidden_loss_strategy=row.get("hidden_loss_strategy", "auto"),
    )


def _remember_best_row(
    best_by_key: dict[str, dict],
    row: dict,
    *,
    source: str,
    path: Path | None = None,
) -> None:
    combo = _combo_from_row(row)
    if combo is None:
        return
    candidate = dict(row)
    candidate["_source"] = source
    candidate["_best_json"] = str(path) if path is not None and source == "best_json" else ""
    current = best_by_key.get(combo.key)
    if current is None or _score_of_row(candidate) > _score_of_row(current):
        best_by_key[combo.key] = candidate


def _load_best_results(out_dir: Path) -> dict[str, dict]:
    best_by_key: dict[str, dict] = {}

    for best_path in sorted(out_dir.glob("*.best.json")):
        with best_path.open("r", encoding="utf-8") as bf:
            row = json.load(bf)
        _remember_best_row(best_by_key, row, source="best_json", path=best_path)

    for summary_path in sorted(out_dir.glob("*.csv")):
        if summary_path.name == "batch_summary.csv" or summary_path.stat().st_size == 0:
            continue
        with summary_path.open("r", newline="", encoding="utf-8") as sf:
            reader = csv.DictReader(sf)
            for row in reader:
                if row.get("status") != "ok" or not row.get("score_value"):
                    continue
                _remember_best_row(best_by_key, row, source="summary_csv", path=summary_path)

    for metrics_path in sorted(out_dir.rglob("metrics.json")):
        try:
            rel_parts = metrics_path.relative_to(out_dir).parts
        except ValueError:
            continue
        if len(rel_parts) < 7:
            continue
        learning_mode, dataset, model, run_name = rel_parts[:4]
        if learning_mode not in VALID_LEARNING_MODES:
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as mf:
                metrics = json.load(mf)
        except (OSError, json.JSONDecodeError):
            continue
        hidden_layer_update_mode = metrics.get("hidden_layer_update_mode")
        if not hidden_layer_update_mode and "_" in run_name:
            hidden_layer_update_mode = run_name.rsplit("_", 1)[-1]
        row = {
            **metrics,
            "dataset": dataset,
            "model": model,
            "learning_mode": learning_mode,
            "hidden_layer_update_mode": hidden_layer_update_mode or "",
            "manual_update_schedule": metrics.get(
                "manual_update_schedule",
                DEFAULT_UPDATE_SCHEDULE,
            ),
            "neg_sample_strategy": metrics.get("neg_sample_strategy", "auto"),
            "goodness_strategy": metrics.get("goodness_strategy", "auto"),
            "hidden_loss_strategy": metrics.get("hidden_loss_strategy", "auto"),
            "score_metric_used": "val_acc_best",
            "score_value": metrics.get("val_acc_best", ""),
            "run_dir": str(metrics_path.parent),
            "T": "",
            "b": "",
            "lr": "",
            "loss_threshold": "",
            "v_threshold": "",
            "dims": "",
            "conv_cfg": "",
        }
        args_path = metrics_path.parent / "args.txt"
        if args_path.exists():
            args_text = args_path.read_text(encoding="utf-8", errors="replace")
            row.update(_extract_args_metadata(args_text))
        _remember_best_row(best_by_key, row, source="metrics_json", path=metrics_path)

    return best_by_key


def _extract_args_metadata(args_text: str) -> dict:
    metadata = {}
    simple_fields = {
        "T": "T",
        "b": "b",
        "lr": "lr",
        "loss_threshold": "loss_threshold",
        "v_threshold": "v_threshold",
    }
    for output_key, arg_key in simple_fields.items():
        token = f"{arg_key}="
        start = args_text.find(token)
        if start < 0:
            continue
        start += len(token)
        end_candidates = [
            idx
            for idx in (args_text.find(",", start), args_text.find(")", start))
            if idx >= 0
        ]
        end = min(end_candidates) if end_candidates else len(args_text)
        metadata[output_key] = args_text[start:end].strip().strip("'\"")

    for output_key, arg_key in (("dims", "dims"), ("conv_cfg", "conv_cfg")):
        token = f"{arg_key}="
        start = args_text.find(token)
        if start < 0:
            continue
        start += len(token)
        bracket_start = args_text.find("[", start)
        if bracket_start != start:
            continue
        depth = 0
        for idx in range(bracket_start, len(args_text)):
            char = args_text[idx]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    metadata[output_key] = args_text[bracket_start : idx + 1]
                    break
    return metadata


def _compact_params(row: dict) -> str:
    parts = [
        f"T={row.get('T', '')}",
        f"b={row.get('b', '')}",
        f"lr={row.get('lr', '')}",
        f"loss={row.get('loss_threshold', '')}",
        f"v={row.get('v_threshold', '')}",
    ]
    if row.get("dims"):
        parts.append(f"dims={row['dims']}")
    if row.get("conv_cfg"):
        parts.append(f"conv={row['conv_cfg']}")
    return "; ".join(parts)


def write_reports(out_dir: Path, combos: list[Combo], records_path: Path) -> None:
    best_by_key = _load_best_results(out_dir)
    run_records = _load_run_records(records_path)
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_path = out_dir / "batch_report.md"
    csv_path = out_dir / "batch_summary.csv"

    headers = [
        "dataset",
        "model",
        "learning_mode",
        "hidden_layer_update_mode",
        "manual_update_schedule",
        "neg_sample_strategy",
        "goodness_strategy",
        "hidden_loss_strategy",
        "status",
        "score",
        "val_acc_best",
        "test_acc",
        "params",
        "run_dir",
        "best_json",
        "source",
    ]

    rows = []
    for combo in combos:
        best = best_by_key.get(combo.key)
        record = run_records.get(combo.key, {})
        if best:
            status = "ok" if best.get("_source") == "best_json" else "partial"
        else:
            status = record.get("status", "pending")
        rows.append(
            {
                "dataset": combo.dataset,
                "model": combo.model,
                "learning_mode": combo.learning_mode,
                "hidden_layer_update_mode": combo.hidden_layer_update_mode,
                "manual_update_schedule": combo.manual_update_schedule,
                "neg_sample_strategy": combo.neg_sample_strategy,
                "goodness_strategy": combo.goodness_strategy,
                "hidden_loss_strategy": combo.hidden_loss_strategy,
                "status": status,
                "score": best.get("score_value", "") if best else "",
                "val_acc_best": best.get("val_acc_best", "") if best else "",
                "test_acc": best.get("test_acc", "") if best else "",
                "params": _compact_params(best) if best else "",
                "run_dir": best.get("run_dir", "") if best else "",
                "best_json": best.get("_best_json", "") if best else "",
                "source": best.get("_source", "") if best else "",
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as mf:
        mf.write("# FF-SNN HPO Batch Report\n\n")
        mf.write(f"Generated at: {generated_at}\n\n")
        mf.write(
            "| Dataset | Model | Learning | Update | Manual Schedule | "
            "Neg Sample | Goodness | Hidden Loss | Status | Score | Val Best | "
            "Test | Source | Run Dir |\n"
        )
        mf.write(
            "|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|---|\n"
        )
        for row in rows:
            mf.write(
                "| {dataset} | {model} | {learning_mode} | "
                "{hidden_layer_update_mode} | {manual_update_schedule} | "
                "{neg_sample_strategy} | {goodness_strategy} | "
                "{hidden_loss_strategy} | {status} | {score} | "
                "{val_acc_best} | {test_acc} | {source} | {run_dir} |\n".format(
                    **row
                )
            )
        mf.write(
            "\nStatus `partial` means a completed HPO summary was not found, "
            "so the row was recovered from an intermediate CSV or metrics file.\n"
        )
        mf.write(
            "Detailed parameters are available in `batch_summary.csv` and each "
            "`.best.json` file when present.\n"
        )


def build_command(args: argparse.Namespace, combo: Combo) -> list[str]:
    search_space = build_search_space(combo.dataset, combo.model)
    cmd = [
        sys.executable,
        str(HPO_SCRIPT),
        "--model",
        combo.model,
        "--dataset",
        combo.dataset,
        "--learning-mode",
        combo.learning_mode,
        "--hidden-layer-update-mode",
        combo.hidden_layer_update_mode,
        "--manual-update-schedule",
        combo.manual_update_schedule,
        "--neg-sample-strategy",
        combo.neg_sample_strategy,
        "--goodness-strategy",
        combo.goodness_strategy,
        "--hidden-loss-strategy",
        combo.hidden_loss_strategy,
        "--epochs",
        str(args.epochs),
        "--out-dir",
        str(args.out_dir),
        "--workers",
        str(args.workers),
        "--search-strategy",
        args.search_strategy,
        "--optimize-metric",
        args.optimize_metric,
        "--random-seed",
        str(args.random_seed),
        "--capture-manual-grad-metrics",
        str(args.capture_manual_grad_metrics).lower(),
        "--capture-autograd-comparison",
        str(args.capture_autograd_comparison).lower(),
        "--top-k-to-print",
        str(args.top_k_to_print),
        "--search-space-json",
        json.dumps(search_space),
        "--csv-note",
        (
            f"batch {combo.dataset} {combo.model} {combo.learning_mode} "
            f"{combo.hidden_layer_update_mode} {combo.manual_update_schedule} "
            f"{combo.neg_sample_strategy} {combo.goodness_strategy} "
            f"{combo.hidden_loss_strategy}"
        ),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.search_strategy == "bayes":
        cmd += [
            "--bayes-max-trials",
            str(args.bayes_max_trials),
            "--bayes-init-random-trials",
            str(args.bayes_init_random_trials),
            "--bayes-acquisition",
            args.bayes_acquisition,
        ]
    elif args.search_strategy == "random":
        cmd += ["--random-search-trials", str(args.random_search_trials)]
    elif args.search_strategy == "successive_halving":
        cmd += [
            "--successive-halving-initial-epochs",
            str(args.successive_halving_initial_epochs),
            "--successive-halving-reduction-factor",
            str(args.successive_halving_reduction_factor),
        ]
    return cmd





def build_combos(args: argparse.Namespace) -> list[Combo]:
    datasets = _select_axis_values(
        args.datasets,
        args.exclude_datasets,
        VALID_DATASETS,
        "datasets",
    )
    models = _select_axis_values(
        args.models,
        args.exclude_models,
        VALID_MODELS,
        "models",
    )
    learning_modes = _select_axis_values(
        args.learning_modes,
        args.exclude_learning_modes,
        VALID_LEARNING_MODES,
        "learning_modes",
    )
    update_modes = _select_axis_values(
        args.update_modes,
        args.exclude_update_modes,
        VALID_UPDATE_MODES,
        "update_modes",
    )
    manual_schedules = _select_axis_values(
        args.manual_update_schedules,
        args.exclude_manual_update_schedules,
        VALID_MANUAL_UPDATE_SCHEDULES,
        "manual_update_schedules",
    )
    neg_sample_strategies = _select_axis_values(
        args.neg_sample_strategies,
        args.exclude_neg_sample_strategies,
        VALID_NEG_SAMPLE_STRATEGIES,
        "neg_sample_strategies",
    )
    goodness_strategies = _select_axis_values(
        args.goodness_strategies,
        args.exclude_goodness_strategies,
        VALID_GOODNESS_STRATEGIES,
        "goodness_strategies",
    )
    hidden_loss_strategies = _select_axis_values(
        args.hidden_loss_strategies,
        args.exclude_hidden_loss_strategies,
        VALID_HIDDEN_LOSS_STRATEGIES,
        "hidden_loss_strategies",
    )

    combos: list[Combo] = []
    for dataset in datasets:
        for model in models:
            for learning_mode in learning_modes:
                for update_mode in update_modes:
                    # The legacy parameter name is retained, but the schedule
                    # controls update timing for manual and autograd modes.
                    for manual_update_schedule in manual_schedules:
                        for neg_sample_strategy in neg_sample_strategies:
                            for goodness_strategy in goodness_strategies:
                                for hidden_loss_strategy in hidden_loss_strategies:
                                    combos.append(
                                        Combo(
                                            dataset=dataset,
                                            model=model,
                                            learning_mode=learning_mode,
                                            hidden_layer_update_mode=update_mode,
                                            manual_update_schedule=manual_update_schedule,
                                            neg_sample_strategy=neg_sample_strategy,
                                            goodness_strategy=goodness_strategy,
                                            hidden_loss_strategy=hidden_loss_strategy,
                                        )
                                    )
    return combos


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "batch_runs.jsonl"
    env = _runtime_env()

    cuda_available = _torch_cuda_available()
    print(f"[BATCH] python={sys.executable}")
    print(f"[BATCH] cuda_available={cuda_available}")
    if args.require_cuda and not cuda_available and not args.report_only:
        print("[BATCH] CUDA is required but is not available. Aborting.")
        return 2

    combos = build_combos(args)
    if args.limit is not None:
        combos = combos[: args.limit]

    if args.report_only:
        write_reports(args.out_dir, combos, records_path)
        print(f"[BATCH] report: {args.out_dir / 'batch_report.md'}")
        print(f"[BATCH] summary: {args.out_dir / 'batch_summary.csv'}")
        return 0

    print(f"[BATCH] combinations={len(combos)}")
    for idx, combo in enumerate(combos, start=1):
        best_path = _expected_best_path(args.out_dir, combo, args.search_strategy)
        if best_path.exists() and not args.force:
            print(f"[BATCH] skip {idx}/{len(combos)} already completed: {combo.key}")
            continue

        cmd = build_command(args, combo)
        print(f"[BATCH] run {idx}/{len(combos)}: {combo.key}")
        print("[BATCH] command:", " ".join(cmd))
        if args.dry_run:
            continue

        started = time.time()
        started_at = dt.datetime.now().isoformat(timespec="seconds")
        result = subprocess.run(cmd, cwd=WORKSPACE_ROOT, env=env)
        ended_at = dt.datetime.now().isoformat(timespec="seconds")
        elapsed_s = round(time.time() - started, 3)
        status = "ok" if result.returncode == 0 else "failed"
        _append_run_record(
            records_path,
            {
                "combo_key": combo.key,
                "dataset": combo.dataset,
                "model": combo.model,
                "learning_mode": combo.learning_mode,
                "hidden_layer_update_mode": combo.hidden_layer_update_mode,
                "manual_update_schedule": combo.manual_update_schedule,
                "neg_sample_strategy": combo.neg_sample_strategy,
                "goodness_strategy": combo.goodness_strategy,
                "hidden_loss_strategy": combo.hidden_loss_strategy,
                "status": status,
                "returncode": result.returncode,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_s": elapsed_s,
                "expected_best_path": str(best_path),
            },
        )
        write_reports(args.out_dir, combos, records_path)
        if result.returncode != 0:
            print(f"[BATCH] failed: {combo.key} returncode={result.returncode}")

    if not args.dry_run:
        write_reports(args.out_dir, combos, records_path)
        print(f"[BATCH] report: {args.out_dir / 'batch_report.md'}")
        print(f"[BATCH] summary: {args.out_dir / 'batch_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
