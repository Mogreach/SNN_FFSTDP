from __future__ import annotations

import argparse
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

DATASET_INPUT_DIMS = {
    "MNIST": 784,
    "FashionMNIST": 784,
    "NMNIST": 2 * 34 * 34,
    "CIFAR10": 3 * 32 * 32,
}
DATASET_CHANNELS = {
    "MNIST": 1,
    "FashionMNIST": 1,
    "NMNIST": 2,
    "CIFAR10": 3,
}


@dataclass(frozen=True)
class Combo:
    dataset: str
    model: str
    learning_mode: str
    hidden_layer_update_mode: str

    @property
    def key(self) -> str:
        return "|".join(
            [
                self.dataset,
                self.model,
                self.learning_mode,
                self.hidden_layer_update_mode,
            ]
        )


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _mlp_dims(dataset: str) -> list[list[int]]:
    input_dim = DATASET_INPUT_DIMS[dataset]
    if dataset == "CIFAR10":
        return [
            [input_dim, 1024, 512, 10],
            [input_dim, 1024, 10],
            [input_dim, 512, 10],
        ]
    if dataset == "NMNIST":
        return [
            [input_dim, 512, 512, 10],
            [input_dim, 512, 10],
            [input_dim, 256, 10],
        ]
    return [
        [input_dim, 512, 512, 10],
        [input_dim, 512, 10],
        [input_dim, 256, 10],
    ]


def _conv_cfgs(dataset: str) -> list[list[list[int]]]:
    in_ch = DATASET_CHANNELS[dataset]
    return [
        [
            [in_ch, 16, 3, 1, 1],
            [16, 32, 3, 1, 1],
            [32, 64, 3, 1, 1],
        ],
        [
            [in_ch, 32, 3, 1, 1],
            [32, 64, 3, 1, 1],
            [64, 128, 3, 1, 1],
        ],
        [
            [in_ch, 32, 3, 1, 1],
            [32, 64, 3, 1, 1],
            [64, 128, 3, 1, 1],
            [128, 256, 3, 1, 1],
        ],
    ]


def build_search_space(dataset: str, model: str) -> dict:
    event_dataset = dataset == "NMNIST"
    larger_input = dataset in {"NMNIST", "CIFAR10"}
    search_space = {
        "loss_threshold": [0.4, 0.8, 1.2, 1.5, 2.0],
        "v_threshold": [1.0, 1.2],
        "T": [20, 32] if event_dataset else [8, 16, 32],
        "lr": [0.0078125, 0.00390625, 0.001953125, 0.0009765625],
    }
    if model == "MLP":
        search_space["b"] = [128, 256, 512] if larger_input else [256, 512, 1024]
        search_space["dims"] = _mlp_dims(dataset)
    else:
        search_space["b"] = [64, 128, 256] if larger_input else [128, 256, 512]
        search_space["conv_cfg"] = _conv_cfgs(dataset)
    return search_space


def _expected_best_path(out_dir: Path, combo: Combo, strategy: str) -> Path:
    return out_dir / (
        f"{combo.learning_mode}-{combo.hidden_layer_update_mode}-"
        f"{combo.dataset}-{combo.model}-{strategy}.best.json"
    )


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
    env.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-cache")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    return env


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


def _load_best_results(out_dir: Path) -> dict[str, dict]:
    best_by_key = {}

    def score_of(row: dict) -> float:
        for key in ("score_value", "val_acc_best", "test_acc"):
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            return value
        return float("-inf")

    def key_of(row: dict) -> str:
        return Combo(
            dataset=row.get("dataset", ""),
            model=row.get("model", ""),
            learning_mode=row.get("learning_mode", ""),
            hidden_layer_update_mode=row.get("hidden_layer_update_mode", ""),
        ).key

    def consider(row: dict, source: str, path: Path | None = None) -> None:
        key = key_of(row)
        if key == "|||":
            return
        row["_source"] = source
        if path is not None and source == "best_json":
            row["_best_json"] = str(path)
        else:
            row.setdefault("_best_json", "")
        current = best_by_key.get(key)
        if current is None or score_of(row) > score_of(current):
            best_by_key[key] = row

    for best_path in sorted(out_dir.glob("*.best.json")):
        with best_path.open("r", encoding="utf-8") as bf:
            row = json.load(bf)
        consider(row, "best_json", best_path)

    for summary_path in sorted(out_dir.glob("*.csv")):
        if summary_path.name == "batch_summary.csv" or summary_path.stat().st_size == 0:
            continue
        with summary_path.open("r", newline="", encoding="utf-8") as sf:
            reader = csv.DictReader(sf)
            for row in reader:
                if row.get("status") != "ok":
                    continue
                if not row.get("score_value"):
                    continue
                consider(row, "summary_csv", summary_path)

    for metrics_path in sorted(out_dir.rglob("metrics.json")):
        try:
            rel_parts = metrics_path.relative_to(out_dir).parts
        except ValueError:
            continue
        if len(rel_parts) < 7:
            continue
        learning_mode, dataset, model, run_name = rel_parts[:4]
        if learning_mode not in {"unsupervised", "supervised"}:
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
        consider(row, "metrics_json", metrics_path)
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
    if row.get("model") == "MLP" and row.get("dims"):
        parts.append(f"dims={row['dims']}")
    if row.get("model") == "CNN" and row.get("conv_cfg"):
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
        mf.write("| Dataset | Model | Learning | Update | Status | Score | Val Best | Test | Source | Run Dir |\n")
        mf.write("|---|---|---|---|---|---:|---:|---:|---|---|\n")
        for row in rows:
            mf.write(
                "| {dataset} | {model} | {learning_mode} | {hidden_layer_update_mode} "
                "| {status} | {score} | {val_acc_best} | {test_acc} | {source} | {run_dir} |\n".format(
                    **row
                )
            )
        mf.write("\nStatus `partial` means a completed HPO summary was not found, so the row was recovered from an intermediate CSV or metrics file.\n")
        mf.write("Detailed parameters are available in `batch_summary.csv` and each `.best.json` file when present.\n")


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
        f"batch {combo.dataset} {combo.model} {combo.learning_mode} {combo.hidden_layer_update_mode}",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially run FF-SNN HPO over dataset/model/mode combinations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", default="MNIST,FashionMNIST,NMNIST,CIFAR10")
    parser.add_argument("--models", default="MLP,CNN")
    parser.add_argument("--learning-modes", default="unsupervised,supervised")
    parser.add_argument("--update-modes", default="autograd,manual")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--search-strategy", choices=["grid", "random", "successive_halving", "bayes"], default="bayes")
    parser.add_argument("--optimize-metric", default="val_acc_best")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--bayes-max-trials", type=int, default=50)
    parser.add_argument("--bayes-init-random-trials", type=int, default=4)
    parser.add_argument("--bayes-acquisition", choices=["ucb", "ei"], default="ucb")
    parser.add_argument("--random-search-trials", type=int, default=16)
    parser.add_argument("--successive-halving-initial-epochs", type=int, default=50)
    parser.add_argument("--successive-halving-reduction-factor", type=int, default=2)
    parser.add_argument("--capture-manual-grad-metrics", action="store_true")
    parser.add_argument("--capture-autograd-comparison", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun combinations even if a .best.json already exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild batch_report.md and batch_summary.csv from existing results.")
    parser.add_argument("--limit", type=int, help="Run only the first N combinations.")
    parser.add_argument("--top-k-to-print", type=int, default=5)
    return parser.parse_args(argv)


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

    combos = [
        Combo(dataset, model, learning_mode, update_mode)
        for dataset in _split_csv(args.datasets)
        for model in _split_csv(args.models)
        for learning_mode in _split_csv(args.learning_modes)
        for update_mode in _split_csv(args.update_modes)
    ]
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
