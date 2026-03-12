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
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "ff-snn.py"
OUT_DIR = ROOT / "logs" / "opt"

# Edit these lists to define the search grid.
SEARCH_SPACE = {
    "loss_threshold": [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "v_threshold": [1.0, 1.2, 1.3],
    "dims": [
        [784, 512, 512, 10]
        [784, 512, 256, 10],
        [784, 256, 256, 10],
    ],
    "T": [16, 32],
    "lr": [0.00390625, 0.00390625/2, 0.00390625/4],
}

# Base training settings.
MODEL = "MLP"
DATASET = "MNIST"
PREDICT_TYPE = "supervised"
EPOCHS = 35


def _dims_to_str(dims: list[int]) -> str:
    return "[" + ",".join(str(d) for d in dims) + "]"


def _iter_grid(space: dict) -> list[dict]:
    keys = ["loss_threshold", "v_threshold", "dims", "T", "lr"]
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
        "-T",
        str(params["T"]),
        "-lr",
        str(params["lr"]),
    ]
    if DATASET:
        cmd += ["-dataset", DATASET]
    if PREDICT_TYPE:
        cmd += ["-predict_type", PREDICT_TYPE]
    if MODEL == "MLP":
        cmd += ["-dims"] + [str(v) for v in params["dims"]]
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = OUT_DIR / f"{timestamp}.csv"

    fieldnames = [
        "loss_threshold",
        "v_threshold",
        "dims",
        "T",
        "lr",
        "test_acc",
        "last_epoch_loss_mean",
        "last_epoch_goodness_pos_mean",
        "last_epoch_goodness_neg_mean",
        "last_epoch_firing_pos_mean",
        "last_epoch_firing_neg_mean",
        "run_dir",
        "status",
        "error",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for params in _iter_grid(SEARCH_SPACE):
            row = {
                "loss_threshold": params["loss_threshold"],
                "v_threshold": params["v_threshold"],
                "dims": _dims_to_str(params["dims"]),
                "T": params["T"],
                "lr": params["lr"],
                "status": "ok",
                "error": "",
                "run_dir": "",
            }
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
