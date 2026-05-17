from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiment import (
    ExperimentModeConfig,
    ExperimentStrategyConfig,
    TrainMemorySnapshot,
    StepResult,
)
from src.ff_strategies.objectives import (
    compute_hidden_pair_loss,
)


def _to_float(value, *, nan_for_none: bool = False):
    if value is None:
        return math.nan if nan_for_none else None
    if torch.is_tensor(value):
        if value.numel() == 1:
            value = value.detach().cpu().item()
        else:
            value = value.detach().cpu().mean().item()
    elif hasattr(value, "item"):
        value = value.item()
    return float(value)


def _bytes_to_mb(value):
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _mean_last_epoch(metric_per_layer_list):
    values = []
    for layer_metric in metric_per_layer_list:
        if not layer_metric:
            continue
        value = _to_float(layer_metric[-1])
        if value is None or math.isnan(value):
            continue
        values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def _safe_pct_reduction(smaller_value, larger_value):
    if (
        smaller_value is None
        or larger_value is None
        or float(larger_value) <= 0
    ):
        return None
    return 100.0 * (1.0 - float(smaller_value) / float(larger_value))


@dataclass
class TrainMemoryAccumulator:
    current_alloc_sum: float = 0.0
    current_reserved_sum: float = 0.0
    peak_alloc_sum: float = 0.0
    peak_reserved_sum: float = 0.0
    peak_alloc_max: float = 0.0
    peak_reserved_max: float = 0.0
    count: int = 0

    def add(self, snapshot: TrainMemorySnapshot | None) -> None:
        # Skip missing samples so CPU runs and profiling-disabled runs can
        # still share the same recording path.
        if snapshot is None:
            return
        if (
            snapshot.current_alloc_bytes is None
            and snapshot.current_reserved_bytes is None
            and snapshot.peak_alloc_bytes is None
            and snapshot.peak_reserved_bytes is None
        ):
            return
        self.count += 1
        if snapshot.current_alloc_bytes is not None:
            self.current_alloc_sum += float(snapshot.current_alloc_bytes)
        if snapshot.current_reserved_bytes is not None:
            self.current_reserved_sum += float(snapshot.current_reserved_bytes)
        if snapshot.peak_alloc_bytes is not None:
            peak_alloc = float(snapshot.peak_alloc_bytes)
            self.peak_alloc_sum += peak_alloc
            self.peak_alloc_max = max(self.peak_alloc_max, peak_alloc)
        if snapshot.peak_reserved_bytes is not None:
            peak_reserved = float(snapshot.peak_reserved_bytes)
            self.peak_reserved_sum += peak_reserved
            self.peak_reserved_max = max(self.peak_reserved_max, peak_reserved)

    def summary(self) -> dict:
        if self.count == 0:
            return {
                "train_gpu_mem_alloc_mean_mb": None,
                "train_gpu_mem_reserved_mean_mb": None,
                "bp_gpu_mem_peak_alloc_mean_mb": None,
                "bp_gpu_mem_peak_reserved_mean_mb": None,
                "bp_gpu_mem_peak_alloc_max_mb": None,
                "bp_gpu_mem_peak_reserved_max_mb": None,
            }
        return {
            "train_gpu_mem_alloc_mean_mb": _bytes_to_mb(
                self.current_alloc_sum / self.count
            ),
            "train_gpu_mem_reserved_mean_mb": _bytes_to_mb(
                self.current_reserved_sum / self.count
            ),
            "bp_gpu_mem_peak_alloc_mean_mb": _bytes_to_mb(
                self.peak_alloc_sum / self.count
            ),
            "bp_gpu_mem_peak_reserved_mean_mb": _bytes_to_mb(
                self.peak_reserved_sum / self.count
            ),
            "bp_gpu_mem_peak_alloc_max_mb": _bytes_to_mb(self.peak_alloc_max),
            "bp_gpu_mem_peak_reserved_max_mb": _bytes_to_mb(
                self.peak_reserved_max
            ),
        }


@dataclass
class GradientMetricsAccumulator:
    alloc_sum: float = 0.0
    reserved_sum: float = 0.0
    alloc_max: float = 0.0
    reserved_max: float = 0.0
    time_sum_ms: float = 0.0
    ops_sum: float = 0.0
    count: int = 0
    sample_total: int = 0

    def add(
        self,
        *,
        alloc_bytes=None,
        reserved_bytes=None,
        time_ms=None,
        sample_count: int = 0,
        ops_est=None,
    ) -> None:
        if (
            alloc_bytes is None
            and reserved_bytes is None
            and time_ms is None
            and ops_est is None
        ):
            return
        self.count += 1
        self.sample_total += int(sample_count)
        if alloc_bytes is not None:
            alloc_value = float(alloc_bytes)
            self.alloc_sum += alloc_value
            self.alloc_max = max(self.alloc_max, alloc_value)
        if reserved_bytes is not None:
            reserved_value = float(reserved_bytes)
            self.reserved_sum += reserved_value
            self.reserved_max = max(self.reserved_max, reserved_value)
        if time_ms is not None:
            self.time_sum_ms += float(time_ms)
        if ops_est is not None:
            self.ops_sum += float(ops_est)

    def summary(
        self,
        name_prefix: str,
        *,
        include_time: bool = True,
        include_ops: bool = False,
        include_throughput: bool = True,
    ) -> dict:
        result = {
            f"{name_prefix}_peak_alloc_mean_mb": None,
            f"{name_prefix}_peak_reserved_mean_mb": None,
            f"{name_prefix}_peak_alloc_max_mb": None,
            f"{name_prefix}_peak_reserved_max_mb": None,
        }
        if self.count > 0:
            result.update(
                {
                    f"{name_prefix}_peak_alloc_mean_mb": _bytes_to_mb(
                        self.alloc_sum / self.count
                    ),
                    f"{name_prefix}_peak_reserved_mean_mb": _bytes_to_mb(
                        self.reserved_sum / self.count
                    ),
                    f"{name_prefix}_peak_alloc_max_mb": _bytes_to_mb(
                        self.alloc_max
                    ),
                    f"{name_prefix}_peak_reserved_max_mb": _bytes_to_mb(
                        self.reserved_max
                    ),
                }
            )

        if include_time:
            result[f"{name_prefix}_time_mean_ms"] = (
                self.time_sum_ms / self.count if self.count > 0 else None
            )
            result[f"{name_prefix}_peak_alloc_per_sample_kb"] = (
                (self.alloc_sum / self.sample_total) / 1024.0
                if self.sample_total > 0
                else None
            )
            result[f"{name_prefix}_peak_reserved_per_sample_kb"] = (
                (self.reserved_sum / self.sample_total) / 1024.0
                if self.sample_total > 0
                else None
            )
            result[f"{name_prefix}_time_per_sample_us"] = (
                (self.time_sum_ms * 1000.0) / self.sample_total
                if self.sample_total > 0
                else None
            )
            if include_throughput:
                result[f"{name_prefix}_samples_per_s"] = (
                    self.sample_total / (self.time_sum_ms / 1000.0)
                    if self.time_sum_ms > 0
                    else None
                )

        if include_ops:
            result[f"{name_prefix}_ops_est_total"] = (
                self.ops_sum if self.ops_sum > 0 else None
            )
            result[f"{name_prefix}_ops_est_gops"] = (
                self.ops_sum / 1e9 if self.ops_sum > 0 else None
            )
            result[f"{name_prefix}_ops_est_per_sample"] = (
                self.ops_sum / self.sample_total
                if self.sample_total > 0 and self.ops_sum > 0
                else None
            )
            result[f"{name_prefix}_ops_est_gops_per_s"] = (
                (self.ops_sum / 1e9) / (self.time_sum_ms / 1000.0)
                if self.ops_sum > 0 and self.time_sum_ms > 0
                else None
            )
        return result


class ExperimentMetricsTracker:
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        mode_config: ExperimentModeConfig,
        strategy_config: ExperimentStrategyConfig | None = None,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.mode_config = mode_config
        self.strategy_config = strategy_config or ExperimentStrategyConfig()
        self.val_acc_history = []
        # Keep the old attribute name as an alias so existing code and exported
        # metrics do not break while we migrate naming toward validation accuracy.
        self.train_acc_history = self.val_acc_history
        self.epoch_summaries = []
        self.test_summary = {}
        self.layer_histories = {
            # All per-layer curves are stored in the same shape so plotting and
            # metrics export do not need to know which training branch produced them.
            "loss": [[] for _ in range(num_hidden_layers)],
            "goodness_pos": [[] for _ in range(num_hidden_layers)],
            "goodness_neg": [[] for _ in range(num_hidden_layers)],
            "cos_pos": [[] for _ in range(num_hidden_layers)],
            "cos_neg": [[] for _ in range(num_hidden_layers)],
            "spike_out_pos": [[] for _ in range(num_hidden_layers)],
            "spike_out_neg": [[] for _ in range(num_hidden_layers)],
        }
        self.train_memory = TrainMemoryAccumulator()
        self.backward_profile = GradientMetricsAccumulator()
        self.manual_grad_profile = GradientMetricsAccumulator()
        self.autograd_cmp_profile = GradientMetricsAccumulator()
        self.begin_epoch()

    def _reconstruct_epoch_loss(
        self,
        *,
        avg_goodness_pos: torch.Tensor,
        avg_goodness_neg: torch.Tensor,
        loss_threshold: float,
    ) -> torch.Tensor:
        """
        Rebuild the epoch loss with the same objective family as the current
        experiment branch.

        This keeps the logged `loss_mean` aligned with the actual hidden-layer
        training objective instead of hard-coding the unsupervised pairwise form.
        """
        per_layer_losses = []
        for pos_value, neg_value in zip(avg_goodness_pos, avg_goodness_neg):
            layer_loss = compute_hidden_pair_loss(
                pos_value.view(1),
                neg_value.view(1),
                threshold=loss_threshold,
                strategy_name=self.strategy_config.hidden_loss_strategy,
                mode_config=self.mode_config,
            )
            per_layer_losses.append(layer_loss.reshape(()))
        return torch.stack(per_layer_losses)

    def begin_epoch(self) -> None:
        self.epoch_batch_count = 0
        self.epoch_goodness_pos_sum = None
        self.epoch_goodness_neg_sum = None
        self.epoch_cos_pos_sum = None
        self.epoch_cos_neg_sum = None
        self.epoch_spike_out_pos_sum = None
        self.epoch_spike_out_neg_sum = None

    def _tensorize(self, values, expected_layers: int) -> torch.Tensor:
        tensor = torch.tensor(
            [_to_float(v, nan_for_none=True) for v in values],
            dtype=torch.float32,
        )
        if tensor.numel() != expected_layers:
            raise ValueError(
                f"Expected {expected_layers} values, but received {tensor.numel()}"
            )
        return tensor

    def record_train_step(
        self,
        step_result: StepResult,
        *,
        batch_size: int,
        train_memory_snapshot: TrainMemorySnapshot | None = None,
    ) -> None:
        # Convert one training step into epoch accumulators first, then derive
        # human-readable curves and summary metrics only at epoch boundaries.
        self.epoch_batch_count += 1
        if self.num_hidden_layers > 0:
            goodness_pos = self._tensorize(
                step_result.goodness_pos,
                self.num_hidden_layers,
            )
            goodness_neg = self._tensorize(
                step_result.goodness_neg,
                self.num_hidden_layers,
            )
            cos_pos = self._tensorize(step_result.cos_pos, self.num_hidden_layers)
            cos_neg = self._tensorize(step_result.cos_neg, self.num_hidden_layers)
            spike_out_pos = self._tensorize(
                step_result.spike_out_pos,
                self.num_hidden_layers,
            )
            spike_out_neg = self._tensorize(
                step_result.spike_out_neg,
                self.num_hidden_layers,
            )

            if self.epoch_goodness_pos_sum is None:
                self.epoch_goodness_pos_sum = torch.zeros_like(goodness_pos)
                self.epoch_goodness_neg_sum = torch.zeros_like(goodness_neg)
                self.epoch_cos_pos_sum = torch.zeros_like(cos_pos)
                self.epoch_cos_neg_sum = torch.zeros_like(cos_neg)
                self.epoch_spike_out_pos_sum = torch.zeros_like(spike_out_pos)
                self.epoch_spike_out_neg_sum = torch.zeros_like(spike_out_neg)

            self.epoch_goodness_pos_sum += goodness_pos
            self.epoch_goodness_neg_sum += goodness_neg
            self.epoch_cos_pos_sum += cos_pos
            self.epoch_cos_neg_sum += cos_neg
            self.epoch_spike_out_pos_sum += spike_out_pos
            self.epoch_spike_out_neg_sum += spike_out_neg

        profiler = step_result.profiler
        self.train_memory.add(train_memory_snapshot)
        self.backward_profile.add(
            alloc_bytes=profiler.backward_peak_alloc_bytes,
            reserved_bytes=profiler.backward_peak_reserved_bytes,
        )
        self.manual_grad_profile.add(
            alloc_bytes=profiler.manual_grad_peak_alloc_bytes,
            reserved_bytes=profiler.manual_grad_peak_reserved_bytes,
            time_ms=profiler.manual_grad_time_ms,
            sample_count=batch_size,
            ops_est=profiler.manual_grad_ops_est,
        )
        self.autograd_cmp_profile.add(
            alloc_bytes=profiler.backward_cmp_peak_alloc_bytes,
            reserved_bytes=profiler.backward_cmp_peak_reserved_bytes,
            time_ms=profiler.backward_cmp_time_ms,
            sample_count=batch_size,
        )

    def finalize_epoch(self, *, loss_threshold: float, train_acc: float):
        if self.epoch_batch_count == 0:
            self.val_acc_history.append(float(train_acc))
            self.epoch_summaries.append(
                {
                    "epoch": len(self.val_acc_history),
                    "val_acc": float(train_acc),
                    "loss_mean": None,
                }
            )
            return torch.empty(0)

        # Reconstruct the epoch loss from averaged hidden-layer goodness with
        # the same loss family used by the current experiment mode.
        avg_goodness_pos = self.epoch_goodness_pos_sum / self.epoch_batch_count
        avg_goodness_neg = self.epoch_goodness_neg_sum / self.epoch_batch_count
        avg_cos_pos = self.epoch_cos_pos_sum / self.epoch_batch_count
        avg_cos_neg = self.epoch_cos_neg_sum / self.epoch_batch_count
        avg_spike_out_pos = self.epoch_spike_out_pos_sum / self.epoch_batch_count
        avg_spike_out_neg = self.epoch_spike_out_neg_sum / self.epoch_batch_count

        loss = self._reconstruct_epoch_loss(
            avg_goodness_pos=avg_goodness_pos,
            avg_goodness_neg=avg_goodness_neg,
            loss_threshold=loss_threshold,
        )

        for layer_idx in range(self.num_hidden_layers):
            self.layer_histories["loss"][layer_idx].append(
                _to_float(loss[layer_idx], nan_for_none=True)
            )
            self.layer_histories["goodness_pos"][layer_idx].append(
                _to_float(avg_goodness_pos[layer_idx], nan_for_none=True)
            )
            self.layer_histories["goodness_neg"][layer_idx].append(
                _to_float(avg_goodness_neg[layer_idx], nan_for_none=True)
            )
            self.layer_histories["cos_pos"][layer_idx].append(
                _to_float(avg_cos_pos[layer_idx], nan_for_none=True)
            )
            self.layer_histories["cos_neg"][layer_idx].append(
                _to_float(avg_cos_neg[layer_idx], nan_for_none=True)
            )
            self.layer_histories["spike_out_pos"][layer_idx].append(
                _to_float(avg_spike_out_pos[layer_idx], nan_for_none=True)
            )
            self.layer_histories["spike_out_neg"][layer_idx].append(
                _to_float(avg_spike_out_neg[layer_idx], nan_for_none=True)
            )

        self.val_acc_history.append(float(train_acc))
        self.epoch_summaries.append(
            {
                "epoch": len(self.val_acc_history),
                "val_acc": float(train_acc),
                "loss_mean": float(loss.mean().item()),
                "goodness_pos_mean": float(avg_goodness_pos.mean().item()),
                "goodness_neg_mean": float(avg_goodness_neg.mean().item()),
                "cos_pos_mean": _to_float(avg_cos_pos.mean(), nan_for_none=True),
                "cos_neg_mean": _to_float(avg_cos_neg.mean(), nan_for_none=True),
                "firing_pos_mean": float(avg_spike_out_pos.mean().item()),
                "firing_neg_mean": float(avg_spike_out_neg.mean().item()),
            }
        )
        return loss

    def _plot_single_metric(
        self,
        metric_per_layer_list,
        *,
        save_path: Path,
        ylabel: str,
        title: str,
    ) -> None:
        if not metric_per_layer_list:
            return
        plt.figure(figsize=(10, 6))
        for layer_idx, values in enumerate(metric_per_layer_list):
            plt.plot(values, "o-", label=f"Layer {layer_idx + 1}")
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _plot_pos_neg_metric(
        self,
        pos_metric_per_layer_list,
        neg_metric_per_layer_list,
        *,
        save_path: Path,
        ylabel: str,
        title: str,
    ) -> None:
        if not pos_metric_per_layer_list:
            return
        plt.figure(figsize=(10, 6))
        for layer_idx, (pos_values, neg_values) in enumerate(
            zip(pos_metric_per_layer_list, neg_metric_per_layer_list)
        ):
            plt.plot(pos_values, "o-", label=f"Layer {layer_idx + 1} Positive")
            plt.plot(neg_values, "x--", label=f"Layer {layer_idx + 1} Negative")
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def save_plots(self, out_dir) -> None:
        out_dir = Path(out_dir)
        if self.val_acc_history:
            plt.figure(figsize=(8, 6))
            plt.plot(
                range(1, len(self.val_acc_history) + 1),
                self.val_acc_history,
                marker="o",
                label="Validation Accuracy",
            )
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy (%)")
            plt.title("Validation Accuracy Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(out_dir / "training_accuracy_curve.png", dpi=300)
            plt.close()

        self._plot_single_metric(
            self.layer_histories["loss"],
            save_path=out_dir / "loss_of_each_layer.png",
            ylabel="Loss",
            title="Loss vs Epoch for Each Layer",
        )
        self._plot_single_metric(
            self.layer_histories["cos_pos"],
            save_path=out_dir / "cosine_similarity_positive.png",
            ylabel="Positive Cosine similarity",
            title="Positive Cosine similarity vs Epoch for Each Layer",
        )
        self._plot_single_metric(
            self.layer_histories["cos_neg"],
            save_path=out_dir / "cosine_similarity_negative.png",
            ylabel="Negative Cosine similarity",
            title="Negative Cosine similarity vs Epoch for Each Layer",
        )
        self._plot_single_metric(
            self.layer_histories["goodness_pos"],
            save_path=out_dir / "goodness_positive.png",
            ylabel="Positive Goodness",
            title="Positive Goodness vs Epoch for Each Layer",
        )
        self._plot_single_metric(
            self.layer_histories["goodness_neg"],
            save_path=out_dir / "goodness_negative.png",
            ylabel="Negative Goodness",
            title="Negative Goodness vs Epoch for Each Layer",
        )
        self._plot_pos_neg_metric(
            self.layer_histories["spike_out_pos"],
            self.layer_histories["spike_out_neg"],
            save_path=out_dir / "spike_out_positive.png",
            ylabel="Average Firing Rate",
            title="Positive / Negative Firing Rate vs Epoch",
        )

    def build_metrics(
        self,
        *,
        test_acc: float,
        test_duration_s: float | None = None,
        test_batches: int | None = None,
    ) -> dict:
        # Export both experiment identity and aggregated statistics together so
        # downstream HPO/analysis code only needs to read one file.
        self.test_summary = {
            "test_acc": float(test_acc),
            "test_duration_s": (
                float(test_duration_s) if test_duration_s is not None else None
            ),
            "test_batches": int(test_batches) if test_batches is not None else None,
        }
        metrics = {
            "learning_mode": self.mode_config.learning_mode,
            "hidden_layer_update_mode": self.mode_config.hidden_layer_update_mode,
            "neg_sample_strategy": self.strategy_config.neg_sample_strategy,
            "goodness_strategy": self.strategy_config.goodness_strategy,
            "hidden_loss_strategy": self.strategy_config.hidden_loss_strategy,
            "capture_manual_grad_metrics": (
                self.mode_config.profiling.capture_manual_grad_metrics
            ),
            "capture_autograd_comparison": (
                self.mode_config.profiling.capture_autograd_comparison
            ),
            "test_acc": float(test_acc),
            "test_duration_s": self.test_summary["test_duration_s"],
            "test_batches": self.test_summary["test_batches"],
            "train_acc_last": (
                float(self.train_acc_history[-1]) if self.train_acc_history else None
            ),
            "train_acc_best": (
                float(max(self.train_acc_history)) if self.train_acc_history else None
            ),
            "val_acc_last": (
                float(self.val_acc_history[-1]) if self.val_acc_history else None
            ),
            "val_acc_best": (
                float(max(self.val_acc_history)) if self.val_acc_history else None
            ),
            "validation_accuracy_history": [float(v) for v in self.val_acc_history],
            "epoch_summaries": self.epoch_summaries,
            "test_summary": self.test_summary,
            "last_epoch_loss_mean": _mean_last_epoch(
                self.layer_histories["loss"]
            ),
            "last_epoch_goodness_pos_mean": _mean_last_epoch(
                self.layer_histories["goodness_pos"]
            ),
            "last_epoch_goodness_neg_mean": _mean_last_epoch(
                self.layer_histories["goodness_neg"]
            ),
            "last_epoch_firing_pos_mean": _mean_last_epoch(
                self.layer_histories["spike_out_pos"]
            ),
            "last_epoch_firing_neg_mean": _mean_last_epoch(
                self.layer_histories["spike_out_neg"]
            ),
        }
        metrics.update(self.train_memory.summary())
        metrics.update(
            self.backward_profile.summary(
                "bp_only_gpu_mem",
                include_time=False,
                include_ops=False,
                include_throughput=False,
            )
        )
        metrics.update(
            self.manual_grad_profile.summary(
                "manual_grad",
                include_time=True,
                include_ops=True,
                include_throughput=True,
            )
        )
        metrics.update(
            self.autograd_cmp_profile.summary(
                "autograd_cmp",
                include_time=True,
                include_ops=False,
                include_throughput=True,
            )
        )
        metrics["manual_vs_autograd_alloc_reduction_pct"] = _safe_pct_reduction(
            metrics.get("manual_grad_peak_alloc_mean_mb"),
            metrics.get("autograd_cmp_peak_alloc_mean_mb"),
        )
        metrics["manual_vs_autograd_time_reduction_pct"] = _safe_pct_reduction(
            metrics.get("manual_grad_time_mean_ms"),
            metrics.get("autograd_cmp_time_mean_ms"),
        )
        metrics["manual_vs_autograd_alloc_per_sample_reduction_pct"] = (
            _safe_pct_reduction(
                metrics.get("manual_grad_peak_alloc_per_sample_kb"),
                metrics.get("autograd_cmp_peak_alloc_per_sample_kb"),
            )
        )
        metrics["manual_vs_autograd_time_per_sample_reduction_pct"] = (
            _safe_pct_reduction(
                metrics.get("manual_grad_time_per_sample_us"),
                metrics.get("autograd_cmp_time_per_sample_us"),
            )
        )

        manual_samples_per_s = metrics.get("manual_grad_samples_per_s")
        autograd_samples_per_s = metrics.get("autograd_cmp_samples_per_s")
        if (
            manual_samples_per_s is not None
            and autograd_samples_per_s is not None
            and autograd_samples_per_s > 0
        ):
            metrics["manual_vs_autograd_throughput_gain_pct"] = 100.0 * (
                manual_samples_per_s / autograd_samples_per_s - 1.0
            )
        else:
            metrics["manual_vs_autograd_throughput_gain_pct"] = None
        return metrics
