from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


LEARNING_MODE_UNSUPERVISED = "unsupervised"
LEARNING_MODE_SUPERVISED = "supervised"

HIDDEN_LAYER_UPDATE_AUTOGRAD = "autograd"
HIDDEN_LAYER_UPDATE_MANUAL = "manual"
UPDATE_SCHEDULE_SEPARATE = "separate"
UPDATE_SCHEDULE_PAIRED = "paired"

VALID_LEARNING_MODES = (
    LEARNING_MODE_UNSUPERVISED,
    LEARNING_MODE_SUPERVISED,
)
VALID_HIDDEN_LAYER_UPDATE_MODES = (
    HIDDEN_LAYER_UPDATE_AUTOGRAD,
    HIDDEN_LAYER_UPDATE_MANUAL,
)
VALID_UPDATE_SCHEDULES = (
    UPDATE_SCHEDULE_SEPARATE,
    UPDATE_SCHEDULE_PAIRED,
)


@dataclass(frozen=True)
class ProfilingOptions:
    # These flags only control metric collection; they do not decide the actual update rule.
    capture_manual_grad_metrics: bool = True
    capture_autograd_comparison: bool = True


@dataclass(frozen=True)
class ExperimentModeConfig:
    # learning_mode selects the supervised / unsupervised experiment branch,
    # while hidden_layer_update_mode selects the hidden-layer update rule that
    # both MLP implementations support. update_schedule is kept as the
    # public name for backward compatibility, but it controls update timing for
    # both analytical/manual and autograd hidden-layer updates.
    learning_mode: str = LEARNING_MODE_UNSUPERVISED
    hidden_layer_update_mode: str = HIDDEN_LAYER_UPDATE_AUTOGRAD
    update_schedule: str = UPDATE_SCHEDULE_SEPARATE
    profiling: ProfilingOptions = field(default_factory=ProfilingOptions)

    def __post_init__(self) -> None:
        if self.learning_mode not in VALID_LEARNING_MODES:
            raise ValueError(f"Unsupported learning_mode={self.learning_mode}")
        if self.hidden_layer_update_mode not in VALID_HIDDEN_LAYER_UPDATE_MODES:
            raise ValueError(
                "Unsupported hidden_layer_update_mode="
                f"{self.hidden_layer_update_mode}"
            )
        if self.update_schedule not in VALID_UPDATE_SCHEDULES:
            raise ValueError(
                "Unsupported update_schedule="
                f"{self.update_schedule}"
            )

    @property
    def is_unsupervised(self) -> bool:
        return self.learning_mode == LEARNING_MODE_UNSUPERVISED

    @property
    def is_supervised(self) -> bool:
        return self.learning_mode == LEARNING_MODE_SUPERVISED

    @property
    def uses_manual_update(self) -> bool:
        return self.hidden_layer_update_mode == HIDDEN_LAYER_UPDATE_MANUAL

    @property
    def uses_autograd_update(self) -> bool:
        return self.hidden_layer_update_mode == HIDDEN_LAYER_UPDATE_AUTOGRAD

    @property
    def uses_separate_update_schedule(self) -> bool:
        return self.uses_separate_update_schedule

    @property
    def uses_paired_update_schedule(self) -> bool:
        return self.uses_paired_update_schedule

    @property
    def uses_separate_update_schedule(self) -> bool:
        return self.update_schedule == UPDATE_SCHEDULE_SEPARATE

    @property
    def uses_paired_update_schedule(self) -> bool:
        return self.update_schedule == UPDATE_SCHEDULE_PAIRED

    @property
    def run_name(self) -> str:
        return f"{self.learning_mode}_{self.hidden_layer_update_mode}"

    def to_dict(self) -> dict:
        return {
            "learning_mode": self.learning_mode,
            "hidden_layer_update_mode": self.hidden_layer_update_mode,
            "update_schedule": self.update_schedule,
            "capture_manual_grad_metrics": self.profiling.capture_manual_grad_metrics,
            "capture_autograd_comparison": self.profiling.capture_autograd_comparison,
        }

    @classmethod
    def from_args(cls, args) -> "ExperimentModeConfig":
        learning_mode = getattr(args, "learning_mode", LEARNING_MODE_UNSUPERVISED)
        hidden_layer_update_mode = getattr(
            args,
            "hidden_layer_update_mode",
            HIDDEN_LAYER_UPDATE_AUTOGRAD,
        )
        update_schedule = getattr(
            args,
            "update_schedule",
            UPDATE_SCHEDULE_SEPARATE,
        )
        profiling = ProfilingOptions(
            capture_manual_grad_metrics=getattr(
                args,
                "capture_manual_grad_metrics",
                True,
            ),
            capture_autograd_comparison=getattr(
                args,
                "capture_autograd_comparison",
                True,
            ),
        )
        return cls(
            learning_mode=learning_mode,
            hidden_layer_update_mode=hidden_layer_update_mode,
            update_schedule=update_schedule,
            profiling=profiling,
        )


@dataclass(frozen=True)
class ExperimentStrategyConfig:
    # These three knobs cover the main FF experiment variations that tend to
    # diverge together in research code.
    neg_sample_strategy: str = "auto"
    goodness_strategy: str = "auto"
    hidden_loss_strategy: str = "auto"

    def to_dict(self) -> dict:
        return {
            "neg_sample_strategy": self.neg_sample_strategy,
            "goodness_strategy": self.goodness_strategy,
            "hidden_loss_strategy": self.hidden_loss_strategy,
        }

    @classmethod
    def from_args(cls, args) -> "ExperimentStrategyConfig":
        return cls(
            neg_sample_strategy=getattr(args, "neg_sample_strategy", "auto"),
            goodness_strategy=getattr(args, "goodness_strategy", "auto"),
            hidden_loss_strategy=getattr(args, "hidden_loss_strategy", "auto"),
        )


@dataclass
class GradientProfilingSnapshot:
    # One step may contain both the real training backward and auxiliary
    # comparison/profiling branches, so we keep them in one transport object.
    backward_peak_alloc_bytes: Optional[float] = None
    backward_peak_reserved_bytes: Optional[float] = None
    manual_grad_peak_alloc_bytes: Optional[float] = None
    manual_grad_peak_reserved_bytes: Optional[float] = None
    manual_grad_time_ms: Optional[float] = None
    manual_grad_ops_est: Optional[float] = None
    backward_cmp_peak_alloc_bytes: Optional[float] = None
    backward_cmp_peak_reserved_bytes: Optional[float] = None
    backward_cmp_time_ms: Optional[float] = None


@dataclass
class TrainMemorySnapshot:
    current_alloc_bytes: Optional[float] = None
    current_reserved_bytes: Optional[float] = None
    peak_alloc_bytes: Optional[float] = None
    peak_reserved_bytes: Optional[float] = None


@dataclass
class StepResult:
    # The runner consumes one uniform step result no matter which learning
    # branch / hidden-layer update mode produced it.
    goodness_pos: list[float]
    goodness_neg: list[float]
    cos_pos: list[Optional[float]]
    cos_neg: list[Optional[float]]
    spike_out_pos: list[float]
    spike_out_neg: list[float]
    profiler: GradientProfilingSnapshot = field(
        default_factory=GradientProfilingSnapshot
    )
