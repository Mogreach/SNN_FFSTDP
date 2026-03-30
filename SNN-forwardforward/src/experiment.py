from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


LEARNING_MODE_UNSUPERVISED = "unsupervised"
LEARNING_MODE_SUPERVISED = "supervised"

UNSUPERVISED_UPDATE_AUTOGRAD = "autograd"
UNSUPERVISED_UPDATE_MANUAL = "manual"

VALID_LEARNING_MODES = (
    LEARNING_MODE_UNSUPERVISED,
    LEARNING_MODE_SUPERVISED,
)
VALID_UNSUPERVISED_UPDATE_MODES = (
    UNSUPERVISED_UPDATE_AUTOGRAD,
    UNSUPERVISED_UPDATE_MANUAL,
)


@dataclass(frozen=True)
class ProfilingOptions:
    # These flags only control metric collection; they do not decide the actual update rule.
    capture_manual_grad_metrics: bool = True
    capture_autograd_comparison: bool = True


@dataclass(frozen=True)
class ExperimentModeConfig:
    # learning_mode reserves the top-level experiment branch, while
    # unsupervised_update_mode selects the concrete hidden-layer update rule.
    learning_mode: str = LEARNING_MODE_UNSUPERVISED
    unsupervised_update_mode: str = UNSUPERVISED_UPDATE_AUTOGRAD
    profiling: ProfilingOptions = field(default_factory=ProfilingOptions)

    def __post_init__(self) -> None:
        if self.learning_mode not in VALID_LEARNING_MODES:
            raise ValueError(f"Unsupported learning_mode={self.learning_mode}")
        if self.unsupervised_update_mode not in VALID_UNSUPERVISED_UPDATE_MODES:
            raise ValueError(
                f"Unsupported unsupervised_update_mode={self.unsupervised_update_mode}"
            )

    @property
    def is_unsupervised(self) -> bool:
        return self.learning_mode == LEARNING_MODE_UNSUPERVISED

    @property
    def uses_manual_update(self) -> bool:
        return (
            self.learning_mode == LEARNING_MODE_UNSUPERVISED
            and self.unsupervised_update_mode == UNSUPERVISED_UPDATE_MANUAL
        )

    @property
    def uses_autograd_update(self) -> bool:
        return (
            self.learning_mode == LEARNING_MODE_UNSUPERVISED
            and self.unsupervised_update_mode == UNSUPERVISED_UPDATE_AUTOGRAD
        )

    @property
    def run_name(self) -> str:
        if self.is_unsupervised:
            return f"{self.learning_mode}_{self.unsupervised_update_mode}"
        return self.learning_mode

    def to_dict(self) -> dict:
        return {
            "learning_mode": self.learning_mode,
            "unsupervised_update_mode": self.unsupervised_update_mode,
            "capture_manual_grad_metrics": self.profiling.capture_manual_grad_metrics,
            "capture_autograd_comparison": self.profiling.capture_autograd_comparison,
        }

    @classmethod
    def from_args(cls, args) -> "ExperimentModeConfig":
        learning_mode = getattr(args, "learning_mode", None) or getattr(
            args,
            "predict_type",
            LEARNING_MODE_UNSUPERVISED,
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
            unsupervised_update_mode=getattr(
                args,
                "unsupervised_update_mode",
                UNSUPERVISED_UPDATE_AUTOGRAD,
            ),
            profiling=profiling,
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
class UnsupervisedStepResult:
    # The runner consumes one uniform step result no matter which unsupervised
    # implementation branch produced it.
    goodness_pos: list[float]
    goodness_neg: list[float]
    cos_pos: list[Optional[float]]
    cos_neg: list[Optional[float]]
    spike_out_pos: list[float]
    spike_out_neg: list[float]
    profiler: GradientProfilingSnapshot = field(
        default_factory=GradientProfilingSnapshot
    )
