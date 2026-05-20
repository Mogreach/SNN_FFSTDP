from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import torch


GOODNESS_AUTO = "auto"
GOODNESS_SPIKE_SQUARE = "spike_square"
GOODNESS_SPIKE_SQUARE_MEAN = "spike_square_mean"
GOODNESS_FREQ_SQUARE = "freq_square"
GOODNESS_FREQ_SQUARE_MEAN = "freq_square_mean"
GOODNESS_MEMBRANE_POTENTIAL_SQUARE_MEAN = "membrane_potential_square_mean"

# Backward-compatible constant aliases kept for older imports.
GOODNESS_SQUARE = GOODNESS_SPIKE_SQUARE
GOODNESS_SQUARE_MEAN = GOODNESS_SPIKE_SQUARE_MEAN

_LEGACY_GOODNESS_ALIASES = {
    "square": GOODNESS_SPIKE_SQUARE,
    "square_mean": GOODNESS_SPIKE_SQUARE_MEAN,
    # Spike frequencies in the current code path are non-negative, so the old
    # signed variant is equivalent to spike_square_mean and can be treated as a
    # backward-compatible alias.
    "signed_square_mean": GOODNESS_SPIKE_SQUARE_MEAN,
}


@dataclass(frozen=True)
class GoodnessContext:
    freq: torch.Tensor
    T: int
    membrane_potential: torch.Tensor | None = None


GoodnessFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class GoodnessStrategySpec:
    fn: GoodnessFn
    manual_activity_transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    manual_input_gradient_transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None


_GOODNESS_REGISTRY: dict[str, GoodnessStrategySpec] = {}


def register_goodness_strategy(
    name: str,
    fn: GoodnessFn,
    *,
    manual_activity_transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    manual_input_gradient_transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> None:
    _GOODNESS_REGISTRY[name] = GoodnessStrategySpec(
        fn=fn,
        manual_activity_transform=manual_activity_transform,
        manual_input_gradient_transform=manual_input_gradient_transform,
    )


def list_goodness_strategies() -> tuple[str, ...]:
    return tuple(sorted(_GOODNESS_REGISTRY))


def resolve_goodness_strategy_name(
    strategy_name: str | None,
    *,
    default_strategy_name: str,
) -> str:
    if strategy_name in (None, "", GOODNESS_AUTO):
        return default_strategy_name
    strategy_name = _LEGACY_GOODNESS_ALIASES.get(strategy_name, strategy_name)
    if strategy_name not in _GOODNESS_REGISTRY:
        raise ValueError(
            "Unknown goodness strategy="
            f"{strategy_name!r}. Registered: {sorted(_GOODNESS_REGISTRY)}"
        )
    return strategy_name


def resolve_goodness_strategy_spec(
    strategy_name: str | None,
    *,
    default_strategy_name: str,
) -> GoodnessStrategySpec:
    resolved_name = resolve_goodness_strategy_name(
        strategy_name,
        default_strategy_name=default_strategy_name,
    )
    return _GOODNESS_REGISTRY[resolved_name]


def compute_goodness(
    freq: torch.Tensor,
    *,
    T: int,
    strategy_name: str | None,
    default_strategy_name: str,
    membrane_potential: torch.Tensor | None = None,
) -> torch.Tensor:
    strategy_spec = resolve_goodness_strategy_spec(
        strategy_name,
        default_strategy_name=default_strategy_name,
    )
    context = GoodnessContext(
        freq=freq,
        T=T,
        membrane_potential=membrane_potential,
    )
    return _invoke_goodness_strategy(strategy_spec.fn, context)


def supports_manual_goodness_gradient(
    strategy_name: str | None,
    *,
    default_strategy_name: str,
) -> bool:
    strategy_spec = resolve_goodness_strategy_spec(
        strategy_name,
        default_strategy_name=default_strategy_name,
    )
    return (
        strategy_spec.manual_activity_transform is not None
        and strategy_spec.manual_input_gradient_transform is not None
    )


# Backward-compatible alias. The newer name is more precise because manual
# updates need both the activity transform and the local gradient factor.
supports_manual_goodness_activity = supports_manual_goodness_gradient


def prepare_manual_goodness_activity(
    freq: torch.Tensor,
    *,
    T: int,
    strategy_name: str | None,
    default_strategy_name: str,
) -> torch.Tensor:
    strategy_spec = resolve_goodness_strategy_spec(
        strategy_name,
        default_strategy_name=default_strategy_name,
    )
    if strategy_spec.manual_activity_transform is None:
        resolved_name = resolve_goodness_strategy_name(
            strategy_name,
            default_strategy_name=default_strategy_name,
        )
        raise NotImplementedError(
            "Goodness strategy "
            f"{resolved_name!r} does not define how manual analytical "
            "gradients should transform firing activity. Use autograd mode "
            "or register the strategy with manual_activity_transform=..."
    )
    return strategy_spec.manual_activity_transform(freq, T)


def prepare_manual_goodness_input_gradient(
    freq: torch.Tensor,
    *,
    T: int,
    strategy_name: str | None,
    default_strategy_name: str,
) -> torch.Tensor:
    strategy_spec = resolve_goodness_strategy_spec(
        strategy_name,
        default_strategy_name=default_strategy_name,
    )
    if strategy_spec.manual_input_gradient_transform is None:
        resolved_name = resolve_goodness_strategy_name(
            strategy_name,
            default_strategy_name=default_strategy_name,
        )
        raise NotImplementedError(
            "Goodness strategy "
            f"{resolved_name!r} does not define the local manual gradient "
            "factor for analytical updates. Use autograd mode or register the "
            "strategy with manual_input_gradient_transform=..."
        )
    return strategy_spec.manual_input_gradient_transform(freq, T)


def _invoke_goodness_strategy(
    fn: GoodnessFn,
    context: GoodnessContext,
) -> torch.Tensor:
    signature = inspect.signature(fn)
    parameters = tuple(signature.parameters.values())
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if len(parameters) == 1 and not accepts_kwargs:
        return fn(context)

    kwargs = {}
    if accepts_kwargs or "membrane_potential" in signature.parameters:
        kwargs["membrane_potential"] = context.membrane_potential
    if accepts_kwargs or "context" in signature.parameters:
        kwargs["context"] = context
    return fn(context.freq, context.T, **kwargs)


def _spike_square_goodness(freq: torch.Tensor, T: int) -> torch.Tensor:
    return (T * freq).pow(2)


def _spike_square_mean_goodness(freq: torch.Tensor, T: int) -> torch.Tensor:
    return (T * freq).pow(2).flatten(1).mean(1, keepdim=True)


def _freq_square_goodness(freq: torch.Tensor, T: int) -> torch.Tensor:
    return T * freq.pow(2)


def _freq_square_mean_goodness(freq: torch.Tensor, T: int) -> torch.Tensor:
    return T * freq.pow(2).flatten(1).mean(1, keepdim=True)


def _manual_spike_activity(freq: torch.Tensor, T: int) -> torch.Tensor:
    return T * freq


def _manual_freq_activity(freq: torch.Tensor, T: int) -> torch.Tensor:
    del T
    return freq


def _manual_spike_square_input_gradient(freq: torch.Tensor, T: int) -> torch.Tensor:
    return 2 * _manual_spike_activity(freq, T)


def _manual_freq_square_input_gradient(freq: torch.Tensor, T: int) -> torch.Tensor:
    return 2 * _manual_freq_activity(freq, T)


def _membrane_potential_square_mean_goodness(
    freq: torch.Tensor,
    T: int,
    *,
    membrane_potential: torch.Tensor | None = None,
) -> torch.Tensor:
    del freq
    if membrane_potential is None:
        raise ValueError(
            "Goodness strategy 'membrane_potential_square_mean' requires "
            "membrane_potential context, but the current layer did not provide it."
    )
    return (T * membrane_potential.pow(2)).flatten(1).mean(1, keepdim=True)


register_goodness_strategy(
    GOODNESS_SPIKE_SQUARE,
    _spike_square_goodness,
    manual_activity_transform=_manual_spike_activity,
    manual_input_gradient_transform=_manual_spike_square_input_gradient,
)
register_goodness_strategy(
    GOODNESS_SPIKE_SQUARE_MEAN,
    _spike_square_mean_goodness,
    manual_activity_transform=_manual_spike_activity,
    manual_input_gradient_transform=_manual_spike_square_input_gradient,
)
register_goodness_strategy(
    GOODNESS_FREQ_SQUARE,
    _freq_square_goodness,
    manual_activity_transform=_manual_freq_activity,
    manual_input_gradient_transform=_manual_freq_square_input_gradient,
)
register_goodness_strategy(
    GOODNESS_FREQ_SQUARE_MEAN,
    _freq_square_mean_goodness,
    manual_activity_transform=_manual_freq_activity,
    manual_input_gradient_transform=_manual_freq_square_input_gradient,
)
register_goodness_strategy(
    GOODNESS_MEMBRANE_POTENTIAL_SQUARE_MEAN,
    _membrane_potential_square_mean_goodness,
)
