from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import torch


GOODNESS_AUTO = "auto"
GOODNESS_SQUARE = "square"
GOODNESS_SQUARE_MEAN = "square_mean"
GOODNESS_MEMBRANE_POTENTIAL_SQUARE_MEAN = "membrane_potential_square_mean"

_LEGACY_GOODNESS_ALIASES = {
    # Spike frequencies in the current code path are non-negative, so the old
    # signed variant is equivalent to square_mean and can be treated as a
    # backward-compatible alias.
    "signed_square_mean": GOODNESS_SQUARE_MEAN,
}


@dataclass(frozen=True)
class GoodnessContext:
    freq: torch.Tensor
    T: int
    membrane_potential: torch.Tensor | None = None


GoodnessFn = Callable[..., torch.Tensor]

_GOODNESS_REGISTRY: dict[str, GoodnessFn] = {}


def register_goodness_strategy(name: str, fn: GoodnessFn) -> None:
    _GOODNESS_REGISTRY[name] = fn


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


def compute_goodness(
    freq: torch.Tensor,
    *,
    T: int,
    strategy_name: str | None,
    default_strategy_name: str,
    membrane_potential: torch.Tensor | None = None,
) -> torch.Tensor:
    resolved_name = resolve_goodness_strategy_name(
        strategy_name,
        default_strategy_name=default_strategy_name,
    )
    context = GoodnessContext(
        freq=freq,
        T=T,
        membrane_potential=membrane_potential,
    )
    return _invoke_goodness_strategy(_GOODNESS_REGISTRY[resolved_name], context)


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


def _square_goodness(freq: torch.Tensor, T: int) -> torch.Tensor:
    return (T*freq).pow(2)


def _square_mean_goodness(freq: torch.Tensor, T: int) -> torch.Tensor:
    return (T * freq).pow(2).flatten(1).mean(1, keepdim=True)


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


register_goodness_strategy(GOODNESS_SQUARE, _square_goodness)
register_goodness_strategy(GOODNESS_SQUARE_MEAN, _square_mean_goodness)
register_goodness_strategy(
    GOODNESS_MEMBRANE_POTENTIAL_SQUARE_MEAN,
    _membrane_potential_square_mean_goodness,
)
