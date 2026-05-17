from __future__ import annotations

from collections.abc import Callable

import torch


GOODNESS_AUTO = "auto"
GOODNESS_SQUARE = "square"
GOODNESS_SQUARE_MEAN = "square_mean"
GOODNESS_SIGNED_SQUARE_MEAN = "signed_square_mean"

GoodnessFn = Callable[[torch.Tensor, int], torch.Tensor]

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
) -> torch.Tensor:
    resolved_name = resolve_goodness_strategy_name(
        strategy_name,
        default_strategy_name=default_strategy_name,
    )
    return _GOODNESS_REGISTRY[resolved_name](freq, T)


def _square_goodness(freq: torch.Tensor, T: int) -> torch.Tensor:
    return T * freq.pow(2)


def _square_mean_goodness(freq: torch.Tensor, T: int) -> torch.Tensor:
    return (T * freq.pow(2)).flatten(1).mean(1, keepdim=True)


def _signed_square_mean_goodness(freq: torch.Tensor, T: int) -> torch.Tensor:
    # Keeping the sign in the registry makes it easy to experiment with
    # non-negative spike rates today and signed-rate variants later.
    signed = T * freq.abs().pow(2) * freq.sign()
    return signed.flatten(1).mean(1, keepdim=True)


register_goodness_strategy(GOODNESS_SQUARE, _square_goodness)
register_goodness_strategy(GOODNESS_SQUARE_MEAN, _square_mean_goodness)
register_goodness_strategy(
    GOODNESS_SIGNED_SQUARE_MEAN,
    _signed_square_mean_goodness,
)
