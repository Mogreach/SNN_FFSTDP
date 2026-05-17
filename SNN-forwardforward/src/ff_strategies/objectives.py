from __future__ import annotations

from collections.abc import Callable

import torch

from src.loss import (
    ff_pairwise_goodness_loss,
    ff_scaled_supervised_delta_loss,
    ff_supervised_delta_loss,
)


HIDDEN_LOSS_AUTO = "auto"
HIDDEN_LOSS_PAIRWISE = "pairwise_goodness"
HIDDEN_LOSS_SUPERVISED_DELTA = "supervised_delta"
HIDDEN_LOSS_SCALED_SUPERVISED_DELTA = "scaled_supervised_delta"

HiddenLossFn = Callable[[torch.Tensor, torch.Tensor, float, object], torch.Tensor]

_HIDDEN_LOSS_REGISTRY: dict[str, HiddenLossFn] = {}


def register_hidden_loss_strategy(name: str, fn: HiddenLossFn) -> None:
    _HIDDEN_LOSS_REGISTRY[name] = fn


def list_hidden_loss_strategies() -> tuple[str, ...]:
    return tuple(sorted(_HIDDEN_LOSS_REGISTRY))


def resolve_hidden_loss_strategy_name(strategy_name: str | None, mode_config) -> str:
    if strategy_name in (None, "", HIDDEN_LOSS_AUTO):
        return (
            HIDDEN_LOSS_SUPERVISED_DELTA
            if mode_config.is_supervised
            else HIDDEN_LOSS_PAIRWISE
        )
    if strategy_name not in _HIDDEN_LOSS_REGISTRY:
        raise ValueError(
            "Unknown hidden loss strategy="
            f"{strategy_name!r}. Registered: {sorted(_HIDDEN_LOSS_REGISTRY)}"
        )
    return strategy_name


def compute_hidden_pair_loss(
    pos_goodness: torch.Tensor,
    neg_goodness: torch.Tensor,
    *,
    threshold: float,
    strategy_name: str | None,
    mode_config,
) -> torch.Tensor:
    resolved_name = resolve_hidden_loss_strategy_name(strategy_name, mode_config)
    return _HIDDEN_LOSS_REGISTRY[resolved_name](
        pos_goodness,
        neg_goodness,
        threshold,
        mode_config,
    )


def _pairwise_goodness_loss(
    pos_goodness: torch.Tensor,
    neg_goodness: torch.Tensor,
    threshold: float,
    mode_config,
) -> torch.Tensor:
    del mode_config
    return ff_pairwise_goodness_loss(pos_goodness, neg_goodness, threshold)


def _supervised_delta_loss(
    pos_goodness: torch.Tensor,
    neg_goodness: torch.Tensor,
    threshold: float,
    mode_config,
) -> torch.Tensor:
    del mode_config
    return ff_supervised_delta_loss(pos_goodness, neg_goodness, threshold)


def _scaled_supervised_delta_loss(
    pos_goodness: torch.Tensor,
    neg_goodness: torch.Tensor,
    threshold: float,
    mode_config,
) -> torch.Tensor:
    del mode_config
    return ff_scaled_supervised_delta_loss(pos_goodness, neg_goodness, threshold)


register_hidden_loss_strategy(HIDDEN_LOSS_PAIRWISE, _pairwise_goodness_loss)
register_hidden_loss_strategy(HIDDEN_LOSS_SUPERVISED_DELTA, _supervised_delta_loss)
register_hidden_loss_strategy(
    HIDDEN_LOSS_SCALED_SUPERVISED_DELTA,
    _scaled_supervised_delta_loss,
)
