from src.ff_strategies.goodness import (
    GOODNESS_AUTO,
    GOODNESS_MEMBRANE_POTENTIAL_SQUARE_MEAN,
    compute_goodness,
    list_goodness_strategies,
    register_goodness_strategy,
    resolve_goodness_strategy_name,
)
from src.ff_strategies.negative_sampling import (
    NEG_SAMPLE_AUTO,
    NEG_SAMPLE_GLOBAL_FOURIER_LABEL,
    generate_pos_n_neg_sample,
    list_negative_sampling_strategies,
    register_negative_sampling_strategy,
    resolve_negative_sampling_strategy_name,
)
from src.ff_strategies.objectives import (
    HIDDEN_LOSS_AUTO,
    compute_hidden_pair_loss,
    list_hidden_loss_strategies,
    register_hidden_loss_strategy,
    resolve_hidden_loss_strategy_name,
)

__all__ = [
    "GOODNESS_AUTO",
    "GOODNESS_MEMBRANE_POTENTIAL_SQUARE_MEAN",
    "HIDDEN_LOSS_AUTO",
    "NEG_SAMPLE_AUTO",
    "NEG_SAMPLE_GLOBAL_FOURIER_LABEL",
    "compute_goodness",
    "compute_hidden_pair_loss",
    "generate_pos_n_neg_sample",
    "list_goodness_strategies",
    "list_hidden_loss_strategies",
    "list_negative_sampling_strategies",
    "register_goodness_strategy",
    "register_hidden_loss_strategy",
    "register_negative_sampling_strategy",
    "resolve_goodness_strategy_name",
    "resolve_hidden_loss_strategy_name",
    "resolve_negative_sampling_strategy_name",
]

# Import the local customization template for optional side-effect
# registrations. Keeping this at package init means researchers can add a new
# strategy in one place and use it immediately from CLI arguments.
from src.ff_strategies import custom_strategies as _custom_strategies  # noqa: E402,F401
