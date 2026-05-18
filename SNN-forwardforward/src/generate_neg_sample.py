"""
Legacy compatibility wrapper for negative-sample generation.

The project now keeps the actual strategy implementations in
`src.ff_strategies.negative_sampling`, but many model files still import from
this module path. Re-exporting here lets us migrate incrementally without
breaking those imports.
"""

from src.ff_strategies.negative_sampling import (
    NEG_SAMPLE_AUTO,
    NEG_SAMPLE_EMBED_LABEL_ONEHOT,
    NEG_SAMPLE_EMBED_ZERO_ONEHOT,
    NEG_SAMPLE_GLOBAL_FOURIER_LABEL,
    NEG_SAMPLE_SCFF,
    generate_continuous_mask,
    generate_global_fourier_label_pos_n_neg_sample,
    generate_negative_samples_continuous,
    generate_pos_n_neg_sample,
    get_y_neg,
    list_negative_sampling_strategies,
    minmax_norm,
    overlay_label_on_x,
    overlay_y_on_x,
    overlay_zero_on_x,
    register_negative_sampling_strategy,
    resolve_negative_sampling_strategy_name,
    stdnorm,
)

__all__ = [
    "NEG_SAMPLE_AUTO",
    "NEG_SAMPLE_EMBED_LABEL_ONEHOT",
    "NEG_SAMPLE_EMBED_ZERO_ONEHOT",
    "NEG_SAMPLE_GLOBAL_FOURIER_LABEL",
    "NEG_SAMPLE_SCFF",
    "generate_continuous_mask",
    "generate_global_fourier_label_pos_n_neg_sample",
    "generate_negative_samples_continuous",
    "generate_pos_n_neg_sample",
    "get_y_neg",
    "list_negative_sampling_strategies",
    "minmax_norm",
    "overlay_label_on_x",
    "overlay_y_on_x",
    "overlay_zero_on_x",
    "register_negative_sampling_strategy",
    "resolve_negative_sampling_strategy_name",
    "stdnorm",
]
