"""
Local customization template for FF experiments.

Add your own strategy functions here, then register them with a unique name.
Because this module is imported by `src.ff_strategies`, new registrations
become available automatically to the rest of the project and to CLI args.

Examples
--------
1. Custom goodness:

    def cube_mean_goodness(freq, T, *, membrane_potential=None, context=None):
        del membrane_potential, context
        return (T * freq.pow(3)).flatten(1).mean(1, keepdim=True)

    register_goodness_strategy("cube_mean", cube_mean_goodness)

   If the strategy should also support analytical `manual` gradients for
   MLP/CNN layers, also provide:
   - `manual_activity_transform(freq, T)`
   - `manual_input_gradient_transform(freq, T)`

2. Custom hidden-layer loss:

    def margin_pair_loss(pos_goodness, neg_goodness, threshold, mode_config):
        del mode_config
        margin = threshold - (pos_goodness - neg_goodness)
        return torch.relu(margin).mean()

    register_hidden_loss_strategy("margin_pair", margin_pair_loss)

3. Custom negative sampling:

    def mixup_neg_sampler(x, y, num_classes):
        del y, num_classes
        shuffled = x[torch.randperm(x.shape[0], device=x.device)]
        return x, 0.5 * x + 0.5 * shuffled

    register_negative_sampling_strategy("mixup_neg", mixup_neg_sampler)

After registration, you can run experiments like:
    -goodness_strategy cube_mean
    -hidden_loss_strategy margin_pair
    -neg_sample_strategy mixup_neg
"""

from __future__ import annotations

import torch

from src.ff_strategies.goodness import register_goodness_strategy
from src.ff_strategies.negative_sampling import register_negative_sampling_strategy
from src.ff_strategies.objectives import register_hidden_loss_strategy


# Keep the file import-safe by default. Uncomment and adapt the examples above
# when you want to activate a custom strategy.
_ = (
    torch,
    register_goodness_strategy,
    register_hidden_loss_strategy,
    register_negative_sampling_strategy,
)
