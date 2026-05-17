from __future__ import annotations

import random
from collections.abc import Callable

import torch
import torch.nn.functional as F


NEG_SAMPLE_AUTO = "auto"
NEG_SAMPLE_EMBED_LABEL_ONEHOT = "embed_label_onehot"
NEG_SAMPLE_EMBED_ZERO_ONEHOT = "embed_zero_onehot"
NEG_SAMPLE_SCFF = "SCFF"

NegativeSamplingFn = Callable[[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]

_NEGATIVE_SAMPLING_REGISTRY: dict[str, NegativeSamplingFn] = {}


def register_negative_sampling_strategy(name: str, fn: NegativeSamplingFn) -> None:
    _NEGATIVE_SAMPLING_REGISTRY[name] = fn


def list_negative_sampling_strategies() -> tuple[str, ...]:
    return tuple(sorted(_NEGATIVE_SAMPLING_REGISTRY))


def resolve_negative_sampling_strategy_name(strategy_name: str | None, mode_config) -> str:
    if strategy_name in (None, "", NEG_SAMPLE_AUTO):
        return (
            NEG_SAMPLE_SCFF
            if mode_config.is_supervised
            else NEG_SAMPLE_EMBED_LABEL_ONEHOT
        )
    if strategy_name not in _NEGATIVE_SAMPLING_REGISTRY:
        raise ValueError(
            "Unknown negative sampling strategy="
            f"{strategy_name!r}. Registered: {sorted(_NEGATIVE_SAMPLING_REGISTRY)}"
        )
    return strategy_name


def stdnorm(x, dims=(1, 2, 3)):
    x = x - torch.mean(x, dim=dims, keepdim=True)
    x = x / (1e-10 + torch.std(x, dim=dims, keepdim=True))
    return x


def minmax_norm(x, dims=(1, 2, 3), eps=1e-10):
    x_min = torch.amin(x, dim=dims, keepdim=True)
    x_max = torch.amax(x, dim=dims, keepdim=True)
    x = (x - x_min) / (x_max - x_min + eps)
    return x


def generate_pos_n_neg_sample(
    x,
    y,
    num_classes=10,
    type="SCFF",
    *,
    strategy_name: str | None = None,
    mode_config=None,
):
    """
    Backward-compatible wrapper around the new negative-sampling registry.

    Legacy callers can keep passing `type=...`; new code should prefer
    `strategy_name=...` plus `mode_config` so one experiment setting flows
    through the whole project consistently.
    """
    if strategy_name is None:
        strategy_name = type
    if mode_config is not None:
        strategy_name = resolve_negative_sampling_strategy_name(
            strategy_name,
            mode_config,
        )
    if strategy_name not in _NEGATIVE_SAMPLING_REGISTRY:
        raise ValueError(
            "Unknown negative sampling strategy="
            f"{strategy_name!r}. Registered: {sorted(_NEGATIVE_SAMPLING_REGISTRY)}"
        )
    return _NEGATIVE_SAMPLING_REGISTRY[strategy_name](x, y, num_classes)


def get_y_neg(y, num_classes, device):
    y_neg = y.clone()
    for idx, y_samp in enumerate(y):
        allowed_indices = list(range(num_classes))
        allowed_indices.remove(y_samp.item())
        y_neg[idx] = torch.tensor(allowed_indices)[
            torch.randint(len(allowed_indices), size=(1,))
        ].item()
    return y_neg.to(device)


def overlay_y_on_x(x, y, classes=10):
    x_ = x.clone()
    batch_size = x.shape[0]
    x_[:, :, 0, :classes] *= 0.0
    max_value = x_.max()
    for i in range(batch_size):
        x_[i, :, 0, y[i].item()] = max_value
    return x_


def overlay_label_on_x(x, classes=10):
    x_ = x.clone()
    x_[:, :, 0, :classes] *= 0.0
    x_[:, :, 0, :classes] += 1
    return x_


def overlay_zero_on_x(x, y, classes=10):
    x_ = x.clone()
    batch_size = x.shape[0]
    x_[:, :, 0, :classes] *= 0.0
    x_[:, :, 0, :classes] += 1.0
    for i in range(batch_size):
        x_[i, :, 0, y[i].item()] = 0
    return x_


def _embed_label_onehot_sampler(x, y, num_classes):
    x_pos = overlay_y_on_x(x, y, classes=num_classes)
    y_neg = get_y_neg(y, num_classes, x.device)
    x_neg = overlay_y_on_x(x, y_neg, classes=num_classes)
    return x_pos, x_neg


def _embed_zero_onehot_sampler(x, y, num_classes):
    x_pos = overlay_label_on_x(x, classes=num_classes)
    x_neg = overlay_zero_on_x(x, y, classes=num_classes)
    return x_pos, x_neg


def _scff_sampler(x, y, num_classes):
    del y, num_classes
    p = 1
    batch_size = x.shape[0]
    if batch_size <= 1:
        x_pos = minmax_norm(x + x, dims=(2, 3))
        return x_pos, x_pos.clone()
    x_pos = x + x
    random_indices = (torch.randperm(batch_size - 1, device=x.device) + 1)[
        : min(p, batch_size - 1)
    ]
    labels = torch.arange(batch_size, device=x.device)
    batch_negs = []
    for i in random_indices:
        x_neg = x[(labels + i) % batch_size]
        batch_negs.append(x + x_neg)
    x_neg = torch.cat(batch_negs, dim=0)
    x_pos = minmax_norm(x_pos, dims=(2, 3))
    x_neg = minmax_norm(x_neg, dims=(2, 3))
    return x_pos, x_neg


def generate_continuous_mask(shape, block_scale=8, smooth=True, device="cpu"):
    B, C, H, W = shape
    low_H, low_W = H // block_scale, W // block_scale
    noise = torch.rand((B, C, low_H, low_W), device=device)
    mask = F.interpolate(noise, size=(H, W), mode="bilinear", align_corners=False)
    if smooth:
        kernel = torch.ones((1, 1, 5, 5), device=device) / 25.0
        mask = F.conv2d(mask, kernel, padding=2)
    threshold = random.uniform(0.4, 0.6)
    return (mask > threshold).float()


def generate_negative_samples_continuous(
    x,
    y,
    dataset,
    num_classes=10,
    device="cpu",
    visualize=False,
    block_scale=3,
    smooth=False,
):
    B = x.size(0)
    H, W = x.size(2), x.size(3)
    targets = torch.tensor(dataset.targets, device=device)
    data_cpu = dataset.data.float() / 255.0

    offsets = torch.randint(1, num_classes, (B,), device=device)
    neg_labels = (y.to(device) + offsets) % num_classes
    neg_indices = torch.empty(B, dtype=torch.long, device=device)

    for lbl in range(num_classes):
        pos_mask = neg_labels == lbl
        cnt = int(pos_mask.sum().item())
        if cnt == 0:
            continue
        class_indices_cpu = (dataset.targets == lbl).nonzero(as_tuple=True)[0]
        if len(class_indices_cpu) == 0:
            raise RuntimeError(f"No samples for class {lbl} in dataset.")
        sel = torch.randint(0, len(class_indices_cpu), (cnt,), device=device)
        chosen_cpu = class_indices_cpu[sel.cpu()].long()
        neg_indices[pos_mask] = chosen_cpu.to(device)

    neg_img = data_cpu[neg_indices.cpu()].unsqueeze(1).to(device)
    mask = generate_continuous_mask(
        (B, 1, H, W),
        block_scale=block_scale,
        smooth=smooth,
        device=device,
    )
    neg_samples = x.to(device) * mask + neg_img * (1.0 - mask)

    if visualize:
        import matplotlib.pyplot as plt

        n_vis = min(3, B)
        for i in range(n_vis):
            fig, axs = plt.subplots(1, 4, figsize=(8, 2))
            axs[0].imshow(x[i].cpu().squeeze(), cmap="gray")
            axs[0].set_title(f"Positive ({int(y[i].item())})")
            axs[1].imshow(neg_img[i].cpu().squeeze(), cmap="gray")
            axs[1].set_title(f"NegClass ({int(neg_labels[i].item())})")
            axs[2].imshow(mask[i].cpu().squeeze(), cmap="gray")
            axs[2].set_title("Mask")
            axs[3].imshow(neg_samples[i].cpu().squeeze(), cmap="gray")
            axs[3].set_title("Mixed (Negative)")
            for ax in axs:
                ax.axis("off")
            plt.show()

    return neg_samples


register_negative_sampling_strategy(
    NEG_SAMPLE_EMBED_LABEL_ONEHOT,
    _embed_label_onehot_sampler,
)
register_negative_sampling_strategy(
    NEG_SAMPLE_EMBED_ZERO_ONEHOT,
    _embed_zero_onehot_sampler,
)
register_negative_sampling_strategy(NEG_SAMPLE_SCFF, _scff_sampler)
