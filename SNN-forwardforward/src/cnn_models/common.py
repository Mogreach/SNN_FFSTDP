from __future__ import annotations

from dataclasses import dataclass
import torch.nn as nn


CNN_FAMILY_MODELS = {"CNN", "VGG6", "VGG8", "VGG11", "ResNet"}


@dataclass(frozen=True)
class ConvLayerSpec:
    """
    One hidden convolutional layer in the FF-SNN CNN stack.

    The first five fields match the legacy 5-tuple conv_cfg format.
    pool_* fields are optional so VGG-like models can place pooling only
    after the last convolution inside a stage.
    """

    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_kernel_size: int | None = 2
    pool_stride: int | None = 2

    @property
    def uses_pool(self) -> bool:
        return self.pool_kernel_size is not None and self.pool_stride is not None

    def to_legacy_tuple(self) -> tuple[int, int, int, int, int, int | None, int | None]:
        return (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.pool_kernel_size,
            self.pool_stride,
        )


def is_cnn_family_model(model_name: str) -> bool:
    return model_name in CNN_FAMILY_MODELS


class FFNetDelegatorMixin:
    """
    Wrapper models inherit a library backbone class and delegate the project-
    specific FF training / prediction API to `self.ff_net`.
    """

    RUNTIME_STAT_NAMES = (
        "last_backward_peak_alloc_bytes",
        "last_backward_peak_reserved_bytes",
        "last_manual_grad_peak_alloc_bytes",
        "last_manual_grad_peak_reserved_bytes",
        "last_manual_grad_time_ms",
        "last_manual_grad_ops_est",
        "last_backward_cmp_peak_alloc_bytes",
        "last_backward_cmp_peak_reserved_bytes",
        "last_backward_cmp_time_ms",
    )

    @property
    def layers(self):
        return self.ff_net.layers

    def _sync_runtime_stats(self) -> None:
        for stat_name in self.RUNTIME_STAT_NAMES:
            setattr(self, stat_name, getattr(self.ff_net, stat_name, None))

    def predict_multiple(self, x):
        prediction = self.ff_net.predict_multiple(x)
        self._sync_runtime_stats()
        return prediction

    def predict_winner(self, x):
        prediction = self.ff_net.predict_winner(x)
        self._sync_runtime_stats()
        return prediction

    def train_supervised(self, x, label, frozen):
        step_result = self.ff_net.train_supervised(x, label, frozen)
        self._sync_runtime_stats()
        return step_result

    def train_unsupervised(self, x, label, frozen):
        step_result = self.ff_net.train_unsupervised(x, label, frozen)
        self._sync_runtime_stats()
        return step_result

    def train_ff_stdp(self, x, label, frozen):
        step_result = self.ff_net.train_ff_stdp(x, label, frozen)
        self._sync_runtime_stats()
        return step_result

    def set_label_reference_bank(self, label_reference_bank):
        if hasattr(self.ff_net, "set_label_reference_bank"):
            self.ff_net.set_label_reference_bank(label_reference_bank)

    def save(self, args, path):
        self.ff_net.save(args, path)

    def load(self, path):
        self.ff_net.load(path)
        self._sync_runtime_stats()

    def forward(self, x):
        # Keep `net(x)` usable in ad-hoc debugging; the real training path uses
        # train_* / predict_* methods from the delegated FF core.
        return self.predict_multiple(x)


def normalize_conv_layer_spec(spec) -> ConvLayerSpec:
    """
    Accept legacy tuples, richer tuples, dictionaries or dataclass instances.
    """

    if isinstance(spec, ConvLayerSpec):
        return spec

    if isinstance(spec, dict):
        return ConvLayerSpec(
            in_channels=int(spec["in_channels"]),
            out_channels=int(spec["out_channels"]),
            kernel_size=int(spec.get("kernel_size", 3)),
            stride=int(spec.get("stride", 1)),
            padding=int(spec.get("padding", 1)),
            pool_kernel_size=(
                None
                if spec.get("pool_kernel_size", 2) is None
                else int(spec.get("pool_kernel_size", 2))
            ),
            pool_stride=(
                None
                if spec.get("pool_stride", 2) is None
                else int(spec.get("pool_stride", 2))
            ),
        )

    if isinstance(spec, (list, tuple)):
        if len(spec) == 5:
            in_ch, out_ch, kernel_size, stride, padding = spec
            pool_kernel_size = 2
            pool_stride = 2
        elif len(spec) == 7:
            (
                in_ch,
                out_ch,
                kernel_size,
                stride,
                padding,
                pool_kernel_size,
                pool_stride,
            ) = spec
        else:
            raise ValueError(
                "conv_cfg item must be a 5-tuple, 7-tuple, dict or ConvLayerSpec."
            )
        return ConvLayerSpec(
            in_channels=int(in_ch),
            out_channels=int(out_ch),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            pool_kernel_size=(
                None if pool_kernel_size is None else int(pool_kernel_size)
            ),
            pool_stride=None if pool_stride is None else int(pool_stride),
        )

    raise TypeError(f"Unsupported conv layer spec type: {type(spec)!r}")


def compute_conv_output_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel_size) // stride + 1


def compute_pool_output_size(
    size: int,
    pool_kernel_size: int | None,
    pool_stride: int | None,
) -> int:
    if pool_kernel_size is None or pool_stride is None:
        return size
    return (size - pool_kernel_size) // pool_stride + 1


def build_vgg_stage_specs(
    *,
    in_channels: int,
    stage_channels: tuple[int, ...],
    stage_depths: tuple[int, ...],
    input_height: int,
    input_width: int,
    pool_after_stage: tuple[bool, ...] | None = None,
) -> list[ConvLayerSpec]:
    """
    Build VGG-like stage specs and automatically disable late pooling when the
    feature map is already too small for another 2x2 reduction.
    """

    if len(stage_channels) != len(stage_depths):
        raise ValueError("stage_channels and stage_depths must have the same length.")

    if pool_after_stage is None:
        pool_after_stage = tuple(True for _ in stage_channels)
    elif len(pool_after_stage) != len(stage_channels):
        raise ValueError("pool_after_stage must match the number of stages.")

    specs: list[ConvLayerSpec] = []
    current_in_channels = in_channels
    current_height = input_height
    current_width = input_width

    for out_channels, depth, should_pool_stage in zip(
        stage_channels,
        stage_depths,
        pool_after_stage,
    ):
        for layer_index in range(depth):
            is_last_conv_in_stage = layer_index == depth - 1
            can_pool_now = (
                should_pool_stage
                and is_last_conv_in_stage
                and current_height >= 2
                and current_width >= 2
            )
            specs.append(
                ConvLayerSpec(
                    in_channels=current_in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pool_kernel_size=2 if can_pool_now else None,
                    pool_stride=2 if can_pool_now else None,
                )
            )
            current_in_channels = out_channels

        if should_pool_stage and current_height >= 2 and current_width >= 2:
            current_height = compute_pool_output_size(current_height, 2, 2)
            current_width = compute_pool_output_size(current_width, 2, 2)

    return specs


def build_conv_cfg_from_vgg_cfg(
    cfg: list[int | str] | tuple[int | str, ...],
    *,
    in_channels: int,
) -> list[ConvLayerSpec]:
    """
    Convert the standard VGG config notation into per-convolution FF specs.

    Example:
        [64, 'M', 128, 'M'] ->
        conv(1->64, pool) + conv(64->128, pool)
    """

    specs: list[ConvLayerSpec] = []
    current_in_channels = in_channels

    for item in cfg:
        if item == "M":
            if not specs:
                raise ValueError("VGG cfg cannot start with a pooling marker.")
            last_spec = specs[-1]
            specs[-1] = ConvLayerSpec(
                in_channels=last_spec.in_channels,
                out_channels=last_spec.out_channels,
                kernel_size=last_spec.kernel_size,
                stride=last_spec.stride,
                padding=last_spec.padding,
                pool_kernel_size=2,
                pool_stride=2,
            )
            continue

        out_channels = int(item)
        specs.append(
            ConvLayerSpec(
                in_channels=current_in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_kernel_size=None,
                pool_stride=None,
            )
        )
        current_in_channels = out_channels

    return specs


def extract_vgg_feature_blocks(features: nn.Sequential):
    """
    Parse a SpikingJelly VGG feature stack into per-convolution FF blocks:
    (conv, spiking neuron, optional pool).
    """

    blocks = []
    current_conv = None
    current_spike = None

    for module in features:
        if isinstance(module, nn.Conv2d):
            if current_conv is not None:
                blocks.append((current_conv, current_spike, nn.Identity()))
            current_conv = module
            current_spike = None
            continue

        if isinstance(module, nn.MaxPool2d):
            if current_conv is None or current_spike is None:
                raise RuntimeError("Unexpected VGG feature layout while parsing pooling.")
            blocks.append((current_conv, current_spike, module))
            current_conv = None
            current_spike = None
            continue

        current_spike = module

    if current_conv is not None:
        if current_spike is None:
            raise RuntimeError("Unexpected VGG feature layout: missing spiking neuron.")
        blocks.append((current_conv, current_spike, nn.Identity()))

    return blocks
