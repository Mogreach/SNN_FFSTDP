from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from spikingjelly.activation_based import layer, neuron, surrogate
from spikingjelly.activation_based.model import spiking_vgg

from src.cnn_models.common import (
    FFNetDelegatorMixin,
    build_conv_cfg_from_vgg_cfg,
    extract_vgg_feature_blocks,
)
from src.ff_snn_cnn_sup import ConvNet as SupervisedConvNet
from src.ff_snn_cnn_unsup import ConvNet as UnsupervisedConvNet


class OfficialSpikingVGGBackbone(spiking_vgg.SpikingVGG):
    """
    Thin subclass over SpikingJelly's official SpikingVGG.

    We override `make_layers` only to support arbitrary input channels and to
    keep `bias=False`, which matches the current FF hidden-layer math.
    """

    def __init__(
        self,
        *,
        cfg,
        in_channels: int,
        num_classes: int,
        v_threshold: float,
    ):
        self.input_channels = in_channels
        self.backbone_cfg = list(cfg)
        super().__init__(
            cfg=list(cfg),
            batch_norm=False,
            num_classes=num_classes,
            init_weights=True,
            spiking_neuron=neuron.IFNode,
            v_reset=None,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            step_mode="s",
        )

    def make_layers(
        self,
        cfg,
        batch_norm=False,
        norm_layer=None,
        neuron: callable = None,
        **kwargs,
    ):
        del batch_norm, norm_layer
        layers = []
        in_channels = self.input_channels
        for v in cfg:
            if v == "M":
                layers.append(layer.MaxPool2d(kernel_size=2, stride=2))
                continue

            conv2d = layer.Conv2d(
                in_channels,
                int(v),
                kernel_size=3,
                padding=1,
                bias=False,
            )
            layers.extend([conv2d, neuron(**deepcopy(kwargs))])
            in_channels = int(v)
        return nn.Sequential(*layers)


def _migrate_official_vgg_modules_to_ff_net(ff_net, features: nn.Sequential) -> None:
    feature_blocks = extract_vgg_feature_blocks(features)
    hidden_layers = ff_net.layers[:-1]
    if len(hidden_layers) != len(feature_blocks):
        raise RuntimeError(
            "Official VGG backbone and FF hidden-layer stack disagree on the "
            f"number of convolution blocks: {len(feature_blocks)} vs {len(hidden_layers)}."
        )

    for ff_layer, (conv_module, spike_module, pool_module) in zip(
        hidden_layers,
        feature_blocks,
    ):
        ff_layer.layer[0] = conv_module
        ff_layer.layer[1] = spike_module
        ff_layer.layer[2] = pool_module


class OfficialVGGFFAdapter(OfficialSpikingVGGBackbone, FFNetDelegatorMixin):
    """
    Official backbone + current-project FF training core.

    The hidden-layer modules come from SpikingJelly's VGG implementation, while
    the output head and FF training logic remain the project-specific part.
    """

    def __init__(
        self,
        *,
        cfg,
        ff_net_cls=None,
        in_channels: int,
        H: int,
        W: int,
        mode_config,
        device=None,
        **kwargs,
    ):
        self.mode_config = mode_config
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        super().__init__(
            cfg=cfg,
            in_channels=in_channels,
            num_classes=kwargs["num_classes"],
            v_threshold=kwargs["v_threshold"],
        )

        if ff_net_cls is None:
            ff_net_cls = (
                UnsupervisedConvNet if mode_config.is_unsupervised else SupervisedConvNet
            )
        conv_cfg = build_conv_cfg_from_vgg_cfg(cfg, in_channels=in_channels)
        self.ff_net = ff_net_cls(
            conv_cfg=conv_cfg,
            H=H,
            W=W,
            mode_config=mode_config,
            device=self.device,
            **kwargs,
        )
        _migrate_official_vgg_modules_to_ff_net(self.ff_net, self.features)

        # The official classifier is not used because the current experiments
        # concatenate hidden-layer spike features into a project-specific head.
        self.features = nn.Identity()
        self.avgpool = nn.Identity()
        self.classifier = nn.Identity()
        self.to(self.device)


def build_official_vgg_model(
    *,
    cfg,
    mode_config,
    in_channels: int,
    H: int,
    W: int,
    **kwargs,
):
    ff_net_cls = UnsupervisedConvNet if mode_config.is_unsupervised else SupervisedConvNet
    return OfficialVGGFFAdapter(
        cfg=cfg,
        ff_net_cls=ff_net_cls,
        in_channels=in_channels,
        H=H,
        W=W,
        mode_config=mode_config,
        **kwargs,
    )
