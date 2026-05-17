from __future__ import annotations

from spikingjelly.activation_based.model import spiking_vgg

from src.cnn_models.official_vgg_base import OfficialVGGFFAdapter


# Reuse SpikingJelly's official VGG11 / VGG-A stage definition directly.
VGG11_CFG = list(spiking_vgg.cfgs["A"])


class VGG11Net(OfficialVGGFFAdapter):
    def __init__(self, *, in_channels: int, H: int, W: int, mode_config, **kwargs):
        super().__init__(
            cfg=VGG11_CFG,
            in_channels=in_channels,
            H=H,
            W=W,
            mode_config=mode_config,
            **kwargs,
        )


def build_vgg11_model(*, mode_config, in_channels: int, H: int, W: int, **kwargs):
    return VGG11Net(
        in_channels=in_channels,
        H=H,
        W=W,
        mode_config=mode_config,
        **kwargs,
    )
