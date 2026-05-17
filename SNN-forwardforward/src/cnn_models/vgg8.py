from __future__ import annotations

from src.cnn_models.official_vgg_base import OfficialVGGFFAdapter


# 8 convolution layers, still expressed with the official VGG cfg notation.
VGG8_CFG = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M"]


class VGG8Net(OfficialVGGFFAdapter):
    def __init__(self, *, in_channels: int, H: int, W: int, mode_config, **kwargs):
        super().__init__(
            cfg=VGG8_CFG,
            in_channels=in_channels,
            H=H,
            W=W,
            mode_config=mode_config,
            **kwargs,
        )


def build_vgg8_model(*, mode_config, in_channels: int, H: int, W: int, **kwargs):
    return VGG8Net(
        in_channels=in_channels,
        H=H,
        W=W,
        mode_config=mode_config,
        **kwargs,
    )
