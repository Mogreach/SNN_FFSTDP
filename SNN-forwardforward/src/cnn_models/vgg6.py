from __future__ import annotations

from src.cnn_models.official_vgg_base import OfficialVGGFFAdapter


# Official VGG cfg notation:
#   integer -> one 3x3 convolution
#   "M"     -> 2x2 max-pooling
#
# VGG6 here means 6 convolution layers in a VGG-style stage arrangement.
VGG6_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]


class VGG6Net(OfficialVGGFFAdapter):
    def __init__(self, *, in_channels: int, H: int, W: int, mode_config, **kwargs):
        super().__init__(
            cfg=VGG6_CFG,
            in_channels=in_channels,
            H=H,
            W=W,
            mode_config=mode_config,
            **kwargs,
        )


def build_vgg6_model(*, mode_config, in_channels: int, H: int, W: int, **kwargs):
    return VGG6Net(
        in_channels=in_channels,
        H=H,
        W=W,
        mode_config=mode_config,
        **kwargs,
    )
