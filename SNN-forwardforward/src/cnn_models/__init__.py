from __future__ import annotations

from src.cnn_models.common import is_cnn_family_model


PREDEFINED_CNN_MODEL_NAMES = ("VGG6", "VGG8", "VGG11", "ResNet")


def _get_predefined_cnn_builders():
    # Import lazily to avoid a package-init circular dependency with
    # ff_snn_cnn_sup / ff_snn_cnn_unsup importing cnn_models.common.
    from src.cnn_models.resnet import build_resnet_model
    from src.cnn_models.vgg11 import build_vgg11_model
    from src.cnn_models.vgg6 import build_vgg6_model
    from src.cnn_models.vgg8 import build_vgg8_model

    return {
        "VGG6": build_vgg6_model,
        "VGG8": build_vgg8_model,
        "VGG11": build_vgg11_model,
        "ResNet": build_resnet_model,
    }


def build_predefined_cnn_model(
    model_name,
    *,
    args,
    num_classes,
    mode_config,
    sample_batch,
    device=None,
):
    builders = _get_predefined_cnn_builders()
    if model_name not in builders:
        raise ValueError(f"Unsupported predefined CNN model: {model_name}")

    _, in_channels, height, width = sample_batch.shape
    builder = builders[model_name]
    return builder(
        in_channels=in_channels,
        H=height,
        W=width,
        tau=args.tau,
        epoch=args.epochs,
        T=args.T,
        lr=args.lr,
        v_threshold=args.v_threshold,
        v_threshold_neg=args.v_threshold_neg,
        opt=args.opt,
        loss_threshold=args.loss_threshold,
        num_classes=num_classes,
        mode_config=mode_config,
        device=device,
    )


__all__ = [
    "PREDEFINED_CNN_MODEL_NAMES",
    "build_predefined_cnn_model",
    "is_cnn_family_model",
]
