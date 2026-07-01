import torch.nn as nn

from src.models.registry import MODEL_REGISTRY


def build_ghostnetv2_100(num_classes: int, pretrained: bool = True) -> nn.Module:
    import timm

    return timm.create_model(
        "ghostnetv2_100",
        pretrained=pretrained,
        num_classes=num_classes,
    )


MODEL_REGISTRY.register("ghostnetv2_100", factory=build_ghostnetv2_100)
