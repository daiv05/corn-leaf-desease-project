import torch.nn as nn
import torchvision.models as tv_models

from src.models.registry import MODEL_REGISTRY


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = tv_models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_lite0(num_classes: int, pretrained: bool = True) -> nn.Module:
    import timm

    return timm.create_model(
        "efficientnet_lite0",
        pretrained=pretrained,
        num_classes=num_classes,
    )


MODEL_REGISTRY.register("efficientnet_b0", factory=build_efficientnet_b0)
MODEL_REGISTRY.register("efficientnet_lite0", factory=build_efficientnet_lite0)
