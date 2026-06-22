import torch.nn as nn
import torchvision.models as tv_models

from src.models.registry import MODEL_REGISTRY


def build_mobilenet_v3_large(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


MODEL_REGISTRY.register("mobilenet_v3_large", factory=build_mobilenet_v3_large)
