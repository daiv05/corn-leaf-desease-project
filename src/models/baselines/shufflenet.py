import torch.nn as nn
import torchvision.models as tv_models

from src.models.registry import MODEL_REGISTRY


def build_shufflenet_v2_x1_0(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
    model = tv_models.shufflenet_v2_x1_0(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


MODEL_REGISTRY.register("shufflenet_v2_x1_0", factory=build_shufflenet_v2_x1_0)
