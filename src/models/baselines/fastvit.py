import torch.nn as nn

from src.models.registry import MODEL_REGISTRY


def build_fastvit_t8(num_classes: int, pretrained: bool = True) -> nn.Module:
    import timm

    return timm.create_model(
        "fastvit_t8",
        pretrained=pretrained,
        num_classes=num_classes,
    )


MODEL_REGISTRY.register("fastvit_t8", factory=build_fastvit_t8)
