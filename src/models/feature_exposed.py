from __future__ import annotations

import torch
import torch.nn as nn

# Nombres de modelos del registry construidos con torchvision (no exponen
# forward_features/forward_head como timm), junto con la dimensión de su
# vector de features pooled antes del head de clasificación.
_TORCHVISION_FEATURE_DIMS = {
    "efficientnet_b0": 1280,
    "shufflenet_v2_x1_0": 1024,
}


class FeatureExposedModel(nn.Module):
    """
    Envuelve un modelo del registry para que su forward retorne
    `(logits, pooled_features)` en vez de solo `logits`.

    `pooled_features` es el vector de la penúltima capa (antes del head de
    clasificación), usado para calcular distancia de Mahalanobis en el
    detector de imágenes fuera de dominio (OOD).
    """

    def __init__(self, model: nn.Module, model_name: str) -> None:
        super().__init__()
        self.model = model
        self.model_name = model_name
        # `num_features` de timm describe el canal del backbone, no siempre la
        # dimensión real de `forward_head(..., pre_logits=True)` (ej. ghostnetv2,
        # mobilenet_v3: tienen una conv-head intermedia que cambia el ancho). Se
        # deriva del propio forward con un tensor dummy para no adivinar mal.
        self.feature_dim = self._probe_feature_dim(model_name)

    def _probe_feature_dim(self, model_name: str) -> int:
        if model_name in _TORCHVISION_FEATURE_DIMS:
            return _TORCHVISION_FEATURE_DIMS[model_name]
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                _, features = self._forward_timm(dummy)
            return int(features.shape[1])
        finally:
            self.model.train(was_training)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_name == "efficientnet_b0":
            return self._forward_torchvision_efficientnet(x)
        if self.model_name == "shufflenet_v2_x1_0":
            return self._forward_torchvision_shufflenet(x)
        return self._forward_timm(x)

    def _forward_timm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_features = self.model.forward_features(x)
        pooled_features = self.model.forward_head(raw_features, pre_logits=True)
        logits = self.model.forward_head(raw_features)
        return logits, pooled_features

    def _forward_torchvision_efficientnet(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.features(x)
        pooled = self.model.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        logits = self.model.classifier(pooled)
        return logits, pooled

    def _forward_torchvision_shufflenet(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        pooled = x.mean([2, 3])
        logits = self.model.fc(pooled)
        return logits, pooled
