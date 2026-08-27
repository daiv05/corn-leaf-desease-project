import pytest
import torch

from src.models import build_model, list_models
from src.models.feature_exposed import FeatureExposedModel

NUM_CLASSES = 9


@pytest.mark.parametrize("model_name", list_models())
def test_forward_returns_logits_and_pooled_features(model_name):
    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    wrapped = FeatureExposedModel(model, model_name)

    with torch.no_grad():
        logits, features = wrapped(torch.randn(1, 3, 224, 224))

    assert logits.shape == (1, NUM_CLASSES)
    assert features.shape == (1, wrapped.feature_dim)


def test_unknown_torchvision_model_falls_back_to_probed_dim():
    model = build_model("shufflenet_v2_x1_0", num_classes=NUM_CLASSES, pretrained=False)
    wrapped = FeatureExposedModel(model, "shufflenet_v2_x1_0")

    assert wrapped.feature_dim == 1024
