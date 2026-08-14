import importlib
import os

import pytest
import torch
from torch.utils.data import DataLoader

import src.config

pytest.importorskip("ai_edge_torch")


def _build_test_loader(splits_dir, dataset_root, image_size=(32, 32)):
    os.environ["DATASET_ROOT"] = str(dataset_root)
    importlib.reload(src.config)

    from src.data.dataset import CornDataset
    from src.data.transforms import CornTransformFactory

    factory = CornTransformFactory(target_size=image_size)
    test_dataset = CornDataset(
        csv_path=str(splits_dir / "test.csv"),
        transform=factory.get_pipeline("test"),
    )
    loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return test_dataset, loader


def test_export_to_tflite_produce_archivo(tmp_path, tmp_splits_dir, fake_image_root):
    from src.export.tflite_export import export_to_tflite
    from src.models import build_model

    test_dataset, _ = _build_test_loader(tmp_splits_dir, fake_image_root)
    model = build_model(
        "shufflenet_v2_x1_0", num_classes=len(test_dataset.class_to_idx), pretrained=False
    )
    model.eval()

    output_path = tmp_path / "export" / "model.tflite"
    result = export_to_tflite(model, output_path, (32, 32), torch.device("cpu"))

    assert result == output_path
    assert output_path.exists()
