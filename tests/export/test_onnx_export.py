import importlib
import os

import pytest
import torch
from torch.utils.data import DataLoader

import src.config

pytest.importorskip("onnxruntime")


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


def test_export_to_onnx_produce_archivo(tmp_path, tmp_splits_dir, fake_image_root):
    from src.export.onnx_export import export_to_onnx
    from src.models import build_model

    test_dataset, _ = _build_test_loader(tmp_splits_dir, fake_image_root)
    model = build_model(
        "shufflenet_v2_x1_0", num_classes=len(test_dataset.class_to_idx), pretrained=False
    )
    model.eval()

    output_path = tmp_path / "export" / "model.onnx"
    result = export_to_onnx(model, output_path, (32, 32), torch.device("cpu"))

    assert result == output_path
    assert output_path.exists()


def test_validate_onnx_parity_pasa_para_el_mismo_modelo(tmp_path, tmp_splits_dir, fake_image_root):
    from src.export.onnx_export import export_to_onnx
    from src.export.parity import validate_onnx_parity
    from src.models import build_model

    test_dataset, test_loader = _build_test_loader(tmp_splits_dir, fake_image_root)
    model = build_model(
        "shufflenet_v2_x1_0", num_classes=len(test_dataset.class_to_idx), pretrained=False
    )
    model.eval()

    output_path = tmp_path / "export" / "model.onnx"
    export_to_onnx(model, output_path, (32, 32), torch.device("cpu"))

    result = validate_onnx_parity(
        model, output_path, test_loader, torch.device("cpu"), sample_size=8, tolerance=1e-3
    )

    assert result.passed
    assert result.max_abs_prob_diff < 1e-3
    assert result.agreement_rate == 1.0
