"""Genera evidencia Grad-CAM y LIME sobre la selección final de ETAPA 2.

Grad-CAM se calcula en cada componente seleccionado y, cuando la selección es un
ensemble, se combinan mapas ya normalizados con los mismos pesos de soft voting.
La combinación es una explicación aproximada del voto, no una atribución causal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from lime import lime_image
from PIL import Image, ImageOps
from skimage.segmentation import mark_boundaries
from torchvision import transforms

from scripts.etapa_2.stage2_experiments import (
    CLASSES,
    _as_feature_extractor,
    load_numeric_probe,
)
from src.models import resolve_input_size


TARGET_LAYERS = {
    "efficientnet_b0": "features.8.0",
    "shufflenet_v2_x1_0": "conv5.0",
    "mobilenet_v3_small": "conv_head",
    "fastvit_t8": "final_conv.conv_kxk.0.conv",
}


class ExplainableProbe(nn.Module):
    def __init__(self, model_name: str, probe_path: Path):
        super().__init__()
        self.model_name = model_name
        self.backbone = _as_feature_extractor(model_name, pretrained=True)
        probe = load_numeric_probe(probe_path)
        self.feature_norm = probe.feature_norm
        self.register_buffer("mean", torch.as_tensor(probe.mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(probe.scale, dtype=torch.float32))
        self.register_buffer(
            "coefficients", torch.as_tensor(probe.coefficients, dtype=torch.float32)
        )
        self.register_buffer("intercept", torch.as_tensor(probe.intercept, dtype=torch.float32))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        features = (features - self.mean) / self.scale
        if self.feature_norm == "l2":
            features = functional.normalize(features, dim=1)
        return functional.linear(features, self.coefficients, self.intercept)


def image_transform(model_name: str) -> transforms.Compose:
    size = resolve_input_size(model_name, (224, 224))
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def gradcam(model: ExplainableProbe, image: Image.Image, target: int) -> np.ndarray:
    layer_name = TARGET_LAYERS[model.model_name]
    layer = dict(model.backbone.named_modules())[layer_name]
    captured: dict[str, torch.Tensor] = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = layer.register_forward_hook(save_activation)
    tensor = image_transform(model.model_name)(image).unsqueeze(0).requires_grad_(True)
    logits = model(tensor)
    model.zero_grad(set_to_none=True)
    logits[0, target].backward()
    activation = captured["activation"]
    gradient = activation.grad
    weights = gradient.mean(dim=(-2, -1), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = functional.interpolate(
        cam, size=(image.height, image.width), mode="bilinear", align_corners=False
    )[0, 0]
    cam -= cam.min()
    cam /= cam.max().clamp_min(1e-12)
    handle.remove()
    return cam.detach().cpu().numpy()


def choose_cases(predictions: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    chosen: list[tuple[str, pd.Series]] = []
    used: set[int] = set()

    def pick(name: str, candidates: pd.DataFrame, ascending: bool) -> None:
        candidates = candidates.loc[~candidates.index.isin(used)].sort_values(
            "confidence", ascending=ascending
        )
        if candidates.empty:
            return
        row = candidates.iloc[0]
        used.add(int(row.name))
        chosen.append((name, row))

    pick("Acierto fácil", predictions[predictions["correct"]], False)
    pick("Acierto difícil", predictions[predictions["correct"]], True)
    pick("Error de alta confianza", predictions[~predictions["correct"]], False)
    nutritional = predictions[
        predictions["label"].isin(
            ["nitrogen_deficiency", "phosphorus_deficiency", "potassium_deficiency"]
        )
    ]
    potassium = nutritional[nutritional["label"].eq("potassium_deficiency")]
    pick("Deficiencia nutricional (K)", potassium if not potassium.empty else nutritional, True)
    lab = predictions[predictions["environment"].eq("lab")]
    pick("Fondo controlado: atajo potencial", lab, False)
    return chosen


def load_models(output_dir: Path, selection: dict) -> tuple[dict[str, ExplainableProbe], np.ndarray]:
    models = {}
    weights = []
    for model_name in selection["models"]:
        models[model_name] = ExplainableProbe(
            model_name, output_dir / "final" / f"{model_name}_probe.npz"
        ).eval()
        weights.append(selection.get("weights", {}).get(model_name, 1.0))
    weights_array = np.asarray(weights, dtype=float)
    return models, weights_array / weights_array.sum()


def predict_arrays(
    images: np.ndarray,
    models: dict[str, ExplainableProbe],
    weights: np.ndarray,
    batch_size: int = 16,
) -> np.ndarray:
    result = np.zeros((len(images), len(CLASSES)), dtype=np.float64)
    for weight, (model_name, model) in zip(weights, models.items()):
        transform = image_transform(model_name)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = torch.stack(
                    [transform(Image.fromarray(np.uint8(item))) for item in images[start : start + batch_size]]
                )
                probabilities = torch.sigmoid(model(batch))
                probabilities /= probabilities.sum(dim=1, keepdim=True)
                chunks.append(probabilities.cpu().numpy())
        result += weight * np.concatenate(chunks, axis=0)
    return result


def generate(output_dir: Path) -> None:
    torch.set_num_threads(16)
    prediction_path = output_dir / "final" / "predictions.csv"
    predictions = pd.read_csv(prediction_path)
    selection = json.loads((output_dir / "ensemble" / "selection.json").read_text())
    dataset_root = Path(json.loads((output_dir / "dataset_summary.json").read_text())["dataset_root"])
    models, weights = load_models(output_dir, selection)
    cases = choose_cases(predictions)
    figure_dir = output_dir / "interpretability"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(cases), 2, figsize=(10, 3.15 * len(cases)))
    metadata = []
    for row_index, (case_name, row) in enumerate(cases):
        image_path = dataset_root / row["image_path"]
        with Image.open(image_path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            original = source.copy()
        target = int(row["pred_idx"])
        aggregate = np.zeros((original.height, original.width), dtype=np.float64)
        for weight, model in zip(weights, models.values()):
            aggregate += weight * gradcam(model, original, target)
        aggregate -= aggregate.min()
        aggregate /= max(float(aggregate.max()), 1e-12)

        axes[row_index, 0].imshow(original)
        axes[row_index, 0].set_title(
            f"{case_name}\nReal: {row['label']} | Pred.: {row['pred_label']}"
        )
        axes[row_index, 1].imshow(original)
        axes[row_index, 1].imshow(aggregate, cmap="jet", alpha=0.43, vmin=0, vmax=1)
        axes[row_index, 1].set_title(f"Grad-CAM ponderado | confianza={row['confidence']:.3f}")
        for axis in axes[row_index]:
            axis.axis("off")
        metadata.append(
            {
                "case": case_name,
                "image_path": row["image_path"],
                "true_label": row["label"],
                "predicted_label": row["pred_label"],
                "confidence": float(row["confidence"]),
                "environment": row["environment"],
                "source": row["source"],
                "method": "weighted component Grad-CAM" if len(models) > 1 else "Grad-CAM",
            }
        )
    fig.tight_layout()
    fig.savefig(figure_dir / "final_gradcam_cases.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(metadata).to_csv(figure_dir / "cases.csv", index=False)

    # LIME se limita a un caso difícil y al componente con mayor peso. Explicar
    # las 32 perturbaciones con los cuatro backbones no cambia el tipo de evidencia
    # y multiplica el costo de CPU; Grad-CAM ya cubre el ensemble completo.
    lime_name, lime_row = next(
        ((name, row) for name, row in cases if name == "Error de alta confianza"), cases[1]
    )
    lime_path = dataset_root / lime_row["image_path"]
    with Image.open(lime_path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        source.thumbnail((512, 512))
        lime_source = np.asarray(source)
    explainer = lime_image.LimeImageExplainer(random_state=42)
    dominant_name = max(
        selection["models"], key=lambda name: selection.get("weights", {}).get(name, 1.0)
    )
    dominant_models = {dominant_name: models[dominant_name]}
    explanation = explainer.explain_instance(
        lime_source,
        lambda batch: predict_arrays(batch, dominant_models, np.asarray([1.0])),
        top_labels=1,
        hide_color=0,
        num_samples=32,
        batch_size=16,
    )
    label = int(explanation.top_labels[0])
    explained, mask = explanation.get_image_and_mask(
        label, positive_only=False, num_features=8, hide_rest=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.7))
    axes[0].imshow(lime_source)
    axes[0].set_title("Original: fósforo")
    axes[1].imshow(mark_boundaries(explained / 255.0, mask))
    axes[1].set_title("LIME B0 → potasio (32 perturbaciones)")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{lime_name} | confianza del ensemble={lime_row['confidence']:.3f}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figure_dir / "final_lime_case.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (figure_dir / "method.json").write_text(
        json.dumps(
            {
                "selection": selection,
                "gradcam_target_layers": TARGET_LAYERS,
                "lime_case": str(lime_row["image_path"]),
                "lime_samples": 32,
                "lime_component": dominant_name,
                "lime_max_side_pixels": 512,
                "seed": 42,
                "caveat": "Grad-CAM del ensemble combina mapas normalizados de sus componentes con los pesos del soft voting.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/etapa_2"))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
