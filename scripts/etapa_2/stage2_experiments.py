"""Experimentos reproducibles para el informe de ETAPA 2.

El protocolo mantiene el holdout fuera de tuning, comparación, ensemble y CV. Dado
que la máquina local no expone CUDA, la evaluación usa backbones ImageNet congelados
y optimiza una cabeza lineal incremental sobre sus embeddings. Esto permite ejecutar
el corpus completo en CPU sin presentar el resultado como fine-tuning end-to-end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from PIL import Image, ImageOps
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models import build_model, resolve_input_size


SEED = 42
HOLDOUT_SEED = 4202
CLASSES = [
    "common_rust",
    "fall_armyworm",
    "gray_leaf_spot",
    "healthy",
    "lethal_necrosis",
    "nitrogen_deficiency",
    "northern_corn_leaf_blight",
    "phosphorus_deficiency",
    "potassium_deficiency",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
DEFAULT_MODELS = [
    "efficientnet_b0",
    "shufflenet_v2_x1_0",
    "mobilenet_v3_small",
    "fastvit_t8",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_from_filename(filename: str, label: str, environment: str) -> str:
    stem = Path(filename).stem
    prefix = f"{label}_"
    body = stem[len(prefix) :] if stem.startswith(prefix) else stem
    match = re.match(rf"(.+?)_{re.escape(environment)}_(.+)$", body)
    return match.group(1) if match else "unknown"


def _inspect_image(path: Path) -> tuple[int, int, float, str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        width, height = image.size
        image.thumbnail((256, 256))
        gray = np.asarray(image)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return width, height, blur_score, _sha256(path)


def _fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    columns = ["image_path", "label", "environment", "sha256"]
    for row in frame.sort_values("image_path")[columns].itertuples(index=False, name=None):
        digest.update(("\t".join(map(str, row)) + "\n").encode("utf-8"))
    return digest.hexdigest()


def prepare(dataset_root: Path, output_dir: Path) -> None:
    clean_dir = dataset_root / "clean"
    if not clean_dir.is_dir():
        raise SystemExit(f"No existe el dataset esperado: {clean_dir}")

    candidates: list[tuple[str, str, Path]] = []
    records: list[dict] = []
    invalid: list[dict] = []
    for label in CLASSES:
        for environment in ("lab", "real"):
            env_dir = clean_dir / label / environment
            if not env_dir.is_dir():
                continue
            candidates.extend(
                (label, environment, path)
                for path in sorted(env_dir.iterdir())
                if path.suffix.lower() in IMAGE_SUFFIXES
            )

    workers = max(1, min(16, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inspected = pool.map(_inspect_image, (path for _, _, path in candidates))
        for index, ((label, environment, path), result) in enumerate(
            zip(candidates, inspected), start=1
        ):
            try:
                width, height, blur_score, sha256 = result
            except Exception as exc:  # pragma: no cover - depends on external corpus
                invalid.append({"image_path": str(path), "error": repr(exc)})
                continue
            if index % 2000 == 0:
                print(f"[prepare] {index}/{len(candidates)} imágenes verificadas", flush=True)
            records.append(
                {
                    "image_path": path.relative_to(dataset_root).as_posix(),
                    "label": label,
                    "label_idx": CLASS_TO_IDX[label],
                    "environment": environment,
                    "source": _source_from_filename(path.name, label, environment),
                    "width": width,
                    "height": height,
                    "megapixels": width * height / 1_000_000.0,
                    "blur_score": blur_score,
                    "sha256": sha256,
                }
            )

    raw = pd.DataFrame(records).sort_values("image_path").reset_index(drop=True)
    duplicate_rows = raw[raw.duplicated("sha256", keep="first")].copy()
    master = raw.drop_duplicates("sha256", keep="first").reset_index(drop=True)
    strata = master["label"] + "|" + master["environment"]
    dev, holdout = train_test_split(
        master,
        test_size=0.15,
        random_state=HOLDOUT_SEED,
        stratify=strata,
    )
    dev_strata = dev["label"] + "|" + dev["environment"]
    train, val = train_test_split(
        dev,
        test_size=0.15 / 0.85,
        random_state=SEED,
        stratify=dev_strata,
    )

    split_by_index = {}
    split_by_index.update({int(idx): "train" for idx in train.index})
    split_by_index.update({int(idx): "val" for idx in val.index})
    split_by_index.update({int(idx): "holdout" for idx in holdout.index})
    master["split"] = [split_by_index[int(idx)] for idx in master.index]
    master = master.sort_values("image_path").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_dir / "master_manifest.csv", index=False)
    for split in ("train", "val", "holdout"):
        master[master["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)
    master[master["split"].isin(["train", "val"])].to_csv(
        output_dir / "development.csv", index=False
    )
    duplicate_rows.to_csv(output_dir / "exact_duplicates_removed.csv", index=False)
    pd.DataFrame(invalid).to_csv(output_dir / "invalid_images.csv", index=False)

    fingerprint = _fingerprint(master)
    summary = {
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset_root": str(dataset_root),
        "images_scanned": int(len(raw)),
        "images_valid_unique": int(len(master)),
        "invalid_images": int(len(invalid)),
        "exact_duplicates_removed": int(len(duplicate_rows)),
        "classes": CLASSES,
        "counts_by_class": master["label"].value_counts().sort_index().to_dict(),
        "counts_by_environment": master["environment"].value_counts().to_dict(),
        "counts_by_split": master["split"].value_counts().to_dict(),
        "fingerprint_sha256": fingerprint,
        "holdout_seed": HOLDOUT_SEED,
        "development_seed": SEED,
        "protocol": "70% train, 15% validation, 15% frozen holdout; stratified label+environment",
    }
    _json_dump(output_dir / "dataset_summary.json", summary)
    _json_dump(
        output_dir / "holdout.lock.json",
        {
            "status": "frozen-before-model-selection",
            "created_at_utc": summary["created_at_utc"],
            "dataset_fingerprint_sha256": fingerprint,
            "holdout_manifest_sha256": _sha256(output_dir / "holdout.csv"),
            "rows": int((master["split"] == "holdout").sum()),
            "rule": "No usar etiquetas del holdout hasta ejecutar el subcomando final.",
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, dataset_root: Path, size: tuple[int, int]):
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = self.dataset_root / row["image_path"]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label_idx"])


def _as_feature_extractor(model_name: str, pretrained: bool = True) -> nn.Module:
    model = build_model(model_name, num_classes=len(CLASSES), pretrained=pretrained)
    if model_name == "efficientnet_b0":
        model.classifier = nn.Identity()
    elif model_name == "shufflenet_v2_x1_0":
        model.fc = nn.Identity()
    elif hasattr(model, "reset_classifier"):
        model.reset_classifier(0, global_pool="avg")
    else:  # pragma: no cover - guards future registry entries
        raise ValueError(f"No se sabe retirar el clasificador de {model_name}")
    return model.eval()


def extract_features(
    dataset_root: Path,
    output_dir: Path,
    models: list[str],
    batch_size: int,
    workers: int,
) -> None:
    manifest = pd.read_csv(output_dir / "master_manifest.csv")
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

    pending = [model for model in models if not (feature_dir / f"{model}.npy").exists()]
    if not pending:
        print("[skip] todos los embeddings solicitados ya existen")
        return

    # Se decodifica cada fotografía una sola vez. FastViT consume 256 px; los demás
    # reciben una interpolación a 224 px del mismo batch ya normalizado.
    max_height = max(resolve_input_size(model, (224, 224))[0] for model in pending)
    max_width = max(resolve_input_size(model, (224, 224))[1] for model in pending)
    dataset = ManifestDataset(manifest, dataset_root, (max_height, max_width))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    loaded_models = {model: _as_feature_extractor(model, pretrained=True) for model in pending}
    chunks: dict[str, list[np.ndarray]] = {model: [] for model in pending}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader, start=1):
            for model_name, model in loaded_models.items():
                target_size = resolve_input_size(model_name, (224, 224))
                model_images = images
                if tuple(images.shape[-2:]) != tuple(target_size):
                    model_images = torch_functional.interpolate(
                        images, size=target_size, mode="bilinear", align_corners=False
                    )
                features = model(model_images)
                if features.ndim > 2:
                    features = torch.flatten(features, 1)
                chunks[model_name].append(
                    features.cpu().numpy().astype(np.float32, copy=False)
                )
            if batch_index % 50 == 0:
                print(
                    f"[features] {min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                    flush=True,
                )

    total_seconds = time.perf_counter() - started
    for model_name, model in loaded_models.items():
        matrix = np.concatenate(chunks[model_name], axis=0)
        np.save(feature_dir / f"{model_name}.npy", matrix)
        size = resolve_input_size(model_name, (224, 224))
        params = sum(parameter.numel() for parameter in model.parameters())
        sample = dataset[0][0].unsqueeze(0)
        if tuple(sample.shape[-2:]) != tuple(size):
            sample = torch_functional.interpolate(sample, size=size, mode="bilinear", align_corners=False)
        timings = []
        with torch.inference_mode():
            for _ in range(5):
                model(sample)
            for _ in range(30):
                tick = time.perf_counter()
                model(sample)
                timings.append((time.perf_counter() - tick) * 1000.0)
        info = {
            "model": model_name,
            "pretrained": "ImageNet",
            "feature_dimension": int(matrix.shape[1]),
            "rows": int(matrix.shape[0]),
            "parameters": int(params),
            "fp32_parameter_size_mb": params * 4 / (1024**2),
            "input_size": list(size),
            "cpu_latency_ms_median": statistics.median(timings),
            "cpu_latency_ms_p90": float(np.percentile(timings, 90)),
            "joint_feature_extraction_seconds": total_seconds,
            "torch_version": torch.__version__,
        }
        _json_dump(feature_dir / f"{model_name}.json", info)
        print(json.dumps(info, indent=2))


@dataclass
class ProbeBundle:
    scaler: StandardScaler
    classifier: SGDClassifier
    feature_norm: str
    classes: list[str]

    def transform(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        return normalize(scaled) if self.feature_norm == "l2" else scaled

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(features))


@dataclass
class NumericProbe:
    """Portable inference view of a fitted probe, independent of pickle imports."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    feature_norm: str

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        transformed = (np.asarray(features) - self.mean) / self.scale
        if self.feature_norm == "l2":
            transformed = normalize(transformed)
        logits = transformed @ self.coefficients.T + self.intercept
        # SGDClassifier(log_loss) usa una regresión logística one-vs-rest en
        # multiclase: aplica sigmoide a cada margen y normaliza la fila.
        probabilities = np.empty_like(logits, dtype=np.float64)
        positive = logits >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        negative_exp = np.exp(logits[~positive])
        probabilities[~positive] = negative_exp / (1.0 + negative_exp)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def save_numeric_probe(bundle: ProbeBundle, path: Path) -> None:
    """Persist only numeric state so external audit/interpretability scripts can load it."""

    np.savez_compressed(
        path,
        mean=bundle.scaler.mean_,
        scale=bundle.scaler.scale_,
        coefficients=bundle.classifier.coef_,
        intercept=bundle.classifier.intercept_,
        feature_norm=np.asarray(bundle.feature_norm),
        classes=np.asarray(bundle.classes),
    )


def load_numeric_probe(path: Path) -> NumericProbe:
    artifact = np.load(path, allow_pickle=False)
    return NumericProbe(
        mean=artifact["mean"],
        scale=artifact["scale"],
        coefficients=artifact["coefficients"],
        intercept=artifact["intercept"],
        feature_norm=str(artifact["feature_norm"].item()),
    )


def _class_sample_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    base = np.ones(len(CLASSES), dtype=float)
    present = counts > 0
    base[present] = len(labels) / (present.sum() * counts[present])
    return np.power(base[labels], power)


def fit_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    params: dict,
    val_features: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    trial=None,
) -> tuple[ProbeBundle, list[dict]]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features) if val_features is not None else None
    feature_norm = params.get("feature_norm", "l2")
    if feature_norm == "l2":
        train_scaled = normalize(train_scaled)
        if val_scaled is not None:
            val_scaled = normalize(val_scaled)

    classifier = SGDClassifier(
        loss="log_loss",
        penalty=params.get("penalty", "l2"),
        alpha=float(params.get("alpha", 1e-4)),
        l1_ratio=float(params.get("l1_ratio", 0.15)),
        learning_rate="constant",
        eta0=float(params.get("eta0", 0.01)),
        average=bool(params.get("average", True)),
        random_state=SEED,
    )
    batch_size = int(params.get("batch_size", 256))
    epochs = int(params.get("epochs", 20))
    sample_weights = _class_sample_weights(
        train_labels, float(params.get("balance_power", 0.0))
    )
    rng = np.random.default_rng(SEED)
    history: list[dict] = []
    all_classes = np.arange(len(CLASSES))
    for epoch in range(epochs):
        order = rng.permutation(len(train_labels))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            classifier.partial_fit(
                train_scaled[indices],
                train_labels[indices],
                classes=all_classes,
                sample_weight=sample_weights[indices],
            )
        if val_scaled is not None and val_labels is not None:
            predictions = classifier.predict(val_scaled)
            score = f1_score(val_labels, predictions, average="macro", zero_division=0)
            history.append({"epoch": epoch + 1, "val_macro_f1": float(score)})
            if trial is not None:
                trial.report(score, step=epoch + 1)
                if trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()
    return ProbeBundle(scaler, classifier, feature_norm, CLASSES), history


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def _split_arrays(manifest: pd.DataFrame, matrix: np.ndarray, split: str) -> tuple[np.ndarray, np.ndarray]:
    mask = manifest["split"].eq(split).to_numpy()
    return matrix[mask], manifest.loc[mask, "label_idx"].to_numpy(dtype=int)


def tune(output_dir: Path, model_name: str, trials: int, epochs: int) -> None:
    import optuna

    manifest = pd.read_csv(output_dir / "master_manifest.csv")
    matrix = np.load(output_dir / "features" / f"{model_name}.npy", mmap_mode="r")
    train_x, train_y = _split_arrays(manifest, matrix, "train")
    val_x, val_y = _split_arrays(manifest, matrix, "val")

    baseline_params = {
        "alpha": 1e-4,
        "eta0": 0.01,
        "penalty": "l2",
        "l1_ratio": 0.15,
        "batch_size": 256,
        "balance_power": 0.0,
        "average": True,
        "feature_norm": "l2",
        "epochs": epochs,
    }
    baseline_bundle, baseline_history = fit_probe(
        train_x, train_y, baseline_params, val_x, val_y
    )
    baseline_metrics = _metrics(val_y, baseline_bundle.predict_proba(val_x))

    def objective(trial) -> float:
        penalty = trial.suggest_categorical("penalty", ["l2", "elasticnet"])
        params = {
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-3, log=True),
            "eta0": trial.suggest_float("eta0", 3e-4, 5e-2, log=True),
            "penalty": penalty,
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 0.5)
            if penalty == "elasticnet"
            else 0.0,
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "balance_power": trial.suggest_categorical("balance_power", [0.0, 0.5, 1.0]),
            "average": trial.suggest_categorical("average", [True, False]),
            "feature_norm": trial.suggest_categorical("feature_norm", ["none", "l2"]),
            "epochs": epochs,
        }
        bundle, history = fit_probe(train_x, train_y, params, val_x, val_y, trial=trial)
        trial.set_user_attr("best_epoch", max(history, key=lambda row: row["val_macro_f1"])["epoch"])
        return _metrics(val_y, bundle.predict_proba(val_x))["macro_f1"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4),
        study_name=f"etapa2_{model_name}",
    )
    study.optimize(objective, n_trials=trials, gc_after_trial=True)
    best_params = dict(study.best_trial.params)
    if best_params["penalty"] == "l2":
        best_params["l1_ratio"] = 0.0
    best_params["epochs"] = epochs
    best_bundle, best_history = fit_probe(train_x, train_y, best_params, val_x, val_y)
    best_metrics = _metrics(val_y, best_bundle.predict_proba(val_x))

    tuning_dir = output_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(tuning_dir / "trials.csv", index=False)
    pd.DataFrame(baseline_history).to_csv(tuning_dir / "baseline_history.csv", index=False)
    pd.DataFrame(best_history).to_csv(tuning_dir / "best_history.csv", index=False)
    comparison = pd.DataFrame(
        [
            {"configuration": "baseline_probe", **baseline_metrics},
            {"configuration": "optuna_tuned_probe", **best_metrics},
        ]
    )
    comparison.to_csv(tuning_dir / "baseline_vs_tuned.csv", index=False)
    _json_dump(tuning_dir / "baseline_params.json", baseline_params)
    _json_dump(tuning_dir / "best_params.json", best_params)
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        importance = {}
    pd.DataFrame(
        [{"parameter": key, "importance": value} for key, value in importance.items()]
    ).to_csv(tuning_dir / "parameter_importance.csv", index=False)
    _json_dump(
        tuning_dir / "summary.json",
        {
            "model": model_name,
            "objective": "validation macro-F1",
            "trials_requested": trials,
            "trials_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
            "trials_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
            "best_trial": study.best_trial.number,
            "baseline_metrics": baseline_metrics,
            "best_metrics": best_metrics,
            "best_params": best_params,
        },
    )
    print(json.dumps({"baseline": baseline_metrics, "best": best_metrics, "params": best_params}, indent=2))


def compare_models(output_dir: Path, models: list[str]) -> None:
    manifest = pd.read_csv(output_dir / "master_manifest.csv")
    params = json.loads((output_dir / "tuning" / "best_params.json").read_text())
    comparison_dir = output_dir / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    val_mask = manifest["split"].eq("val").to_numpy()
    train_mask = manifest["split"].eq("train").to_numpy()
    train_y = manifest.loc[train_mask, "label_idx"].to_numpy(dtype=int)
    val_y = manifest.loc[val_mask, "label_idx"].to_numpy(dtype=int)

    for model_name in models:
        matrix = np.load(output_dir / "features" / f"{model_name}.npy", mmap_mode="r")
        bundle, _ = fit_probe(matrix[train_mask], train_y, params)
        probabilities = bundle.predict_proba(matrix[val_mask])
        metrics = _metrics(val_y, probabilities)
        info = json.loads((output_dir / "features" / f"{model_name}.json").read_text())
        predictions = probabilities.argmax(axis=1)
        np.save(comparison_dir / f"{model_name}_val_probabilities.npy", probabilities)
        joblib.dump(bundle, comparison_dir / f"{model_name}_probe.joblib")
        save_numeric_probe(bundle, comparison_dir / f"{model_name}_probe.npz")
        pred_frame = manifest.loc[val_mask].copy()
        pred_frame["pred_idx"] = predictions
        pred_frame["pred_label"] = [CLASSES[index] for index in predictions]
        pred_frame["confidence"] = probabilities.max(axis=1)
        pred_frame.to_csv(comparison_dir / f"{model_name}_val_predictions.csv", index=False)
        rows.append(
            {
                "model": model_name,
                **metrics,
                "parameters": info["parameters"],
                "fp32_parameter_size_mb": info["fp32_parameter_size_mb"],
                "cpu_latency_ms_median": info["cpu_latency_ms_median"],
                "feature_dimension": info["feature_dimension"],
            }
        )
    pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(
        comparison_dir / "models.csv", index=False
    )
    print(pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_string(index=False))


def ensemble(output_dir: Path, models: list[str], trials: int) -> None:
    import optuna

    manifest = pd.read_csv(output_dir / "master_manifest.csv")
    val_y = manifest.loc[manifest["split"].eq("val"), "label_idx"].to_numpy(dtype=int)
    comparison_dir = output_dir / "model_comparison"
    probabilities = {
        model: np.load(comparison_dir / f"{model}_val_probabilities.npy") for model in models
    }
    model_metrics = pd.read_csv(comparison_dir / "models.csv")
    best_row = model_metrics.sort_values("macro_f1", ascending=False).iloc[0]
    uniform = np.mean(np.stack(list(probabilities.values())), axis=0)
    uniform_metrics = _metrics(val_y, uniform)

    def objective(trial) -> float:
        raw = np.array([trial.suggest_float(f"weight_{model}", 0.0, 1.0) for model in models])
        if raw.sum() == 0:
            return 0.0
        weights = raw / raw.sum()
        combined = sum(weights[i] * probabilities[model] for i, model in enumerate(models))
        return _metrics(val_y, combined)["macro_f1"]

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=trials)
    raw_weights = np.array([study.best_params[f"weight_{model}"] for model in models])
    weights = raw_weights / raw_weights.sum()
    weighted = sum(weights[i] * probabilities[model] for i, model in enumerate(models))
    weighted_metrics = _metrics(val_y, weighted)

    error_sets = {
        model: set(np.flatnonzero(probabilities[model].argmax(axis=1) != val_y).tolist())
        for model in models
    }
    complementarity = []
    for i, first in enumerate(models):
        for second in models[i + 1 :]:
            union = error_sets[first] | error_sets[second]
            intersection = error_sets[first] & error_sets[second]
            complementarity.append(
                {
                    "model_a": first,
                    "model_b": second,
                    "errors_a": len(error_sets[first]),
                    "errors_b": len(error_sets[second]),
                    "shared_errors": len(intersection),
                    "jaccard_errors": len(intersection) / len(union) if union else 1.0,
                }
            )
    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(complementarity).to_csv(ensemble_dir / "complementarity.csv", index=False)
    methods = pd.DataFrame(
        [
            {"method": str(best_row["model"]), "type": "best_individual", **best_row.to_dict()},
            {"method": "uniform_soft_voting", "type": "ensemble", **uniform_metrics},
            {"method": "weighted_soft_voting", "type": "ensemble", **weighted_metrics},
        ]
    )
    methods.to_csv(ensemble_dir / "methods.csv", index=False)
    candidates = [
        (str(best_row["model"]), "individual", float(best_row["macro_f1"])),
        ("uniform_soft_voting", "ensemble", uniform_metrics["macro_f1"]),
        ("weighted_soft_voting", "ensemble", weighted_metrics["macro_f1"]),
    ]
    selected_name, selected_type, _ = max(candidates, key=lambda item: item[2])
    selection = {
        "selected_name": selected_name,
        "selected_type": selected_type,
        "models": [selected_name] if selected_type == "individual" else models,
        "weights": {model: float(weights[i]) for i, model in enumerate(models)}
        if selected_name == "weighted_soft_voting"
        else ({model: 1.0 / len(models) for model in models} if selected_type == "ensemble" else {}),
        "validation_macro_f1": max(candidates, key=lambda item: item[2])[2],
        "best_individual": str(best_row["model"]),
        "best_individual_macro_f1": float(best_row["macro_f1"]),
        "uniform_metrics": uniform_metrics,
        "weighted_metrics": weighted_metrics,
        "weights_optimized_on": "validation only",
    }
    _json_dump(ensemble_dir / "selection.json", selection)
    _json_dump(ensemble_dir / "weighted_params.json", study.best_params)
    print(json.dumps(selection, indent=2))


def _selected_probabilities(
    output_dir: Path,
    selection: dict,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    labels: np.ndarray,
    params: dict,
) -> np.ndarray:
    model_probabilities = []
    weights = []
    for model_name in selection["models"]:
        matrix = np.load(output_dir / "features" / f"{model_name}.npy", mmap_mode="r")
        bundle, _ = fit_probe(matrix[train_indices], labels[train_indices], params)
        model_probabilities.append(bundle.predict_proba(matrix[eval_indices]))
        weights.append(selection.get("weights", {}).get(model_name, 1.0))
    weights_array = np.asarray(weights, dtype=float)
    weights_array /= weights_array.sum()
    return sum(weight * probs for weight, probs in zip(weights_array, model_probabilities))


def cross_validate(output_dir: Path, folds: int) -> None:
    manifest = pd.read_csv(output_dir / "master_manifest.csv")
    selection = json.loads((output_dir / "ensemble" / "selection.json").read_text())
    params = json.loads((output_dir / "tuning" / "best_params.json").read_text())
    dev_indices = np.flatnonzero(manifest["split"].isin(["train", "val"]).to_numpy())
    labels = manifest["label_idx"].to_numpy(dtype=int)
    strata = (
        manifest.loc[dev_indices, "label"].astype(str)
        + "|"
        + manifest.loc[dev_indices, "environment"].astype(str)
    ).to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    per_class_rows = []
    for fold, (train_local, eval_local) in enumerate(splitter.split(dev_indices, strata), start=1):
        train_indices = dev_indices[train_local]
        eval_indices = dev_indices[eval_local]
        probs = _selected_probabilities(
            output_dir, selection, train_indices, eval_indices, labels, params
        )
        fold_metrics = _metrics(labels[eval_indices], probs)
        rows.append({"fold": fold, **fold_metrics, "n": len(eval_indices)})
        pred = probs.argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels[eval_indices], pred, labels=np.arange(len(CLASSES)), zero_division=0
        )
        for idx, class_name in enumerate(CLASSES):
            per_class_rows.append(
                {
                    "fold": fold,
                    "class": class_name,
                    "precision": precision[idx],
                    "recall": recall[idx],
                    "f1": f1[idx],
                    "support": support[idx],
                }
            )
    folds_frame = pd.DataFrame(rows)
    summary = {
        metric: {"mean": float(folds_frame[metric].mean()), "std": float(folds_frame[metric].std(ddof=1))}
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    }
    cv_dir = output_dir / "cross_validation"
    cv_dir.mkdir(parents=True, exist_ok=True)
    folds_frame.to_csv(cv_dir / "folds.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(cv_dir / "per_class_folds.csv", index=False)
    _json_dump(cv_dir / "summary.json", {"folds": folds, "selection": selection, "metrics": summary})
    print(folds_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


def final_evaluation(output_dir: Path) -> None:
    final_dir = output_dir / "final"
    if (final_dir / "metrics.json").exists():
        raise SystemExit("La evaluación final ya existe; el holdout no se vuelve a abrir.")
    manifest = pd.read_csv(output_dir / "master_manifest.csv")
    selection = json.loads((output_dir / "ensemble" / "selection.json").read_text())
    params = json.loads((output_dir / "tuning" / "best_params.json").read_text())
    labels = manifest["label_idx"].to_numpy(dtype=int)
    dev_indices = np.flatnonzero(manifest["split"].isin(["train", "val"]).to_numpy())
    holdout_indices = np.flatnonzero(manifest["split"].eq("holdout").to_numpy())
    probabilities_list = []
    weights = []
    bundles = {}
    for model_name in selection["models"]:
        matrix = np.load(output_dir / "features" / f"{model_name}.npy", mmap_mode="r")
        bundle, _ = fit_probe(matrix[dev_indices], labels[dev_indices], params)
        probabilities_list.append(bundle.predict_proba(matrix[holdout_indices]))
        weights.append(selection.get("weights", {}).get(model_name, 1.0))
        bundles[model_name] = bundle
    weights_array = np.asarray(weights, dtype=float)
    weights_array /= weights_array.sum()
    probabilities = sum(
        weight * probs for weight, probs in zip(weights_array, probabilities_list)
    )
    holdout_y = labels[holdout_indices]
    predictions = probabilities.argmax(axis=1)
    metrics = _metrics(holdout_y, probabilities)

    final_dir.mkdir(parents=True, exist_ok=False)
    for model_name, bundle in bundles.items():
        joblib.dump(bundle, final_dir / f"{model_name}_probe.joblib")
        save_numeric_probe(bundle, final_dir / f"{model_name}_probe.npz")
    np.save(final_dir / "probabilities.npy", probabilities)
    pred_frame = manifest.iloc[holdout_indices].copy().reset_index(drop=True)
    pred_frame["pred_idx"] = predictions
    pred_frame["pred_label"] = [CLASSES[index] for index in predictions]
    pred_frame["confidence"] = probabilities.max(axis=1)
    pred_frame["correct"] = pred_frame["label_idx"] == pred_frame["pred_idx"]
    pred_frame.to_csv(final_dir / "predictions.csv", index=False)
    report = classification_report(
        holdout_y,
        predictions,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(final_dir / "classification_report.csv")
    matrix = confusion_matrix(holdout_y, predictions, labels=np.arange(len(CLASSES)))
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(final_dir / "confusion_matrix.csv")
    _json_dump(
        final_dir / "metrics.json",
        {
            **metrics,
            "n": len(holdout_y),
            "selection": selection,
            "evaluated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "holdout_manifest_sha256": _sha256(output_dir / "holdout.csv"),
            "evaluation_count": 1,
        },
    )
    print(json.dumps(metrics, indent=2))


def _group_metrics(frame: pd.DataFrame, dimension: str, min_support: int = 1) -> pd.DataFrame:
    rows = []
    for group_name, group in frame.groupby(dimension, observed=True):
        if len(group) < min_support:
            continue
        true = group["label_idx"].to_numpy(dtype=int)
        pred = group["pred_idx"].to_numpy(dtype=int)
        present = sorted(set(true.tolist()))
        rows.append(
            {
                "dimension": dimension,
                "group": str(group_name),
                "n": len(group),
                "classes_present": len(present),
                "accuracy": accuracy_score(true, pred),
                "precision": precision_score(true, pred, labels=present, average="macro", zero_division=0),
                "recall": recall_score(true, pred, labels=present, average="macro", zero_division=0),
                "f1": f1_score(true, pred, labels=present, average="macro", zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def fairness(output_dir: Path) -> None:
    frame = pd.read_csv(output_dir / "final" / "predictions.csv")
    frame["resolution_bucket"] = pd.qcut(
        frame["megapixels"], q=3, labels=["baja", "media", "alta"], duplicates="drop"
    )
    frame["blur_bucket"] = pd.qcut(
        frame["blur_score"], q=3, labels=["más desenfocada", "media", "más nítida"], duplicates="drop"
    )
    pieces = [
        _group_metrics(frame, "environment", min_support=20),
        _group_metrics(frame, "source", min_support=30),
        _group_metrics(frame, "resolution_bucket", min_support=20),
        _group_metrics(frame, "blur_bucket", min_support=20),
    ]
    group_frame = pd.concat(pieces, ignore_index=True)

    true = frame["label_idx"].to_numpy(dtype=int)
    pred = frame["pred_idx"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=np.arange(len(CLASSES)), zero_division=0
    )
    class_frame = pd.DataFrame(
        {
            "class": CLASSES,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    # El entorno no es independiente de la clase: solamente tres clases aparecen
    # tanto en laboratorio como en campo. Este archivo permite leer el cambio de
    # recall dentro de cada una sin confundirlo con la composición global.
    environment_class_rows = []
    for class_name, class_group in frame.groupby("label"):
        environments = class_group["environment"].value_counts()
        if not {"lab", "real"}.issubset(environments.index):
            continue
        if min(int(environments["lab"]), int(environments["real"])) < 10:
            continue
        for environment, environment_group in class_group.groupby("environment"):
            environment_class_rows.append(
                {
                    "class": class_name,
                    "environment": environment,
                    "n": len(environment_group),
                    "recall": float(
                        (environment_group["label_idx"] == environment_group["pred_idx"]).mean()
                    ),
                }
            )
    environment_within_class = pd.DataFrame(environment_class_rows)
    gaps = []
    class_gaps = {
        metric: float(class_frame[metric].max() - class_frame[metric].min())
        for metric in ("precision", "recall", "f1")
    }
    gaps.append({"dimension": "class", **class_gaps})
    for dimension, dimension_frame in group_frame.groupby("dimension"):
        gaps.append(
            {
                "dimension": dimension,
                **{
                    metric: float(dimension_frame[metric].max() - dimension_frame[metric].min())
                    for metric in ("precision", "recall", "f1")
                },
            }
        )
    fairness_dir = output_dir / "fairness"
    fairness_dir.mkdir(parents=True, exist_ok=True)
    class_frame.to_csv(fairness_dir / "by_class.csv", index=False)
    group_frame.to_csv(fairness_dir / "by_group.csv", index=False)
    environment_within_class.to_csv(
        fairness_dir / "environment_within_class.csv", index=False
    )
    pd.DataFrame(gaps).rename(
        columns={"precision": "precision_gap", "recall": "recall_gap", "f1": "f1_gap"}
    ).to_csv(fairness_dir / "gaps.csv", index=False)
    _json_dump(
        fairness_dir / "metadata_limitations.json",
        {
            "environment_is_class_confounder": True,
            "lab_classes": sorted(frame.loc[frame["environment"].eq("lab"), "label"].unique()),
            "real_classes": sorted(frame.loc[frame["environment"].eq("real"), "label"].unique()),
            "interpretation": (
                "Los gaps globales describen el corpus, pero no estiman un efecto causal del entorno. "
                "environment_within_class.csv controla parcialmente la composición mediante recall por clase."
            ),
            "no_location_or_farm_identifier": True,
        },
    )
    print(pd.DataFrame(gaps).to_string(index=False))


def predict_image(output_dir: Path, image_path: Path) -> dict:
    selection = json.loads((output_dir / "ensemble" / "selection.json").read_text())
    component_probabilities = []
    weights = []
    timings = []
    for model_name in selection["models"]:
        size = resolve_input_size(model_name, (224, 224))
        dataset = ManifestDataset(
            pd.DataFrame([{"image_path": image_path.name, "label_idx": 0}]),
            image_path.parent,
            size,
        )
        tensor, _ = dataset[0]
        model = _as_feature_extractor(model_name, pretrained=True)
        tick = time.perf_counter()
        with torch.inference_mode():
            features = model(tensor.unsqueeze(0)).cpu().numpy()
        bundle = load_numeric_probe(output_dir / "final" / f"{model_name}_probe.npz")
        component_probabilities.append(bundle.predict_proba(features)[0])
        timings.append((time.perf_counter() - tick) * 1000.0)
        weights.append(selection.get("weights", {}).get(model_name, 1.0))
    weights_array = np.asarray(weights, dtype=float)
    weights_array /= weights_array.sum()
    probabilities = sum(
        weight * probs for weight, probs in zip(weights_array, component_probabilities)
    )
    top = np.argsort(probabilities)[::-1][:3]
    result = {
        "image": str(image_path),
        "model": selection["selected_name"],
        "latency_ms": float(sum(timings)),
        "top_k": [
            {"class": CLASSES[index], "probability": float(probabilities[index])} for index in top
        ],
        "low_confidence_warning": bool(probabilities[top[0]] < 0.60),
        "disclaimer": "Apoyo orientativo; no sustituye la evaluación de un profesional agrónomo.",
    }
    return result


def demo(output_dir: Path, image_path: Path) -> None:
    result = predict_image(output_dir, image_path)
    demo_dir = output_dir / "prototype"
    demo_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(demo_dir / "demo_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/etapa_2"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset-root", type=Path, required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--dataset-root", type=Path, required=True)
    extract_parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    extract_parser.add_argument("--batch-size", type=int, default=64)
    extract_parser.add_argument("--workers", type=int, default=8)

    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("--model", default="efficientnet_b0")
    tune_parser.add_argument("--trials", type=int, default=30)
    tune_parser.add_argument("--epochs", type=int, default=20)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)

    ensemble_parser = subparsers.add_parser("ensemble")
    ensemble_parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ensemble_parser.add_argument("--trials", type=int, default=250)

    cv_parser = subparsers.add_parser("cv")
    cv_parser.add_argument("--folds", type=int, default=5)

    subparsers.add_parser("final")
    subparsers.add_parser("fairness")
    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if args.command == "prepare":
        prepare(args.dataset_root.resolve(), output_dir)
    elif args.command == "extract":
        extract_features(
            args.dataset_root.resolve(), output_dir, args.models, args.batch_size, args.workers
        )
    elif args.command == "tune":
        tune(output_dir, args.model, args.trials, args.epochs)
    elif args.command == "compare":
        compare_models(output_dir, args.models)
    elif args.command == "ensemble":
        ensemble(output_dir, args.models, args.trials)
    elif args.command == "cv":
        cross_validate(output_dir, args.folds)
    elif args.command == "final":
        final_evaluation(output_dir)
    elif args.command == "fairness":
        fairness(output_dir)
    elif args.command == "demo":
        demo(output_dir, args.image.resolve())


if __name__ == "__main__":
    main()
