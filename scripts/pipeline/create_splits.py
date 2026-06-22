import argparse
import hashlib
import logging
import os
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from src.config import DATASET_ROOT
from src.data.splitter import HierarchicalStratifiedSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_manifest(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction == 1.0:
        return df
    stratify_col = df["label"] + "_" + df["environment"]
    return (
        df.groupby(stratify_col, group_keys=False)
        .apply(lambda g: g.sample(frac=fraction, random_state=seed))
        .reset_index(drop=True)
    )


def _split_output_dir(base: Path, fraction: float) -> Path:
    if fraction == 1.0:
        return base
    pct = int(round(fraction * 100))
    return base.parent / (base.name + f"_sample{pct}")


def run_data_preparation_pipeline(config_path: str, sample_fraction: float | None = None) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not DATASET_ROOT.exists():
        raise SystemExit(
            f"DATASET_ROOT no encontrado: {DATASET_ROOT}. Verifica DATASET_ROOT en .env"
        )

    if sample_fraction is None:
        sample_fraction = config.get("sampling", {}).get("fraction", 1.0)
    sampling_seed = config.get("sampling", {}).get("seed", 42)

    clean_dir = DATASET_ROOT / config["paths"]["raw_dir"]
    base_output_dir = DATASET_ROOT / config["paths"]["split_output_dir"]
    output_dir = _split_output_dir(base_output_dir, sample_fraction)
    seed = config["dataset"]["seed"]
    allowed_classes = config["dataset"]["classes"]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Escaneando directorios para calcular la carga de trabajo...")
    raw_image_paths: list[tuple[str, str, Path, str]] = []

    for class_name in os.listdir(clean_dir):
        if class_name not in allowed_classes:
            continue
        class_path = clean_dir / class_name
        if not class_path.is_dir():
            continue

        for environment in os.listdir(class_path):
            env_path = class_path / environment
            if environment not in ("real", "lab") or not env_path.is_dir():
                continue

            for img_name in os.listdir(env_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    abs_path = env_path / img_name
                    rel_path = str(abs_path.relative_to(DATASET_ROOT))
                    raw_image_paths.append((class_name, environment, abs_path, rel_path))

    if not raw_image_paths:
        raise ValueError(f"El pipeline no pudo indexar ninguna imagen válida en '{clean_dir}'.")

    all_records: list[dict] = []
    seen_hashes: set[str] = set()
    duplicates_found = 0
    corrupt_found = 0

    logger.info(
        f"Indexando {len(raw_image_paths)} imágenes con verificación SHA-256 y validación PIL..."
    )

    for class_name, environment, abs_path, rel_path in tqdm(
        raw_image_paths, desc="Indexando", unit="img"
    ):
        try:
            with Image.open(abs_path) as img:
                img.verify()

            digest = _sha256(abs_path)
            if digest in seen_hashes:
                logger.warning(f"Duplicado exacto detectado y omitido: {rel_path}")
                duplicates_found += 1
                continue
            seen_hashes.add(digest)
            all_records.append(
                {"image_path": rel_path, "label": class_name, "environment": environment}
            )
        except Exception as e:
            tqdm.write(f"⚠️  Imagen corrupta o ilegible, omitida: {rel_path} - {e}")
            corrupt_found += 1

    df_manifest = pd.DataFrame(all_records)
    logger.info(
        f"Manifiesto construido: {len(df_manifest)} imágenes válidas "
        f"(duplicados exactos omitidos: {duplicates_found} | corruptas omitidas: {corrupt_found})"
    )

    if sample_fraction < 1.0:
        before = len(df_manifest)
        df_manifest = _sample_manifest(df_manifest, sample_fraction, sampling_seed)
        logger.info(
            f"Muestreo estratificado {int(sample_fraction * 100)}%: "
            f"{before} - {len(df_manifest)} imágenes"
        )

    logger.info("Ejecutando división jerárquica estratificada (70% Train, 15% Val, 15% Test)...")
    splitter = HierarchicalStratifiedSplitter(seed=seed)
    train_df, val_df, test_df = splitter.split(
        df_manifest, train_size=0.70, val_size=0.15, test_size=0.15
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Pipeline finalizado. Splits guardados en {output_dir}")
    logger.info(
        f"Distribución -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )

    logger.info("Generando reporte de auditoría del split...")
    report_dir = DATASET_ROOT / "reports" / "class_distribution"
    report_dir.mkdir(parents=True, exist_ok=True)

    train_counts = train_df.groupby(["label", "environment"]).size().rename("train_count")
    val_counts = val_df.groupby(["label", "environment"]).size().rename("val_count")
    test_counts = test_df.groupby(["label", "environment"]).size().rename("test_count")

    report_df = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int)
    report_df["total_count"] = report_df.sum(axis=1)
    report_df = report_df.reset_index()
    report_df.to_csv(report_dir / "split_audit_report.csv", index=False)

    logger.info(f"Reporte de auditoría guardado en: {report_dir}/split_audit_report.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera splits CSV estratificados.")
    parser.add_argument(
        "--config",
        default="config/dataset.yaml",
        help="Ruta al archivo de configuración (default: config/dataset.yaml)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        dest="sample_fraction",
        help="Fracción del dataset a usar (0.0-1.0). Si se omite, usa sampling.fraction del YAML.",
    )
    args = parser.parse_args()
    run_data_preparation_pipeline(config_path=args.config, sample_fraction=args.sample_fraction)
