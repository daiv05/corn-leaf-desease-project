import hashlib
import logging
import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(str(PROJECT_ROOT))

from src.config import DATASET_ROOT
from src.data.splitter import HierarchicalStratifiedSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_data_preparation_pipeline(config_path: str) -> None:
    """Orquesta todo el flujo de preparación e indexación de datos."""

    # 1. Cargar configuraciones del proyecto
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not DATASET_ROOT.exists():
        raise SystemExit(
            f"DATASET_ROOT no encontrado: {DATASET_ROOT}. Verifica DATASET_ROOT en .env"
        )

    clean_dir = DATASET_ROOT / config["paths"]["raw_dir"]
    output_dir = DATASET_ROOT / config["paths"]["split_output_dir"]
    seed = config["dataset"]["seed"]
    allowed_classes = config["dataset"]["classes"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Recopilación de rutas candidatas
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

    # 3. Construcción del manifiesto con red de seguridad SHA-256.
    # Detecta duplicados exactos (mismo contenido binario, distinto nombre) que pudieran
    # haberse añadido a clean/ sin pasar por el flujo de deduplicación perceptual.
    all_records: list[dict] = []
    seen_hashes: set[str] = set()
    duplicates_found = 0
    corrupt_found = 0

    logger.info(f"Indexando {len(raw_image_paths)} imágenes con verificación SHA-256 y validación PIL...")

    for class_name, environment, abs_path, rel_path in tqdm(
        raw_image_paths, desc="Indexando", unit="img"
    ):
        try:
            # Verificación de integridad a nivel de imagen (cabecera + estructura interna).
            # Image.verify() es barato — solo lee metadatos sin decodificar píxeles.
            # Requiere re-abrir porque verify() deja el objeto en estado inválido.
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
            tqdm.write(f"⚠️  Imagen corrupta o ilegible, omitida: {rel_path} — {e}")
            corrupt_found += 1

    df_manifest = pd.DataFrame(all_records)
    logger.info(
        f"Manifiesto construido: {len(df_manifest)} imágenes válidas "
        f"(duplicados exactos omitidos: {duplicates_found} | corruptas omitidas: {corrupt_found})"
    )

    # 4. Partición balanceada jerárquica estratificada (70 / 15 / 15)
    logger.info("Ejecutando división jerárquica estratificada (70% Train, 15% Val, 15% Test)...")
    splitter = HierarchicalStratifiedSplitter(seed=seed)
    train_df, val_df, test_df = splitter.split(
        df_manifest, train_size=0.70, val_size=0.15, test_size=0.15
    )

    # 5. Persistir manifiestos CSV
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Pipeline finalizado. Splits guardados en {output_dir}")
    logger.info(
        f"Distribución -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )

    # 6. Reporte de auditoría de estratificación
    logger.info("Generando reporte de auditoría del split...")
    report_dir = DATASET_ROOT / "reports" / "class_distribution"
    report_dir.mkdir(parents=True, exist_ok=True)

    train_counts = train_df.groupby(["label", "environment"]).size().rename("train_count")
    val_counts = val_df.groupby(["label", "environment"]).size().rename("val_count")
    test_counts = test_df.groupby(["label", "environment"]).size().rename("test_count")

    report_df = (
        pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int)
    )
    report_df["total_count"] = report_df.sum(axis=1)
    report_df = report_df.reset_index()
    report_df.to_csv(report_dir / "split_audit_report.csv", index=False)

    logger.info(f"Reporte de auditoría guardado en: {report_dir}/split_audit_report.csv")


if __name__ == "__main__":
    run_data_preparation_pipeline(config_path="config/dataset.yaml")
