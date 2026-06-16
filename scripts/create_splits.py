import os
import sys
import yaml
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm  # Librería estándar de la industria para barras de progreso

# Inyección de dependencias de la raíz del proyecto
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(PROJECT_ROOT))

from src.config import DATASET_ROOT
from src.preprocessing.deduplication import PerceptualHasher
from src.data.splitter import HierarchicalStratifiedSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_preparation_pipeline(config_path: str) -> None:
    """Orquesta todo el flujo de preparación e indexación de datos."""

    # 1. Cargar configuraciones del proyecto
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT no encontrado: {DATASET_ROOT}. Verifica DATASET_ROOT en .env")

    raw_dir = DATASET_ROOT / config['paths']['raw_dir']
    output_dir = DATASET_ROOT / config['paths']['split_output_dir']
    seed = config['dataset']['seed']
    allowed_classes = config['dataset']['classes']

    # Forzar la creación segura de toda la estructura de directorios necesaria
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Recopilación previa de rutas para inicializar la barra de progreso de forma precisa
    logger.info("Escaneando directorios para calcular la carga de trabajo...")
    raw_image_paths = []

    for class_name in os.listdir(raw_dir):
        if class_name not in allowed_classes:
            continue
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for environment in os.listdir(class_path):
            env_path = os.path.join(class_path, environment)
            if environment not in ['real', 'lab'] or not os.path.isdir(env_path):
                continue

            for img_name in os.listdir(env_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    abs_path = os.path.join(env_path, img_name)
                    # Ruta relativa a DATASET_ROOT para que el manifest sea portable
                    # entre entornos (servidor con volumen vs. local con otra ruta).
                    rel_path = os.path.relpath(abs_path, DATASET_ROOT)
                    raw_image_paths.append((class_name, environment, abs_path, rel_path))

    if not raw_image_paths:
        raise ValueError(f"El pipeline no pudo indexar ninguna imagen válida en la ruta raíz '{raw_dir}'.")

    # 3. Procesamiento y deduplicación visual iterativa por pHash
    all_records = []
    hasher = PerceptualHasher()
    seen_hashes = {} 
    
    logger.info(f"Iniciando normalización y deduplicación visual por pHash sobre {len(raw_image_paths)} imágenes...")
    
    # tqmd envuelve la lista y genera la barra dinámica en la terminal
    for class_name, environment, abs_path, rel_path in tqdm(raw_image_paths, desc="Procesando imágenes de maíz", unit="img"):
        try:
            # Calcular huella digital visual (pHash) en caliente
            img_hash = hasher.calculate_phash(abs_path)

            # Control de redundancia: Validar si existe duplicado exacto o cercano
            is_duplicate = False
            if (class_name, environment) in seen_hashes:
                for existing_hash in seen_hashes[(class_name, environment)]:
                    if hasher.compute_hamming_distance(img_hash, existing_hash) <= 5:
                        is_duplicate = True
                        break

            if is_duplicate:
                continue

            # Registrar hash único
            seen_hashes.setdefault((class_name, environment), []).append(img_hash)

            # Añadir registro limpio al inventario (ruta relativa a DATASET_ROOT)
            all_records.append({
                'image_path': rel_path,
                'label': class_name,
                'environment': environment
            })
        except Exception as e:
            # Usamos tqdm.write para que los warnings no rompan la estética de la barra de progreso
            tqdm.write(f"⚠️ Saltando archivo dañado o inválido en {abs_path}: {str(e)}")

    df_manifest = pd.DataFrame(all_records)
    logger.info(f"Saneamiento completado. Imágenes únicas retenidas: {len(df_manifest)} (Descartadas por redundancia: {len(raw_image_paths) - len(df_manifest)})")

    # 4. Partición balanceada y controlada utilizando la estrategia multi-nivel
    logger.info("Ejecutando división jerárquica estratificada (70% Train, 15% Val, 15% Test)...")
    splitter = HierarchicalStratifiedSplitter(seed=seed)
    train_df, val_df, test_df = splitter.split(df_manifest, train_size=0.70, val_size=0.15, test_size=0.15)
    
    # 5. Consolidación de archivos de manifiesto CSV inmutables
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    logger.info(f"Pipeline finalizado con éxito. Registros salvados en {output_dir}")
    logger.info(f"Distribución neta final -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ==========================================
    # GENERACIÓN DE EVIDENCIA DE ESTRATIFICACIÓN
    # ==========================================
    logger.info("Generando reporte de auditoría del split...")

    report_dir = DATASET_ROOT / "reports" / "class_distribution"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    train_counts = train_df.groupby(['label', 'environment']).size().rename('train_count')
    val_counts = val_df.groupby(['label', 'environment']).size().rename('val_count')
    test_counts = test_df.groupby(['label', 'environment']).size().rename('test_count')
    
    report_df = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int)
    report_df['total_count'] = report_df.sum(axis=1)
    report_df = report_df.reset_index()
    
    report_df.to_csv(os.path.join(report_dir, 'split_audit_report.csv'), index=False)
    logger.info(f"Evidencia de estratificación guardada en: {report_dir}/split_audit_report.csv")

if __name__ == "__main__":
    run_data_preparation_pipeline(config_path="config/dataset.yaml")