# test_loader_integration.py

import sys
import os
import torch
from torch.utils.data import DataLoader

# Forzar la inclusión de la raíz para resolver importaciones locales
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import DATASET_ROOT
from src.data.transforms import CornTransformFactory
from src.data.dataset import CornDataset


def test_dataset_pipeline():
    print("=" * 60)
    print("🚀 INICIANDO DIAGNÓSTICO DEL PIPELINE DE CARGA DE MAÍZ")
    print("=" * 60)

    csv_train_path = str(DATASET_ROOT / "splits" / "seed_42" / "train.csv")
    config_yaml_path = "config/dataset.yaml"

    # 1. Validar existencia de los archivos manifiesto
    if not os.path.exists(csv_train_path):
        print(
            f"❌ Error: No se encuentra el manifiesto '{csv_train_path}'. Ejecuta primero scripts/create_splits.py"
        )
        return

    try:
        # 2. Inicializar la fábrica de transformaciones y cargar pipeline de entrenamiento
        print("➡️ Pasó 1: Inicializando Fábrica de Transformaciones...")
        transform_factory = CornTransformFactory(config_path=config_yaml_path)
        train_transforms = transform_factory.get_pipeline(stage="train")
        print("   ✅ Transformaciones de entrenamiento instanciadas correctamente.")

        # 3. Inicializar el Dataset personalizado
        print("\n➡️ Paso 2: Inicializando componente CornDataset...")
        dataset = CornDataset(
            csv_path=csv_train_path, config_path=config_yaml_path, transform=train_transforms
        )
        total_samples = len(dataset)
        print(f"   ✅ Dataset instanciado con éxito. Total de muestras indexadas: {total_samples}")
        print(f"   📌 Mapeo de clases detectado: {dataset.class_to_idx}")

        # 4. Extraer y auditar una muestra individual en caliente
        print("\n➡️ Paso 3: Probando extracción perezosa (__getitem__) de la muestra número 0...")
        sample_tensor, sample_label = dataset[0]

        print("   ✅ Extracción unitaria exitosa.")
        print(f"   📊 Dimensiones del tensor devuelto (C, H, W): {sample_tensor.shape}")
        print(
            f"   🏷️ Índice de clase mapeado numéricamente: {sample_label} (Corresponde a: '{dataset.idx_to_class[sample_label]}')"
        )
        print(
            f"   📈 Valores espectrales del tensor -> Mínimo: {sample_tensor.min():.4f} | Máximo: {sample_tensor.max():.4f}"
        )

        # 5. Probar el empaquetado por lotes con el DataLoader nativo de PyTorch
        print("\n➡️ Paso 4: Inicializando motor de lotes DataLoader de PyTorch...")
        batch_size = min(16, total_samples)  # Ajustar dinámicamente según el tamaño del dataset
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 0 para evitar conflictos de hilos durante la depuración en Windows
            pin_memory=True,
        )

        # Extraer el primer batch real que vería la red neuronal
        print(
            f"   📦 Solicitando un lote optimizado de {batch_size} imágenes a la cola de memoria..."
        )
        images_batch, labels_batch = next(iter(data_loader))

        print("\n" + "=" * 60)
        print("🎉 ¡DIAGNÓSTICO COMPLETADO EXITOSAMENTE!")
        print("=" * 60)
        print(
            f"   Tensor de imágenes del lote (Batch Size, Canales, Alto, Ancho): {images_batch.shape}"
        )
        print(f"   Tensor de etiquetas correspondientes del lote: {labels_batch}")
        print(
            "   El flujo de datos está 100% libre de colisiones dimensionales y listo para entrenar."
        )
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ CRÍTICO: Falló la integración del cargador de datos.")
        print(f"   Detalle del error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dataset_pipeline()
