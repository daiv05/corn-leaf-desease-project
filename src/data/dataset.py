import os
import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data.loader import load_and_normalize_image
from src.config import DATASET_ROOT

class CornDataset(Dataset):
    """
    Componente Dataset personalizado para el mapeo y consumo indexado
    de imágenes de patologías y deficiencias en hojas de maíz.
    """
    def __init__(self, csv_path: str, config_path: str = "config/dataset.yaml", transform=None):
        """
        Args:
            csv_path (str): Ruta al manifiesto del split (train.csv, val.csv o test.csv).
            config_path (str): Ruta al archivo de configuración paramétrica.
            transform (callable, optional): Pipeline de transformaciones de torchvision.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontró el archivo de manifiesto: {csv_path}")

        # 1. Cargar el manifiesto indexado ligero (Lazy Loading)
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform

        # Las rutas del manifiesto son relativas a DATASET_ROOT, para que el
        # mismo CSV sirva tanto en el servidor (volumen montado) como en local.
        self.dataset_root = DATASET_ROOT

        # 2. Cargar mapeo de clases desde la configuración centralizada
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.allowed_classes = config['dataset']['classes']
        
        # Generar diccionario bidireccional de codificación: {'healthy': 0, 'common_rust': 1, ...}
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.allowed_classes)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def __len__(self) -> int:
        """Devuelve el tamaño neto total de la muestra actual."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Obtiene y procesa un elemento del dataset en caliente bajo demanda.
        
        Garantiza:
        1. Carga perezosa desde disco duro para optimizar la memoria RAM.
        2. Normalización física y corrección de color/rotación en el milisegundo de lectura.
        3. Transformación dimensional homogénea y conversión a tensor para la GPU.
        """
        # Extraer metadatos de la fila correspondiente del CSV
        img_path = self.dataset_root / self.data_frame.iloc[idx]['image_path']
        class_name = self.data_frame.iloc[idx]['label']

        # 1. Carga y normalización en caliente del formato (RGB, corrección EXIF de smartphones)
        image = load_and_normalize_image(img_path)
        
        # 2. Mapear la etiqueta de texto a su correspondiente índice entero codificado
        label_idx = self.class_to_idx[class_name]
        
        # 3. Aplicar transformaciones geométricas/espectrales y conversión a tensor de PyTorch
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx