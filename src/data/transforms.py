from abc import ABC, abstractmethod
import torchvision.transforms as T
import yaml

class TransformPipelineFactory(ABC):
    """
    Interfaz abstracta para la creación de pipelines de transformación (DIP).
    """
    @abstractmethod
    def create_transforms(self) -> T.Compose:
        pass


class CornTrainingTransforms(TransformPipelineFactory):
    """
    Pipeline de transformaciones y augmentations específicas para el entrenamiento.
    
    Aplica distorsiones geométricas seguras para fitopatología, evitando
    alteraciones agresivas de color que arruinen el diagnóstico de deficiencias.
    """
    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size
        # Estadísticas espectrales estándar para normalización (pueden actualizarse con el EDA)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def create_transforms(self) -> T.Compose:
        return T.Compose([
            # 1. Homogeneización de la dimensionalidad (Paso crítico de tamaño)
            T.Resize(self.target_size),
            
            # 2. Augmentations Geométricas (Simula capturas del productor en campo)
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
            
            # 3. Augmentation sutil de iluminación (Evita alterar drásticamente el Hue/Tono)
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0),
            
            # 4. Conversión a Tensor y Normalización Z-score espectral
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])


class CornValidationTransforms(TransformPipelineFactory):
    """
    Pipeline de transformaciones deterministas para validación y prueba.
    
    No aplica augmentations aleatorias para garantizar una evaluación justa y reproducible.
    """
    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def create_transforms(self) -> T.Compose:
        return T.Compose([
            # 1. Homogeneización de la dimensionalidad estricta
            T.Resize(self.target_size),
            
            # 2. Conversión y Normalización directa
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])


class CornTransformFactory:
    """
    Punto de acceso único (Factory) para obtener los pipelines según la etapa del flujo.
    """
    def __init__(self, config_path: str = "config/dataset.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Lee las dimensiones configuradas en tu archivo paramétrico
        width, height = config['dataset']['target_size']
        self.target_size = (width, height)

    def get_pipeline(self, stage: str) -> T.Compose:
        """Retorna el pipeline de transformación correspondiente a la etapa."""
        if stage.lower() == 'train':
            return CornTrainingTransforms(self.target_size).create_transforms()
        elif stage.lower() in ['val', 'test', 'inference']:
            return CornValidationTransforms(self.target_size).create_transforms()
        else:
            raise ValueError(f"Etapa de pipeline desconocida: {stage}. Use 'train', 'val' o 'test'.")