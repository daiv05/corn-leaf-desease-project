from src.data.transforms import CornTransformFactory
from src.data.dataset import CornDataset
from torch.utils.data import DataLoader

# 1. Obtener los pipelines de transformación desde la Fábrica
factory = CornTransformFactory(config_path="config/dataset.yaml")
train_transforms = factory.get_pipeline(stage='train')
val_transforms = factory.get_pipeline(stage='val')

# 2. Instanciar los datasets apuntando a los archivos de manifiesto consolidados
train_dataset = CornDataset(csv_path="dataset/splits/seed_42/train.csv", transform=train_transforms)
val_dataset = CornDataset(csv_path="dataset/splits/seed_42/val.csv", transform=val_transforms)

# 3. Empaquetar en los motores de lotes (DataLoader)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)