import logging
import os

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_and_normalize_image(image_path: str) -> Image.Image:
    """
    Carga una imagen de disco y aplica normalización de formato en caliente.

    Garantiza:
    1. Corrección de orientación física mediante metadatos EXIF.
    2. Conversión estricta a 3 canales (RGB), eliminando canales Alfa (RGBA)
       o expandiendo imágenes monocromáticas.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La ruta de la imagen no existe: {image_path}")

    try:
        img = Image.open(image_path)

        # Corregir orientación EXIF en caliente (Smartphones)
        img = ImageOps.exif_transpose(img)

        # Normalización estricta del espacio de color a RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img

    except Exception as e:
        logger.error(f"Error crítico al normalizar en caliente la imagen {image_path}: {str(e)}")
        raise RuntimeError(f"Archivo corrupto o ilegible: {image_path}") from e
