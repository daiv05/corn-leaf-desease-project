import numpy as np
from PIL import Image
from src.data.loader import load_and_normalize_image

class PerceptualHasher:
    """Calcula el hash perceptual (pHash) de imágenes para comparación visual."""
    
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    def calculate_phash(self, image_path: str) -> str:
        """Genera un hash hexadecimal estable basado en la estructura de baja frecuencia."""
        img = load_and_normalize_image(image_path)
        
        # 1. Reducir dimensionalidad y convertir a escala de grises para aislar estructura macro
        img_resized = img.resize((self.hash_size * 4, self.hash_size * 4), Image.Resampling.LANCZOS).convert('L')
        pixels = np.array(img_resized, dtype=np.float32)
        
        # 2. Transformada Coseno Discreta (DCT) básica 2D por filas y columnas
        def dct1d(x):
            N = len(x)
            n = np.arange(N)
            k = n.reshape((N, 1))
            M = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
            return np.dot(M, x)

        dct_row = np.apply_along_axis(dct1d, 1, pixels)
        dct_col = np.apply_along_axis(dct1d, 0, dct_row)
        
        # 3. Extraer la submatriz de bajas frecuencias superiores
        sub_dct = dct_col[:self.hash_size, :self.hash_size]
        
        # 4. Calcular la mediana y construir la huella binaria
        mediana = np.median(sub_dct)
        binary_mask = sub_dct > mediana
        
        # Convertir la máscara a cadena hexadecimal
        return ''.join(['1' if bit else '0' for bit in binary_mask.flatten()])

    @staticmethod
    def compute_hamming_distance(hash1: str, hash2: str) -> int:
        """Calcula cuántos bits difieren entre dos hashes perceptuales."""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))