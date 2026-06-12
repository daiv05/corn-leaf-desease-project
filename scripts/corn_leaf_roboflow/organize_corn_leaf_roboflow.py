"""
Reorganiza corn-leaf-roboflow desde formato YOLO (train/valid/test + labels)
hacia una carpeta por clase en data/clean/, sin anotaciones.

Fuente:  data/raw/corn-leaf-roboflow/{train,valid,test}/{images,labels}/
Destino: data/clean/<clase>/real/

Reglas:
  - Cada imagen se asigna a la clase con más bboxes en su .txt (clase dominante).
  - Imágenes sin label (.txt vacío o ausente) se omiten.
  - El nombre destino sigue el patrón: <clase>_corn_leaf_roboflow_real_<random8>.jpg
  - Idempotente: si la imagen fuente ya fue copiada (mismo stem en destino), se omite.
  - El dataset mezcla formato YOLO bbox (5 campos) y segmentación (N>5 campos);
    ambos son válidos — el class_id sigue siendo el primer token en ambos casos.
"""

import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "corn-leaf-roboflow"
CLEAN_DIR = ROOT / "data" / "clean"

CLASS_NAMES = {
    0: "fall_armyworm_damage",
    1: "healthy",
    2: "leaf_spot",
    3: "magnesium_deficiency",
    4: "nitrogen_deficiency",
    5: "northern_corn_leaf_blight",
    6: "phosphorus_deficiency",
    7: "potassium_deficiency",
}

DEST_FOLDER = {
    "fall_armyworm_damage":    "fall_armyworm_damage",
    "healthy":                 "healthy",
    "leaf_spot":               "leaf_spot",
    "magnesium_deficiency":    "magnesium_deficiency",
    "nitrogen_deficiency":     "nitrogen_deficiency",
    "northern_corn_leaf_blight": "northern_corn_leaf_blight",
    "phosphorus_deficiency":   "phosphorus_deficiency",
    "potassium_deficiency":    "potassium_deficiency",
}

SPLITS = ["train", "valid", "test"]
RANDOM_DIGITS = 8


def dominant_class(label_path: Path) -> int | None:
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return None
    ids = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            ids.append(int(parts[0]))
        except ValueError:
            print(f"  [warn] token no entero en {label_path.name}: '{parts[0]}'")
    if not ids:
        return None
    return Counter(ids).most_common(1)[0][0]


def collect_images() -> dict[str, list[Path]]:
    """Devuelve {clase: [rutas de imagen]} para todo el dataset."""
    by_class: dict[str, list[Path]] = {name: [] for name in CLASS_NAMES.values()}

    for split in SPLITS:
        labels_dir = RAW_DIR / split / "labels"
        images_dir = RAW_DIR / split / "images"

        if not labels_dir.exists():
            print(f"  [skip] {split}/labels no existe")
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            cls_id = dominant_class(label_file)
            if cls_id is None:
                continue

            cls_name = CLASS_NAMES.get(cls_id)
            if cls_name is None:
                print(f"  [warn] class_id desconocido {cls_id} en {label_file.name}")
                continue

            img_path = images_dir / f"{label_file.stem}.jpg"
            if not img_path.exists():
                print(f"  [warn] imagen no encontrada: {img_path.name}")
                continue

            by_class[cls_name].append(img_path)

    return by_class


def _load_existing_numbers(dest_dir: Path) -> set[int]:
    """Devuelve los números aleatorios ya usados en archivos del destino."""
    numbers: set[int] = set()
    for p in dest_dir.glob("*_corn_leaf_roboflow_real_*"):
        try:
            numbers.add(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return numbers


def copy_images(by_class: dict[str, list[Path]]) -> None:
    # used_numbers es global entre clases para garantizar nombres únicos en todo el destino
    used_numbers: set[int] = set()

    # Pre-cargar números ya usados en todos los destinos (guard contra re-ejecución)
    for cls_name in CLASS_NAMES.values():
        dest_dir = CLEAN_DIR / DEST_FOLDER[cls_name] / "real"
        used_numbers.update(_load_existing_numbers(dest_dir))

    for cls_name, images in by_class.items():
        if not images:
            print(f"  [skip] {cls_name}: sin imágenes")
            continue

        dest_folder = DEST_FOLDER[cls_name]
        dest_dir = CLEAN_DIR / dest_folder / "real"
        dest_dir.mkdir(parents=True, exist_ok=True)

        already_in_dest = len(list(dest_dir.glob("*_corn_leaf_roboflow_real_*")))
        if already_in_dest >= len(images):
            print(f"  [skip] {cls_name:30s}: {already_in_dest} archivos ya presentes, omitiendo")
            continue

        copied = 0
        for src in images:
            ext = src.suffix.lower()
            while True:
                number = random.randint(10 ** (RANDOM_DIGITS - 1), 10**RANDOM_DIGITS - 1)
                if number not in used_numbers:
                    used_numbers.add(number)
                    break

            dst_name = f"{cls_name}_corn_leaf_roboflow_real_{number}{ext}"
            shutil.copy2(src, dest_dir / dst_name)
            copied += 1

        print(f"  {cls_name:30s} → {dest_folder}/real/  ({copied} copiadas)")


def main() -> None:
    print("Recolectando imágenes por clase...")
    by_class = collect_images()

    total = sum(len(v) for v in by_class.values())
    print(f"Total encontradas: {total}\n")

    print("Copiando imágenes...")
    copy_images(by_class)

    print("\nListo.")


if __name__ == "__main__":
    main()
