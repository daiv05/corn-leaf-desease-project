"""Corrige el sufijo de entorno en imágenes de `clean/<clase>/lab/` cuyo nombre dice `_real_`.

El entorno está codificado en la carpeta y en el marcador `_<env>_` del nombre. En
`gray_leaf_spot/lab/` y `northern_corn_leaf_blight/lab/` ambos se contradicen: son imágenes
de PlantVillage clasificadas como `lab` que conservaron el `_real_` del renombrado original.
Manda la carpeta, que es lo que consume `create_splits.py`.

Requisito previo a aplanar `clean/`: sin esto, derivar el entorno del nombre etiquetaría
esas 1401 imágenes como `real`. Idempotente.
"""

import argparse
import re
import sys
from pathlib import Path

from src.config import get_dataset_root

ENV_MARKER = re.compile(r"_real_")

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def find_mismatches(clean_dir: Path) -> list[tuple[Path, Path]]:
    """Localiza imágenes bajo `<clase>/lab/` cuyo nombre contiene el marcador `_real_`.

    @param {Path} clean_dir Raíz de `clean/`.
    @returns {list[tuple[Path, Path]]} Pares (origen, destino) del renombrado.
    """
    renames: list[tuple[Path, Path]] = []

    for class_dir in sorted(clean_dir.iterdir()):
        lab_dir = class_dir / "lab"
        if not lab_dir.is_dir():
            continue

        for image_path in sorted(lab_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            occurrences = len(ENV_MARKER.findall(image_path.name))
            if occurrences == 0:
                continue
            # Un nombre ambiguo se reporta y se omite en vez de adivinar cuál sustituir.
            if occurrences > 1:
                print(f"  AMBIGUO (omitido, {occurrences} marcadores): {image_path.name}")
                continue

            new_name = ENV_MARKER.sub("_lab_", image_path.name)
            renames.append((image_path, image_path.with_name(new_name)))

    return renames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Renombra a `_lab_` las imágenes de clean/<clase>/lab/ marcadas como `_real_`."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplica los renombrados. Sin este flag solo muestra lo que haría (dry-run).",
    )
    args = parser.parse_args()

    clean_dir = get_dataset_root() / "clean"
    if not clean_dir.is_dir():
        print(f"ERROR: no existe '{clean_dir}'.", file=sys.stderr)
        return 1

    renames = find_mismatches(clean_dir)
    if not renames:
        print("Nada que renombrar: ninguna imagen en lab/ tiene el marcador `_real_`.")
        return 0

    # Una colisión sobrescribiría una imagen distinta, así que se aborta antes de tocar disco.
    collisions = [dst for _, dst in renames if dst.exists()]
    if collisions:
        print(f"ERROR: {len(collisions)} destino(s) ya existen. Abortado:", file=sys.stderr)
        for dst in collisions[:10]:
            print(f"  {dst}", file=sys.stderr)
        return 1

    by_class: dict[str, int] = {}
    for src, _ in renames:
        class_name = src.parent.parent.name
        by_class[class_name] = by_class.get(class_name, 0) + 1

    print(f"{'APLICANDO' if args.apply else 'DRY-RUN'}: {len(renames)} renombrados")
    for class_name, count in sorted(by_class.items()):
        print(f"  {class_name}: {count}")

    print("\nEjemplos:")
    for src, dst in renames[:3]:
        print(f"  {src.name}\n  -> {dst.name}")

    if not args.apply:
        print("\nDry-run: no se modificó nada. Reejecuta con --apply para aplicar.")
        return 0

    for src, dst in renames:
        src.rename(dst)

    print(f"\nListo: {len(renames)} archivos renombrados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
