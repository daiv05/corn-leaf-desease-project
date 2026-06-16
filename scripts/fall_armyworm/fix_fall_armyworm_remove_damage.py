"""
Renombra los archivos en data/clean/fall_armyworm/real que contienen
'_damage_' en el nombre, eliminando esa parte.

Ejemplo:
  fall_armyworm_damage_corn_leaf_roboflow_real_32534600.jpg
  -> fall_armyworm_corn_leaf_roboflow_real_32534600.jpg
"""

import os

TARGET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "clean", "fall_armyworm", "real"
)


def rename_files(target_dir: str) -> None:
    target_dir = os.path.abspath(target_dir)

    if not os.path.isdir(target_dir):
        print(f"ERROR: directorio no existe: {target_dir}")
        return

    candidates = [
        f for f in os.listdir(target_dir)
        if "_damage_" in f and os.path.isfile(os.path.join(target_dir, f))
    ]

    if not candidates:
        print("No se encontraron archivos con '_damage_' en el nombre.")
        return

    print(f"Archivos encontrados: {len(candidates)}")
    confirm = input("Continuar? [s/N]: ").strip().lower()
    if confirm != "s":
        print("Operacion cancelada.")
        return

    renamed = 0
    skipped = 0

    for filename in candidates:
        new_name = filename.replace("_damage_", "_")
        src = os.path.join(target_dir, filename)
        dst = os.path.join(target_dir, new_name)

        if os.path.exists(dst):
            print(f"OMITIDO (ya existe): {new_name}")
            skipped += 1
            continue

        os.rename(src, dst)
        renamed += 1

    print(f"Renombrados: {renamed} archivos")
    if skipped:
        print(f"Omitidos (destino ya existe): {skipped}")


if __name__ == "__main__":
    rename_files(TARGET_DIR)
