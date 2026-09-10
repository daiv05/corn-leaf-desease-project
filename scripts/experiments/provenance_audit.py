"""Auditoría de procedencia del corpus: fuente por imagen, duplicados y fuga entre splits.

Deriva el identificador de fuente desde el nombre de archivo, lo verifica contra la lista
de datasets documentada, y mide dos mecanismos de fuga independientes: duplicados exactos
por contenido y casi-duplicados por hash perceptual repartidos entre particiones.

Uso:
    python scripts/experiments/provenance_audit.py --output-dir outputs/experiments/provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import get_dataset_root, get_output_root

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ENVIRONMENTS = ("lab", "real")
POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)

DOCUMENTED_SOURCES = {
    "maize_field": "maize-in-field-dataset",
    "maize_desease": "maize-diseases",
    "maize_desease_v1.1": "maize-diseases",
    "cropdg": "cropdg-unified-multidomain",
    "maize_africa": "maize-beans-tomatoes-africa",
    "maize_africa_v1": "maize-beans-tomatoes-africa",
    "maize_africa_v1.2": "maize-beans-tomatoes-africa",
    "multi_desease": "multicrop-disease-maiz-disease-pests-and-disease",
    "maize_nutrient": "maize-nutrient-deficiency",
    "corn_leaf_roboflow": "corn-leaf-roboflow",
    "maize_2_roboflow": "maize-2-roboflow",
    "maize_leaf_roboflow": "maize-leaf-roboflow",
    "maize_deficiency_scanner_roboflow": "maize-deficiency-scanner-roboflow",
    "corn_leaf_diseases_classification_roboflow": "corn-leaf-diseases-classification-roboflow",
}

DOCUMENTED_DATASETS = set(DOCUMENTED_SOURCES.values()) | {"corn-leaf-diseases"}


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos de la auditoría."""
    parser = argparse.ArgumentParser(description="Auditoría de procedencia y fuga del corpus.")
    parser.add_argument("--clean-dir", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def parse_source(class_name: str, environment: str, filename: str) -> str | None:
    """Extrae el token de fuente del nombre de archivo.

    El convenio del corpus es ``<clase>_<fuente>_<entorno>_<id>.<ext>``. Se ancla el prefijo
    de clase y el sufijo de entorno porque los nombres de clase contienen guiones bajos.

    @param {str} class_name Clase a la que pertenece la imagen.
    @param {str} environment Entorno de captura, lab o real.
    @param {str} filename Nombre de archivo con extensión.
    @returns {str|None} Token de fuente, o None si el nombre no sigue el convenio.
    """
    stem = filename.rsplit(".", 1)[0]
    prefix = class_name + "_"
    if not stem.startswith(prefix):
        return None
    match = re.match(rf"^(?P<source>.+)_{environment}_[0-9a-fA-F]+$", stem[len(prefix):])
    return match.group("source") if match else None


def build_manifest(clean_dir: Path) -> pd.DataFrame:
    """Recorre el árbol limpio y devuelve un manifiesto con clase, entorno y fuente."""
    records = []
    for class_dir in sorted(p for p in clean_dir.iterdir() if p.is_dir()):
        for environment in ENVIRONMENTS:
            env_dir = class_dir / environment
            if not env_dir.exists():
                continue
            for path in sorted(env_dir.rglob("*")):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                records.append({
                    "image_path": path.relative_to(clean_dir.parent).as_posix(),
                    "label": class_dir.name,
                    "environment": environment,
                    "source": parse_source(class_dir.name, environment, path.name),
                    "absolute_path": str(path),
                })
    return pd.DataFrame(records)


def verify_sources(manifest: pd.DataFrame) -> dict[str, object]:
    """Contrasta los tokens hallados contra la lista de datasets documentada."""
    found = set(manifest.source.dropna().unique())
    undocumented = sorted(found - set(DOCUMENTED_SOURCES))
    mapped = {DOCUMENTED_SOURCES[token] for token in found if token in DOCUMENTED_SOURCES}
    return {
        "images_total": int(len(manifest)),
        "images_without_source": int(manifest.source.isna().sum()),
        "tokens_found": len(found),
        "tokens_undocumented": undocumented,
        "datasets_documented": len(DOCUMENTED_DATASETS),
        "datasets_with_images": len(mapped),
        "datasets_without_images": sorted(DOCUMENTED_DATASETS - mapped),
    }


def content_hash(path: str) -> str:
    """Calcula el SHA-256 del contenido del archivo."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perceptual_hash(path: str) -> np.ndarray:
    """Calcula un hash perceptual de diferencia de 64 bits, empaquetado en bytes."""
    with Image.open(path) as raw:
        if raw.format == "JPEG":
            raw.draft("L", (128, 128))
        image = raw.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.int16)
    return np.packbits((array[:, 1:] > array[:, :-1]).flatten())


def _union_find(size: int):
    """Devuelve las funciones find y union de una estructura de conjuntos disjuntos."""
    parent = list(range(size))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    return find, union


def group_near_duplicates(hashes: np.ndarray, max_distance: int) -> dict[int, list[int]]:
    """Agrupa índices cuyo hash perceptual difiere en como mucho ``max_distance`` bits.

    Con distancia cero basta agrupar por igualdad exacta; para distancias mayores se generan
    candidatos por bandas de 16 bits y sólo esos pares se comparan bit a bit.

    @param {np.ndarray} hashes Vector de hashes de 64 bits sin signo.
    @param {int} max_distance Distancia de Hamming máxima admitida.
    @returns {dict[int, list[int]]} Grupos con más de un miembro, indexados por representante.
    """
    find, union = _union_find(len(hashes))
    band_count, band_width = (1, 64) if max_distance == 0 else (4, 16)

    for band in range(band_count):
        key = (hashes >> np.uint64(band * band_width)) & np.uint64((1 << band_width) - 1)
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, value in enumerate(key.tolist()):
            buckets[value].append(index)
        for members in buckets.values():
            if not 1 < len(members) <= 400:
                continue
            indices = np.asarray(members)
            values = hashes[indices]
            for offset in range(len(indices)):
                xor = np.bitwise_xor(values[offset], values[offset + 1:])
                if xor.size == 0:
                    continue
                distance = POPCOUNT[
                    np.frombuffer(xor.astype(">u8").tobytes(), dtype=np.uint8).reshape(-1, 8)
                ].sum(axis=1)
                for position in np.where(distance <= max_distance)[0]:
                    union(int(indices[offset]), int(indices[offset + 1 + position]))

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(hashes)):
        groups[find(index)].append(index)
    return {root: members for root, members in groups.items() if len(members) > 1}


def summarize_leakage(groups, manifest, split_of, max_distance) -> dict[str, object]:
    """Resume cuántos grupos de duplicados quedan repartidos entre particiones distintas."""
    paths = manifest.image_path.tolist()
    labels = manifest.label.tolist()
    cross_split = cross_class = images_in_groups = 0
    detail = []
    for members in groups.values():
        present = [(paths[i], labels[i], split_of[paths[i]])
                   for i in members if paths[i] in split_of]
        if len(present) < 2:
            continue
        images_in_groups += len(present)
        if len({split for _, _, split in present}) > 1:
            cross_split += 1
            if len({label for _, label, _ in present}) > 1:
                cross_class += 1
            detail.append(present)
    return {
        "max_distance": max_distance,
        "groups_cross_split": cross_split,
        "groups_cross_split_and_class": cross_class,
        "images_in_multi_member_groups": images_in_groups,
        "detail": detail,
    }


def main() -> None:
    """Ejecuta la auditoría completa y persiste tablas y resumen."""
    args = parse_args()
    clean_dir = args.clean_dir or (get_dataset_root() / "clean")
    splits_dir = args.splits_dir or (get_output_root() / "splits" / "seed_42")
    output_dir = args.output_dir or (get_output_root() / "experiments" / "provenance")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(clean_dir)
    verification = verify_sources(manifest)
    print(json.dumps(verification, indent=2, ensure_ascii=False), flush=True)

    splits = pd.concat(
        [pd.read_csv(splits_dir / f"{name}.csv").assign(split=name)
         for name in ("train", "val", "test")],
        ignore_index=True,
    )
    split_of = dict(zip(splits.image_path, splits.split))
    manifest["split"] = manifest.image_path.map(split_of)

    manifest.drop(columns=["absolute_path"]).to_csv(
        output_dir / "manifest_with_source.csv", index=False)
    pd.crosstab(manifest.source, manifest.label).to_csv(output_dir / "source_by_class.csv")
    pd.crosstab(manifest.source, manifest.split.fillna("(fuera de splits)")).to_csv(
        output_dir / "source_by_split.csv")

    print("[*] hashing por contenido...", flush=True)
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        content = list(pool.map(content_hash, manifest.absolute_path.tolist()))
    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(content):
        exact_groups[digest].append(index)
    exact = {digest: members for digest, members in exact_groups.items() if len(members) > 1}
    exact_summary = summarize_leakage(exact, manifest, split_of, max_distance=-1)

    print("[*] hashing perceptual...", flush=True)
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        packed = list(pool.map(perceptual_hash, manifest.absolute_path.tolist()))
    hashes = np.stack(packed).view(">u8").ravel()

    sweep = []
    for distance in range(args.max_distance + 1):
        summary = summarize_leakage(
            group_near_duplicates(hashes, distance), manifest, split_of, distance)
        summary.pop("detail")
        sweep.append(summary)
        print(f"    distancia {distance}: {summary['groups_cross_split']} grupos cross-split",
              flush=True)

    report = {
        "source_verification": verification,
        "images_outside_splits": sorted(
            manifest.loc[manifest.split.isna(), "image_path"].tolist()),
        "exact_duplicate_groups": len(exact),
        "exact_duplicate_groups_cross_split": exact_summary["groups_cross_split"],
        "near_duplicate_sweep": sweep,
    }
    (output_dir / "provenance_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[*] resultados en {output_dir}", flush=True)


if __name__ == "__main__":
    main()
