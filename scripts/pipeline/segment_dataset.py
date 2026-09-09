"""Pre-segmentación batch del dataset de hojas de maíz."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.config import get_dataset_root, get_output_root
from src.segmentation.detector import MaizeLeafSegmenter
from src.segmentation.leaf_processor import (
    CROP_MASK_LETTERBOX,
    SUPPORTED_MASK_PROFILES,
    LeafMaskProcessorConfig,
    SegmentedLeafProcessor,
    build_comparison_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-segmenta el dataset de imágenes de maíz.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Directorio del dataset limpio de entrada (por defecto: DATASET_ROOT/clean)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio destino para el dataset segmentado",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Ruta al checkpoint de pesos YOLO (.pt)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=CROP_MASK_LETTERBOX,
        choices=sorted(SUPPORTED_MASK_PROFILES),
        help=f"Estrategia de enmascaramiento/recorte (default: {CROP_MASK_LETTERBOX})",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Tamaño objetivo (alto ancho) para perfiles letterbox (default: 224 224)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Máximo de imágenes a procesar (0 = todas las encontradas)",
    )
    parser.add_argument(
        "--max-previews",
        type=int,
        default=50,
        help="Número de paneles comparativos visuales a guardar en previews/ (default: 50)",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Directorio donde guardar los paneles de preview",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo para inferencia (ej. 'cuda', 'cuda:0', 'cpu')",
    )
    return parser.parse_args()


def run_segmentation(
    args: argparse.Namespace,
    commit_callback: object = None,
    commit_interval: int = 500,
) -> dict[str, object]:
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = get_dataset_root() / "clean"

    output_dir = args.output_dir
    if output_dir is None:
        env_out = os.getenv("SEGMENTED_DATASET_ROOT")
        if env_out:
            output_dir = Path(env_out) / "clean"
        else:
            output_dir = get_dataset_root() / "clean_segmented"

    preview_dir = args.preview_dir
    if preview_dir is None:
        preview_dir = get_output_root() / "segmentation_previews"

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Entrada dataset:   {dataset_dir}")
    print(f"[*] Destino segmentado: {output_dir}")
    print(f"[*] Previews dir:       {preview_dir}")
    print(f"[*] Checkpoint:         {args.checkpoint}")
    print(f"[*] Perfil:             {args.profile}")
    print(f"[*] Target size:        {args.target_size}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Directorio de dataset no encontrado: {dataset_dir}")

    # Escanear archivos preservando jerarquía: <clase>/<ambiente>/<archivo>
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths: list[Path] = []
    for ext in valid_exts:
        image_paths.extend(dataset_dir.rglob(f"*{ext}"))
        image_paths.extend(dataset_dir.rglob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))

    print(f"[*] Total imágenes encontradas en {dataset_dir}: {len(image_paths)}")
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
        print(f"[*] Limitado por --max-images a: {len(image_paths)}")

    if not image_paths:
        print("[!] No se encontraron imágenes para procesar.")
        return {}

    # Inicializar modelos
    segmenter = MaizeLeafSegmenter(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    config = LeafMaskProcessorConfig(
        processing_profile=args.profile,
        target_size=(args.target_size[0], args.target_size[1]),
    )
    processor = SegmentedLeafProcessor(config=config)

    t0 = time.time()
    processed_count = 0
    skipped_count = 0
    fallback_count = 0
    saved_previews = 0

    print("[*] Iniciando procesamiento batch...")
    for img_path in tqdm(image_paths, desc="Segmentando"):
        rel_path = img_path.relative_to(dataset_dir)
        dest_path = output_dir / rel_path

        # Si el perfil usa letterbox/mask_black, guardar en PNG para evitar
        # artefactos de compresión JPEG en fondo negro.
        dest_path = dest_path.with_suffix(".png")

        if dest_path.exists():
            skipped_count += 1
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(img_path) as raw:
                img = raw.convert("RGB")
        except Exception as err:
            print(f"[!] Error abriendo {img_path}: {err}", file=sys.stderr)
            continue

        instances = segmenter.segment(img)
        res = processor.process(img, instances, source_image=str(img_path))

        out_img = res.processed_image or res.original_image
        out_img.save(dest_path, format="PNG")

        if res.fallback_used:
            fallback_count += 1

        # Guardar paneles comparativos de muestra
        if saved_previews < args.max_previews:
            panel = build_comparison_panel(res)
            preview_filename = (
                f"preview_{saved_previews:03d}_{rel_path.parts[0]}_{rel_path.stem}.jpg"
            )
            panel.save(preview_dir / preview_filename, quality=90)
            saved_previews += 1

        processed_count += 1
        if commit_callback and processed_count % commit_interval == 0:
            try:
                commit_callback(processed_count)
            except Exception as exc:
                print(f"[!] Aviso en commit_callback: {exc}", file=sys.stderr)

    elapsed = time.time() - t0
    stats = {
        "total_discovered": len(image_paths),
        "processed": processed_count,
        "skipped_existing": skipped_count,
        "fallbacks": fallback_count,
        "previews_generated": saved_previews,
        "elapsed_seconds": round(elapsed, 2),
        "ms_per_image": round((elapsed / max(processed_count, 1)) * 1000, 2),
        "profile": args.profile,
        "target_size": args.target_size,
        "checkpoint": str(args.checkpoint),
    }

    summary_file = get_output_root() / "segmentation_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("Resumen de Pre-segmentación:")
    print(json.dumps(stats, indent=2))
    print(f"[*] Estadísticas guardadas en: {summary_file}")
    print("=" * 50)
    return stats


if __name__ == "__main__":
    run_segmentation(parse_args())
