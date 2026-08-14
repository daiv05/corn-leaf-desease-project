import argparse
import json
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, get_output_root
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory
from src.export.common import (
    export_model,
    load_checkpoint_for_export,
    parse_export_formats,
    resolve_export_inputs,
    write_export_summary,
)
from src.models import list_models
from src.training.common import resolve_run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta un checkpoint del pipeline principal a ONNX/TFLite."
    )
    parser.add_argument("--model", required=True, choices=list_models())
    parser.add_argument("--run", default=None, help="run_id; por defecto usa latest.json.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta explicita a un checkpoint .pth (ignora --run si se pasa).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio de runs del pipeline principal (default: <outputs>/main).",
    )
    parser.add_argument(
        "--formats", default="onnx", help="Formatos a exportar, CSV (ej: 'onnx,tflite')."
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con test.csv (default: el 'splits_dir' de summary.json).",
    )
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--parity-sample-size", type=int, default=30, dest="parity_sample_size"
    )
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--no-parity",
        action="store_true",
        dest="no_parity",
        help="Omite la validacion de paridad numerica (no recomendado).",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    output_root = get_output_root()
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "main"

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        run_dir = checkpoint_path.parent
    else:
        run_dir = resolve_run_dir(output_dir, args.model, args.run)
        checkpoint_path = run_dir / "best.pth"

    formats = parse_export_formats(args.formats)
    if not formats:
        raise SystemExit("Debes indicar al menos un formato en --formats.")

    class_to_idx, _, image_size = resolve_export_inputs(run_dir, args.model, config_path)
    device = select_device()
    model = load_checkpoint_for_export(checkpoint_path, args.model, class_to_idx, device)

    test_loader = None
    if not args.no_parity:
        summary = json.loads((run_dir / "summary.json").read_text())
        splits_dir = (
            Path(args.splits_dir)
            if args.splits_dir
            else Path(summary.get("splits_dir", output_root / "splits" / "seed_42"))
        )
        test_csv = splits_dir / "test.csv"
        if not test_csv.exists():
            raise SystemExit(
                f"No existe {test_csv}. Pasa --splits-dir o usa --no-parity."
            )
        factory = CornTransformFactory(config_path=str(config_path), target_size=image_size)
        test_dataset = CornDataset(
            csv_path=str(test_csv),
            config_path=str(config_path),
            transform=factory.get_pipeline("test"),
            class_to_idx=class_to_idx,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

    report = export_model(
        model=model,
        run_dir=run_dir,
        model_name=args.model,
        class_to_idx=class_to_idx,
        image_size=image_size,
        formats=formats,
        test_loader=test_loader,
        device=device,
        tolerance=args.tolerance,
        parity_sample_size=args.parity_sample_size,
        skip_parity=args.no_parity,
    )
    write_export_summary(run_dir, report)

    print(f"Modelo: {args.model}")
    print(f"Run: {run_dir}")
    failed = False
    for result in report.formats:
        status = "OK" if result.succeeded else "FALLO"
        print(f"  [{status}] {result.format} -> {result.output_path}")
        if result.error:
            print(f"    error: {result.error}")
        if result.parity is not None:
            parity_status = "paso" if result.parity.passed else "NO PASO"
            print(
                f"    paridad: {parity_status} "
                f"(max_abs_prob_diff={result.parity.max_abs_prob_diff:.6f}, "
                f"tolerance={result.parity.tolerance:.6f}, "
                f"agreement_rate={result.parity.agreement_rate:.4f})"
            )
        if not result.succeeded or (result.parity is not None and not result.parity.passed):
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
