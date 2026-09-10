"""Validación dejando una fuente fuera: la partición honesta del corpus.

Cada pliegue retiene un grupo de procedencia completo como prueba y otro distinto como
validación, de modo que ni el entrenamiento ni la selección de modelo ven el dominio sobre
el que se mide. Cada imagen se evalúa exactamente una vez, en el pliegue donde su fuente
queda retenida, así que las predicciones agrupadas cubren el corpus entero fuera de fuente.

Antes de particionar se eliminan los casi-duplicados: de cada componente con hash perceptual
idéntico se conserva un representante. Sin eso, la misma imagen podría aparecer a ambos
lados de la frontera aunque las fuentes estén separadas.

Uso:
    python scripts/experiments/leave_one_source_out.py --output outputs/experiments/loso.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from scripts.experiments.provenance_leak import LeakDataset, predict
from src.config import get_dataset_root, get_output_root
from src.data.provenance import provenance_from_path

POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos del experimento."""
    parser = argparse.ArgumentParser(description="Validación dejando una fuente fuera.")
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="efficientnet_lite0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-cap", type=int, default=1000)
    parser.add_argument("--val-cap", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", type=str, default="original")
    parser.add_argument("--balance-groups", action="store_true")
    parser.add_argument("--backmix", type=float, default=0.0)
    parser.add_argument("--folds", type=str, default="")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def perceptual_hash(path: str) -> np.ndarray:
    """Calcula un hash perceptual de diferencia de 64 bits, empaquetado en bytes."""
    with Image.open(path) as raw:
        if raw.format == "JPEG":
            raw.draft("L", (128, 128))
        image = raw.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.int16)
    return np.packbits((array[:, 1:] > array[:, :-1]).flatten())


def deduplicate(manifest: pd.DataFrame, dataset_root: Path, workers: int) -> pd.DataFrame:
    """Conserva un representante por componente de hash perceptual idéntico.

    @param {pd.DataFrame} manifest Manifiesto con columna image_path.
    @param {Path} dataset_root Raíz del dataset para resolver rutas.
    @param {int} workers Hilos de decodificación.
    @returns {pd.DataFrame} Manifiesto sin casi-duplicados, con la columna kept_of.
    """
    paths = [str(dataset_root / p) for p in manifest.image_path]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        packed = list(pool.map(perceptual_hash, paths))
    hashes = np.stack(packed).view(">u8").ravel()

    first_index: dict[int, int] = {}
    keep = np.ones(len(hashes), dtype=bool)
    duplicates_of = [None] * len(hashes)
    for index, value in enumerate(hashes.tolist()):
        if value in first_index:
            keep[index] = False
            duplicates_of[index] = manifest.image_path.iloc[first_index[value]]
        else:
            first_index[value] = index

    result = manifest.copy()
    result["duplicate_of"] = duplicates_of
    dropped = int((~keep).sum())
    print(f"[*] casi-duplicados eliminados: {dropped} de {len(result)}", flush=True)
    return result.loc[keep].reset_index(drop=True), dropped


def build_folds(manifest: pd.DataFrame) -> list[dict[str, object]]:
    """Construye un pliegue por grupo de procedencia, con validación rotada y válida.

    La fuente de validación se elige rotando entre los grupos restantes y se descarta
    cualquier candidata que deje alguna clase sin ejemplos de entrenamiento.

    @param {pd.DataFrame} manifest Manifiesto con columnas label y provenance.
    @returns {list[dict]} Descripción de cada pliegue.
    """
    groups = sorted(manifest.provenance.unique())
    classes = sorted(manifest.label.unique())
    folds = []
    for position, test_group in enumerate(groups):
        candidates = [groups[(position + offset) % len(groups)] for offset in range(1, len(groups))]
        chosen = None
        for candidate in candidates:
            train = manifest[~manifest.provenance.isin({test_group, candidate})]
            if set(train.label.unique()) == set(classes):
                chosen = candidate
                break
        if chosen is None:
            print(f"[!] {test_group}: ninguna fuente de validación deja las 9 clases en train",
                  flush=True)
            continue
        test = manifest[manifest.provenance == test_group]
        folds.append({
            "test_group": test_group,
            "val_group": chosen,
            "n_test": int(len(test)),
            "test_classes": sorted(test.label.unique()),
        })
    return folds


def balance_by_group(train: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Reparte el cupo de cada clase entre sus fuentes en lugar de tomarlo del total.

    Una clase cuyo 80 % procede de una sola fuente entrena, sin esto, casi sólo con esa
    fuente. El reparto es por llenado progresivo: las celdas pequeñas aportan todo lo que
    tienen y liberan su remanente para las demás.

    @param {pd.DataFrame} train Subconjunto de entrenamiento con columnas label y provenance.
    @param {int} cap Cupo total por clase.
    @returns {pd.DataFrame} Subconjunto reequilibrado por celda fuente-clase.
    """
    selected = []
    for _, class_frame in train.groupby("label"):
        cells = sorted(
            (cell for _, cell in class_frame.groupby("provenance")),
            key=len,
        )
        remaining = cap
        for position, cell in enumerate(cells):
            quota = remaining // (len(cells) - position)
            take = min(len(cell), quota)
            if take:
                selected.append(cell.sample(take, random_state=42))
            remaining -= take
    return pd.concat(selected, ignore_index=True)


def run_fold(fold, manifest, dataset_root, classes, args, device) -> dict[str, object]:
    """Entrena con las fuentes restantes y evalúa sobre la fuente retenida."""
    class_to_idx = {name: index for index, name in enumerate(classes)}
    test = manifest[manifest.provenance == fold["test_group"]]
    val = manifest[manifest.provenance == fold["val_group"]]
    if args.val_cap > 0 and len(val) > args.val_cap:
        val = val.sample(args.val_cap, random_state=42)
    train = manifest[~manifest.provenance.isin({fold["test_group"], fold["val_group"]})]
    if args.train_cap > 0:
        train = (balance_by_group(train, args.train_cap) if args.balance_groups else pd.concat(
            [group.sample(min(len(group), args.train_cap), random_state=42)
             for _, group in train.groupby("label")],
            ignore_index=True,
        ))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    loaders = {}
    parts = (("train", train, True), ("val", val, False), ("test", test, False))
    for name, frame, is_train in parts:
        loaders[name] = DataLoader(
            LeakDataset(frame, dataset_root, class_to_idx, args.arm, 0.10,
                        args.image_size, is_train, args.seed,
                        backmix_probability=args.backmix if is_train else 0.0),
            batch_size=args.batch_size, shuffle=is_train,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    model = timm.create_model(args.model, pretrained=True, num_classes=len(classes)).to(device)
    counts = np.bincount([class_to_idx[label] for label in train.label],
                         minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(np.sqrt(counts.sum() / np.maximum(counts, 1.0)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for images, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(images.to(device, non_blocking=True)),
                      targets.to(device, non_blocking=True)).backward()
            optimizer.step()
        scheduler.step()
        trues, preds = predict(model, loaders["val"], device)
        val_f1 = float(f1_score(trues, preds, average="macro"))
        print(f"    [{fold['test_group']}] epoca {epoch + 1} val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, stale, best_state = val_f1, 0, {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    trues, preds = predict(model, loaders["test"], device)
    present = sorted({int(t) for t in trues})
    return {
        "arm": args.arm,
        "balance_groups": bool(args.balance_groups),
        "backmix": float(args.backmix),
        "test_group": fold["test_group"],
        "val_group": fold["val_group"],
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(trues)),
        "val_macro_f1": best_f1,
        "test_macro_f1": float(f1_score(trues, preds, average="macro")),
        "test_accuracy": float((trues == preds).mean()),
        "per_class_f1": {
            classes[index]: float(value)
            for index, value in zip(present, f1_score(trues, preds, average=None, labels=present))
        },
        "true": trues.tolist(),
        "pred": preds.tolist(),
        "paths": test.image_path.tolist(),
    }


def main() -> None:
    """Ejecuta todos los pliegues y agrupa las predicciones fuera de fuente."""
    args = parse_args()
    splits_dir = args.splits_dir or (get_output_root() / "splits" / "seed_42")
    output = args.output or (get_output_root() / "experiments" / "leave_one_source_out.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = get_dataset_root()

    manifest = pd.concat(
        [pd.read_csv(splits_dir / f"{name}.csv") for name in ("train", "val", "test")],
        ignore_index=True,
    )
    manifest["provenance"] = manifest.image_path.map(provenance_from_path)
    unresolved = int(manifest.provenance.isna().sum())
    if unresolved:
        raise SystemExit(f"{unresolved} imágenes sin grupo de procedencia")

    manifest, dropped = deduplicate(manifest, dataset_root, args.num_workers)
    classes = sorted(manifest.label.unique())
    folds = build_folds(manifest)
    if args.folds:
        wanted = set(args.folds.split(","))
        folds = [f for f in folds if f["test_group"] in wanted]

    print(f"[*] grupos de procedencia: {manifest.provenance.nunique()} | pliegues: {len(folds)}",
          flush=True)
    for fold in folds:
        print(f"    test={fold['test_group']} ({fold['n_test']}) val={fold['val_group']} "
              f"clases={len(fold['test_classes'])}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for fold in folds:
        result = run_fold(fold, manifest, dataset_root, classes, args, device)
        results.append(result)
        print(f"[{result['test_group']}] TEST macro_f1={result['test_macro_f1']:.4f} "
              f"accuracy={result['test_accuracy']:.4f}", flush=True)
        output.write_text(json.dumps(
            {"classes": classes, "duplicates_dropped": dropped, "folds": results},
            indent=2, ensure_ascii=False), encoding="utf-8")

    pooled_true = [t for r in results for t in r["true"]]
    pooled_pred = [p for r in results for p in r["pred"]]
    pooled = {
        "n": len(pooled_true),
        "macro_f1": float(f1_score(pooled_true, pooled_pred, average="macro")),
        "accuracy": float(np.mean(np.asarray(pooled_true) == np.asarray(pooled_pred))),
        "per_class_f1": dict(zip(classes, f1_score(
            pooled_true, pooled_pred, average=None, labels=range(len(classes))).tolist())),
    }
    print(json.dumps(pooled, indent=2, ensure_ascii=False), flush=True)
    output.write_text(json.dumps(
        {"classes": classes, "duplicates_dropped": dropped, "pooled": pooled, "folds": results},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[*] resultados en {output}", flush=True)


if __name__ == "__main__":
    main()
