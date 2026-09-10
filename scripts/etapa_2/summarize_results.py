"""Consolida números ya producidos para redactar y auditar el informe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize(output_dir: Path) -> dict:
    tuning = load_json(output_dir / "tuning" / "summary.json")
    models = pd.read_csv(output_dir / "model_comparison" / "models.csv")
    ensemble = load_json(output_dir / "ensemble" / "selection.json")
    complementarity = pd.read_csv(output_dir / "ensemble" / "complementarity.csv")
    cv = pd.read_csv(output_dir / "cross_validation" / "folds.csv")
    cv_summary = load_json(output_dir / "cross_validation" / "summary.json")
    cv_classes = pd.read_csv(output_dir / "cross_validation" / "per_class_folds.csv")
    final = load_json(output_dir / "final" / "metrics.json")
    report = pd.read_csv(output_dir / "final" / "classification_report.csv", index_col=0)
    confusion = pd.read_csv(output_dir / "final" / "confusion_matrix.csv", index_col=0)
    predictions = pd.read_csv(output_dir / "final" / "predictions.csv")
    gaps = pd.read_csv(output_dir / "fairness" / "gaps.csv")
    groups = pd.read_csv(output_dir / "fairness" / "by_group.csv")
    env_class = pd.read_csv(output_dir / "fairness" / "environment_within_class.csv")

    potassium_confusions = confusion.loc["potassium_deficiency"].drop(
        "potassium_deficiency"
    ).sort_values(ascending=False)
    npk = ["nitrogen_deficiency", "phosphorus_deficiency", "potassium_deficiency"]
    npk_cross = {
        actual: {
            predicted: int(confusion.loc[actual, predicted])
            for predicted in npk
            if predicted != actual
        }
        for actual in npk
    }
    high_confidence_errors = predictions[
        (~predictions["correct"]) & (predictions["confidence"] >= 0.90)
    ].sort_values("confidence", ascending=False)
    class_cv = cv_classes.groupby("class")["f1"].agg(["mean", "std"])
    best_pair = complementarity.sort_values("jaccard_errors").iloc[0]
    fairness_extremes = {}
    for dimension, frame in groups.groupby("dimension"):
        fairness_extremes[dimension] = {}
        for metric in ("precision", "recall", "f1"):
            low = frame.loc[frame[metric].idxmin()]
            high = frame.loc[frame[metric].idxmax()]
            fairness_extremes[dimension][metric] = {
                "lowest_group": str(low["group"]),
                "lowest_value": float(low[metric]),
                "highest_group": str(high["group"]),
                "highest_value": float(high[metric]),
            }

    payload = {
        "tuning": {
            **tuning,
            "macro_f1_absolute_gain": tuning["best_metrics"]["macro_f1"]
            - tuning["baseline_metrics"]["macro_f1"],
        },
        "models": models.to_dict(orient="records"),
        "best_model": models.sort_values("macro_f1", ascending=False).iloc[0].to_dict(),
        "ensemble": ensemble,
        "most_complementary_error_pair": best_pair.to_dict(),
        "cross_validation": {
            **cv_summary,
            "lowest_fold": cv.sort_values("macro_f1").iloc[0].to_dict(),
            "highest_fold": cv.sort_values("macro_f1", ascending=False).iloc[0].to_dict(),
            "lowest_mean_class_f1": class_cv.sort_values("mean").iloc[0].to_dict()
            | {"class": class_cv.sort_values("mean").index[0]},
            "most_variable_class_f1": class_cv.sort_values("std", ascending=False).iloc[0].to_dict()
            | {"class": class_cv.sort_values("std", ascending=False).index[0]},
        },
        "final": final,
        "potassium": {
            "support": int(report.loc["potassium_deficiency", "support"]),
            "precision": float(report.loc["potassium_deficiency", "precision"]),
            "recall": float(report.loc["potassium_deficiency", "recall"]),
            "f1": float(report.loc["potassium_deficiency", "f1-score"]),
            "correct": int(confusion.loc["potassium_deficiency", "potassium_deficiency"]),
            "errors": int(potassium_confusions.sum()),
            "confusions": {key: int(value) for key, value in potassium_confusions.items() if value},
        },
        "npk_cross_confusions": npk_cross,
        "gray_leaf_spot_to_nclb": int(
            confusion.loc["gray_leaf_spot", "northern_corn_leaf_blight"]
        ),
        "nclb_to_gray_leaf_spot": int(
            confusion.loc["northern_corn_leaf_blight", "gray_leaf_spot"]
        ),
        "errors": {
            "total": int((~predictions["correct"]).sum()),
            "confidence_ge_0_90": int(len(high_confidence_errors)),
            "maximum_confidence": float(high_confidence_errors["confidence"].max())
            if len(high_confidence_errors)
            else None,
            "top_high_confidence": high_confidence_errors.head(5)[
                ["image_path", "label", "pred_label", "confidence"]
            ].to_dict(orient="records"),
        },
        "fairness": {
            "gaps": gaps.to_dict(orient="records"),
            "extremes": fairness_extremes,
            "environment_within_class": env_class.to_dict(orient="records"),
        },
    }
    path = output_dir / "report_analysis.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/etapa_2"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.output_dir), indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
