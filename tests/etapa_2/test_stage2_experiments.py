import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts" / "etapa_2" / "stage2_experiments.py"
SPEC = importlib.util.spec_from_file_location("stage2_experiments", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_source_parser_preserves_compound_source_name():
    name = "potassium_deficiency_maize_2_roboflow_real_ab12.jpg"
    assert (
        MODULE._source_from_filename(name, "potassium_deficiency", "real")
        == "maize_2_roboflow"
    )


def test_fingerprint_is_independent_of_row_order():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"image_path": "clean/b.jpg", "label": "healthy", "environment": "real", "sha256": "b"},
            {"image_path": "clean/a.jpg", "label": "healthy", "environment": "real", "sha256": "a"},
        ]
    )
    assert MODULE._fingerprint(frame) == MODULE._fingerprint(frame.iloc[::-1])


def test_probe_smoke_test_returns_nine_probabilities():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(9), 8)
    features = rng.normal(size=(len(labels), 24)).astype(np.float32)
    features[np.arange(len(labels)), labels] += 3.0
    params = {
        "epochs": 2,
        "batch_size": 32,
        "alpha": 1e-4,
        "eta0": 0.02,
        "penalty": "l2",
        "average": True,
        "feature_norm": "l2",
        "balance_power": 0.5,
    }
    bundle, history = MODULE.fit_probe(
        features[:54], labels[:54], params, features[54:], labels[54:]
    )
    probabilities = bundle.predict_proba(features[54:])
    assert probabilities.shape == (18, 9)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert len(history) == 2


def test_numeric_probe_roundtrip_matches_sklearn(tmp_path):
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(9), 6)
    features = rng.normal(size=(len(labels), 18)).astype(np.float32)
    bundle, _ = MODULE.fit_probe(
        features,
        labels,
        {
            "epochs": 2,
            "batch_size": 32,
            "alpha": 1e-4,
            "eta0": 0.01,
            "penalty": "l2",
            "average": True,
            "feature_norm": "l2",
            "balance_power": 0.0,
        },
    )
    artifact = tmp_path / "probe.npz"
    MODULE.save_numeric_probe(bundle, artifact)
    portable = MODULE.load_numeric_probe(artifact)
    assert np.allclose(
        portable.predict_proba(features), bundle.predict_proba(features), atol=1e-6
    )
