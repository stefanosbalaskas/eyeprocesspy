from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_example():
    path = Path(__file__).parents[1] / "examples" / "aoi_perturbation_failure_clinic.py"
    spec = importlib.util.spec_from_file_location("aoi_failure_clinic_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_clinic_example_outputs_and_audit(tmp_path):
    paths = _load_example().run(tmp_path)
    assert all(path.exists() for path in paths.values())
    audit = pd.read_csv(paths["audit"])
    failures = pd.read_csv(paths["failures"])
    models = pd.read_csv(paths["models"])
    assignments = pd.read_csv(paths["assignments"])
    report = paths["report"].read_text(encoding="utf-8")
    assert audit.set_index("perturbation_id").loc["erode_6_px", "status"] == "failed"
    assert set(failures["stage"]) == {"geometry", "model"}
    assert models["model_converged"].tolist() == [True, False]
    ambiguous = assignments.loc[
        (assignments["perturbation_id"] == "dilate_1_px") & (assignments["observation_id"] == "o2"),
        "aoi_assignment",
    ].iloc[0]
    assert ambiguous == "__ambiguous__"
    assert "1/2 converged" in report
    assert "not probabilities" in report
