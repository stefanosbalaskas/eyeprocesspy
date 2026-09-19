from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from aoi_reporting_bundle import write_reporting_bundle


def dummy_result():
    return {
        "grid_result": {
            "audit": pd.DataFrame(
                {
                    "perturbation_id": ["baseline", "dilate_0.5_deg"],
                    "status": ["completed", "completed"],
                    "message": ["", ""],
                }
            )
        },
        "stability": {
            "overall": pd.DataFrame(
                {
                    "perturbation_id": ["baseline", "dilate_0.5_deg"],
                    "proportion_unchanged": [1.0, 0.95],
                }
            )
        },
        "assignment_table": pd.DataFrame(
            {
                "perturbation_id": ["baseline", "dilate_0.5_deg"],
                "observation_id": [1, 1],
                "aoi_assignment": ["claim", "claim"],
            }
        ),
        "assignment_probability": pd.DataFrame(
            {"observation_id": [1], "aoi_assignment": ["claim"], "frequency": [1.0]}
        ),
        "models": pd.DataFrame(
            {
                "perturbation_id": ["baseline"],
                "term": ["condition"],
                "estimate": [0.4],
                "SE": [0.1],
                "CI_low": [0.2],
                "CI_high": [0.6],
                "p_value": [0.01],
                "model_converged": [True],
                "N": [24],
                "direction": ["positive"],
            }
        ),
        "failures": pd.DataFrame(columns=["perturbation_id", "message", "stage"]),
        "provenance": {
            "observation_level": "fixation",
            "overlap_policy": "ambiguous",
            "software": {"package": "eyeprocesspy", "version": "test"},
        },
        "caveat": "Robustness proportions are not probabilities of truth.",
    }


def test_reporting_bundle(tmp_path: Path):
    result = dummy_result()
    inference = pd.DataFrame(
        {
            "term": ["condition"],
            "n_models": [1],
            "n_converged": [1],
            "median_N": [24.0],
            "min_N": [24.0],
            "max_N": [24.0],
        }
    )
    figure = tmp_path / "source.svg"
    figure.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
    out = tmp_path / "bundle"
    paths = write_reporting_bundle(
        result,
        out,
        inference_stability=inference,
        report_text="## report\n\nSynthetic.",
        analysis_plan_text="analysis_id: test\n",
        figure_paths={"geometry": figure},
    )

    expected = {
        "analysis-plan.yml",
        "perturbation-audit.csv",
        "assignment-stability.csv",
        "assignment-table.csv",
        "assignment-frequency.csv",
        "model-results.csv",
        "inference-stability.csv",
        "failures.csv",
        "report.md",
        "provenance.json",
        "source.svg",
        "manifest.json",
    }
    assert set(paths) == expected
    assert all(path.exists() for path in paths.values())

    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["scientific_contract"]["observation_level"] == "fixation"
    assert manifest["scientific_contract"]["overlap_policy"] == "ambiguous"
    names = [row["path"] for row in manifest["files"]]
    assert names == sorted(names)
    assert "manifest.json" not in names
    assert all(len(row["sha256"]) == 64 for row in manifest["files"])
    assert "not probabilities" in paths["provenance.json"].read_text(encoding="utf-8")

    out2 = tmp_path / "bundle-2"
    paths2 = write_reporting_bundle(
        result,
        out2,
        inference_stability=inference,
        report_text="## report\n\nSynthetic.",
        analysis_plan_text="analysis_id: test\n",
        figure_paths={"geometry": figure},
    )
    manifest2 = json.loads(paths2["manifest.json"].read_text(encoding="utf-8"))
    assert manifest == manifest2
