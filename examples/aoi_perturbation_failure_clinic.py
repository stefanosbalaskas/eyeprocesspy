"""Synthetic failure-clinic example for AOI perturbation sensitivity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    aois = pd.DataFrame(
        {
            "aoi_id": ["claim", "cta"],
            "xmin": [0.0, 11.0],
            "xmax": [10.0, 20.0],
            "ymin": [0.0, 0.0],
            "ymax": [10.0, 10.0],
        }
    )
    observations = pd.DataFrame(
        {
            "obs": ["o1", "o2", "o3", "o4", "o5"],
            "participant": ["p1", "p1", "p1", "p2", "p2"],
            "trial": [1, 1, 1, 1, 1],
            "x": [5.0, 10.5, 15.0, np.nan, 30.0],
            "y": [5.0, 5.0, 5.0, 5.0, 5.0],
            "duration": [0.20, 0.15, 0.22, 0.18, 0.10],
            "time": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )
    return observations, aois


def diagnostic_model(features: pd.DataFrame, assigned: pd.DataFrame, spec) -> pd.DataFrame:
    pid = str(spec["perturbation_id"])
    if pid == "shift_x_1_px":
        raise RuntimeError("synthetic optimizer failure for documentation")
    if pid == "dilate_1_px":
        return pd.DataFrame(
            [
                {
                    "term": "condition",
                    "estimate": 0.30,
                    "SE": 0.20,
                    "CI_low": -0.09,
                    "CI_high": 0.69,
                    "p_value": 0.13,
                    "model_converged": False,
                    "N": 4,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "term": "condition",
                "estimate": 0.42,
                "SE": 0.12,
                "CI_low": 0.18,
                "CI_high": 0.66,
                "p_value": 0.01,
                "model_converged": True,
                "N": 4,
            }
        ]
    )


def run(output_dir: Path) -> dict[str, Path]:
    observations, aois = synthetic_inputs()
    grid = ep.create_aoi_perturbation_grid(
        dilations=[1.0],
        erosions=[6.0],
        translations_x=[1.0],
        include_baseline=True,
        unit="px",
    )
    result = ep.run_aoi_sensitivity_analysis(
        observations,
        aois,
        grid,
        x_col="x",
        y_col="y",
        observation_id_col="obs",
        participant_col="participant",
        trial_col="trial",
        duration_col="duration",
        time_col="time",
        observation_level="fixation",
        overlap_policy="ambiguous",
        model_callback=diagnostic_model,
        preprocessing_specification={"duration_unit": "seconds"},
        event_detector="synthetic_fixations",
        quality_rules={"missing_coordinates": "preserve", "overlap": "ambiguous"},
        model_specification={"purpose": "documentation-only failure clinic"},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": output_dir / "aoi_failure_audit.csv",
        "failures": output_dir / "aoi_failure_failures.csv",
        "models": output_dir / "aoi_failure_models.csv",
        "assignments": output_dir / "aoi_failure_assignments.csv",
        "report": output_dir / "aoi_failure_report.md",
        "plot": output_dir / "aoi_failure_geometry.svg",
    }
    result["grid_result"]["audit"].to_csv(paths["audit"], index=False)
    result["failures"].to_csv(paths["failures"], index=False)
    result["models"].to_csv(paths["models"], index=False)
    result["assignment_table"].to_csv(paths["assignments"], index=False)
    paths["report"].write_text(ep.report_aoi_sensitivity(result), encoding="utf-8")

    ax = ep.plot_aoi_perturbations(
        result,
        perturbation_id="dilate_1_px",
        data=observations,
        x_col="x",
        y_col="y",
    )
    ax.figure.savefig(paths["plot"], bbox_inches="tight")
    plt.close(ax.figure)

    audit = result["grid_result"]["audit"].set_index("perturbation_id")
    assert audit.loc["erode_6_px", "status"] == "failed"
    assert set(result["failures"]["stage"]) == {"geometry", "model"}
    assert result["assignments"]["dilate_1_px"].tolist()[1] == ep.AMBIGUOUS
    assert pd.isna(result["assignments"]["baseline"].tolist()[3])
    inference = ep.assess_aoi_inference_stability(result, term="condition")
    assert int(inference.iloc[0]["n_models"]) == 2
    assert int(inference.iloc[0]["n_converged"]) == 1
    assert "not probabilities" in paths["report"].read_text(encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workflow-output/aoi-failure-clinic"),
    )
    args = parser.parse_args()
    for name, path in run(args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
