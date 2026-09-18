"""Synthetic AOI perturbation sensitivity analysis.

The example is intentionally small enough for CI. It demonstrates a five-AOI
advertising/interface layout, degree-based dilation/erosion and translations,
feature recomputation, an explicit model callback, failure-aware summaries,
and four visual diagnostics. No private or empirical participant data are used.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep

OUTPUT = Path(__file__).resolve().parents[1] / "workflow-output"
OUTPUT.mkdir(exist_ok=True)


def make_aois() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "aoi_id": ["headline", "image", "claim", "disclosure", "cta"],
            "xmin": [100, 300, 300, 300, 680],
            "xmax": [400, 700, 700, 700, 920],
            "ymin": [80, 190, 360, 500, 600],
            "ymax": [180, 340, 480, 590, 710],
        }
    )


def make_fixations(seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    centers = [
        (250, 130, "headline"),
        (500, 270, "image"),
        (500, 430, "claim"),
        (500, 560, "disclosure"),
        (800, 650, "cta"),
    ]
    for participant in range(1, 9):
        condition = participant % 2
        for trial in range(1, 4):
            for order, (cx, cy, aoi) in enumerate(centers):
                n = 5 + (3 if aoi == "disclosure" and condition else 0)
                for fixation in range(n):
                    rows.append(
                        {
                            "obs": len(rows) + 1,
                            "participant": f"p{participant}",
                            "trial": trial,
                            "condition": condition,
                            "x": rng.normal(cx, 35),
                            "y": rng.normal(cy, 22),
                            "duration": rng.uniform(0.06, 0.18),
                            "time": order + fixation / 20,
                        }
                    )
    return pd.DataFrame(rows)


def explicit_ols_callback(
    features: pd.DataFrame, assigned: pd.DataFrame, spec
) -> pd.DataFrame:
    """Explicit illustrative OLS callback for disclosure dwell."""
    target = features.loc[features["aoi"].eq("disclosure")].copy()
    condition_map = assigned[["participant", "condition"]].drop_duplicates()
    target = target.merge(
        condition_map, on="participant", how="left", validate="many_to_one"
    )
    y = target["dwell"].to_numpy(float)
    X = np.column_stack(
        [np.ones(len(target)), target["condition"].to_numpy(float)]
    )
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    df = len(y) - X.shape[1]
    if df <= 0:
        return pd.DataFrame(
            [{
                "term": "condition", "estimate": np.nan, "SE": np.nan,
                "CI_low": np.nan, "CI_high": np.nan, "p_value": np.nan,
                "model_converged": False, "N": len(y),
            }]
        )
    sigma2 = float(np.sum(residual**2) / df)
    covariance = sigma2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(float(covariance[1, 1]))
    estimate = float(beta[1])
    return pd.DataFrame(
        [{
            "term": "condition", "estimate": estimate, "SE": se,
            "CI_low": estimate - 1.96 * se, "CI_high": estimate + 1.96 * se,
            "p_value": np.nan, "model_converged": True, "N": len(y),
        }]
    )


def main() -> None:
    data = make_fixations()
    aois = make_aois()
    viewing = dict(
        screen_width_px=1024,
        screen_height_px=768,
        viewing_distance=60,
        physical_screen_size=(53.1, 29.9),
        boundary_policy="allow",
    )
    grid = ep.create_aoi_perturbation_grid(
        dilations=[0.25, 0.50, 1.00],
        erosions=[0.25],
        translations_x=[0.50],
        translations_y=[0.50],
        unit="deg",
        include_baseline=True,
        **viewing,
    )
    result = ep.run_aoi_sensitivity_analysis(
        data,
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
        model_callback=explicit_ols_callback,
        preprocessing_specification={"duration_unit": "seconds"},
        event_detector="synthetic_fixations",
        quality_rules={"missing": "preserve", "overlap": "ambiguous"},
        model_specification={
            "family": "OLS",
            "outcome": "disclosure_dwell",
            "predictor": "condition",
        },
    )

    summary = ep.summarise_aoi_sensitivity(result)
    print(summary["assignment_stability"].to_string(index=False))
    print(summary["inference_stability"].to_string(index=False))
    print(ep.report_aoi_sensitivity(result))

    ax = ep.plot_aoi_perturbations(
        result,
        perturbation_id="dilate_0.5_deg",
        data=data.iloc[::12].copy(),
        x_col="x",
        y_col="y",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT / "aoi-perturbations.svg", bbox_inches="tight")
    plt.close(ax.figure)

    ax = ep.plot_aoi_assignment_stability(result)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT / "aoi-assignment-stability.svg", bbox_inches="tight")
    plt.close(ax.figure)

    ax = ep.plot_aoi_coefficient_stability(result, term="condition")
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT / "aoi-coefficient-stability.svg", bbox_inches="tight")
    plt.close(ax.figure)

    pairs = [
        (x, y)
        for x in (-0.25, 0.0, 0.25, 0.50)
        for y in (-0.25, 0.0, 0.25, 0.50)
    ]
    surface_grid = ep.create_aoi_perturbation_grid(
        anisotropic=pairs,
        unit="deg",
        include_baseline=True,
        **viewing,
    )
    surface = ep.run_aoi_sensitivity_analysis(
        data,
        aois,
        surface_grid,
        x_col="x",
        y_col="y",
        observation_id_col="obs",
        participant_col="participant",
        trial_col="trial",
        duration_col="duration",
        time_col="time",
    )
    ax = ep.plot_aoi_robustness_surface(surface)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT / "aoi-robustness-surface.svg", bbox_inches="tight")
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
