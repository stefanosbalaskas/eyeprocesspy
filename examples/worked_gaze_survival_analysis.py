"""End-to-end synthetic censored gaze-latency analysis.

The example deliberately begins with trial windows plus fixation/AOI-visit rows,
so right censoring is constructed rather than hidden inside a prepared fixture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from eyeprocesspy.survival import (
    check_gaze_proportional_hazards,
    compare_gaze_survival_models,
    compare_gaze_survival_specifications,
    fit_gaze_aft_model,
    fit_gaze_mixed_cox_model,
    plot_gaze_cumulative_incidence,
    plot_gaze_hazard,
    plot_gaze_survival_curve,
    prepare_gaze_survival_data,
    report_gaze_survival_model,
    simulate_gaze_survival_inputs,
    summarise_gaze_censoring,
    tidy_gaze_survival_model,
    validate_gaze_survival_data,
)


def _json_ready(report: dict[str, Any]) -> dict[str, Any]:
    out = dict(report)
    if hasattr(out.get("effects"), "to_dict"):
        out["effects"] = out["effects"].to_dict(orient="records")
    return out


def run_worked_example(output_dir: str | Path = "gaze-survival-output") -> dict[str, Any]:
    """Run the disclosure-inspection workflow and write manuscript-ready outputs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw = simulate_gaze_survival_inputs(
        "disclosure", seed=20260918, n_participants=36, trials_per_participant=3
    )
    data = prepare_gaze_survival_data(
        raw["trials"],
        raw["events"],
        target_aoi="disclosure",
        event_type="first_fixation",
        condition_col="condition_id",
        observation_end_reason_col="observation_end_reason",
        min_valid_fraction=0.90,
        source_data="synthetic_disclosure_trial_event_inputs",
        preprocessing_specification="synthetic_truth_no_filtering",
        event_detector="synthetic_truth",
        aoi_specification="fixed synthetic disclosure AOI",
        quality_rules={"minimum_fixation_ms": 80, "valid_fraction_min": 0.90},
    )

    issues = validate_gaze_survival_data(data, raise_on_error=False)
    censoring = summarise_gaze_censoring(data, by="condition")

    # Repeated trials: cluster-robust Cox in Python. Latent frailty is exposed in
    # R eyeprocess via coxme rather than being silently approximated here.
    cox = fit_gaze_mixed_cox_model(
        data,
        "C(condition)",
        participant_col="participant_id",
        structure="cluster_robust",
    )
    weibull = fit_gaze_aft_model(data, "C(condition)", distribution="weibull")
    lognormal = fit_gaze_aft_model(data, "C(condition)", distribution="lognormal")
    ph = check_gaze_proportional_hazards(cox)
    comparison = compare_gaze_survival_models(cox, weibull, lognormal)

    # One transparent sensitivity branch: same data, explicitly different AOI
    # specification metadata. Real analyses should rebuild the table from the
    # alternative geometry/detector rather than merely relabel it.
    expanded_aoi = data.copy()
    expanded_aoi["aoi_specification"] = "synthetic expanded disclosure AOI"
    sensitivity = compare_gaze_survival_specifications(
        {"primary": data, "expanded_aoi_demo": expanded_aoi},
        "C(condition)",
        model_families=["cox_cluster_robust", "aft_weibull"],
    )

    censoring.to_csv(output / "censoring-summary.csv", index=False)
    tidy_gaze_survival_model(cox).to_csv(output / "cox-effects.csv", index=False)
    tidy_gaze_survival_model(weibull).to_csv(output / "weibull-aft-effects.csv", index=False)
    ph.to_csv(output / "cox-ph-diagnostics.csv", index=False)
    comparison.to_csv(output / "model-comparison.csv", index=False)
    sensitivity.to_csv(output / "sensitivity-specifications.csv", index=False)

    report = report_gaze_survival_model(cox)
    (output / "cox-report.json").write_text(
        json.dumps(_json_ready(report), indent=2, default=str), encoding="utf-8"
    )

    for filename, plotter in (
        ("km-disclosure.svg", lambda: plot_gaze_survival_curve(data, group="condition")),
        (
            "cumulative-incidence-disclosure.svg",
            lambda: plot_gaze_cumulative_incidence(data, group="condition"),
        ),
        ("hazard-disclosure.svg", lambda: plot_gaze_hazard(data, group="condition")),
    ):
        ax = plotter()
        ax.figure.tight_layout()
        ax.figure.savefig(output / filename, bbox_inches="tight")
        plt.close(ax.figure)

    return {
        "data": data,
        "validation": issues,
        "censoring": censoring,
        "cox": cox,
        "weibull": weibull,
        "lognormal": lognormal,
        "ph_diagnostics": ph,
        "comparison": comparison,
        "sensitivity": sensitivity,
        "report": report,
        "output_dir": output,
    }


if __name__ == "__main__":
    result = run_worked_example()
    print(result["censoring"].to_string(index=False))
    print(tidy_gaze_survival_model(result["cox"]).to_string(index=False))
    print(result["ph_diagnostics"].to_string(index=False))
    print(f"Outputs written to {result['output_dir'].resolve()}")