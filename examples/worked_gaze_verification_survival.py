"""End-to-end synthetic evidence-verification survival analysis.

The example models time to first entry into a source/evidence AOI. It begins
with trial windows plus AOI-visit rows so right censoring is constructed from
an explicit observation window rather than hidden in a pre-censored fixture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from eyeprocesspy.survival import (
    check_gaze_proportional_hazards,
    compare_gaze_survival_models,
    fit_gaze_aft_model,
    fit_gaze_mixed_cox_model,
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


def run_verification_example(
    output_dir: str | Path = "gaze-verification-survival-output",
) -> dict[str, Any]:
    """Run the evidence-verification workflow and write reporting outputs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw = simulate_gaze_survival_inputs(
        "verification",
        seed=20260918,
        n_participants=36,
        trials_per_participant=3,
    )
    data = prepare_gaze_survival_data(
        raw["trials"],
        raw["events"],
        target_aoi="source_evidence",
        event_type="first_aoi_entry",
        condition_col="condition_id",
        observation_end_reason_col="observation_end_reason",
        min_valid_fraction=0.90,
        source_data="synthetic_verification_trial_event_inputs",
        preprocessing_specification="synthetic_truth_no_filtering",
        event_detector="synthetic_truth",
        aoi_specification="fixed synthetic source/evidence AOI",
        quality_rules={"valid_fraction_min": 0.90},
    )

    validation = validate_gaze_survival_data(data, raise_on_error=False)
    censoring = summarise_gaze_censoring(data, by="condition")

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
    report = report_gaze_survival_model(cox)

    censoring.to_csv(output / "verification-censoring-summary.csv", index=False)
    tidy_gaze_survival_model(cox).to_csv(
        output / "verification-cox-effects.csv", index=False
    )
    tidy_gaze_survival_model(weibull).to_csv(
        output / "verification-weibull-aft-effects.csv", index=False
    )
    ph.to_csv(output / "verification-cox-ph-diagnostics.csv", index=False)
    comparison.to_csv(output / "verification-model-comparison.csv", index=False)
    (output / "verification-cox-report.json").write_text(
        json.dumps(_json_ready(report), indent=2, default=str), encoding="utf-8"
    )

    ax = plot_gaze_survival_curve(data, group="condition")
    ax.set_title("Time to first source/evidence AOI entry")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "km-evidence-verification.svg", bbox_inches="tight")
    plt.close(ax.figure)

    return {
        "data": data,
        "validation": validation,
        "censoring": censoring,
        "cox": cox,
        "weibull": weibull,
        "lognormal": lognormal,
        "ph_diagnostics": ph,
        "comparison": comparison,
        "report": report,
        "output_dir": output,
    }


if __name__ == "__main__":
    result = run_verification_example()
    print(result["censoring"].to_string(index=False))
    print(tidy_gaze_survival_model(result["cox"]).to_string(index=False))
    print(result["ph_diagnostics"].to_string(index=False))
    print(f"Outputs written to {result['output_dir'].resolve()}")
