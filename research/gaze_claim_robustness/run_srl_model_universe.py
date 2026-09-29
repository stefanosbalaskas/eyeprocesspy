#!/usr/bin/env python3
"""Fit the frozen SRL Prompt transition-rate model across 144 specifications."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import prepare_srl_model_table
import srl_glmm_bridge

UNIVERSE_REQUIRED = {
    "universe_id",
    "detector_id",
    "eye",
    "viewing_distance_cm",
    "aoi_convention",
    "quality_rule",
    "cohort_id",
    "model_id",
    "estimand_id",
    "specification_hash",
}
MEASUREMENT_REQUIRED = {
    "participant_id",
    "stimulus_id",
    "detector_id",
    "eye",
    "viewing_distance_cm",
    "aoi_convention",
    "transition_count",
    "status",
    "experiment_condition",
    "metadata_tracking_ratio_percent",
}


def _require(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}.")


def cohort_sets(
    identity_presence: pd.DataFrame,
    timebase_summary: pd.DataFrame,
) -> dict[str, set[str]]:
    """Reconstruct the two frozen participant cohorts from audit evidence."""
    exact = set(
        identity_presence.loc[
            identity_presence["exact_cross_source_raw_eligible"].astype(bool),
            "participant_id",
        ].astype(str)
    )
    nominal = set(
        timebase_summary.loc[
            timebase_summary["all_8_tasks_nominal_250hz"].astype(bool),
            "participant_id",
        ].astype(str)
    )
    exact_nominal = exact & nominal

    if len(exact) != 82:
        raise ValueError(f"Expected exact_raw82=82; found {len(exact)}.")
    if len(exact_nominal) != 77:
        raise ValueError(
            f"Expected nominal250_77=77; found {len(exact_nominal)}."
        )
    return {
        "exact_raw82": exact,
        "nominal250_77": exact_nominal,
    }


def _append_status(status: object, reason: str) -> str:
    base = str(status).strip()
    if base == "ok":
        return reason
    if not base:
        return reason
    return base + "|" + reason


def apply_quality_rule(
    branch: pd.DataFrame,
    quality_rule: str,
) -> pd.DataFrame:
    """Apply one frozen quality branch without silently dropping trial rows."""
    out = branch.copy()
    if quality_rule == "released_sample":
        return out
    if quality_rule != "trial_80_sensitivity":
        raise ValueError(f"Unknown quality rule: {quality_rule!r}.")

    ratio = pd.to_numeric(
        out["metadata_tracking_ratio_percent"],
        errors="coerce",
    )
    missing = ~np.isfinite(ratio)
    below = np.isfinite(ratio) & ratio.lt(80.0)

    out.loc[missing, "status"] = out.loc[missing, "status"].map(
        lambda value: _append_status(value, "quality_missing_tracking_ratio")
    )
    out.loc[below, "status"] = out.loc[below, "status"].map(
        lambda value: _append_status(
            value,
            "quality_excluded_tracking_ratio_lt80",
        )
    )
    return out


def select_measurement_branch(
    measurement: pd.DataFrame,
    spec: pd.Series,
    participant_ids: set[str],
) -> pd.DataFrame:
    """Select one detector/eye/distance/AOI branch and frozen cohort."""
    distance = float(spec["viewing_distance_cm"])
    mask = (
        measurement["detector_id"].astype(str).eq(str(spec["detector_id"]))
        & measurement["eye"].astype(str).eq(str(spec["eye"]))
        & np.isclose(
            pd.to_numeric(
                measurement["viewing_distance_cm"],
                errors="coerce",
            ),
            distance,
            atol=1e-12,
            rtol=0,
        )
        & measurement["aoi_convention"].astype(str).eq(
            str(spec["aoi_convention"])
        )
        & measurement["participant_id"].astype(str).isin(participant_ids)
    )
    out = measurement.loc[mask].copy()
    expected = len(participant_ids) * 8
    if len(out) != expected:
        raise ValueError(
            f"{spec['universe_id']}: expected {expected} trial rows; "
            f"found {len(out)}."
        )
    key = ["participant_id", "stimulus_id"]
    if out.duplicated(key).any():
        raise ValueError(
            f"{spec['universe_id']}: duplicate participant × stimulus rows."
        )
    return out


def _fit_one(
    *,
    universe_row: pd.Series,
    branch: pd.DataFrame,
    stimuli: pd.DataFrame,
    r_script: Path,
    temp_root: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    quality_rule = str(universe_row["quality_rule"])
    quality_applied = apply_quality_rule(branch, quality_rule)
    model_table, audit = prepare_srl_model_table.prepare_model_table(
        quality_applied,
        stimuli,
    )

    universe_id = str(universe_row["universe_id"])
    model_dir = temp_root / universe_id
    model_dir.mkdir(parents=True, exist_ok=True)
    input_csv = model_dir / "model_table.csv"
    model_table.to_csv(input_csv, index=False)

    command = [
        "Rscript",
        str(r_script),
        "--input",
        str(input_csv),
        "--output-dir",
        str(model_dir),
        "--model-id",
        universe_id,
    ]
    proc = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    coef_path = model_dir / f"{universe_id}_prompt_coefficient.csv"
    status_path = model_dir / f"{universe_id}_status.csv"

    result: dict[str, object] = {
        **universe_row.to_dict(),
        "planned_trial_rows": len(model_table),
        "model_evaluable_rows": int(model_table["model_evaluable"].sum()),
        "r_return_code": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "status": "fit_failed",
        "failure_reason": pd.NA,
        "estimate": np.nan,
        "SE": np.nan,
        "CI_lower": np.nan,
        "CI_upper": np.nan,
        "N": np.nan,
        "converged": False,
        "rate_ratio": np.nan,
        "CI_lower_rate_ratio": np.nan,
        "CI_upper_rate_ratio": np.nan,
    }

    if coef_path.exists():
        fitted = srl_glmm_bridge.read_primary_glmm_result(coef_path)
        result.update(fitted)
        result["status"] = (
            "ok" if bool(fitted["converged"]) and proc.returncode == 0
            else "non_converged"
        )
    else:
        result["failure_reason"] = (
            proc.stderr[-2000:]
            or proc.stdout[-2000:]
            or "No coefficient artifact produced."
        )

    if status_path.exists():
        status = pd.read_csv(status_path)
        if len(status) == 1:
            for column in (
                "optimizer_convergence_code",
                "positive_definite_hessian",
                "participant_random_intercept_variance",
                "glmmTMB_version",
            ):
                if column in status:
                    result[column] = status.iloc[0][column]

    audit = audit.copy()
    audit.insert(0, "universe_id", universe_id)
    audit.insert(1, "quality_rule", quality_rule)
    audit.insert(2, "cohort_id", str(universe_row["cohort_id"]))
    return result, audit


def fit_universe(
    *,
    measurement: pd.DataFrame,
    stimuli: pd.DataFrame,
    universe: pd.DataFrame,
    identity_presence: pd.DataFrame,
    timebase_summary: pd.DataFrame,
    r_script: Path,
    limit_specs: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit all frozen specifications in deterministic universe order."""
    _require(measurement, MEASUREMENT_REQUIRED, "measurement")
    _require(universe, UNIVERSE_REQUIRED, "universe")
    cohorts = cohort_sets(identity_presence, timebase_summary)

    if limit_specs < 0:
        raise ValueError("limit_specs must be >= 0.")
    selected = universe.copy()
    if limit_specs:
        selected = selected.iloc[:limit_specs].copy()

    results: list[dict[str, object]] = []
    audits: list[pd.DataFrame] = []

    with tempfile.TemporaryDirectory(prefix="srl_glmm_") as temp:
        temp_root = Path(temp)
        for index, (_, spec) in enumerate(selected.iterrows(), start=1):
            cohort_id = str(spec["cohort_id"])
            if cohort_id not in cohorts:
                raise ValueError(f"Unknown cohort_id: {cohort_id!r}.")

            branch = select_measurement_branch(
                measurement,
                spec,
                cohorts[cohort_id],
            )
            result, audit = _fit_one(
                universe_row=spec,
                branch=branch,
                stimuli=stimuli,
                r_script=r_script,
                temp_root=temp_root,
            )
            results.append(result)
            audits.append(audit)
            print(
                f"[{index}/{len(selected)}] {spec['universe_id']} "
                f"status={result['status']} "
                f"evaluable={result['model_evaluable_rows']}",
                flush=True,
            )

    result_frame = pd.DataFrame(results)
    audit_frame = (
        pd.concat(audits, ignore_index=True, sort=False)
        if audits
        else pd.DataFrame()
    )
    return result_frame, audit_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_csv", type=Path)
    parser.add_argument("stimuli_csv", type=Path)
    parser.add_argument("universe_csv", type=Path)
    parser.add_argument("identity_presence_csv", type=Path)
    parser.add_argument("timebase_summary_csv", type=Path)
    parser.add_argument(
        "--r-script",
        type=Path,
        default=Path(
            "research/gaze_claim_robustness/fit_srl_primary_glmm.R"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_multiverse_models"),
    )
    parser.add_argument("--limit-specs", type=int, default=0)
    args = parser.parse_args()

    results, audits = fit_universe(
        measurement=pd.read_csv(args.measurement_csv),
        stimuli=pd.read_csv(args.stimuli_csv),
        universe=pd.read_csv(args.universe_csv),
        identity_presence=pd.read_csv(args.identity_presence_csv),
        timebase_summary=pd.read_csv(args.timebase_summary_csv),
        r_script=args.r_script,
        limit_specs=args.limit_specs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        args.output_dir / "srl_multiverse_model_results.csv",
        index=False,
    )
    audits.to_csv(
        args.output_dir / "srl_multiverse_row_audit.csv",
        index=False,
    )

    failures = results.loc[~results["status"].eq("ok")].copy()
    failures.to_csv(
        args.output_dir / "srl_multiverse_model_failures.csv",
        index=False,
    )

    print(
        f"Model specifications: {len(results)}; "
        f"ok={int(results['status'].eq('ok').sum())}; "
        f"non_ok={len(failures)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
