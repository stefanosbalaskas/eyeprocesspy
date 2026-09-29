#!/usr/bin/env python3
"""Evaluate whether the SRL pilot is ready to estimate the focal Prompt effect.

This gate separates:
- BLOCKER: must be resolved before branch-specific focal effects are estimated;
- LIMITATION: analysis may proceed only with the limitation explicitly retained;
- INFO: documented design context that does not block the primary Prompt estimand.

The gate does not inspect or calculate the Prompt effect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _status_row(
    gate_id: str,
    severity: str,
    ready: bool,
    message: str,
) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "severity": severity,
        "ready": bool(ready),
        "message": message,
    }


def _decision_status(
    decisions: pd.DataFrame,
    family: str,
    decision: str,
) -> str | None:
    hit = decisions.loc[
        decisions["decision_family"].astype(str).eq(family)
        & decisions["decision"].astype(str).eq(decision),
        "status",
    ]
    if len(hit) != 1:
        return None
    return str(hit.iloc[0]).strip()


def evaluate_readiness(
    research_dir: Path,
    *,
    archive_audit_dir: Path | None = None,
    identifiability_dir: Path | None = None,
    identity_audit_dir: Path | None = None,
    timebase_audit_dir: Path | None = None,
) -> pd.DataFrame:
    decisions = _read(research_dir / "srl_decision_registry.csv")
    detectors = _read(research_dir / "srl_detector_plan.csv")
    quality = _read(research_dir / "srl_quality_plan.csv")
    models = _read(research_dir / "srl_model_plan.csv")
    cohorts = _read(research_dir / "srl_cohort_plan.csv")

    rows: list[dict[str, object]] = []
    dense_timebase_ok = False
    dense_timebase_message = "actual archive audit not supplied"

    if archive_audit_dir is None:
        rows.append(
            _status_row(
                "actual_archive_audit",
                "BLOCKER",
                False,
                (
                    "Actual Figshare release has not been audited in this execution. "
                    "Run audit_srl_dataset.py on the extracted archive."
                ),
            )
        )
    else:
        raw_manifest_path = archive_audit_dir / "raw_sample_manifest.csv"
        raw_failures_path = archive_audit_dir / "raw_sample_failures.csv"
        stimulus_audit_path = archive_audit_dir / "stimulus_file_audit.csv"
        condition_path = archive_audit_dir / "condition_counts.csv"

        try:
            raw_manifest = _read(raw_manifest_path)
            raw_failures = _read(raw_failures_path)
            stimulus_audit = _read(stimulus_audit_path)
            condition_counts = _read(condition_path)

            raw_ok = len(raw_manifest) == 83 and raw_failures.empty
            stimulus_ok = (
                "present" in stimulus_audit
                and stimulus_audit["present"].astype(bool).all()
            )
            condition_levels = set(
                condition_counts["experiment_condition"].astype(str)
            )
            condition_ok = {"Prompt", "Non-prompt"}.issubset(condition_levels)

            archive_ok = bool(raw_ok and stimulus_ok and condition_ok)
            rows.append(
                _status_row(
                    "actual_archive_audit",
                    "BLOCKER",
                    archive_ok,
                    (
                        f"raw_manifest_rows={len(raw_manifest)} (expected 83); "
                        f"raw_failures={len(raw_failures)}; "
                        f"stimuli_complete={stimulus_ok}; "
                        f"prompt_conditions_present={condition_ok}."
                    ),
                )
            )
        except Exception as exc:
            rows.append(
                _status_row(
                    "actual_archive_audit",
                    "BLOCKER",
                    False,
                    f"Archive audit outputs are incomplete or invalid: {type(exc).__name__}: {exc}",
                )
            )

    if identifiability_dir is None:
        rows.append(
            _status_row(
                "prompt_identifiability",
                "BLOCKER",
                False,
                (
                    "Run audit_srl_identifiability.py on participants.csv and stimuli.csv "
                    "before estimating focal effects."
                ),
            )
        )
    else:
        try:
            estimands = _read(
                identifiability_dir / "estimand_identifiability.csv"
            )
            hit = estimands.loc[
                estimands["estimand"].astype(str).eq("Prompt_vs_Non-prompt")
            ]
            prompt_ok = (
                len(hit) == 1
                and bool(hit.iloc[0]["primary_eligible"])
            )
            rows.append(
                _status_row(
                    "prompt_identifiability",
                    "BLOCKER",
                    prompt_ok,
                    (
                        str(hit.iloc[0]["design_status"])
                        if len(hit) == 1
                        else "Prompt estimand row missing."
                    ),
                )
            )
        except Exception as exc:
            rows.append(
                _status_row(
                    "prompt_identifiability",
                    "BLOCKER",
                    False,
                    f"Identifiability audit outputs invalid: {type(exc).__name__}: {exc}",
                )
            )

    aoi_status = _decision_status(
        decisions,
        "AOI",
        "nominal geometric AOIs",
    )
    rows.append(
        _status_row(
            "aoi_exact_boundary",
            "BLOCKER",
            aoi_status in {"frozen", "frozen_multiverse"},
            (
                f"status={aoi_status!r}. Exact source-compatible pixel boundary "
                "convention must be frozen before focal effects."
            ),
        )
    )

    ivt_rows = detectors.loc[
        detectors["detector_id"].astype(str).isin(
            ["ivt_30_100_simple", "ivt_40_50_simple"]
        )
    ]
    ivt_ok = (
        len(ivt_rows) == 2
        and not ivt_rows["status"].astype(str).str.contains(
            "pending",
            case=False,
            regex=False,
        ).any()
    )
    rows.append(
        _status_row(
            "ivt_gap_semantics",
            "BLOCKER",
            ivt_ok,
            (
                "I-VT maximum-gap/interpolation semantics must be frozen; "
                f"current statuses={ivt_rows['status'].astype(str).tolist()}."
            ),
        )
    )

    if identity_audit_dir is None or timebase_audit_dir is None:
        rows.append(
            _status_row(
                "cohort_timebase_validation",
                "BLOCKER",
                False,
                (
                    "Run audit_srl_identity.py and audit_srl_task_timebase.py on the "
                    "actual release before executing the frozen cohort universe."
                ),
            )
        )
    else:
        try:
            identity_summary = _read(identity_audit_dir / "identity_summary.csv")
            presence = _read(identity_audit_dir / "id_source_presence.csv")
            participant_timebase = _read(
                timebase_audit_dir / "participant_timebase_summary.csv"
            )

            identity_values = {
                str(row["metric"]): int(row["value"])
                for _, row in identity_summary.iterrows()
            }
            exact_ids = set(
                presence.loc[
                    presence["exact_cross_source_raw_eligible"].astype(bool),
                    "participant_id",
                ].astype(str)
            )
            nominal_ids = set(
                participant_timebase.loc[
                    participant_timebase["all_8_tasks_nominal_250hz"].astype(bool),
                    "participant_id",
                ].astype(str)
            )
            dense_ids = set(
                participant_timebase.loc[
                    participant_timebase[
                        "all_8_tasks_dense_regular_250hz"
                    ].astype(bool),
                    "participant_id",
                ].astype(str)
            )

            exact_nominal_ids = exact_ids & nominal_ids
            cohort_values = {
                str(row["cohort_id"]): int(row["n_participants"])
                for _, row in cohorts.iterrows()
                if str(row["status"]).strip() == "frozen"
            }

            cohort_ok = bool(
                identity_values.get("raw_file_ids") == 83
                and identity_values.get("exact_raw_metadata_ids") == 82
                and identity_values.get("exact_raw_prompt") == 41
                and identity_values.get("exact_raw_non_prompt") == 41
                and len(nominal_ids) == 78
                and len(exact_nominal_ids) == 77
                and len(dense_ids) == 0
                and cohort_values.get("exact_raw82") == 82
                and cohort_values.get("nominal250_77") == 77
            )
            rows.append(
                _status_row(
                    "cohort_timebase_validation",
                    "BLOCKER",
                    cohort_ok,
                    (
                        f"raw_ids={identity_values.get('raw_file_ids')}; "
                        f"exact_raw={len(exact_ids)}; "
                        f"exact_prompt={identity_values.get('exact_raw_prompt')}; "
                        f"exact_non_prompt={identity_values.get('exact_raw_non_prompt')}; "
                        f"raw_nominal250={len(nominal_ids)}; "
                        f"exact_nominal250={len(exact_nominal_ids)}; "
                        f"all8_dense250={len(dense_ids)}. "
                        "REMoDNaV remains outside the primary denominator because "
                        "the actual task-level timebase is not strictly dense/regular."
                    ),
                )
            )
        except Exception as exc:
            rows.append(
                _status_row(
                    "cohort_timebase_validation",
                    "BLOCKER",
                    False,
                    f"Identity/timebase audit outputs invalid: {type(exc).__name__}: {exc}",
                )
            )

    recovery = quality.loc[
        quality["quality_id"].astype(str).eq("transition_recovery_gate")
    ]
    recovery_status = (
        str(recovery.iloc[0]["status"]).strip()
        if len(recovery) == 1
        else "missing"
    )
    rows.append(
        _status_row(
            "quality_recovery_calibration",
            "INFO",
            True,
            (
                f"status={recovery_status!r}. Recovery-calibrated quality filtering is "
                "an optional extension; the minimum primary quality multiverse is the "
                "released-sample branch plus the documented 80% reference/sensitivity."
            ),
        )
    )

    primary_model = models.loc[
        models["model_id"].astype(str).eq("primary_nb_glmm")
    ]
    model_ok = (
        len(primary_model) == 1
        and str(primary_model.iloc[0]["status"]).strip() == "engine_frozen"
        and "glmmTMB" in str(primary_model.iloc[0]["family"])
    )
    rows.append(
        _status_row(
            "primary_model_engine",
            "BLOCKER",
            model_ok,
            (
                "Primary estimator must remain the frozen glmmTMB NB2 random-intercept "
                "transition-rate model."
            ),
        )
    )

    correction_status = _decision_status(
        decisions,
        "preprocessing",
        "manual gaze-offset correction",
    )
    rows.append(
        _status_row(
            "manual_offset_provenance",
            "LIMITATION",
            correction_status in {"accepted_limitation", "resolved"},
            (
                f"status={correction_status!r}. If unrecoverable, analysis can proceed "
                "only with scope explicitly limited to uncertainty downstream of the "
                "released sample coordinates."
            ),
        )
    )

    modality_status = _decision_status(
        decisions,
        "scientific_claim",
        "material-type contrast identifiability",
    )
    rows.append(
        _status_row(
            "material_type_secondary_scope",
            "INFO",
            True,
            (
                f"status={modality_status!r}. This does not block the randomized Prompt "
                "primary estimand; it constrains wording of secondary material-type analyses."
            ),
        )
    )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--research-dir",
        type=Path,
        default=Path("research/gaze_claim_robustness"),
    )
    parser.add_argument("--archive-audit-dir", type=Path)
    parser.add_argument("--identifiability-dir", type=Path)
    parser.add_argument("--identity-audit-dir", type=Path)
    parser.add_argument("--timebase-audit-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("srl_execution_readiness.csv"),
    )
    args = parser.parse_args()

    status = evaluate_readiness(
        args.research_dir,
        archive_audit_dir=args.archive_audit_dir,
        identifiability_dir=args.identifiability_dir,
        identity_audit_dir=args.identity_audit_dir,
        timebase_audit_dir=args.timebase_audit_dir,
    )
    status.to_csv(args.output, index=False)

    print(status.to_string(index=False))
    blockers = status.loc[
        status["severity"].eq("BLOCKER") & ~status["ready"].astype(bool)
    ]
    print(f"\nUnresolved blockers: {len(blockers)}")
    return 2 if len(blockers) else 0


if __name__ == "__main__":
    raise SystemExit(main())
