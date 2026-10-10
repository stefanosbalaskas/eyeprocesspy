#!/usr/bin/env python3
"""Validate the SRL measurement-only artifact before any focal model is fit.

This validator is intentionally structural. It does not inspect, summarize, or
rank the Prompt effect because no focal model has been fit at this stage.

Planned measurement structure:
82 participants
x 8 Tasks
x 3 detectors
x 2 eyes
x 3 viewing-distance assumptions
x 2 AOI geometries
= 23,616 rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_PARTICIPANTS = 82
EXPECTED_TASKS = tuple(f"Task_{i}" for i in range(1, 9))
EXPECTED_DETECTORS = (
    "ivt_30_100_simple",
    "ivt_40_50_simple",
    "idt_1_100",
)
EXPECTED_EYES = ("left", "right")
EXPECTED_DISTANCES = (60.0, 65.0, 70.0)
EXPECTED_AOIS = (
    "exact_quarters",
    "reported_area_center_seam",
)
EXPECTED_ROWS = (
    EXPECTED_PARTICIPANTS
    * len(EXPECTED_TASKS)
    * len(EXPECTED_DETECTORS)
    * len(EXPECTED_EYES)
    * len(EXPECTED_DISTANCES)
    * len(EXPECTED_AOIS)
)
ROWS_PER_PARTICIPANT = (
    len(EXPECTED_TASKS)
    * len(EXPECTED_DETECTORS)
    * len(EXPECTED_EYES)
    * len(EXPECTED_DISTANCES)
    * len(EXPECTED_AOIS)
)
ROWS_PER_PARTICIPANT_TASK = (
    len(EXPECTED_DETECTORS)
    * len(EXPECTED_EYES)
    * len(EXPECTED_DISTANCES)
    * len(EXPECTED_AOIS)
)
ROWS_PER_MEASUREMENT_BRANCH = EXPECTED_PARTICIPANTS * len(EXPECTED_TASKS)

KEY = [
    "participant_id",
    "stimulus_id",
    "detector_id",
    "eye",
    "viewing_distance_cm",
    "aoi_convention",
]
REQUIRED = set(KEY) | {
    "transition_count",
    "status",
    "experiment_condition",
    "metadata_tracking_ratio_percent",
    "metadata_stimulus_time_seconds",
}
FORBIDDEN_MODEL_FIELDS = {
    "prompt_indicator",
    "estimate",
    "SE",
    "CI_lower",
    "CI_upper",
    "rate_ratio",
    "p_value",
    "p_value_reference_only",
    "z_value",
}


def _require(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}.")


def _exact_levels(
    observed: set[str],
    expected: tuple[str, ...],
    label: str,
) -> None:
    expected_set = set(expected)
    if observed != expected_set:
        raise ValueError(
            f"{label} levels differ from frozen plan; "
            f"expected={sorted(expected_set)}, observed={sorted(observed)}."
        )


def validate_measurement_artifact(
    measurement: pd.DataFrame,
    *,
    identity_presence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate exact measurement structure and return a compact audit table."""
    _require(measurement, REQUIRED, "measurement")

    if len(measurement) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} measurement rows; found {len(measurement)}."
        )
    if measurement.duplicated(KEY).any():
        dup = measurement.loc[measurement.duplicated(KEY, keep=False), KEY]
        raise ValueError(
            "Duplicate measurement branch keys detected: "
            + dup.head(10).astype(str).agg("|".join, axis=1).str.cat(sep=", ")
        )

    forbidden = sorted(FORBIDDEN_MODEL_FIELDS & set(measurement.columns))
    if forbidden:
        raise ValueError(
            "Measurement artifact contains forbidden focal-model fields: "
            + ", ".join(forbidden)
        )

    participant_ids = set(measurement["participant_id"].astype(str))
    if len(participant_ids) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} participants; "
            f"found {len(participant_ids)}."
        )

    if identity_presence is not None:
        required_identity = {
            "participant_id",
            "exact_cross_source_raw_eligible",
        }
        _require(identity_presence, required_identity, "identity presence")
        exact_ids = set(
            identity_presence.loc[
                identity_presence["exact_cross_source_raw_eligible"].astype(bool),
                "participant_id",
            ].astype(str)
        )
        if len(exact_ids) != EXPECTED_PARTICIPANTS:
            raise ValueError(
                f"Identity evidence expected {EXPECTED_PARTICIPANTS} exact IDs; "
                f"found {len(exact_ids)}."
            )
        if participant_ids != exact_ids:
            raise ValueError(
                "Measurement participants differ from the frozen exact raw cohort; "
                f"missing={sorted(exact_ids - participant_ids)}, "
                f"unexpected={sorted(participant_ids - exact_ids)}."
            )

    _exact_levels(
        set(measurement["stimulus_id"].astype(str)),
        EXPECTED_TASKS,
        "stimulus_id",
    )
    _exact_levels(
        set(measurement["detector_id"].astype(str)),
        EXPECTED_DETECTORS,
        "detector_id",
    )
    _exact_levels(
        set(measurement["eye"].astype(str)),
        EXPECTED_EYES,
        "eye",
    )
    _exact_levels(
        set(measurement["aoi_convention"].astype(str)),
        EXPECTED_AOIS,
        "aoi_convention",
    )

    distances = pd.to_numeric(
        measurement["viewing_distance_cm"],
        errors="coerce",
    )
    if not np.isfinite(distances).all():
        raise ValueError("viewing_distance_cm contains non-finite values.")
    observed_distances = sorted(set(distances.astype(float)))
    if observed_distances != list(EXPECTED_DISTANCES):
        raise ValueError(
            "viewing_distance_cm levels differ from frozen plan; "
            f"expected={list(EXPECTED_DISTANCES)}, observed={observed_distances}."
        )

    participant_counts = measurement.groupby(
        "participant_id",
        dropna=False,
    ).size()
    if not participant_counts.eq(ROWS_PER_PARTICIPANT).all():
        bad = participant_counts.loc[
            ~participant_counts.eq(ROWS_PER_PARTICIPANT)
        ].to_dict()
        raise ValueError(
            f"Each participant must have {ROWS_PER_PARTICIPANT} rows; bad={bad}."
        )

    participant_task_counts = measurement.groupby(
        ["participant_id", "stimulus_id"],
        dropna=False,
    ).size()
    if not participant_task_counts.eq(ROWS_PER_PARTICIPANT_TASK).all():
        bad = participant_task_counts.loc[
            ~participant_task_counts.eq(ROWS_PER_PARTICIPANT_TASK)
        ].head(20)
        raise ValueError(
            f"Each participant x Task must have {ROWS_PER_PARTICIPANT_TASK} rows; "
            f"bad={bad.to_dict()}."
        )

    branch_counts = measurement.groupby(
        [
            "detector_id",
            "eye",
            "viewing_distance_cm",
            "aoi_convention",
        ],
        dropna=False,
    ).size()
    if len(branch_counts) != (
        len(EXPECTED_DETECTORS)
        * len(EXPECTED_EYES)
        * len(EXPECTED_DISTANCES)
        * len(EXPECTED_AOIS)
    ):
        raise ValueError(
            f"Expected 36 measurement branches; found {len(branch_counts)}."
        )
    if not branch_counts.eq(ROWS_PER_MEASUREMENT_BRANCH).all():
        bad = branch_counts.loc[
            ~branch_counts.eq(ROWS_PER_MEASUREMENT_BRANCH)
        ].to_dict()
        raise ValueError(
            f"Each measurement branch must have {ROWS_PER_MEASUREMENT_BRANCH} rows; "
            f"bad={bad}."
        )

    condition_by_participant = (
        measurement[["participant_id", "experiment_condition"]]
        .drop_duplicates()
        .groupby("participant_id", dropna=False)
        .size()
    )
    if not condition_by_participant.eq(1).all():
        raise ValueError(
            "At least one participant maps to multiple experiment conditions."
        )
    conditions = set(measurement["experiment_condition"].astype(str))
    if conditions != {"Prompt", "Non-prompt"}:
        raise ValueError(
            "experiment_condition must contain exactly Prompt and Non-prompt; "
            f"found={sorted(conditions)}."
        )

    status_counts = (
        measurement["status"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="rows")
    )
    status_counts.insert(0, "audit_type", "measurement_status")
    return status_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_csv", type=Path)
    parser.add_argument("--identity-presence-csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("srl_measurement_structure_audit.csv"),
    )
    args = parser.parse_args()

    audit = validate_measurement_artifact(
        pd.read_csv(args.measurement_csv),
        identity_presence=(
            pd.read_csv(args.identity_presence_csv)
            if args.identity_presence_csv is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    print(
        f"Measurement artifact structurally valid: {EXPECTED_ROWS} rows; "
        f"{EXPECTED_PARTICIPANTS} participants; 36 measurement branches."
    )
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
