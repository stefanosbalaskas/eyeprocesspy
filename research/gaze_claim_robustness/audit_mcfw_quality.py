#!/usr/bin/env python3
"""Audit published MCFW-Gaze quality summaries before robustness analysis.

The script is intentionally descriptive. It does not choose exclusion
thresholds, clip unusual values, or reinterpret published columns whose
semantics need clarification from the source documentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRIAL_REQUIRED = {
    "pid",
    "trial",
    "eye",
    "n_trial_samples",
    "n_valid_trial_samples",
    "prop_invalid_samples",
    "prop_missing_data",
}

VALIDATION_REQUIRED = {
    "pid",
    "accuracy_left_eye (deg)",
    "accuracy_right_eye (deg)",
    "RMS_S2S_left_eye (deg)",
    "RMS_S2S_right_eye (deg)",
    "SD_left_eye (deg)",
    "SD_right_eye (deg)",
    "Prop_data_loss_left_eye",
    "Prop_data_loss_right_eye",
}


def _require(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _trial_family(value: object) -> str:
    trial = str(value)
    if trial.startswith("image_"):
        return "natural_image"
    if trial.startswith("gaze_pattern_auth_"):
        return "gaze_pattern_auth"
    if trial.startswith("password"):
        return "password"
    if trial in {"shopping", "news", "video"}:
        return f"web_{trial}"
    return "other"


def audit_trial_quality(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize sample validity without selecting a quality threshold."""
    _require(frame, TRIAL_REQUIRED, "trial-quality table")
    data = frame.copy()
    data["trial_family"] = data["trial"].map(_trial_family)

    for column in (
        "n_trial_samples",
        "n_valid_trial_samples",
        "prop_invalid_samples",
        "prop_missing_data",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    summary = (
        data.groupby(["trial_family", "eye"], dropna=False, sort=True)
        .agg(
            rows=("trial", "size"),
            participants=("pid", "nunique"),
            median_samples=("n_trial_samples", "median"),
            median_valid_samples=("n_valid_trial_samples", "median"),
            median_invalid_fraction=("prop_invalid_samples", "median"),
            max_invalid_fraction=("prop_invalid_samples", "max"),
            median_published_missing=("prop_missing_data", "median"),
            min_published_missing=("prop_missing_data", "min"),
            max_published_missing=("prop_missing_data", "max"),
        )
        .reset_index()
    )

    published_missing = data["prop_missing_data"]
    invalid_fraction = data["prop_invalid_samples"]
    audit = pd.DataFrame(
        [
            {
                "check": "prop_invalid_samples outside [0,1]",
                "n_rows": int(((invalid_fraction < 0) | (invalid_fraction > 1)).sum()),
                "interpretation": (
                    "Rows outside [0,1] require source clarification; values are not clipped."
                ),
            },
            {
                "check": "prop_missing_data outside [0,1]",
                "n_rows": int(((published_missing < 0) | (published_missing > 1)).sum()),
                "interpretation": (
                    "Published field contains values outside conventional proportion bounds; "
                    "do not reinterpret or clip without source justification."
                ),
            },
            {
                "check": "valid samples exceed total samples",
                "n_rows": int(
                    (data["n_valid_trial_samples"] > data["n_trial_samples"]).sum()
                ),
                "interpretation": (
                    "Any occurrence requires source clarification before quality filtering."
                ),
            },
        ]
    )
    return summary, audit


def audit_validation_quality(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize calibration/precision columns and flag nonstandard ranges."""
    _require(frame, VALIDATION_REQUIRED, "validation-quality table")
    data = frame.copy()
    numeric = sorted(VALIDATION_REQUIRED - {"pid"})
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    long_rows: list[dict[str, object]] = []
    for column in numeric:
        values = data[column]
        finite = values[np.isfinite(values)]
        long_rows.append(
            {
                "metric": column,
                "n_rows": len(values),
                "n_finite": int(np.isfinite(values).sum()),
                "median": float(finite.median()) if len(finite) else np.nan,
                "minimum": float(finite.min()) if len(finite) else np.nan,
                "maximum": float(finite.max()) if len(finite) else np.nan,
            }
        )
    summary = pd.DataFrame(long_rows)

    loss_columns = ["Prop_data_loss_left_eye", "Prop_data_loss_right_eye"]
    checks = []
    for column in loss_columns:
        values = data[column]
        checks.append(
            {
                "check": f"{column} outside [0,1]",
                "n_rows": int(((values < 0) | (values > 1)).sum()),
                "interpretation": (
                    "The published label suggests a proportion, but out-of-range values "
                    "must be clarified rather than clipped or silently treated as percentages."
                ),
            }
        )
    return summary, pd.DataFrame(checks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_quality_csv", type=Path)
    parser.add_argument("--validation-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("mcfw_quality_audit"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial = pd.read_csv(args.trial_quality_csv)
    trial_summary, trial_audit = audit_trial_quality(trial)
    trial_summary.to_csv(args.output_dir / "trial_quality_summary.csv", index=False)
    trial_audit.to_csv(args.output_dir / "trial_quality_semantic_audit.csv", index=False)

    print(trial_summary.to_string(index=False))
    print()
    print(trial_audit.to_string(index=False))

    warning_count = int(trial_audit["n_rows"].sum())
    if args.validation_csv is not None:
        validation = pd.read_csv(args.validation_csv)
        validation_summary, validation_audit = audit_validation_quality(validation)
        validation_summary.to_csv(
            args.output_dir / "validation_quality_summary.csv", index=False
        )
        validation_audit.to_csv(
            args.output_dir / "validation_quality_semantic_audit.csv", index=False
        )
        print()
        print(validation_summary.to_string(index=False))
        print()
        print(validation_audit.to_string(index=False))
        warning_count += int(validation_audit["n_rows"].sum())

    if warning_count:
        print(
            "\nPublished quality fields contain values requiring semantic clarification. "
            "No values were clipped and no quality threshold was selected."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
