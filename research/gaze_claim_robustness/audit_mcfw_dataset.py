#!/usr/bin/env python3
"""Inventory a local MCFW-Gaze archive without altering the gaze signal.

The audit preserves left/right eyes separately and does not interpolate,
average binocular coordinates, detect events, or exclude low-quality trials.
Those are later prespecified analytical decisions.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    "device_time_stamp",
    "system_time_stamp",
    "left_gaze_point_on_display_area_x",
    "left_gaze_point_on_display_area_y",
    "left_gaze_point_valid",
    "left_gaze_point_available",
    "right_gaze_point_on_display_area_x",
    "right_gaze_point_on_display_area_y",
    "right_gaze_point_valid",
    "right_gaze_point_available",
}

PARTICIPANT_RE = re.compile(r"participant_(\d+)$")


def _family(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("image_"):
        return "natural_image"
    if stem.startswith("gaze_pattern_auth_trial"):
        return "gaze_pattern_auth"
    if stem.startswith("password_experiment"):
        return "password"
    if stem in {"shopping", "news", "video"}:
        return f"web_{stem}"
    return "other"


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "1.0"})


def audit_file(path: Path, dataset_root: Path) -> dict[str, object]:
    frame = pd.read_csv(path, sep="\t")
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

    participant_dir = path.parent.name
    match = PARTICIPANT_RE.match(participant_dir)
    participant = participant_dir if match else path.parent.as_posix()

    left_valid = _bool_series(frame, "left_gaze_point_valid")
    left_available = _bool_series(frame, "left_gaze_point_available")
    right_valid = _bool_series(frame, "right_gaze_point_valid")
    right_available = _bool_series(frame, "right_gaze_point_available")

    left_usable = left_valid & left_available
    right_usable = right_valid & right_available
    either_usable = left_usable | right_usable
    both_usable = left_usable & right_usable

    device_time = pd.to_numeric(frame["device_time_stamp"], errors="coerce")
    finite_time = device_time[np.isfinite(device_time)]
    diffs = finite_time.diff().dropna()

    return {
        "participant": participant,
        "trial": path.stem,
        "trial_family": _family(path.name),
        "path": path.relative_to(dataset_root).as_posix(),
        "n_samples": len(frame),
        "left_usable_fraction": float(left_usable.mean()) if len(frame) else np.nan,
        "right_usable_fraction": float(right_usable.mean()) if len(frame) else np.nan,
        "either_eye_usable_fraction": float(either_usable.mean()) if len(frame) else np.nan,
        "both_eyes_usable_fraction": float(both_usable.mean()) if len(frame) else np.nan,
        "nonfinite_device_timestamps": int((~np.isfinite(device_time)).sum()),
        "nonincreasing_device_steps": int((diffs <= 0).sum()),
        "median_device_timestamp_step_raw_units": (
            float(diffs.median()) if len(diffs) else np.nan
        ),
    }


def build_manifest(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        raise ValueError(f"Expected MCFW data directory at {data_root}")

    files = sorted(data_root.glob("participant_*/*.tsv"))
    if not files:
        raise ValueError(f"No participant TSV files found under {data_root}")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for path in files:
        try:
            rows.append(audit_file(path, dataset_root))
        except Exception as exc:
            failures.append(
                {
                    "path": path.relative_to(dataset_root).as_posix(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(
        failures, columns=["path", "error_type", "error"]
    )


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(
            columns=[
                "trial_family",
                "files",
                "participants",
                "median_samples",
                "median_left_usable",
                "median_right_usable",
                "median_either_eye_usable",
                "median_both_eyes_usable",
                "files_with_nonincreasing_timestamps",
            ]
        )
    return (
        manifest.groupby("trial_family", sort=True)
        .agg(
            files=("trial", "size"),
            participants=("participant", "nunique"),
            median_samples=("n_samples", "median"),
            median_left_usable=("left_usable_fraction", "median"),
            median_right_usable=("right_usable_fraction", "median"),
            median_either_eye_usable=("either_eye_usable_fraction", "median"),
            median_both_eyes_usable=("both_eyes_usable_fraction", "median"),
            files_with_nonincreasing_timestamps=(
                "nonincreasing_device_steps",
                lambda x: int((x > 0).sum()),
            ),
        )
        .reset_index()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mcfw_dataset_audit"),
    )
    args = parser.parse_args()

    manifest, failures = build_manifest(args.dataset_root)
    summary = summarize_manifest(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "mcfw_file_manifest.csv", index=False)
    summary.to_csv(args.output_dir / "mcfw_family_summary.csv", index=False)
    failures.to_csv(args.output_dir / "mcfw_failures.csv", index=False)

    print(summary.to_string(index=False))
    if len(failures):
        print("\nNon-evaluable files:")
        print(failures.to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
