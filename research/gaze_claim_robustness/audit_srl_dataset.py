#!/usr/bin/env python3
"""Pre-analysis audit for the open SRL eye-tracking dataset.

The audit verifies structure and measurement availability before any focal
effect is estimated. It intentionally does not compute Prompt/Non-prompt or
Text/Multimedia gaze effects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PARTICIPANT_REQUIRED = {
    "part_ID",
    "experiment_condition",
}
STIMULI_REQUIRED = {
    "part_ID",
    "stimulus_name",
    "stimulus_type",
    "stimulus_time",
    "tracking_ratio",
}
RAW_REQUIRED = {
    "RecordingTime [ms]",
    "Trial",
    "Stimulus",
    "Participant",
    "Tracking Ratio [%]",
    "Point of Regard Right X [px]",
    "Point of Regard Right Y [px]",
    "Point of Regard Left X [px]",
    "Point of Regard Left Y [px]",
    "AOI Name Right",
    "AOI Name Left",
}


def _unique_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {name!r} under {root}; found {len(matches)}."
        )
    return matches[0]


def _raw_dir(root: Path) -> Path:
    matches = sorted(path for path in root.rglob("ET_data_raw") if path.is_dir())
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one ET_data_raw directory under {root}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _read_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    return pd.read_csv(path, sep=None, engine="python", nrows=nrows)


def _require(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _metadata_audit(
    participants: pd.DataFrame,
    stimuli: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require(participants, PARTICIPANT_REQUIRED, "participants.csv")
    _require(stimuli, STIMULI_REQUIRED, "stimuli.csv")

    condition_counts = (
        participants.groupby("experiment_condition", dropna=False)
        .agg(participants=("part_ID", "nunique"))
        .reset_index()
    )
    stimulus_cells = (
        stimuli.groupby(["part_ID", "stimulus_type"], dropna=False)
        .agg(
            n_rows=("stimulus_name", "size"),
            n_stimuli=("stimulus_name", "nunique"),
            median_tracking_ratio=("tracking_ratio", "median"),
        )
        .reset_index()
    )
    return condition_counts, stimulus_cells


def _raw_files(directory: Path) -> list[Path]:
    allowed = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )


def _raw_structure_audit(
    directory: Path,
    *,
    sample_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = _raw_files(directory)
    if not files:
        raise ValueError(f"No tabular raw files found under {directory}.")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for path in files:
        try:
            frame = _read_table(path, nrows=sample_rows)
            _require(frame, RAW_REQUIRED, str(path))
            time = pd.to_numeric(frame["RecordingTime [ms]"], errors="coerce")
            finite_time = time[np.isfinite(time)]
            diffs = finite_time.diff().dropna()

            right_x = pd.to_numeric(
                frame["Point of Regard Right X [px]"], errors="coerce"
            )
            right_y = pd.to_numeric(
                frame["Point of Regard Right Y [px]"], errors="coerce"
            )
            left_x = pd.to_numeric(
                frame["Point of Regard Left X [px]"], errors="coerce"
            )
            left_y = pd.to_numeric(
                frame["Point of Regard Left Y [px]"], errors="coerce"
            )
            right_finite = np.isfinite(right_x) & np.isfinite(right_y)
            left_finite = np.isfinite(left_x) & np.isfinite(left_y)

            rows.append(
                {
                    "file": path.name,
                    "sampled_rows": len(frame),
                    "participants_in_sample": frame["Participant"].nunique(
                        dropna=True
                    ),
                    "trials_in_sample": frame["Trial"].nunique(dropna=True),
                    "stimuli_in_sample": frame["Stimulus"].nunique(dropna=True),
                    "median_timestamp_step_ms": (
                        float(diffs.median()) if len(diffs) else np.nan
                    ),
                    "nonfinite_timestamps": int((~np.isfinite(time)).sum()),
                    "nonincreasing_timestamp_steps": int((diffs <= 0).sum()),
                    "right_finite_fraction": (
                        float(right_finite.mean()) if len(frame) else np.nan
                    ),
                    "left_finite_fraction": (
                        float(left_finite.mean()) if len(frame) else np.nan
                    ),
                    "right_aoi_levels": frame["AOI Name Right"].nunique(
                        dropna=True
                    ),
                    "left_aoi_levels": frame["AOI Name Left"].nunique(
                        dropna=True
                    ),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "file": path.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(
        failures, columns=["file", "error_type", "error"]
    )


def _stimulus_files(root: Path) -> pd.DataFrame:
    names = {
        path.name
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    expected = (
        {"Intro.jpg", "Outro.jpg"}
        | {f"Task_{i}.jpg" for i in range(1, 9)}
        | {f"Prompt_{i}.jpg" for i in range(1, 4)}
        | {f"Non-prompt_{i}.jpg" for i in range(1, 4)}
    )
    return pd.DataFrame(
        [
            {
                "expected_file": name,
                "present": name in names,
            }
            for name in sorted(expected)
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_dataset_audit"),
    )
    args = parser.parse_args()
    if args.sample_rows < 2:
        raise ValueError("sample_rows must be >= 2.")

    participants = _read_table(_unique_file(args.dataset_root, "participants.csv"))
    stimuli = _read_table(_unique_file(args.dataset_root, "stimuli.csv"))
    raw_directory = _raw_dir(args.dataset_root)

    condition_counts, stimulus_cells = _metadata_audit(participants, stimuli)
    raw_manifest, raw_failures = _raw_structure_audit(
        raw_directory,
        sample_rows=args.sample_rows,
    )
    stimulus_files = _stimulus_files(args.dataset_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition_counts.to_csv(
        args.output_dir / "condition_counts.csv",
        index=False,
    )
    stimulus_cells.to_csv(
        args.output_dir / "participant_stimulus_cells.csv",
        index=False,
    )
    raw_manifest.to_csv(
        args.output_dir / "raw_sample_manifest.csv",
        index=False,
    )
    raw_failures.to_csv(
        args.output_dir / "raw_sample_failures.csv",
        index=False,
    )
    stimulus_files.to_csv(
        args.output_dir / "stimulus_file_audit.csv",
        index=False,
    )

    print(condition_counts.to_string(index=False))
    print()
    print(
        "Raw participant files:",
        len(raw_manifest),
        "evaluable;",
        len(raw_failures),
        "failed.",
    )
    print(
        "Expected stimulus files present:",
        int(stimulus_files["present"].sum()),
        "/",
        len(stimulus_files),
    )

    if len(raw_failures) or not bool(stimulus_files["present"].all()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
