#!/usr/bin/env python3
"""Audit SRL raw sampling within Task_1..Task_8 segments.

REMoDNaV requires dense regular sampling. Whole-recording timestamp gaps are
not the relevant unit because the empirical pipeline analyzes learning slides.
This audit therefore evaluates timestamp regularity within each released Task
segment and distinguishes native 250 Hz, lower-rate, and irregular segments.

No gaze outcome or experimental-group effect is calculated.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = {f"Task_{i}" for i in range(1, 9)}
REQUIRED = {"RecordingTime [ms]", "Trial", "Stimulus", "Participant"}


def _delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(8192)
    return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter


def _read(path: Path) -> pd.DataFrame:
    delimiter = _delimiter(path)
    frame = pd.read_csv(
        path,
        sep=delimiter,
        usecols=lambda col: col in REQUIRED,
        low_memory=False,
    )
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}.")
    return frame


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _task(value: object) -> str:
    if pd.isna(value):
        return ""
    return Path(str(value).strip()).stem


def audit_task_timebase(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("*.txt")):
        frame = _read(path)
        frame["participant_id"] = frame["Participant"].map(_clean_id)
        frame["task"] = frame["Stimulus"].map(_task)
        task_frame = frame.loc[frame["task"].isin(TASKS)].copy()

        for (participant_id, task), z in task_frame.groupby(
            ["participant_id", "task"],
            sort=True,
            observed=True,
        ):
            time = pd.to_numeric(z["RecordingTime [ms]"], errors="coerce")
            finite = time[np.isfinite(time)].to_numpy(dtype=float)
            diffs = np.diff(finite)
            finite_diffs = diffs[np.isfinite(diffs)]
            positive = finite_diffs[finite_diffs > 0]

            median = float(np.median(positive)) if positive.size else np.nan
            is_250 = bool(np.isfinite(median) and abs(median - 4.0) <= 0.5)
            is_60 = bool(np.isfinite(median) and abs(median - (1000 / 60)) <= 1.0)
            sampling_class = (
                "nominal_250hz"
                if is_250
                else "nominal_60hz"
                if is_60
                else "other_or_unresolved"
            )

            close_4 = (
                np.isclose(finite_diffs, 4.0, atol=0.25)
                if finite_diffs.size
                else np.asarray([], dtype=bool)
            )
            multiple_4 = (
                np.isclose(
                    finite_diffs / 4.0,
                    np.round(finite_diffs / 4.0),
                    atol=0.125,
                )
                if finite_diffs.size
                else np.asarray([], dtype=bool)
            )
            positive_only = bool(
                finite_diffs.size and bool((finite_diffs > 0).all())
            )
            dense_250 = bool(
                is_250
                and positive_only
                and close_4.size
                and bool(close_4.all())
            )
            lossless_grid_250 = bool(
                is_250
                and positive_only
                and multiple_4.size
                and bool(multiple_4.all())
            )

            rows.append(
                {
                    "file": path.name,
                    "participant_id": participant_id,
                    "task": task,
                    "rows": len(z),
                    "finite_timestamps": len(finite),
                    "median_positive_step_ms": median,
                    "minimum_step_ms": (
                        float(np.min(finite_diffs))
                        if finite_diffs.size
                        else np.nan
                    ),
                    "maximum_step_ms": (
                        float(np.max(finite_diffs))
                        if finite_diffs.size
                        else np.nan
                    ),
                    "nonpositive_steps": (
                        int((finite_diffs <= 0).sum())
                        if finite_diffs.size
                        else 0
                    ),
                    "fraction_exact_4ms": (
                        float(close_4.mean()) if close_4.size else np.nan
                    ),
                    "sampling_class": sampling_class,
                    "dense_regular_250hz": dense_250,
                    "lossless_nan_grid_250hz": lossless_grid_250,
                    "n_missing_4ms_slots_if_regularized": (
                        int(
                            np.round(finite_diffs[finite_diffs > 0] / 4.0).sum()
                            - len(finite_diffs[finite_diffs > 0])
                        )
                        if lossless_grid_250
                        else pd.NA
                    ),
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No Task_1..Task_8 raw segments were found.")

    participant = (
        detail.groupby("participant_id", sort=True)
        .agg(
            n_tasks=("task", "nunique"),
            nominal_250hz_tasks=(
                "sampling_class",
                lambda s: int((s == "nominal_250hz").sum()),
            ),
            nominal_60hz_tasks=(
                "sampling_class",
                lambda s: int((s == "nominal_60hz").sum()),
            ),
            dense_regular_250hz_tasks=(
                "dense_regular_250hz",
                "sum",
            ),
            lossless_nan_grid_250hz_tasks=(
                "lossless_nan_grid_250hz",
                "sum",
            ),
            minimum_fraction_exact_4ms=("fraction_exact_4ms", "min"),
            maximum_within_task_gap_ms=("maximum_step_ms", "max"),
            total_nonpositive_steps=("nonpositive_steps", "sum"),
        )
        .reset_index()
    )
    participant["all_8_tasks_nominal_250hz"] = (
        participant["n_tasks"].eq(8)
        & participant["nominal_250hz_tasks"].eq(8)
    )
    participant["all_8_tasks_dense_regular_250hz"] = (
        participant["n_tasks"].eq(8)
        & participant["dense_regular_250hz_tasks"].eq(8)
    )
    participant["all_8_tasks_lossless_nan_grid_250hz"] = (
        participant["n_tasks"].eq(8)
        & participant["lossless_nan_grid_250hz_tasks"].eq(8)
    )
    return detail, participant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_task_timebase_audit"),
    )
    args = parser.parse_args()

    detail, participants = audit_task_timebase(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "task_timebase.csv", index=False)
    participants.to_csv(
        args.output_dir / "participant_timebase_summary.csv",
        index=False,
    )

    print(
        detail.groupby("sampling_class").size().rename("task_segments").to_string()
    )
    print()
    print(
        "Participants with 8 nominal 250 Hz tasks:",
        int(participants["all_8_tasks_nominal_250hz"].sum()),
        "/",
        len(participants),
    )
    print(
        "Participants with 8 strictly dense 250 Hz tasks:",
        int(participants["all_8_tasks_dense_regular_250hz"].sum()),
        "/",
        len(participants),
    )
    print(
        "Participants losslessly regularizable to 250 Hz with NaN gaps:",
        int(participants["all_8_tasks_lossless_nan_grid_250hz"].sum()),
        "/",
        len(participants),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
