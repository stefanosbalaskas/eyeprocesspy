#!/usr/bin/env python3
"""Execute the frozen MCFW-Gaze detector-agreement validation.

This is an independent measurement-generalization analysis. It does not create
an HCI treatment contrast and it does not select a preferred detector.

The source archive is mapped without interpolation or binocular fusion. Event
detection is run on maximal contiguous runs of source-usable samples so
invalid/unavailable observations cannot bridge events. The complete source
sample timeline is retained for quality accounting and for mapping detector
events back to fixation/non-fixation states.
"""

from __future__ import annotations

import argparse
import itertools
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import eyeprocesspy as ep
import mcfw_adapter
import mcfw_coordinates

EYES = ("left", "right")
DETECTOR_IDS = ("ivt_30_100_simple", "ivt_40_50_simple", "idt_1_100")
DETECTOR_PAIRS = tuple(itertools.combinations(DETECTOR_IDS, 2))
TIMESTAMP_SCALE_SECONDS = 1e-6
NOMINAL_SAMPLING_RATE = 120.0
SHARED_RUN_GAP_MS = 75.0

EVENT_COLUMNS = [
    "participant",
    "trial",
    "trial_family",
    "relative_path",
    "eye",
    "detector_id",
    "detector_spec_hash",
    "status",
    "n_samples_total",
    "n_samples_usable",
    "analysis_usable_fraction",
    "left_usable_fraction",
    "right_usable_fraction",
    "either_eye_usable_fraction",
    "both_eyes_usable_fraction",
    "fixation_count",
    "fixation_rate_per_minute",
    "median_fixation_duration_ms",
    "total_fixation_time_proportion",
]
PAIR_COLUMNS = [
    "participant",
    "trial",
    "trial_family",
    "relative_path",
    "eye",
    "detector_a",
    "detector_b",
    "status",
    "n_samples_total",
    "n_samples_usable",
    "analysis_usable_fraction",
    "left_usable_fraction",
    "right_usable_fraction",
    "either_eye_usable_fraction",
    "both_eyes_usable_fraction",
    "agreement_proportion",
    "fixation_jaccard",
    "cohen_kappa",
    "fixation_count_absolute_difference",
    "fixation_count_symmetric_relative_difference",
    "fixation_rate_absolute_difference",
    "fixation_rate_symmetric_relative_difference",
    "median_duration_absolute_difference_ms",
    "median_duration_symmetric_relative_difference",
    "fixation_time_proportion_absolute_difference",
    "fixation_time_proportion_symmetric_relative_difference",
]
FAILURE_COLUMNS = [
    "participant",
    "trial",
    "trial_family",
    "relative_path",
    "eye",
    "stage",
    "detector_id",
    "error_type",
    "error",
]
WARNING_COLUMNS = [
    "participant",
    "trial",
    "trial_family",
    "relative_path",
    "eye",
    "detector_id",
    "warning",
]


def frozen_detector_specs() -> tuple[Any, ...]:
    """Return the three detector specifications fixed before MCFW results."""
    return (
        ep.define_event_detector_spec(
            "ivt_30_100_simple",
            "ivt",
            velocity_threshold=30.0,
            minimum_duration_ms=100.0,
            maximum_gap_ms=75.0,
            sampling_rate=NOMINAL_SAMPLING_RATE,
            coordinate_unit="degrees",
            merge_rule="none",
            implementation="eyeprocesspy",
            parameters={"include_saccades": False},
        ),
        ep.define_event_detector_spec(
            "ivt_40_50_simple",
            "ivt",
            velocity_threshold=40.0,
            minimum_duration_ms=50.0,
            maximum_gap_ms=75.0,
            sampling_rate=NOMINAL_SAMPLING_RATE,
            coordinate_unit="degrees",
            merge_rule="none",
            implementation="eyeprocesspy",
            parameters={"include_saccades": False},
        ),
        ep.define_event_detector_spec(
            "idt_1_100",
            "idt",
            dispersion_threshold=1.0,
            minimum_duration_ms=100.0,
            sampling_rate=NOMINAL_SAMPLING_RATE,
            coordinate_unit="degrees",
            merge_rule="none",
            implementation="eyeprocesspy",
        ),
    )


def _trial_family(name: str) -> str:
    if name.startswith("image_"):
        return "natural_image"
    if name.startswith("gaze_pattern_auth_trial"):
        return "gaze_pattern_auth"
    if name.startswith("password_experiment"):
        return "password"
    if name in {"shopping", "news", "video"}:
        return f"web_{name}"
    return "other"


def _quality_fields(manifest_row: dict[str, Any]) -> dict[str, float]:
    fields = {}
    for name in (
        "left_usable_fraction",
        "right_usable_fraction",
        "either_eye_usable_fraction",
        "both_eyes_usable_fraction",
    ):
        try:
            fields[name] = float(manifest_row.get(name, np.nan))
        except (TypeError, ValueError):
            fields[name] = np.nan
    return fields


def prepare_detector_input(dataset: ep.EyeDataset) -> ep.EyeDataset:
    """Keep only contiguous source-usable runs for detector execution.

    No samples are interpolated or invented. Invalid/unavailable observations
    remain in the source dataset used for quality accounting and agreement
    mapping, but cannot bridge a fixation event.
    """
    if not ep.is_eye_dataset(dataset):
        raise TypeError("dataset must be an EyeDataset.")
    samples = dataset["gaze_samples"].copy()
    if samples.empty:
        raise ValueError("dataset contains no gaze samples.")

    samples = samples.sort_values("timestamp_seconds", kind="stable").reset_index(drop=True)
    time = pd.to_numeric(samples["timestamp_seconds"], errors="coerce").to_numpy(float)
    x = pd.to_numeric(samples["gaze_x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(samples["gaze_y"], errors="coerce").to_numpy(float)
    valid = samples["valid"].astype("boolean").fillna(False).to_numpy(bool)
    usable = valid & np.isfinite(time) & np.isfinite(x) & np.isfinite(y)
    if not usable.any():
        out = dataset.copy()
        out["gaze_samples"] = samples.iloc[0:0].copy()
        return out

    gap_s = SHARED_RUN_GAP_MS / 1000.0
    run_id = np.full(len(samples), -1, dtype=int)
    current = -1
    previous_usable = False
    previous_time = np.nan
    for i in range(len(samples)):
        if not usable[i]:
            previous_usable = False
            previous_time = np.nan
            continue
        start_run = (
            not previous_usable
            or not np.isfinite(previous_time)
            or not np.isfinite(time[i] - previous_time)
            or time[i] - previous_time > gap_s
        )
        if start_run:
            current += 1
        run_id[i] = current
        previous_usable = True
        previous_time = time[i]

    kept = samples.loc[usable].copy()
    kept_run = run_id[usable]
    source_trial = str(samples["trial_id"].dropna().iloc[0])
    kept["source_trial_id"] = kept["trial_id"]
    kept["detector_input_run"] = kept_run
    kept["trial_id"] = [
        f"{source_trial}__usable_run_{value:05d}" for value in kept_run
    ]

    out = dataset.copy()
    out["gaze_samples"] = kept.reset_index(drop=True)
    out.vendor_metadata = dict(out.vendor_metadata)
    out.vendor_metadata["detector_input_policy"] = {
        "source_usable_only": True,
        "split_at_unusable_samples": True,
        "split_at_gap_ms_above": SHARED_RUN_GAP_MS,
        "interpolation": "none",
        "sample_creation": "none",
    }
    return ep.add_provenance(
        out,
        "prepare_mcfw_detector_input",
        "gaze_samples",
        (
            "source_usable_only=true;split_at_unusable_samples=true;"
            f"split_at_gap_ms_above={SHARED_RUN_GAP_MS:g};"
            "interpolation=none;sample_creation=none"
        ),
        software="eyeprocesspy MCFW research runner",
        reversible=True,
    )


def _fixation_episodes(branch: ep.EyeDataset) -> pd.DataFrame:
    episodes = branch["episodes"]
    if episodes.empty:
        return episodes.copy()
    return episodes.loc[episodes["episode_type"].eq("fixation")].copy()


def fixation_state_on_source_timeline(
    source_samples: pd.DataFrame,
    fixation_episodes: pd.DataFrame,
) -> np.ndarray:
    """Map fixation intervals onto the existing source sample timestamps."""
    timestamps = pd.to_numeric(
        source_samples["timestamp_seconds"], errors="coerce"
    ).to_numpy(float)
    state = np.zeros(len(timestamps), dtype=bool)
    if len(timestamps) == 0 or fixation_episodes.empty:
        return state
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Source timestamps must be finite and strictly increasing.")

    delta = np.zeros(len(timestamps) + 1, dtype=int)
    starts = pd.to_numeric(
        fixation_episodes["start_time"], errors="coerce"
    ).to_numpy(float)
    ends = pd.to_numeric(
        fixation_episodes["end_time"], errors="coerce"
    ).to_numpy(float)
    for start, end in zip(starts, ends, strict=True):
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            continue
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="right"))
        if left >= len(timestamps) or right <= 0 or left >= right:
            continue
        left = max(left, 0)
        right = min(right, len(timestamps))
        delta[left] += 1
        delta[right] -= 1
    return np.cumsum(delta[:-1]) > 0


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return np.nan
    observed = float(np.mean(a == b))
    pa = float(np.mean(a))
    pb = float(np.mean(b))
    expected = pa * pb + (1.0 - pa) * (1.0 - pb)
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return np.nan
    return (observed - expected) / denominator


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return np.nan
    return float(np.logical_and(a, b).sum() / union)


def _symmetric_relative_difference(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    denominator = abs(a) + abs(b)
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return np.nan
    return float(2.0 * abs(a - b) / denominator)


def _event_summary(
    branch: ep.EyeDataset,
    *,
    source_samples: pd.DataFrame,
) -> dict[str, float]:
    fixations = _fixation_episodes(branch)
    durations = pd.to_numeric(
        fixations.get("duration_ms", pd.Series(dtype=float)),
        errors="coerce",
    ).to_numpy(float)
    durations = durations[np.isfinite(durations) & (durations >= 0)]

    timestamps = pd.to_numeric(
        source_samples["timestamp_seconds"], errors="coerce"
    ).to_numpy(float)
    finite_time = timestamps[np.isfinite(timestamps)]
    trial_duration_s = (
        float(finite_time.max() - finite_time.min())
        if len(finite_time) >= 2
        else np.nan
    )
    fixation_count = float(len(fixations))
    fixation_rate = (
        fixation_count / (trial_duration_s / 60.0)
        if np.isfinite(trial_duration_s) and trial_duration_s > 0
        else np.nan
    )
    total_fixation_s = float(durations.sum() / 1000.0) if len(durations) else 0.0
    fixation_prop = (
        total_fixation_s / trial_duration_s
        if np.isfinite(trial_duration_s) and trial_duration_s > 0
        else np.nan
    )
    return {
        "fixation_count": fixation_count,
        "fixation_rate_per_minute": fixation_rate,
        "median_fixation_duration_ms": (
            float(np.median(durations)) if len(durations) else np.nan
        ),
        "total_fixation_time_proportion": fixation_prop,
    }


def _placeholder_event(
    common: dict[str, Any],
    detector_id: str,
    detector_hash: str,
    status: str,
) -> dict[str, Any]:
    return {
        **common,
        "detector_id": detector_id,
        "detector_spec_hash": detector_hash,
        "status": status,
        "fixation_count": np.nan,
        "fixation_rate_per_minute": np.nan,
        "median_fixation_duration_ms": np.nan,
        "total_fixation_time_proportion": np.nan,
    }


def _placeholder_pair(
    common: dict[str, Any],
    detector_a: str,
    detector_b: str,
    status: str,
) -> dict[str, Any]:
    return {
        **common,
        "detector_a": detector_a,
        "detector_b": detector_b,
        "status": status,
        "agreement_proportion": np.nan,
        "fixation_jaccard": np.nan,
        "cohen_kappa": np.nan,
        "fixation_count_absolute_difference": np.nan,
        "fixation_count_symmetric_relative_difference": np.nan,
        "fixation_rate_absolute_difference": np.nan,
        "fixation_rate_symmetric_relative_difference": np.nan,
        "median_duration_absolute_difference_ms": np.nan,
        "median_duration_symmetric_relative_difference": np.nan,
        "fixation_time_proportion_absolute_difference": np.nan,
        "fixation_time_proportion_symmetric_relative_difference": np.nan,
    }


def process_file(
    path: Path,
    *,
    dataset_root: Path,
    manifest_row: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process both eyes for one source trial file."""
    participant = path.parent.name
    trial = path.stem
    family = _trial_family(trial)
    relative_path = path.relative_to(dataset_root).as_posix()
    quality = _quality_fields(manifest_row)
    specs = frozen_detector_specs()
    spec_by_id = {spec.detector_id: spec for spec in specs}

    event_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []

    for eye in EYES:
        try:
            source = mcfw_adapter.load_mcfw_trial(
                path,
                eye=eye,
                timestamp_scale_seconds=TIMESTAMP_SCALE_SECONDS,
                nominal_sampling_rate=NOMINAL_SAMPLING_RATE,
            )
            angular = mcfw_coordinates.convert_mcfw_normalized_to_degrees(source)
        except Exception as exc:
            common = {
                "participant": participant,
                "trial": trial,
                "trial_family": family,
                "relative_path": relative_path,
                "eye": eye,
                "n_samples_total": int(manifest_row.get("n_samples", 0) or 0),
                "n_samples_usable": np.nan,
                "analysis_usable_fraction": np.nan,
                **quality,
            }
            status = f"adapter_or_geometry_failed:{type(exc).__name__}"
            for detector_id in DETECTOR_IDS:
                event_rows.append(
                    _placeholder_event(
                        common,
                        detector_id,
                        spec_by_id[detector_id].fingerprint,
                        status,
                    )
                )
            for detector_a, detector_b in DETECTOR_PAIRS:
                pair_rows.append(
                    _placeholder_pair(common, detector_a, detector_b, status)
                )
            failure_rows.append(
                {
                    "participant": participant,
                    "trial": trial,
                    "trial_family": family,
                    "relative_path": relative_path,
                    "eye": eye,
                    "stage": "adapter_or_geometry",
                    "detector_id": pd.NA,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        source_samples = angular["gaze_samples"].copy()
        usable = (
            source_samples["valid"]
            .astype("boolean")
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        common = {
            "participant": participant,
            "trial": trial,
            "trial_family": family,
            "relative_path": relative_path,
            "eye": eye,
            "n_samples_total": int(len(source_samples)),
            "n_samples_usable": int(usable.sum()),
            "analysis_usable_fraction": (
                float(usable.mean()) if len(usable) else np.nan
            ),
            **quality,
        }

        try:
            detector_input = prepare_detector_input(angular)
            detected = ep.run_detector_multiverse(
                detector_input,
                specs,
                continue_on_error=True,
            )
        except Exception as exc:
            status = f"multiverse_failed:{type(exc).__name__}"
            for detector_id in DETECTOR_IDS:
                event_rows.append(
                    _placeholder_event(
                        common,
                        detector_id,
                        spec_by_id[detector_id].fingerprint,
                        status,
                    )
                )
            for detector_a, detector_b in DETECTOR_PAIRS:
                pair_rows.append(
                    _placeholder_pair(common, detector_a, detector_b, status)
                )
            failure_rows.append(
                {
                    "participant": participant,
                    "trial": trial,
                    "trial_family": family,
                    "relative_path": relative_path,
                    "eye": eye,
                    "stage": "multiverse",
                    "detector_id": pd.NA,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        if not detected.failures.empty:
            for row in detected.failures.to_dict(orient="records"):
                failure_rows.append(
                    {
                        "participant": participant,
                        "trial": trial,
                        "trial_family": family,
                        "relative_path": relative_path,
                        "eye": eye,
                        "stage": "detector",
                        "detector_id": row.get("detector_id", pd.NA),
                        "error_type": row.get("error_type", pd.NA),
                        "error": row.get("error", pd.NA),
                    }
                )
        if not detected.warnings.empty:
            for row in detected.warnings.to_dict(orient="records"):
                warning_rows.append(
                    {
                        "participant": participant,
                        "trial": trial,
                        "trial_family": family,
                        "relative_path": relative_path,
                        "eye": eye,
                        "detector_id": row.get("detector_id", pd.NA),
                        "warning": row.get("warning", row.get("message", pd.NA)),
                    }
                )

        branch_summary: dict[str, dict[str, float]] = {}
        branch_state: dict[str, np.ndarray] = {}
        for detector_id in DETECTOR_IDS:
            spec = spec_by_id[detector_id]
            branch = detected.branches.get(detector_id)
            if branch is None:
                event_rows.append(
                    _placeholder_event(
                        common,
                        detector_id,
                        spec.fingerprint,
                        "detector_failed",
                    )
                )
                continue

            summary = _event_summary(branch, source_samples=source_samples)
            branch_summary[detector_id] = summary
            branch_state[detector_id] = fixation_state_on_source_timeline(
                source_samples,
                _fixation_episodes(branch),
            )
            event_rows.append(
                {
                    **common,
                    "detector_id": detector_id,
                    "detector_spec_hash": spec.fingerprint,
                    "status": "ok",
                    **summary,
                }
            )

        for detector_a, detector_b in DETECTOR_PAIRS:
            if detector_a not in branch_state or detector_b not in branch_state:
                pair_rows.append(
                    _placeholder_pair(
                        common,
                        detector_a,
                        detector_b,
                        "detector_pair_incomplete",
                    )
                )
                continue
            if not usable.any():
                pair_rows.append(
                    _placeholder_pair(
                        common,
                        detector_a,
                        detector_b,
                        "no_source_usable_samples",
                    )
                )
                continue

            a = branch_state[detector_a][usable]
            b = branch_state[detector_b][usable]
            sa = branch_summary[detector_a]
            sb = branch_summary[detector_b]

            pair_rows.append(
                {
                    **common,
                    "detector_a": detector_a,
                    "detector_b": detector_b,
                    "status": "ok",
                    "agreement_proportion": float(np.mean(a == b)),
                    "fixation_jaccard": _jaccard(a, b),
                    "cohen_kappa": _cohen_kappa(a, b),
                    "fixation_count_absolute_difference": abs(
                        sa["fixation_count"] - sb["fixation_count"]
                    ),
                    "fixation_count_symmetric_relative_difference": (
                        _symmetric_relative_difference(
                            sa["fixation_count"],
                            sb["fixation_count"],
                        )
                    ),
                    "fixation_rate_absolute_difference": abs(
                        sa["fixation_rate_per_minute"]
                        - sb["fixation_rate_per_minute"]
                    ),
                    "fixation_rate_symmetric_relative_difference": (
                        _symmetric_relative_difference(
                            sa["fixation_rate_per_minute"],
                            sb["fixation_rate_per_minute"],
                        )
                    ),
                    "median_duration_absolute_difference_ms": abs(
                        sa["median_fixation_duration_ms"]
                        - sb["median_fixation_duration_ms"]
                    ),
                    "median_duration_symmetric_relative_difference": (
                        _symmetric_relative_difference(
                            sa["median_fixation_duration_ms"],
                            sb["median_fixation_duration_ms"],
                        )
                    ),
                    "fixation_time_proportion_absolute_difference": abs(
                        sa["total_fixation_time_proportion"]
                        - sb["total_fixation_time_proportion"]
                    ),
                    "fixation_time_proportion_symmetric_relative_difference": (
                        _symmetric_relative_difference(
                            sa["total_fixation_time_proportion"],
                            sb["total_fixation_time_proportion"],
                        )
                    ),
                }
            )

    return (
        pd.DataFrame(event_rows, columns=EVENT_COLUMNS),
        pd.DataFrame(pair_rows, columns=PAIR_COLUMNS),
        pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS),
        pd.DataFrame(warning_rows, columns=WARNING_COLUMNS),
    )


def _manifest_lookup(
    manifest: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    required = {
        "path",
        "n_samples",
        "left_usable_fraction",
        "right_usable_fraction",
        "either_eye_usable_fraction",
        "both_eyes_usable_fraction",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            "MCFW audit manifest is missing required columns: "
            + ", ".join(missing)
        )
    if manifest["path"].duplicated().any():
        raise ValueError("MCFW audit manifest contains duplicate paths.")
    return {
        str(row["path"]): row
        for row in manifest.to_dict(orient="records")
    }


def _concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=columns)
    return pd.concat(nonempty, ignore_index=True, sort=False).reindex(columns=columns)


def run_validation(
    dataset_root: Path,
    manifest_csv: Path,
    *,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the complete planned MCFW detector-validation universe."""
    dataset_root = Path(dataset_root)
    manifest = pd.read_csv(manifest_csv)
    lookup = _manifest_lookup(manifest)
    files = sorted((dataset_root / "data").glob("participant_*/*.tsv"))
    if not files:
        raise ValueError(f"No participant TSV files found under {dataset_root / 'data'}.")

    relative = [path.relative_to(dataset_root).as_posix() for path in files]
    missing_manifest = sorted(set(relative) - set(lookup))
    if missing_manifest:
        raise ValueError(
            "MCFW audit manifest is missing archive files: "
            + ", ".join(missing_manifest[:10])
        )

    event_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    warning_frames: list[pd.DataFrame] = []

    if workers <= 1:
        for index, path in enumerate(files, start=1):
            key = path.relative_to(dataset_root).as_posix()
            outputs = process_file(
                path,
                dataset_root=dataset_root,
                manifest_row=lookup[key],
            )
            event_frames.append(outputs[0])
            pair_frames.append(outputs[1])
            failure_frames.append(outputs[2])
            warning_frames.append(outputs[3])
            if index % 100 == 0 or index == len(files):
                print(f"processed {index}/{len(files)} files", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            future_to_path = {}
            for path in files:
                key = path.relative_to(dataset_root).as_posix()
                future = executor.submit(
                    process_file,
                    path,
                    dataset_root=dataset_root,
                    manifest_row=lookup[key],
                )
                future_to_path[future] = path

            completed = 0
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    outputs = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Uncaught MCFW worker failure for {path}: {exc}"
                    ) from exc
                event_frames.append(outputs[0])
                pair_frames.append(outputs[1])
                failure_frames.append(outputs[2])
                warning_frames.append(outputs[3])
                completed += 1
                if completed % 100 == 0 or completed == len(files):
                    print(
                        f"processed {completed}/{len(files)} files",
                        flush=True,
                    )

    events = _concat(event_frames, EVENT_COLUMNS)
    pairs = _concat(pair_frames, PAIR_COLUMNS)
    failures = _concat(failure_frames, FAILURE_COLUMNS)
    warnings = _concat(warning_frames, WARNING_COLUMNS)

    expected_event_rows = len(files) * len(EYES) * len(DETECTOR_IDS)
    expected_pair_rows = len(files) * len(EYES) * len(DETECTOR_PAIRS)
    if len(events) != expected_event_rows:
        raise RuntimeError(
            f"Expected {expected_event_rows} event-summary rows; found {len(events)}."
        )
    if len(pairs) != expected_pair_rows:
        raise RuntimeError(
            f"Expected {expected_pair_rows} pairwise rows; found {len(pairs)}."
        )

    event_key = ["relative_path", "eye", "detector_id"]
    pair_key = ["relative_path", "eye", "detector_a", "detector_b"]
    if events.duplicated(event_key).any():
        raise RuntimeError("MCFW event summary contains duplicate planned keys.")
    if pairs.duplicated(pair_key).any():
        raise RuntimeError("MCFW pairwise summary contains duplicate planned keys.")

    return events, pairs, failures, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("mcfw_validation"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    events, pairs, failures, warnings = run_validation(
        args.dataset_root,
        args.manifest_csv,
        workers=args.workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_dir / "mcfw_detector_event_summaries.csv", index=False)
    pairs.to_csv(args.output_dir / "mcfw_detector_pairwise_agreement.csv", index=False)
    failures.to_csv(args.output_dir / "mcfw_detector_failures.csv", index=False)
    warnings.to_csv(args.output_dir / "mcfw_detector_warnings.csv", index=False)

    print(
        f"event rows={len(events)}; pair rows={len(pairs)}; "
        f"failures={len(failures)}; warnings={len(warnings)}"
    )
    print("event status:")
    print(events["status"].value_counts(dropna=False).to_string())
    print("pair status:")
    print(pairs["status"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
