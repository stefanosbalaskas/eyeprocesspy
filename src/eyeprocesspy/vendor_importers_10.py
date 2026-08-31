"""Dedicated Tobii, Pupil Labs, EyeLink, and SMI importers.

This module source-ports the sixteen public contracts in frozen
``R/006-import-other-vendors.R`` from eyeprocess 0.11.1.

The importers reuse eyeprocesspy's canonical generic importer so the resulting
objects follow the same stable table/schema contract as other vendor routes.
SR Research EDF input retains the frozen external-tool boundary: conversion
requires the vendor-provided ``edf2asc`` executable and is never emulated by a
different parser.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .dataset import add_provenance, new_eye_dataset, validate_eye_dataset
from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .importers import (
    _first_existing,
    _read_delimited,
    _safe_numeric,
    infer_eye_mapping,
    read_eye_generic,
)
from .mapping import eye_mapping
from .schema import empty_eye_table, new_coordinate_space
from .timebase import estimate_sampling_rate

__all__ = [
    "is_eyelink_export",
    "is_pupil_labs_export",
    "is_smi_export",
    "is_tobii_export",
    "pupil_labs_format",
    "read_eyelink_asc",
    "read_eyelink_edf",
    "read_eyelink_report",
    "read_pupil_core",
    "read_pupil_neon",
    "read_pupillabs",
    "read_smi",
    "read_smi_aoi_export",
    "read_smi_event_export",
    "read_smi_raw_export",
    "read_tobii",
]


def _stop(message: str) -> None:
    raise EyeProcessValidationError(message)


def _pick(names: Iterable[str], *candidates: str) -> str | None:
    return _first_existing(list(map(str, names)), list(candidates))


def _safe_read_head(path: Path, rows: int) -> pd.DataFrame | None:
    try:
        return _read_delimited(path, nrows=max(1, int(rows)))
    except Exception:
        return None


def _row_mean(a: pd.Series, b: pd.Series) -> pd.Series:
    pair = pd.concat(
        [
            pd.to_numeric(a, errors="coerce"),
            pd.to_numeric(b, errors="coerce"),
        ],
        axis=1,
    )
    result = pair.mean(axis=1, skipna=True)
    result[~np.isfinite(result)] = np.nan
    return result


def _append_table(dataset, name: str, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    current = dataset.get(name)
    if isinstance(current, pd.DataFrame) and not current.empty:
        dataset[name] = pd.concat(
            [current, frame],
            ignore_index=True,
            sort=False,
        )
    else:
        dataset[name] = frame.reset_index(drop=True)


def _set_metadata(dataset, key: str, value: Mapping[str, Any]) -> None:
    metadata = getattr(dataset, "vendor_metadata", None)
    if not isinstance(metadata, dict):
        metadata = dataset.get("vendor_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata[key] = dict(value)
    try:
        dataset.vendor_metadata = metadata
    except Exception:
        dataset["vendor_metadata"] = metadata


def _set_raw(dataset, key: str, value: Any) -> None:
    raw = getattr(dataset, "raw", None)
    if not isinstance(raw, dict):
        raw = dataset.get("raw", {})
    if not isinstance(raw, dict):
        raw = {}
    raw[key] = value
    try:
        dataset.raw = raw
    except Exception:
        dataset["raw"] = raw


def _recording_id(dataset) -> str:
    recordings = dataset["recordings"]
    return str(recordings["recording_id"].iloc[0])


def _coordinate_id(dataset) -> Any:
    spaces = dataset["coordinate_spaces"]
    if isinstance(spaces, pd.DataFrame) and not spaces.empty:
        return spaces["coordinate_space_id"].iloc[0]
    return pd.NA


# Tobii -------------------------------------------------------------------


def is_tobii_export(path, inspect_rows=20):
    """Return a confidence score for Tobii Pro Lab-style delimited exports."""
    source = Path(path)
    if source.is_dir():
        return 0.0
    if source.suffix.lower().lstrip(".") not in {"tsv", "txt", "csv"}:
        return 0.0
    data = _safe_read_head(source, min(3, int(inspect_rows)))
    if data is None:
        return 0.0
    names = [str(name).lower() for name in data.columns]
    patterns = (
        "recording timestamp",
        "gaze point",
        "pupil diameter",
        "tobii",
        "eye movement type",
        "presented stimulus",
    )
    hits = sum(any(pattern in name for pattern in patterns) for name in names)
    signature = any("recording timestamp" in name for name in names) and any("gaze point" in name for name in names)
    if signature:
        return max(0.85, min(1.0, hits / 8.0))
    return min(0.7, hits / 8.0)


def _tobii_valid(values: Any) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    text = series.astype("string").str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[text.str.contains("valid", na=False) | text.eq("0")] = True
    out[text.str.contains("invalid", na=False) | text.isin(["1", "2", "3", "4"])] = False
    return out


def read_tobii(
    path,
    participant_id=None,
    recording_id=None,
    session_id="S001",
    time_unit="microseconds",
    coordinate_space="display_pixels_top_left",
    screen_width=np.nan,
    screen_height=np.nan,
    keep_raw=True,
    quiet=False,
    **kwargs,
):
    """Import a Tobii Pro Lab delimited export."""
    data = _read_delimited(path, **kwargs)
    names = list(map(str, data.columns))
    time_col = _pick(
        names,
        "Recording timestamp",
        "recording_timestamp",
        "system_time_stamp",
        "device_time_stamp",
        "timestamp",
    )
    if time_col is None:
        _stop("Cannot identify a Tobii timestamp column.")

    x_col = _pick(
        names,
        "Gaze point X",
        "Gaze point X (MCSnorm)",
        "gaze_point_x",
        "Gaze2d x",
    )
    y_col = _pick(
        names,
        "Gaze point Y",
        "Gaze point Y (MCSnorm)",
        "gaze_point_y",
        "Gaze2d y",
    )
    left_x = _pick(
        names,
        "Gaze point left X",
        "Gaze point left X (MCSnorm)",
        "left_gaze_point_on_display_area_x",
    )
    left_y = _pick(
        names,
        "Gaze point left Y",
        "Gaze point left Y (MCSnorm)",
        "left_gaze_point_on_display_area_y",
    )
    right_x = _pick(
        names,
        "Gaze point right X",
        "Gaze point right X (MCSnorm)",
        "right_gaze_point_on_display_area_x",
    )
    right_y = _pick(
        names,
        "Gaze point right Y",
        "Gaze point right Y (MCSnorm)",
        "right_gaze_point_on_display_area_y",
    )

    if x_col is None and left_x is not None and right_x is not None:
        data[".eye_tobii_x"] = _row_mean(data[left_x], data[right_x])
        if left_y is not None and right_y is not None:
            data[".eye_tobii_y"] = _row_mean(data[left_y], data[right_y])
        else:
            data[".eye_tobii_y"] = np.nan
        x_col = ".eye_tobii_x"
        y_col = ".eye_tobii_y"
    if x_col is None or y_col is None:
        data[".eye_tobii_x"] = np.nan
        data[".eye_tobii_y"] = np.nan
        x_col = ".eye_tobii_x"
        y_col = ".eye_tobii_y"

    mapping = eye_mapping(
        participant=_pick(
            names,
            "Participant name",
            "Participant",
            "participant_id",
            "subject",
        ),
        recording=_pick(
            names,
            "Recording name",
            "Recording",
            "recording_id",
        ),
        timestamp=time_col,
        x=x_col,
        y=y_col,
        left_x=left_x,
        left_y=left_y,
        right_x=right_x,
        right_y=right_y,
        pupil_left=_pick(
            names,
            "Pupil diameter left",
            "Pupil diameter left [mm]",
            "left_pupil_diameter",
        ),
        pupil_right=_pick(
            names,
            "Pupil diameter right",
            "Pupil diameter right [mm]",
            "right_pupil_diameter",
        ),
        pupil_left_valid=_pick(
            names,
            "Validity left",
            "left_pupil_validity",
        ),
        pupil_right_valid=_pick(
            names,
            "Validity right",
            "right_pupil_validity",
        ),
        fixation_id=_pick(
            names,
            "Fixation index",
            "Eye movement type index",
            "fixation_id",
        ),
        trial=_pick(names, "Trial", "Trial number", "trial_id"),
        stimulus=_pick(
            names,
            "Presented Stimulus name",
            "Presented Media name",
            "stimulus_id",
        ),
        event_name=_pick(names, "Event", "event_name"),
        event_value=_pick(names, "Event value", "event_value"),
    )
    mapping = {key: value for key, value in mapping.items() if value is not None}

    out = read_eye_generic(
        data,
        mapping=mapping,
        time_unit=time_unit,
        coordinate_space=coordinate_space,
        screen_width=screen_width,
        screen_height=screen_height,
        pupil_unit="millimetres",
        vendor="Tobii",
        participant_id=participant_id,
        recording_id=recording_id,
        session_id=session_id,
        keep_raw=keep_raw,
        quiet=True,
    )

    valid_left_col = _pick(
        names,
        "Validity left",
        "left_gaze_point_validity",
    )
    valid_right_col = _pick(
        names,
        "Validity right",
        "right_gaze_point_validity",
    )
    if valid_left_col is not None or valid_right_col is not None:
        valid_left = (
            _tobii_valid(data[valid_left_col])
            if valid_left_col is not None
            else pd.Series(pd.NA, index=data.index, dtype="boolean")
        )
        valid_right = (
            _tobii_valid(data[valid_right_col])
            if valid_right_col is not None
            else pd.Series(pd.NA, index=data.index, dtype="boolean")
        )
        combined = (valid_left | valid_right).fillna(False)
        if len(out["gaze_samples"]) == len(combined):
            out["gaze_samples"]["valid"] = combined.to_numpy(dtype=bool)

    out["recordings"]["software_name"] = "Tobii Pro Lab"
    _set_metadata(
        out,
        "tobii",
        {
            "source_columns": list(map(str, data.columns)),
            "heterogeneous_rows": True,
            "timestamp_role": time_col,
            "export_note": ("Rows may represent gaze observations or events; temporal order uses timestamps."),
        },
    )
    if keep_raw:
        _set_raw(out, "tobii", data.copy())
    out = add_provenance(
        out,
        "import_tobii",
        "dataset",
        f"{len(data)} Tobii rows",
        source_files=str(path),
    )
    return out


# Pupil Labs ---------------------------------------------------------------


def is_pupil_labs_export(path, inspect_rows=20):
    """Return a confidence score for Pupil Labs Neon/Core exports."""
    source = Path(path)
    if source.is_dir():
        files = {file.name.lower() for file in source.iterdir() if file.is_file()}
        if files.intersection(
            {
                "gaze.csv",
                "gaze_positions.csv",
                "3d_eye_states.csv",
                "pupil_positions.csv",
            }
        ):
            return 0.95
        return 0.0

    base = source.name.lower()
    if base in {
        "gaze.csv",
        "fixations.csv",
        "events.csv",
        "3d_eye_states.csv",
        "gaze_positions.csv",
        "pupil_positions.csv",
    }:
        return 0.85
    if source.suffix.lower().lstrip(".") not in {"csv", "tsv", "txt"}:
        return 0.0
    data = _safe_read_head(source, 2)
    if data is None:
        return 0.0
    names = [str(name).lower() for name in data.columns]
    patterns = (
        "gaze_timestamp",
        "world_timestamp",
        "norm_pos",
        "worn",
        "azimuth",
        "elevation",
        "section id",
        "recording id",
    )
    hits = sum(any(pattern in name for pattern in patterns) for name in names)
    return min(0.8, hits / 6.0)


def pupil_labs_format(path):
    """Classify a Pupil Labs path as ``neon``, ``core``, or ``unknown``."""
    source = Path(path)
    if source.is_dir():
        files = {file.name.lower() for file in source.iterdir() if file.is_file()}
    else:
        files = {source.name.lower()}
    if files.intersection({"gaze_positions.csv", "pupil_positions.csv"}):
        return "core"
    if files.intersection({"3d_eye_states.csv", "gaze.csv"}):
        return "neon"
    return "unknown"


def read_pupillabs(path, format=("auto", "neon", "core"), **kwargs):
    """Dispatch to Pupil Labs Neon or Core importers."""
    if isinstance(format, (tuple, list)):
        format = format[0] if format else "auto"
    format = str(format)
    if format not in {"auto", "neon", "core"}:
        _stop("`format` must be 'auto', 'neon', or 'core'.")
    if format == "auto":
        format = pupil_labs_format(path)
    if format == "neon":
        return read_pupil_neon(path, **kwargs)
    if format == "core":
        return read_pupil_core(path, **kwargs)
    _stop('Could not determine Pupil Labs format. Specify `format="neon"` or `format="core"`.')


def read_pupil_neon(
    path,
    participant_id="P001",
    session_id="S001",
    recording_id=None,
    keep_raw=True,
    quiet=False,
    **kwargs,
):
    """Import Pupil Labs Neon gaze plus optional companion exports."""
    source = Path(path)
    gaze_path = source / "gaze.csv" if source.is_dir() else source
    if not gaze_path.is_file():
        _stop("Pupil Labs Neon `gaze.csv` not found.")
    data = _read_delimited(gaze_path, **kwargs)
    names = list(map(str, data.columns))
    time_col = _pick(
        names,
        "timestamp [ns]",
        "timestamp_ns",
        "timestamp",
        "gaze timestamp [ns]",
    )
    x_col = _pick(names, "gaze x [px]", "x [px]", "gaze_x", "x")
    y_col = _pick(names, "gaze y [px]", "y [px]", "gaze_y", "y")
    if time_col is None or x_col is None or y_col is None:
        _stop("Neon gaze export lacks identifiable timestamp/x/y columns.")

    mapping = eye_mapping(
        recording=_pick(
            names,
            "recording id",
            "recording_id",
            "Recording UUID",
        ),
        timestamp=time_col,
        x=x_col,
        y=y_col,
        confidence=_pick(names, "confidence", "worn"),
        fixation_id=_pick(names, "fixation id", "fixation_id"),
        stimulus=_pick(names, "section id", "section_id", "world_index"),
    )
    mapping = {key: value for key, value in mapping.items() if value is not None}
    out = read_eye_generic(
        data,
        mapping=mapping,
        time_unit="nanoseconds",
        coordinate_space="world_camera_pixels",
        vendor="Pupil Labs Neon",
        participant_id=participant_id,
        recording_id=recording_id,
        session_id=session_id,
        keep_raw=keep_raw,
        quiet=True,
    )

    azimuth = _pick(
        names,
        "azimuth [deg]",
        "gaze azimuth [deg]",
        "azimuth_deg",
    )
    elevation = _pick(
        names,
        "elevation [deg]",
        "gaze elevation [deg]",
        "elevation_deg",
    )
    if azimuth is not None:
        out["gaze_samples"]["azimuth_deg"] = _safe_numeric(data[azimuth])
    if elevation is not None:
        out["gaze_samples"]["elevation_deg"] = _safe_numeric(data[elevation])

    if source.is_dir():
        _read_neon_companions(out, source, keep_raw=keep_raw)

    out["recordings"]["device_model"] = "Neon"
    out["recordings"]["software_name"] = "Pupil Cloud/Neon"
    _set_metadata(
        out,
        "pupil_neon",
        {
            "source_columns": list(map(str, data.columns)),
            "timestamp_unit": "UTC nanoseconds",
            "coordinate_space": "world camera pixels",
        },
    )
    out = add_provenance(
        out,
        "import_pupil_neon",
        "dataset",
        f"{len(data)} gaze rows",
        source_files=str(gaze_path),
    )
    return out


def _read_neon_companions(dataset, folder: Path, keep_raw=True) -> None:
    recording = _recording_id(dataset)
    coordinate = _coordinate_id(dataset)

    fixation_path = folder / "fixations.csv"
    if fixation_path.is_file():
        data = _read_delimited(fixation_path)
        names = list(map(str, data.columns))
        start_col = (
            _pick(
                names,
                "start timestamp [ns]",
                "start_timestamp_ns",
                "start timestamp",
            )
            or names[0]
        )
        end_col = _pick(
            names,
            "end timestamp [ns]",
            "end_timestamp_ns",
            "end timestamp",
        )
        duration_col = _pick(
            names,
            "duration [ms]",
            "duration_ms",
            "duration",
        )
        start = _safe_numeric(data[start_col]) * 1e-9
        end = _safe_numeric(data[end_col]) * 1e-9 if end_col is not None else pd.Series(np.nan, index=data.index)
        duration = _safe_numeric(data[duration_col]) if duration_col is not None else (end - start) * 1000
        x_col = _pick(names, "fixation x [px]", "x [px]", "x")
        y_col = _pick(names, "fixation y [px]", "y [px]", "y")
        episodes = pd.DataFrame(
            {
                "episode_id": [f"{recording}_neon_fix_{index:07d}" for index in range(1, len(data) + 1)],
                "recording_id": recording,
                "episode_type": "fixation",
                "eye": "combined",
                "start_time": start,
                "end_time": end,
                "duration_ms": duration,
                "start_x": np.nan,
                "start_y": np.nan,
                "end_x": np.nan,
                "end_y": np.nan,
                "centroid_x": (_safe_numeric(data[x_col]) if x_col else np.nan),
                "centroid_y": (_safe_numeric(data[y_col]) if y_col else np.nan),
                "amplitude": np.nan,
                "peak_velocity": np.nan,
                "dispersion": np.nan,
                "coordinate_space_id": coordinate,
                "source_algorithm": "Pupil Labs Neon",
                "source_parameters": pd.NA,
                "derived_by": "vendor",
                "trial_id": pd.NA,
                "stimulus_id": pd.NA,
                "aoi_id": pd.NA,
            }
        )
        _append_table(dataset, "episodes", episodes)
        if keep_raw:
            _set_raw(dataset, "neon_fixations", data.copy())

    event_path = folder / "events.csv"
    if event_path.is_file():
        data = _read_delimited(event_path)
        names = list(map(str, data.columns))
        time_col = (
            _pick(
                names,
                "timestamp [ns]",
                "timestamp_ns",
                "timestamp",
            )
            or names[0]
        )
        name_col = _pick(names, "name", "event", "event_name")
        if name_col is None:
            name_col = names[min(1, len(names) - 1)]
        seconds = _safe_numeric(data[time_col]) * 1e-9
        events = pd.DataFrame(
            {
                "event_id": [f"{recording}_neon_event_{index:07d}" for index in range(1, len(data) + 1)],
                "recording_id": recording,
                "timestamp_native": seconds / 1e-9,
                "timestamp_seconds": seconds,
                "event_type": "event",
                "event_name": data[name_col].astype("string"),
                "event_value": data[name_col].astype("string"),
                "duration": np.nan,
                "source": "Pupil Labs Neon",
                "native_record": pd.NA,
                "trial_id": pd.NA,
                "stimulus_id": pd.NA,
            }
        )
        _append_table(dataset, "events", events)
        if keep_raw:
            _set_raw(dataset, "neon_events", data.copy())

    eye_path = folder / "3d_eye_states.csv"
    if eye_path.is_file():
        data = _read_delimited(eye_path)
        names = list(map(str, data.columns))
        time_col = (
            _pick(
                names,
                "timestamp [ns]",
                "timestamp_ns",
                "timestamp",
            )
            or names[0]
        )
        seconds = _safe_numeric(data[time_col]) * 1e-9
        rows = []
        for eye in ("left", "right"):
            diameter_col = _pick(
                names,
                f"pupil diameter {eye} [mm]",
                f"diameter_3d_{eye}",
                f"pupil_diameter_{eye}",
            )
            if diameter_col is None:
                continue
            diameter = _safe_numeric(data[diameter_col])
            rows.append(
                pd.DataFrame(
                    {
                        "recording_id": recording,
                        "sample_id": [f"{recording}_neon_eye_{eye}_{index:09d}" for index in range(1, len(data) + 1)],
                        "timestamp_native": seconds / 1e-9,
                        "timestamp_seconds": seconds,
                        "eye": eye,
                        "pupil_diameter": diameter,
                        "pupil_unit": "millimetres",
                        "pupil_valid": np.isfinite(diameter),
                        "eye_openness": np.nan,
                        "gaze_origin_x": np.nan,
                        "gaze_origin_y": np.nan,
                        "gaze_origin_z": np.nan,
                        "gaze_origin_valid": pd.NA,
                        "corneal_reflection_x": np.nan,
                        "corneal_reflection_y": np.nan,
                        "detector_method": "Pupil Labs 3D eye state",
                        "confidence": np.nan,
                        "trial_id": pd.NA,
                        "stimulus_id": pd.NA,
                    }
                )
            )
        if rows:
            dataset["eye_samples"] = pd.concat(
                rows,
                ignore_index=True,
                sort=False,
            )
        if keep_raw:
            _set_raw(dataset, "neon_eye_states", data.copy())


def read_pupil_core(
    path,
    participant_id="P001",
    session_id="S001",
    recording_id=None,
    keep_raw=True,
    quiet=False,
    **kwargs,
):
    """Import Pupil Labs Core gaze plus pupil/fixation companions."""
    source = Path(path)
    gaze_path = source / "gaze_positions.csv" if source.is_dir() else source
    if not gaze_path.is_file():
        _stop("Pupil Core `gaze_positions.csv` not found.")
    data = _read_delimited(gaze_path, **kwargs)
    names = list(map(str, data.columns))
    time_col = _pick(names, "gaze_timestamp", "world_timestamp", "timestamp")
    x_col = _pick(names, "norm_pos_x", "gaze_x", "x")
    y_col = _pick(names, "norm_pos_y", "gaze_y", "y")
    if time_col is None or x_col is None or y_col is None:
        _stop("Pupil Core gaze export lacks identifiable timestamp/x/y columns.")

    mapping = eye_mapping(
        timestamp=time_col,
        x=x_col,
        y=y_col,
        confidence=_pick(names, "confidence"),
        fixation_id=_pick(names, "fixation_id"),
        stimulus=_pick(names, "world_index"),
    )
    mapping = {key: value for key, value in mapping.items() if value is not None}
    out = read_eye_generic(
        data,
        mapping=mapping,
        time_unit="seconds",
        coordinate_space="surface_normalized_bottom_left",
        vendor="Pupil Labs Core",
        participant_id=participant_id,
        recording_id=recording_id,
        session_id=session_id,
        keep_raw=keep_raw,
        quiet=True,
    )
    if source.is_dir():
        _read_core_companions(out, source, keep_raw=keep_raw)

    out["recordings"]["device_model"] = "Pupil Core"
    _set_metadata(
        out,
        "pupil_core",
        {
            "source_columns": list(map(str, data.columns)),
            "coordinate_note": ("Normalized surface coordinates use bottom-left origin."),
        },
    )
    out = add_provenance(
        out,
        "import_pupil_core",
        "dataset",
        f"{len(data)} gaze rows",
        source_files=str(gaze_path),
    )
    return out


def _read_core_companions(dataset, folder: Path, keep_raw=True) -> None:
    recording = _recording_id(dataset)
    coordinate = _coordinate_id(dataset)

    pupil_path = folder / "pupil_positions.csv"
    if pupil_path.is_file():
        data = _read_delimited(pupil_path)
        names = list(map(str, data.columns))
        eye_col = _pick(names, "eye_id", "id", "eye")
        if eye_col is None:
            eye = pd.Series("unknown", index=data.index, dtype="string")
        else:
            raw_eye = data[eye_col].astype("string")
            eye = pd.Series(
                np.where(
                    raw_eye.isin(["0", "right", "R"]),
                    "right",
                    "left",
                ),
                index=data.index,
                dtype="string",
            )
        time_col = _pick(names, "pupil_timestamp", "timestamp") or names[0]
        diameter_col = _pick(
            names,
            "diameter_3d",
            "diameter",
            "pupil_diameter",
        )
        confidence_col = _pick(names, "confidence")
        method_col = _pick(names, "method")
        seconds = _safe_numeric(data[time_col])
        diameter = (
            _safe_numeric(data[diameter_col]) if diameter_col is not None else pd.Series(np.nan, index=data.index)
        )
        confidence = (
            _safe_numeric(data[confidence_col]) if confidence_col is not None else pd.Series(np.nan, index=data.index)
        )
        pupil_unit = "millimetres" if diameter_col is not None and "3d" in diameter_col.lower() else "image_pixels"
        eye_samples = pd.DataFrame(
            {
                "recording_id": recording,
                "sample_id": [f"{recording}_core_eye_{index:09d}" for index in range(1, len(data) + 1)],
                "timestamp_native": seconds,
                "timestamp_seconds": seconds,
                "eye": eye,
                "pupil_diameter": diameter,
                "pupil_unit": pupil_unit,
                "pupil_valid": confidence > 0,
                "eye_openness": np.nan,
                "gaze_origin_x": np.nan,
                "gaze_origin_y": np.nan,
                "gaze_origin_z": np.nan,
                "gaze_origin_valid": pd.NA,
                "corneal_reflection_x": np.nan,
                "corneal_reflection_y": np.nan,
                "detector_method": (data[method_col].astype("string") if method_col is not None else pd.NA),
                "confidence": confidence,
                "trial_id": pd.NA,
                "stimulus_id": pd.NA,
            }
        )
        _append_table(dataset, "eye_samples", eye_samples)
        if keep_raw:
            _set_raw(dataset, "core_pupil_positions", data.copy())

    fixation_path = folder / "fixations.csv"
    if fixation_path.is_file():
        data = _read_delimited(fixation_path)
        names = list(map(str, data.columns))
        start_col = _pick(names, "start_timestamp", "start_time") or names[0]
        duration_col = _pick(names, "duration", "duration_ms")
        start = _safe_numeric(data[start_col])
        duration = (
            _safe_numeric(data[duration_col]) if duration_col is not None else pd.Series(np.nan, index=data.index)
        )
        finite_duration = duration[np.isfinite(duration)]
        duration_ms = duration * 1000 if len(finite_duration) and bool((finite_duration < 100).all()) else duration
        x_col = _pick(names, "norm_pos_x", "x")
        y_col = _pick(names, "norm_pos_y", "y")
        dispersion_col = _pick(names, "dispersion")
        method_col = _pick(names, "method")
        episodes = pd.DataFrame(
            {
                "episode_id": [f"{recording}_core_fix_{index:07d}" for index in range(1, len(data) + 1)],
                "recording_id": recording,
                "episode_type": "fixation",
                "eye": "combined",
                "start_time": start,
                "end_time": start + duration_ms / 1000,
                "duration_ms": duration_ms,
                "start_x": np.nan,
                "start_y": np.nan,
                "end_x": np.nan,
                "end_y": np.nan,
                "centroid_x": (_safe_numeric(data[x_col]) if x_col else np.nan),
                "centroid_y": (_safe_numeric(data[y_col]) if y_col else np.nan),
                "amplitude": np.nan,
                "peak_velocity": np.nan,
                "dispersion": (_safe_numeric(data[dispersion_col]) if dispersion_col else np.nan),
                "coordinate_space_id": coordinate,
                "source_algorithm": (data[method_col].astype("string") if method_col else pd.NA),
                "source_parameters": pd.NA,
                "derived_by": "vendor",
                "trial_id": pd.NA,
                "stimulus_id": pd.NA,
                "aoi_id": pd.NA,
            }
        )
        _append_table(dataset, "episodes", episodes)
        if keep_raw:
            _set_raw(dataset, "core_fixations", data.copy())


# EyeLink -----------------------------------------------------------------


def is_eyelink_export(path, inspect_rows=20):
    """Return a confidence score for EyeLink EDF/ASC/report exports."""
    source = Path(path)
    if source.is_dir():
        return 0.0
    extension = source.suffix.lower().lstrip(".")
    if extension == "edf":
        return 0.98
    if extension == "asc":
        try:
            with source.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                lines = [handle.readline() for _ in range(int(inspect_rows))]
        except OSError:
            lines = []
        markers = (
            "MSG",
            "EFIX",
            "ESACC",
            "EBLINK",
            "START",
            "END",
            "SAMPLES",
            "EVENTS",
        )
        if any(line.strip().startswith(markers) for line in lines if line):
            return 0.95
    if extension not in {"csv", "tsv", "txt"}:
        return 0.0
    data = _safe_read_head(source, 2)
    if data is None:
        return 0.0
    names = [str(name).upper() for name in data.columns]
    patterns = (
        "RECORDING_SESSION_LABEL",
        "CURRENT_FIX",
        "IA_LABEL",
        "TRIAL_INDEX",
        "EYELINK",
        "SACCADE",
        "FIXATION",
    )
    hits = sum(any(pattern in name for pattern in patterns) for name in names)
    return min(0.8, hits / 5.0)


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _parse_eyelink_asc(lines: list[str], recording_id: str):
    samples = []
    eye_samples = []
    episodes = []
    events = []
    calibrations = []
    record_types: dict[str, int] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        tokens = re.split(r"\s+", line)
        first = tokens[0]
        row_type = "SAMPLE" if re.fullmatch(r"\d+", first) else first
        record_types[row_type] = record_types.get(row_type, 0) + 1

        if row_type == "SAMPLE":
            values = [_parse_float(token) for token in tokens]
            if len(values) < 4:
                continue
            timestamp, x, y, pupil = values[:4]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            index = len(samples) + 1
            samples.append(
                {
                    "recording_id": recording_id,
                    "stream_id": f"{recording_id}_gaze",
                    "sample_id": f"{recording_id}_sample_{index:09d}",
                    "timestamp_native": timestamp,
                    "timestamp_seconds": timestamp / 1000,
                    "gaze_x": x,
                    "gaze_y": y,
                    "gaze_z": np.nan,
                    "azimuth_deg": np.nan,
                    "elevation_deg": np.nan,
                    "valid": True,
                    "confidence": np.nan,
                    "fixation_id_source": pd.NA,
                    "blink_id_source": pd.NA,
                    "trial_id": pd.NA,
                    "stimulus_id": pd.NA,
                    "coordinate_space_id": pd.NA,
                }
            )
            eye_samples.append(
                {
                    "recording_id": recording_id,
                    "sample_id": f"{recording_id}_eye_{index:09d}",
                    "timestamp_native": timestamp,
                    "timestamp_seconds": timestamp / 1000,
                    "eye": "recorded",
                    "pupil_diameter": pupil,
                    "pupil_unit": "vendor_units",
                    "pupil_valid": bool(np.isfinite(pupil)),
                    "eye_openness": np.nan,
                    "gaze_origin_x": np.nan,
                    "gaze_origin_y": np.nan,
                    "gaze_origin_z": np.nan,
                    "gaze_origin_valid": pd.NA,
                    "corneal_reflection_x": np.nan,
                    "corneal_reflection_y": np.nan,
                    "detector_method": "EyeLink sample",
                    "confidence": np.nan,
                    "trial_id": pd.NA,
                    "stimulus_id": pd.NA,
                }
            )
            continue

        if row_type in {"EFIX", "ESACC", "EBLINK"}:
            if len(tokens) < 5:
                continue
            eye = tokens[1]
            start = _parse_float(tokens[2])
            end = _parse_float(tokens[3])
            duration = _parse_float(tokens[4])
            index = len(episodes) + 1
            episode_type = {
                "EFIX": "fixation",
                "ESACC": "saccade",
                "EBLINK": "blink",
            }[row_type]
            centroid_x = _parse_float(tokens[5]) if row_type == "EFIX" else np.nan
            centroid_y = _parse_float(tokens[6]) if row_type == "EFIX" else np.nan
            start_x = _parse_float(tokens[5]) if row_type == "ESACC" else np.nan
            start_y = _parse_float(tokens[6]) if row_type == "ESACC" else np.nan
            end_x = _parse_float(tokens[7]) if row_type == "ESACC" else np.nan
            end_y = _parse_float(tokens[8]) if row_type == "ESACC" else np.nan
            amplitude = _parse_float(tokens[9]) if row_type == "ESACC" and len(tokens) > 9 else np.nan
            peak = _parse_float(tokens[10]) if row_type == "ESACC" and len(tokens) > 10 else np.nan
            episodes.append(
                {
                    "episode_id": (f"{recording_id}_{row_type.lower()}_{index:07d}"),
                    "recording_id": recording_id,
                    "episode_type": episode_type,
                    "eye": eye,
                    "start_time": start / 1000,
                    "end_time": end / 1000,
                    "duration_ms": duration,
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "amplitude": amplitude,
                    "peak_velocity": peak,
                    "dispersion": np.nan,
                    "coordinate_space_id": pd.NA,
                    "source_algorithm": "EyeLink online parser",
                    "source_parameters": pd.NA,
                    "derived_by": "vendor",
                    "trial_id": pd.NA,
                    "stimulus_id": pd.NA,
                    "aoi_id": pd.NA,
                }
            )
            continue

        if row_type == "MSG":
            if len(tokens) < 3:
                continue
            timestamp = _parse_float(tokens[1])
            message = " ".join(tokens[2:])
            index = len(events) + 1
            events.append(
                {
                    "event_id": f"{recording_id}_msg_{index:07d}",
                    "recording_id": recording_id,
                    "timestamp_native": timestamp,
                    "timestamp_seconds": timestamp / 1000,
                    "event_type": "message",
                    "event_name": message.split(" ", 1)[0],
                    "event_value": message,
                    "duration": np.nan,
                    "source": "EyeLink MSG",
                    "native_record": line,
                    "trial_id": pd.NA,
                    "stimulus_id": pd.NA,
                }
            )
            if re.search(r"CALIB|VALIDATION|DRIFT", message, re.I):
                numbers = [
                    float(value)
                    for value in re.findall(
                        r"[0-9]+(?:\.[0-9]+)?",
                        message,
                    )
                ]
                calibration_type = (
                    "drift"
                    if re.search("DRIFT", message, re.I)
                    else ("validation" if re.search("VALID", message, re.I) else "calibration")
                )
                calibrations.append(
                    {
                        "calibration_id": (f"{recording_id}_cal_{len(calibrations) + 1:05d}"),
                        "recording_id": recording_id,
                        "timestamp_seconds": timestamp / 1000,
                        "calibration_type": calibration_type,
                        "eye": pd.NA,
                        "point_count": np.nan,
                        "average_error": numbers[0] if numbers else np.nan,
                        "maximum_error": (numbers[1] if len(numbers) > 1 else np.nan),
                        "error_unit": "degrees",
                        "validation_status": ("passed" if re.search(r"GOOD|OK|SUCCESS", message, re.I) else pd.NA),
                        "drift_offset": (numbers[0] if calibration_type == "drift" and numbers else np.nan),
                        "source_record": message,
                    }
                )
            continue

        if row_type in {"BUTTON", "INPUT", "START", "END"}:
            timestamp = _parse_float(tokens[1]) if len(tokens) > 1 else np.nan
            index = len(events) + 1
            events.append(
                {
                    "event_id": f"{recording_id}_event_{index:07d}",
                    "recording_id": recording_id,
                    "timestamp_native": timestamp,
                    "timestamp_seconds": timestamp / 1000,
                    "event_type": row_type.lower(),
                    "event_name": row_type,
                    "event_value": " ".join(tokens[1:]),
                    "duration": np.nan,
                    "source": "EyeLink ASC",
                    "native_record": line,
                    "trial_id": pd.NA,
                    "stimulus_id": pd.NA,
                }
            )

    return {
        "gaze_samples": pd.DataFrame(samples) if samples else empty_eye_table("gaze_samples"),
        "eye_samples": pd.DataFrame(eye_samples) if eye_samples else empty_eye_table("eye_samples"),
        "episodes": pd.DataFrame(episodes) if episodes else empty_eye_table("episodes"),
        "events": pd.DataFrame(events) if events else empty_eye_table("events"),
        "calibrations": pd.DataFrame(calibrations) if calibrations else empty_eye_table("calibrations"),
        "record_types": record_types,
    }


def read_eyelink_asc(
    path,
    participant_id="P001",
    session_id="S001",
    recording_id=None,
    coordinate_space="display_pixels_top_left",
    screen_width=np.nan,
    screen_height=np.nan,
    keep_raw=True,
    quiet=False,
):
    """Import an EyeLink ASC text export."""
    source = Path(path)
    if not source.is_file():
        _stop(f"ASC file does not exist: {source}")
    lines = source.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()
    if not lines:
        _stop("ASC file is empty.")
    recording = recording_id or f"rec_{participant_id}_{session_id}"
    parsed = _parse_eyelink_asc(lines, recording)

    gaze = parsed["gaze_samples"]
    observed_rate = (
        estimate_sampling_rate(gaze["timestamp_seconds"])
        if isinstance(gaze, pd.DataFrame) and not gaze.empty and "timestamp_seconds" in gaze
        else np.nan
    )
    recordings = pd.DataFrame(
        {
            "recording_id": [recording],
            "participant_id": [participant_id],
            "session_id": [session_id],
            "vendor": ["SR Research"],
            "vendor_family": ["EyeLink"],
            "device_model": ["EyeLink"],
            "firmware_version": [pd.NA],
            "software_name": ["EyeLink"],
            "software_version": [pd.NA],
            "experiment_type": [pd.NA],
            "nominal_sampling_rate": [observed_rate],
            "screen_width_px": [screen_width],
            "screen_height_px": [screen_height],
            "recording_start": [pd.NA],
            "source_timezone": [pd.NA],
            "source_file_set": [str(source)],
        }
    )
    coordinate = new_coordinate_space(
        "coord_eyelink_display",
        coordinate_space,
        width=screen_width,
        height=screen_height,
    )
    coordinate_id = coordinate["coordinate_space_id"].iloc[0]
    if not gaze.empty:
        gaze["coordinate_space_id"] = coordinate_id
    episodes = parsed["episodes"]
    if isinstance(episodes, pd.DataFrame) and not episodes.empty:
        episodes["coordinate_space_id"] = coordinate_id
    streams = pd.DataFrame(
        {
            "stream_id": [f"{recording}_gaze"],
            "recording_id": [recording],
            "stream_type": ["gaze_combined"],
            "source_device": ["EyeLink"],
            "source_clock": ["tracker"],
            "sampling_type": ["sampled"],
            "nominal_rate_hz": [observed_rate],
            "observed_rate_hz": [observed_rate],
            "timestamp_unit": ["milliseconds"],
            "value_unit": ["pixels"],
            "coordinate_space_id": [coordinate_id],
            "processing_level": ["raw_imported"],
        }
    )

    out = new_eye_dataset(
        recordings=recordings,
        streams=streams,
        gaze_samples=gaze,
        eye_samples=parsed["eye_samples"],
        episodes=episodes,
        events=parsed["events"],
        calibrations=parsed["calibrations"],
        coordinate_spaces=coordinate,
        raw={"eyelink_asc": lines} if keep_raw else {},
        vendor_metadata={"eyelink": {"record_types": parsed["record_types"]}},
        validate=False,
    )
    out = add_provenance(
        out,
        "import_eyelink_asc",
        "dataset",
        f"{len(lines)} ASC lines",
        source_files=str(source),
    )
    out.validation = validate_eye_dataset(out)
    return out


def read_eyelink_report(
    path,
    mapping=None,
    time_unit="milliseconds",
    coordinate_space="display_pixels_top_left",
    **kwargs,
):
    """Import an EyeLink Data Viewer delimited report."""
    data = _read_delimited(path)
    if mapping is None:
        mapping = infer_eye_mapping(data, vendor="EyeLink Data Viewer")
    return read_eye_generic(
        path,
        mapping=mapping,
        time_unit=time_unit,
        coordinate_space=coordinate_space,
        vendor="EyeLink Data Viewer",
        **kwargs,
    )


def read_eyelink_edf(
    path,
    edf2asc=None,
    output=None,
    keep_asc=False,
    **kwargs,
):
    """Convert an EDF with SR Research EDF2ASC, then import its ASC output."""
    source = Path(path)
    if not source.is_file():
        _stop(f"EDF file does not exist: {source}")
    requested = str(edf2asc or "").strip()
    if requested:
        requested_path = Path(requested).expanduser()
        converter = str(requested_path.resolve()) if requested_path.is_file() else shutil.which(requested)
    else:
        converter = shutil.which("edf2asc")
    if not converter:
        raise EyeProcessBackendError(
            "`edf2asc` executable was not found. Install SR Research EDF2ASC and provide its path."
        )

    temporary = output is None
    if output is None:
        descriptor, temporary_name = tempfile.mkstemp(suffix=".asc")
        os.close(descriptor)
        destination = Path(temporary_name)
    else:
        destination = Path(output)
    try:
        completed = subprocess.run(
            [
                converter,
                "-y",
                str(source.resolve()),
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not destination.is_file():
            detail = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
            raise EyeProcessBackendError("EDF2ASC conversion failed" + (f": {detail}" if detail else "."))
        out = read_eyelink_asc(destination, **kwargs)
        out = add_provenance(
            out,
            "convert_edf2asc",
            "dataset",
            details=f"Converter: {converter}",
            source_files=str(source),
            reversible=False,
        )
        return out
    finally:
        if not keep_asc and destination.exists():
            destination.unlink(missing_ok=True)
        elif temporary and not keep_asc:
            destination.unlink(missing_ok=True)


# SMI BeGaze ---------------------------------------------------------------


def is_smi_export(path, inspect_rows=20):
    """Return a confidence score for SMI BeGaze text exports."""
    source = Path(path)
    if source.is_dir():
        return 0.0
    extension = source.suffix.lower().lstrip(".")
    if extension == "idf":
        return 0.5
    if extension not in {"txt", "csv", "tsv", "asc"}:
        return 0.0
    data = _safe_read_head(source, 3)
    if data is None:
        return 0.0
    names = [str(name).lower() for name in data.columns]
    patterns = (
        "smi",
        "begaze",
        "point of regard",
        "por x",
        "por y",
        "stimulus name",
        "fixation duration",
        "event type",
    )
    hits = sum(any(pattern in name for pattern in patterns) for name in names)
    if any("begaze" in name or "smi" in name for name in names):
        return max(0.85, hits / 6.0)
    return min(0.75, hits / 6.0)


def read_smi(
    path,
    participant_id=None,
    recording_id=None,
    session_id="S001",
    time_unit="microseconds",
    coordinate_space="display_pixels_top_left",
    keep_raw=True,
    quiet=False,
    **kwargs,
):
    """Import an SMI BeGaze text/ASCII export."""
    source = Path(path)
    if source.suffix.lower() == ".idf":
        raise EyeProcessBackendError(
            "Direct SMI IDF import is not supported. Export a text/ASCII file from BeGaze first."
        )

    data = _read_delimited(source, **kwargs)
    names = list(map(str, data.columns))
    time_col = _pick(
        names,
        "Time",
        "Timestamp",
        "Time [ms]",
        "Time [us]",
        "time",
    )
    x_col = _pick(
        names,
        "Point of Regard X [px]",
        "POR X [px]",
        "Mapped gaze data point X",
        "Gaze X",
        "x",
    )
    y_col = _pick(
        names,
        "Point of Regard Y [px]",
        "POR Y [px]",
        "Mapped gaze data point Y",
        "Gaze Y",
        "y",
    )
    if time_col is None or x_col is None or y_col is None:
        _stop("SMI export lacks identifiable timestamp/x/y fields.")

    mapping = eye_mapping(
        participant=_pick(
            names,
            "Participant",
            "Subject",
            "participant_id",
        ),
        recording=_pick(
            names,
            "Recording",
            "Session",
            "recording_id",
        ),
        timestamp=time_col,
        x=x_col,
        y=y_col,
        left_x=_pick(names, "L POR X [px]", "Left POR X [px]"),
        left_y=_pick(names, "L POR Y [px]", "Left POR Y [px]"),
        right_x=_pick(names, "R POR X [px]", "Right POR X [px]"),
        right_y=_pick(names, "R POR Y [px]", "Right POR Y [px]"),
        pupil_left=_pick(
            names,
            "L Pupil Diameter [mm]",
            "Left Pupil Diameter",
            "L Diameter",
        ),
        pupil_right=_pick(
            names,
            "R Pupil Diameter [mm]",
            "Right Pupil Diameter",
            "R Diameter",
        ),
        fixation_id=_pick(names, "Fixation Index", "Fixation ID"),
        trial=_pick(names, "Trial", "Trial Index"),
        stimulus=_pick(names, "Stimulus", "Stimulus Name"),
        event_name=_pick(names, "Event", "Event Type", "Message"),
    )
    mapping = {key: value for key, value in mapping.items() if value is not None}
    out = read_eye_generic(
        data,
        mapping=mapping,
        time_unit=time_unit,
        coordinate_space=coordinate_space,
        vendor="SMI BeGaze",
        participant_id=participant_id,
        recording_id=recording_id,
        session_id=session_id,
        pupil_unit="millimetres",
        keep_raw=keep_raw,
        quiet=True,
    )
    out["recordings"]["software_name"] = "SMI BeGaze"
    _set_metadata(
        out,
        "smi",
        {
            "source_columns": list(map(str, data.columns)),
            "legacy": True,
            "source_type": "BeGaze text export",
        },
    )
    out = add_provenance(
        out,
        "import_smi",
        "dataset",
        f"{len(data)} rows",
        source_files=str(source),
    )
    return out


def read_smi_raw_export(*args, **kwargs):
    """Alias of :func:`read_smi` for SMI raw-text exports."""
    return read_smi(*args, **kwargs)


def read_smi_event_export(*args, **kwargs):
    """Alias of :func:`read_smi` for SMI event exports."""
    return read_smi(*args, **kwargs)


def read_smi_aoi_export(*args, **kwargs):
    """Alias of :func:`read_smi` for SMI AOI exports."""
    return read_smi(*args, **kwargs)
