from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .coordinates import _convert_xy, audit_coordinate_spaces
from .dataset import (
    EyeDataset,
    _assert_eye_dataset,
    _next_id,
    _now_utc,
    add_provenance,
    is_eye_dataset,
    validate_eye_dataset,
)
from .schema import empty_eye_table, schema_table, standardize_eye_table
from .timebase import (
    EyeClockTransform,
    apply_clock_transform,
    audit_timebase,
    estimate_clock_transform,
    estimate_sampling_rate,
)


def _first_nonmissing(values, default=pd.NA):
    if isinstance(values, pd.Series):
        iterable = values.tolist()
    elif isinstance(values, (list, tuple, np.ndarray)):
        iterable = list(values)
    else:
        iterable = [values]
    for value in iterable:
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _finite_numeric(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _mode_value(values):
    series = pd.Series(values).dropna()
    if series.empty:
        return pd.NA
    counts = series.value_counts(sort=False)
    return counts.index[int(np.argmax(counts.to_numpy()))]


def _string_missing(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.isna() | text.str.strip().eq("")


def _as_components(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _safe_min_finite(values) -> float:
    arr = _finite_numeric(values)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else np.nan


def _safe_max_finite(values) -> float:
    arr = _finite_numeric(values)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else np.nan


# R/002-class.R ------------------------------------------------------------


def as_eye_dataset(x, mapping=None, **kwargs):
    """Coerce an object to :class:`EyeDataset` like R ``as_eye_dataset()``."""
    if isinstance(x, EyeDataset):
        return x
    if isinstance(x, pd.DataFrame):
        from .importers import read_eye_generic

        return read_eye_generic(x, mapping=mapping, **kwargs)
    cls = type(x).__name__
    raise TypeError(f"No `as_eye_dataset()` method for class: {cls}.")


# R/007-coordinates-time.R -------------------------------------------------


def convert_xy(
    x,
    y,
    from_,
    to,
    from_width=np.nan,
    from_height=np.nan,
    to_width=np.nan,
    to_height=np.nan,
    clip=False,
):
    """Convert x/y vectors between supported two-dimensional coordinate spaces."""
    return _convert_xy(
        x,
        y,
        from_,
        to,
        from_width=from_width,
        from_height=from_height,
        to_width=to_width,
        to_height=to_height,
        clip=clip,
    )


def synchronize_eye_biometrics(
    gaze,
    biometrics,
    source_markers=None,
    target_markers=None,
    method="linear",
    resolve_ids=False,
):
    """Synchronize biometric time to gaze time and combine two eye datasets."""
    _assert_eye_dataset(gaze)
    _assert_eye_dataset(biometrics)
    if method not in {"linear", "offset", "none"}:
        raise ValueError("`method` must be one of 'linear', 'offset', or 'none'.")

    transform = EyeClockTransform(
        method="none",
        offset=0.0,
        slope=1.0,
        n_markers=0,
        residual_sd=np.nan,
        max_abs_residual=np.nan,
    )
    bio = biometrics
    used_method = method
    if method != "none":
        if source_markers is None or target_markers is None:
            warnings.warn(
                "Marker pairs were not supplied; estimating offset from first observations.",
                RuntimeWarning,
                stacklevel=2,
            )
            source_markers = _safe_min_finite(biometrics["biometrics"]["timestamp_seconds"])
            target_markers = _safe_min_finite(gaze["gaze_samples"]["timestamp_seconds"])
            used_method = "offset"
        transform = estimate_clock_transform(source_markers, target_markers, used_method)
        bio = apply_clock_transform(
            biometrics,
            transform,
            components=("biometrics", "events", "eye_samples"),
        )

    from .adapters import combine_eye_datasets

    out = combine_eye_datasets(gaze, bio, resolve_ids=resolve_ids)
    reversible = used_method != "linear" or np.isfinite(transform.slope)
    return add_provenance(
        out,
        "synchronize_eye_biometrics",
        "biometrics",
        f"method={used_method}",
        reversible=bool(reversible),
    )


def audit_clock_sync(x, channel=None):
    """Audit temporal overlap between gaze and biometric streams."""
    _assert_eye_dataset(x)
    gaze = x["gaze_samples"]
    biometrics = x["biometrics"]
    if gaze.empty or biometrics.empty:
        return pd.DataFrame([{"status": "unavailable", "message": "Gaze and biometric streams are both required."}])

    b = biometrics
    if channel is not None:
        channels = {channel} if isinstance(channel, str) else set(channel)
        b = b[b["channel"].isin(channels)]

    gaze_ids = list(pd.unique(gaze["recording_id"].dropna()))
    bio_ids = set(pd.unique(b["recording_id"].dropna()))
    recs = [recording_id for recording_id in gaze_ids if recording_id in bio_ids]

    rows = []
    for recording_id in recs:
        gt = _finite_numeric(gaze.loc[gaze["recording_id"] == recording_id, "timestamp_seconds"])
        bt = _finite_numeric(b.loc[b["recording_id"] == recording_id, "timestamp_seconds"])
        gt = gt[np.isfinite(gt)]
        bt = bt[np.isfinite(bt)]
        if not gt.size or not bt.size:
            continue
        gaze_start, gaze_end = float(np.min(gt)), float(np.max(gt))
        bio_start, bio_end = float(np.min(bt)), float(np.max(bt))
        overlap = max(0.0, min(gaze_end, bio_end) - max(gaze_start, bio_start))
        union = max(gaze_end, bio_end) - min(gaze_start, bio_start)
        rows.append(
            {
                "recording_id": recording_id,
                "gaze_start": gaze_start,
                "gaze_end": gaze_end,
                "biometric_start": bio_start,
                "biometric_end": bio_end,
                "overlap_seconds": overlap,
                "overlap_fraction": overlap / union if union > 0 else np.nan,
                "start_offset_seconds": bio_start - gaze_start,
                "end_offset_seconds": bio_end - gaze_end,
                "status": "overlap" if overlap > 0 else "no_overlap",
            }
        )
    return pd.DataFrame(rows)


# R/008-trials-aoi.R -------------------------------------------------------


def _event_matches(patterns: Sequence[str], values: pd.Series) -> pd.Series:
    result = pd.Series(False, index=values.index)
    text = values.astype("string")
    for pattern in patterns:
        exact = text.eq(pattern).fillna(False)
        try:
            regex = text.str.contains(pattern, case=False, regex=True, na=False)
        except re.error as exc:
            raise ValueError(f"Invalid event pattern {pattern!r}.") from exc
        result |= exact | regex
    return result


def _participant_for_recording(x: EyeDataset, recording_id):
    recordings = x["recordings"]
    hit = recordings.loc[recordings["recording_id"] == recording_id, "participant_id"]
    return _first_nonmissing(hit, pd.NA)


def build_trials(
    x,
    start_events=("TRIAL_START", "TRIALID", "START_TRIAL"),
    end_events=("TRIAL_END", "TRIAL_RESULT", "END_TRIAL"),
    event_field="event_name",
    trial_id_pattern=None,
    close_open="recording_end",
    overwrite=False,
):
    """Reconstruct trial intervals from event markers."""
    _assert_eye_dataset(x)
    if close_open not in {"recording_end", "next_start", "drop"}:
        raise ValueError("Invalid `close_open`.")
    if event_field not in {"event_name", "event_value"}:
        raise ValueError("Invalid `event_field`.")
    if x["events"].empty:
        raise ValueError("No events are available for trial reconstruction.")

    out = x.copy()
    ev = (
        out["events"]
        .sort_values(
            ["recording_id", "timestamp_seconds"],
            kind="mergesort",
            na_position="last",
        )
        .reset_index(drop=True)
    )
    values = ev[event_field]
    starts = _event_matches(list(start_events), values)
    ends = _event_matches(list(end_events), values)

    built_rows = []
    for recording_id in pd.unique(ev["recording_id"]):
        rec_mask = ev["recording_id"].eq(recording_id)
        start_idx = list(ev.index[rec_mask & starts])
        end_idx = list(ev.index[rec_mask & ends])
        for j, start_idx_value in enumerate(start_idx, start=1):
            start_time = pd.to_numeric(
                pd.Series([ev.at[start_idx_value, "timestamp_seconds"]]),
                errors="coerce",
            ).iloc[0]
            next_start = (
                pd.to_numeric(
                    pd.Series([ev.at[start_idx[j], "timestamp_seconds"]]),
                    errors="coerce",
                ).iloc[0]
                if j < len(start_idx)
                else np.inf
            )
            candidates = [
                idx
                for idx in end_idx
                if idx > start_idx_value
                and pd.to_numeric(pd.Series([ev.at[idx, "timestamp_seconds"]]), errors="coerce").iloc[0] <= next_start
            ]
            if candidates:
                end_time = float(
                    pd.to_numeric(
                        pd.Series([ev.at[candidates[0], "timestamp_seconds"]]),
                        errors="coerce",
                    ).iloc[0]
                )
            elif close_open == "next_start" and np.isfinite(next_start):
                end_time = float(next_start)
            elif close_open == "recording_end":
                all_times = []
                for component in ("gaze_samples", "eye_samples", "biometrics"):
                    table = out[component]
                    if not table.empty:
                        all_times.extend(
                            table.loc[
                                table["recording_id"] == recording_id,
                                "timestamp_seconds",
                            ].tolist()
                        )
                end_time = _safe_max_finite(all_times)
            else:
                continue

            raw_id = ev.at[start_idx_value, "event_value"]
            if pd.isna(raw_id) or not str(raw_id).strip():
                raw_id = ev.at[start_idx_value, "event_name"]
            if trial_id_pattern is not None and not pd.isna(raw_id):
                match = re.search(trial_id_pattern, str(raw_id))
                if match and match.lastindex:
                    raw_id = match.group(1)

            raw_text = "" if pd.isna(raw_id) else str(raw_id)
            if raw_text and raw_text not in set(start_events):
                trial_id = raw_text
            else:
                trial_id = f"{recording_id}_trial_{j:05d}"

            stimulus = ev.at[start_idx_value, "stimulus_id"]
            valid_interval = bool(
                np.isfinite(start_time) and np.isfinite(end_time) and float(end_time) >= float(start_time)
            )
            built_rows.append(
                {
                    "interval_id": f"{recording_id}_interval_trial_{j:05d}",
                    "recording_id": recording_id,
                    "interval_type": "trial",
                    "start_time": start_time,
                    "end_time": end_time,
                    "trial_id": trial_id,
                    "participant_id": _participant_for_recording(out, recording_id),
                    "item_id": pd.NA,
                    "stimulus_id": stimulus,
                    "condition_id": pd.NA,
                    "parent_interval_id": pd.NA,
                    "valid_interval": valid_interval,
                }
            )

    if not built_rows:
        raise ValueError("No trial intervals could be reconstructed from the selected events.")

    built = standardize_eye_table(pd.DataFrame(built_rows), "intervals")
    if overwrite:
        out["intervals"] = out["intervals"][out["intervals"]["interval_type"].ne("trial")].reset_index(drop=True)
    out["intervals"] = standardize_eye_table(
        pd.concat([out["intervals"], built], ignore_index=True, sort=False),
        "intervals",
    )
    out = assign_trials(out)
    return add_provenance(
        out,
        "build_trials",
        "intervals",
        f"{len(built)} trials; close_open={close_open}",
    )


def build_stimulus_intervals(x, source="gaze_samples", overwrite=False):
    """Build contiguous stimulus/media intervals."""
    _assert_eye_dataset(x)
    if source not in {"gaze_samples", "events"}:
        raise ValueError("`source` must be 'gaze_samples' or 'events'.")
    out = x.copy()
    rows = []
    k = 0

    if source == "gaze_samples":
        data = out["gaze_samples"]
        if data.empty or data["stimulus_id"].isna().all():
            raise ValueError("No stimulus ids are available in gaze samples.")
        for recording_id in pd.unique(data["recording_id"]):
            z = data[data["recording_id"].eq(recording_id) & data["stimulus_id"].notna()].copy()
            z = z.sort_values("timestamp_seconds", kind="mergesort")
            if z.empty:
                continue
            change = z["stimulus_id"].astype("string").ne(z["stimulus_id"].astype("string").shift())
            run = np.cumsum(change.fillna(True).to_numpy(dtype=bool))
            for _, q in z.groupby(run, sort=False):
                k += 1
                times = _finite_numeric(q["timestamp_seconds"])
                rows.append(
                    {
                        "interval_id": f"{recording_id}_stim_{k:05d}",
                        "recording_id": recording_id,
                        "interval_type": "stimulus",
                        "start_time": float(np.nanmin(times)),
                        "end_time": float(np.nanmax(times)),
                        "trial_id": _mode_value(q["trial_id"]),
                        "participant_id": _participant_for_recording(out, recording_id),
                        "item_id": pd.NA,
                        "stimulus_id": q["stimulus_id"].iloc[0],
                        "condition_id": pd.NA,
                        "parent_interval_id": pd.NA,
                        "valid_interval": True,
                    }
                )
    else:
        events = out["events"]
        event_type = events["event_type"].astype("string")
        event_name = events["event_name"].astype("string")
        ev = events[event_type.eq("media_change") | event_name.eq("MEDIA_START")]
        if ev.empty:
            raise ValueError("No stimulus/media events found.")
        for recording_id in pd.unique(ev["recording_id"]):
            z = ev[ev["recording_id"].eq(recording_id)].sort_values(
                "timestamp_seconds",
                kind="mergesort",
            )
            for position, (_, row) in enumerate(z.iterrows()):
                k += 1
                if position + 1 < len(z):
                    end_time = float(
                        pd.to_numeric(
                            pd.Series([z.iloc[position + 1]["timestamp_seconds"]]),
                            errors="coerce",
                        ).iloc[0]
                    )
                else:
                    final_times = list(
                        out["gaze_samples"].loc[
                            out["gaze_samples"]["recording_id"].eq(recording_id),
                            "timestamp_seconds",
                        ]
                    )
                    final_times.extend(
                        out["events"]
                        .loc[
                            out["events"]["recording_id"].eq(recording_id),
                            "timestamp_seconds",
                        ]
                        .tolist()
                    )
                    end_time = _safe_max_finite(final_times)
                start_time = float(
                    pd.to_numeric(
                        pd.Series([row["timestamp_seconds"]]),
                        errors="coerce",
                    ).iloc[0]
                )
                rows.append(
                    {
                        "interval_id": f"{recording_id}_stim_{k:05d}",
                        "recording_id": recording_id,
                        "interval_type": "stimulus",
                        "start_time": start_time,
                        "end_time": end_time,
                        "trial_id": row["trial_id"],
                        "participant_id": _participant_for_recording(out, recording_id),
                        "item_id": pd.NA,
                        "stimulus_id": row["event_value"],
                        "condition_id": pd.NA,
                        "parent_interval_id": pd.NA,
                        "valid_interval": bool(np.isfinite(end_time)),
                    }
                )

    built = standardize_eye_table(pd.DataFrame(rows), "intervals")
    if overwrite:
        out["intervals"] = out["intervals"][out["intervals"]["interval_type"].ne("stimulus")].reset_index(drop=True)
    out["intervals"] = standardize_eye_table(
        pd.concat([out["intervals"], built], ignore_index=True, sort=False),
        "intervals",
    )
    out = assign_trials(out)
    return add_provenance(
        out,
        "build_stimulus_intervals",
        "intervals",
        f"{len(built)} intervals",
    )


def _find_interval_id(time, recording, intervals, field):
    times = _finite_numeric(time)
    recording_values = pd.Series(recording).reset_index(drop=True)
    out = np.full(len(recording_values), None, dtype=object)
    if intervals.empty:
        return out
    for recording_id in pd.unique(recording_values):
        row_idx = np.flatnonzero(recording_values.eq(recording_id).to_numpy())
        ints = intervals[
            intervals["recording_id"].eq(recording_id)
            & np.isfinite(pd.to_numeric(intervals["start_time"], errors="coerce").to_numpy(float))
            & np.isfinite(pd.to_numeric(intervals["end_time"], errors="coerce").to_numpy(float))
        ]
        for _, interval in ints.iterrows():
            start = float(interval["start_time"])
            end = float(interval["end_time"])
            hit = row_idx[(times[row_idx] >= start) & (times[row_idx] <= end) & pd.isna(out[row_idx])]
            out[hit] = str(interval[field])
    return out


def assign_trials(x, interval_type="trial", overwrite=False):
    """Assign trial IDs to time-stamped tables from canonical intervals."""
    _assert_eye_dataset(x)
    out = x.copy()
    intervals = out["intervals"][out["intervals"]["interval_type"].eq(interval_type)]
    if intervals.empty:
        return out

    for component in ("gaze_samples", "eye_samples", "events", "biometrics"):
        data = out[component].copy()
        required = {"recording_id", "timestamp_seconds", "trial_id"}
        if data.empty or not required.issubset(data.columns):
            continue
        assigned = _find_interval_id(
            data["timestamp_seconds"],
            data["recording_id"],
            intervals,
            "trial_id",
        )
        if overwrite:
            data["trial_id"] = assigned
        else:
            missing = _string_missing(data["trial_id"]).to_numpy()
            data.loc[missing, "trial_id"] = assigned[missing]
        out[component] = data

    if not out["episodes"].empty:
        data = out["episodes"].copy()
        assigned = _find_interval_id(data["start_time"], data["recording_id"], intervals, "trial_id")
        if overwrite:
            data["trial_id"] = assigned
        else:
            missing = _string_missing(data["trial_id"]).to_numpy()
            data.loc[missing, "trial_id"] = assigned[missing]
        out["episodes"] = data

    return add_provenance(
        out,
        "assign_trials",
        "dataset",
        f"interval_type={interval_type};overwrite={str(bool(overwrite)).upper()}",
    )


def add_responses(x, responses, overwrite=False):
    """Add canonical response rows, optionally replacing matching response keys."""
    _assert_eye_dataset(x)
    if not isinstance(responses, pd.DataFrame):
        raise TypeError("`responses` must be a pandas DataFrame.")
    required = {"participant_id", "item_id", "response"}
    missing = sorted(required - set(responses.columns))
    if missing:
        raise ValueError(f"`responses` is missing required columns: {', '.join(missing)}.")

    out = x.copy()
    incoming = responses.copy()
    if "response_id" not in incoming:
        incoming["response_id"] = [f"response_{index:07d}" for index in range(1, len(incoming) + 1)]
    defaults = {
        "recording_id": pd.NA,
        "trial_id": pd.NA,
        "score": np.nan,
        "response_time": np.nan,
        "response_timestamp": np.nan,
        "response_type": "observed",
        "valid_response": True,
    }
    for column, default in defaults.items():
        if column not in incoming:
            incoming[column] = default
    incoming = standardize_eye_table(incoming, "responses")

    if overwrite and not incoming.empty:
        new_keys = set(
            zip(
                incoming["participant_id"].astype("string"),
                incoming["item_id"].astype("string"),
                incoming["trial_id"].astype("string"),
                strict=False,
            )
        )
        old = out["responses"]
        old_keys = list(
            zip(
                old["participant_id"].astype("string"),
                old["item_id"].astype("string"),
                old["trial_id"].astype("string"),
                strict=False,
            )
        )
        keep = [key not in new_keys for key in old_keys]
        out["responses"] = old.loc[keep].reset_index(drop=True)

    out["responses"] = standardize_eye_table(
        pd.concat([out["responses"], incoming], ignore_index=True, sort=False),
        "responses",
    )
    return add_provenance(out, "add_responses", "responses", f"{len(incoming)} responses")


def build_item_responses(x, score_key=None, response_type="observed"):
    """Create response rows from trial intervals when no responses exist."""
    _assert_eye_dataset(x)
    trials = x["intervals"][x["intervals"]["interval_type"].eq("trial")]
    if trials.empty:
        raise ValueError("Trial intervals are required.")
    if not x["responses"].empty:
        return x
    if score_key is not None and not isinstance(score_key, Mapping):
        raise ValueError("`score_key` must be named by item id.")

    rows = []
    for index, (_, trial) in enumerate(trials.iterrows(), start=1):
        rows.append(
            {
                "response_id": (f"{trial['recording_id']}_response_{index:05d}"),
                "recording_id": trial["recording_id"],
                "participant_id": trial["participant_id"],
                "trial_id": trial["trial_id"],
                "item_id": trial["item_id"],
                "response": pd.NA,
                "score": np.nan,
                "response_time": (
                    float(trial["end_time"]) - float(trial["start_time"])
                    if pd.notna(trial["end_time"]) and pd.notna(trial["start_time"])
                    else np.nan
                ),
                "response_timestamp": trial["end_time"],
                "response_type": response_type,
                "valid_response": True,
            }
        )
    return add_responses(x, pd.DataFrame(rows))


@dataclass(frozen=True)
class EyeAOI:
    """Python representation of the R ``eye_aoi`` list class."""

    definition: pd.DataFrame
    geometry: pd.DataFrame


def new_aoi(
    aoi_id,
    aoi_name=None,
    stimulus_id=pd.NA,
    shape="rectangle",
    x=np.nan,
    y=np.nan,
    width=np.nan,
    height=np.nan,
    polygon=None,
    coordinate_space_id="coord_display_normalized_top_left",
    valid_from=-np.inf,
    valid_to=np.inf,
    frame_id=pd.NA,
    visible=True,
    parent_aoi_id=pd.NA,
    source="user",
):
    """Construct one rectangle, circle, or polygon AOI."""
    if shape not in {"rectangle", "circle", "polygon"}:
        raise ValueError("`shape` must be 'rectangle', 'circle', or 'polygon'.")
    polygon_value = polygon
    if shape == "polygon":
        polygon_array = np.asarray(polygon, dtype=float)
        if polygon_array.ndim != 2 or polygon_array.shape[1] != 2:
            raise ValueError("Polygon AOIs require a two-column coordinate matrix.")
        polygon_value = polygon_array

    definition = pd.DataFrame(
        [
            {
                "aoi_id": str(aoi_id),
                "aoi_name": str(aoi_id if aoi_name is None else aoi_name),
                "stimulus_id": stimulus_id,
                "shape_type": shape,
                "coordinate_space_id": str(coordinate_space_id),
                "parent_aoi_id": parent_aoi_id,
                "source": str(source),
            }
        ]
    )
    geometry = pd.DataFrame(
        [
            {
                "aoi_id": str(aoi_id),
                "valid_from": float(valid_from),
                "valid_to": float(valid_to),
                "frame_id": frame_id,
                "x": float(x) if not pd.isna(x) else np.nan,
                "y": float(y) if not pd.isna(y) else np.nan,
                "width": float(width) if not pd.isna(width) else np.nan,
                "height": float(height) if not pd.isna(height) else np.nan,
                "polygon": polygon_value,
                "visible": bool(visible),
                "coordinate_space_id": str(coordinate_space_id),
            }
        ]
    )
    return EyeAOI(
        standardize_eye_table(definition, "aoi_definitions"),
        standardize_eye_table(geometry, "aoi_geometry"),
    )


def register_aois(x, *aois, overwrite=False):
    """Register one or more :class:`EyeAOI` definitions and geometries."""
    _assert_eye_dataset(x)
    if len(aois) == 1 and isinstance(aois[0], (list, tuple)):
        aois = tuple(aois[0])
    if not aois or not all(isinstance(aoi, EyeAOI) for aoi in aois):
        raise ValueError("Supply one or more `new_aoi()` objects.")

    out = x.copy()
    definitions = pd.concat([aoi.definition for aoi in aois], ignore_index=True, sort=False)
    geometries = pd.concat([aoi.geometry for aoi in aois], ignore_index=True, sort=False)
    ids = set(definitions["aoi_id"].astype(str))

    existing_ids = set(out["aoi_definitions"]["aoi_id"].dropna().astype(str))
    if overwrite:
        out["aoi_definitions"] = out["aoi_definitions"][~out["aoi_definitions"]["aoi_id"].astype(str).isin(ids)]
        out["aoi_geometry"] = out["aoi_geometry"][~out["aoi_geometry"]["aoi_id"].astype(str).isin(ids)]
    elif ids & existing_ids:
        raise ValueError("AOI id already exists; use `overwrite=True`.")

    out["aoi_definitions"] = standardize_eye_table(
        pd.concat(
            [out["aoi_definitions"], definitions],
            ignore_index=True,
            sort=False,
        ),
        "aoi_definitions",
    )
    out["aoi_geometry"] = standardize_eye_table(
        pd.concat(
            [out["aoi_geometry"], geometries],
            ignore_index=True,
            sort=False,
        ),
        "aoi_geometry",
    )
    return add_provenance(
        out,
        "register_aois",
        "aoi_definitions",
        ",".join(definitions["aoi_id"].astype(str)),
    )


def _point_in_polygon(px, py, polygon):
    x = np.asarray(px, dtype=float)
    y = np.asarray(py, dtype=float)
    if polygon is None:
        return np.zeros(len(x), dtype=bool)
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return np.zeros(len(x), dtype=bool)

    inside = np.zeros(len(x), dtype=bool)
    j = len(poly) - 1
    eps = np.finfo(float).eps
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        hit = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + eps) + xi)
        inside ^= hit
        j = i
    return inside


def _aoi_contains(x, y, time, definition, geometry):
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    tv = np.asarray(time, dtype=float)
    active = (tv >= float(geometry["valid_from"])) & (tv <= float(geometry["valid_to"])) & bool(geometry["visible"])
    if not np.any(active):
        return np.zeros(len(xv), dtype=bool)

    shape = definition["shape_type"]
    gx = float(geometry["x"]) if pd.notna(geometry["x"]) else np.nan
    gy = float(geometry["y"]) if pd.notna(geometry["y"]) else np.nan
    width = float(geometry["width"]) if pd.notna(geometry["width"]) else np.nan
    height = float(geometry["height"]) if pd.notna(geometry["height"]) else np.nan
    if shape == "rectangle":
        return active & (xv >= gx) & (xv <= gx + width) & (yv >= gy) & (yv <= gy + height)
    if shape == "circle":
        return active & (((xv - gx) ** 2 / (width / 2) ** 2) + ((yv - gy) ** 2 / (height / 2) ** 2) <= 1)
    if shape == "polygon":
        return active & _point_in_polygon(xv, yv, geometry["polygon"])
    return np.zeros(len(xv), dtype=bool)


def assign_aois(x, component="gaze_samples", overlap="first", overwrite=True):
    """Assign registered AOIs to gaze samples or episodes."""
    _assert_eye_dataset(x)
    if component not in {"gaze_samples", "episodes"}:
        raise ValueError("`component` must be 'gaze_samples' or 'episodes'.")
    if overlap not in {"first", "all", "smallest"}:
        raise ValueError("`overlap` must be 'first', 'all', or 'smallest'.")
    if x["aoi_definitions"].empty or x["aoi_geometry"].empty:
        raise ValueError("No AOIs are registered.")

    out = x.copy()
    definitions = out["aoi_definitions"]
    geometries = out["aoi_geometry"]

    if component == "gaze_samples":
        data = out["gaze_samples"].copy()
        if data.empty:
            return out
        assignments: list[list[str]] = [[] for _ in range(len(data))]
        gx = _finite_numeric(data["gaze_x"])
        gy = _finite_numeric(data["gaze_y"])
        gt = _finite_numeric(data["timestamp_seconds"])

        for _, definition in definitions.iterrows():
            selected = geometries[geometries["aoi_id"].eq(definition["aoi_id"])]
            for _, geometry in selected.iterrows():
                compatible = data["coordinate_space_id"].eq(geometry["coordinate_space_id"]).fillna(False)
                stimulus = definition["stimulus_id"]
                if pd.isna(stimulus) or not str(stimulus).strip():
                    stimulus_ok = pd.Series(True, index=data.index)
                else:
                    stimulus_ok = data["stimulus_id"].eq(stimulus).fillna(False)
                hit = compatible.to_numpy() & stimulus_ok.to_numpy() & _aoi_contains(gx, gy, gt, definition, geometry)
                for index in np.flatnonzero(hit):
                    assignments[index].append(str(definition["aoi_id"]))

        if overlap == "smallest":
            area_map = {}
            for aoi_id in definitions["aoi_id"].astype(str):
                geo = geometries[geometries["aoi_id"].astype(str).eq(aoi_id)]
                areas = pd.to_numeric(geo["width"], errors="coerce") * pd.to_numeric(geo["height"], errors="coerce")
                finite = areas[np.isfinite(areas)]
                area_map[aoi_id] = float(finite.min()) if len(finite) else np.inf
            assigned = [
                (min(values, key=lambda value: area_map.get(value, np.inf)) if values else pd.NA)
                for values in assignments
            ]
        elif overlap == "all":
            assigned = ["|".join(dict.fromkeys(values)) if values else pd.NA for values in assignments]
        else:
            assigned = [values[0] if values else pd.NA for values in assignments]

        if "aoi_id" not in data:
            data["aoi_id"] = pd.NA
        if overwrite:
            data["aoi_id"] = assigned
        else:
            missing = data["aoi_id"].isna()
            data.loc[missing, "aoi_id"] = np.asarray(assigned, dtype=object)[missing.to_numpy()]
        out["gaze_samples"] = data
    else:
        data = out["episodes"].copy()
        if data.empty:
            return out
        assigned = np.full(len(data), None, dtype=object)
        ex = _finite_numeric(data["centroid_x"])
        ey = _finite_numeric(data["centroid_y"])
        et = _finite_numeric(data["start_time"])

        for _, definition in definitions.iterrows():
            selected = geometries[geometries["aoi_id"].eq(definition["aoi_id"])]
            for _, geometry in selected.iterrows():
                compatible = data["coordinate_space_id"].eq(geometry["coordinate_space_id"]).fillna(False)
                hit = compatible.to_numpy() & _aoi_contains(ex, ey, et, definition, geometry)
                hit &= pd.isna(assigned)
                assigned[hit] = str(definition["aoi_id"])

        if overwrite:
            data["aoi_id"] = assigned
        else:
            missing = data["aoi_id"].isna().to_numpy()
            data.loc[missing, "aoi_id"] = assigned[missing]
        out["episodes"] = data

    return add_provenance(out, "assign_aois", component, f"overlap={overlap}")


def build_aoi_visits(
    x,
    gap_tolerance_ms=75,
    minimum_duration_ms=0,
    source="gaze_samples",
):
    """Aggregate contiguous AOI observations into canonical visit episodes."""
    _assert_eye_dataset(x)
    if source not in {"gaze_samples", "episodes"}:
        raise ValueError("`source` must be 'gaze_samples' or 'episodes'.")

    out = x.copy()
    visits = []
    counter = 0

    if source == "gaze_samples":
        data = out["gaze_samples"]
        if "aoi_id" not in data:
            raise ValueError("Assign AOIs to gaze samples first.")
        data = data[data["aoi_id"].notna()].sort_values(
            ["recording_id", "trial_id", "timestamp_seconds"],
            kind="mergesort",
        )
        groups = data.groupby(
            ["recording_id", "trial_id"],
            dropna=False,
            sort=False,
        )
        for _, group in groups:
            z = group.reset_index(drop=True)
            if z.empty:
                continue
            times = _finite_numeric(z["timestamp_seconds"])
            gaps = np.r_[np.inf, np.diff(times) * 1000]
            same = z["aoi_id"].astype("string").eq(z["aoi_id"].astype("string").shift()).fillna(False).to_numpy()
            new_run = (~same) | (gaps > float(gap_tolerance_ms))
            run = np.cumsum(new_run)
            for run_id in pd.unique(run):
                q = z[run == run_id]
                qtime = _finite_numeric(q["timestamp_seconds"])
                duration = (np.nanmax(qtime) - np.nanmin(qtime)) * 1000
                if duration < minimum_duration_ms:
                    continue
                counter += 1
                visits.append(
                    {
                        "episode_id": (f"{q['recording_id'].iloc[0]}_visit_{counter:07d}"),
                        "recording_id": q["recording_id"].iloc[0],
                        "episode_type": "aoi_visit",
                        "eye": "combined",
                        "start_time": float(np.nanmin(qtime)),
                        "end_time": float(np.nanmax(qtime)),
                        "duration_ms": float(duration),
                        "start_x": q["gaze_x"].iloc[0],
                        "start_y": q["gaze_y"].iloc[0],
                        "end_x": q["gaze_x"].iloc[-1],
                        "end_y": q["gaze_y"].iloc[-1],
                        "centroid_x": float(pd.to_numeric(q["gaze_x"], errors="coerce").mean()),
                        "centroid_y": float(pd.to_numeric(q["gaze_y"], errors="coerce").mean()),
                        "amplitude": np.nan,
                        "peak_velocity": np.nan,
                        "dispersion": np.nan,
                        "coordinate_space_id": q["coordinate_space_id"].iloc[0],
                        "source_algorithm": "AOI run-length encoding",
                        "source_parameters": (f"gap_tolerance_ms={gap_tolerance_ms}"),
                        "derived_by": "eyeprocess",
                        "trial_id": q["trial_id"].iloc[0],
                        "stimulus_id": q["stimulus_id"].iloc[0],
                        "aoi_id": q["aoi_id"].iloc[0],
                    }
                )
    else:
        data = out["episodes"]
        data = data[data["episode_type"].eq("fixation") & data["aoi_id"].notna()].sort_values(
            ["recording_id", "trial_id", "start_time"],
            kind="mergesort",
        )
        if data.empty:
            return out
        groups = data.groupby(
            ["recording_id", "trial_id"],
            dropna=False,
            sort=False,
        )
        for _, group in groups:
            z = group.reset_index(drop=True)
            same = z["aoi_id"].astype("string").eq(z["aoi_id"].astype("string").shift()).fillna(False).to_numpy()
            start = _finite_numeric(z["start_time"])
            end = _finite_numeric(z["end_time"])
            gaps = np.r_[np.inf, (start[1:] - end[:-1]) * 1000]
            new_run = (~same) | (gaps > float(gap_tolerance_ms))
            run = np.cumsum(new_run)
            for run_id in pd.unique(run):
                q = z[run == run_id]
                q_start = _finite_numeric(q["start_time"])
                q_end = _finite_numeric(q["end_time"])
                duration = (np.nanmax(q_end) - np.nanmin(q_start)) * 1000
                if duration < minimum_duration_ms:
                    continue
                counter += 1
                row = q.iloc[0].to_dict()
                row.update(
                    {
                        "episode_id": (f"{q['recording_id'].iloc[0]}_visit_{counter:07d}"),
                        "episode_type": "aoi_visit",
                        "start_time": float(np.nanmin(q_start)),
                        "end_time": float(np.nanmax(q_end)),
                        "duration_ms": float(duration),
                        "centroid_x": float(pd.to_numeric(q["centroid_x"], errors="coerce").mean()),
                        "centroid_y": float(pd.to_numeric(q["centroid_y"], errors="coerce").mean()),
                        "source_algorithm": "Fixation AOI aggregation",
                        "source_parameters": (f"gap_tolerance_ms={gap_tolerance_ms}"),
                        "derived_by": "eyeprocess",
                    }
                )
                visits.append(row)

    if visits:
        out["episodes"] = standardize_eye_table(
            pd.concat(
                [out["episodes"], pd.DataFrame(visits)],
                ignore_index=True,
                sort=False,
            ),
            "episodes",
        )
    return add_provenance(out, "build_aoi_visits", "episodes", f"{len(visits)} visits")


# R/011-quality-governance.R ----------------------------------------------


def _quality_row(
    recording_id,
    trial_id=pd.NA,
    stream_id=pd.NA,
    *,
    metric,
    value,
    threshold,
    status,
    message,
):
    return pd.DataFrame(
        [
            {
                "quality_id": _next_id("quality"),
                "recording_id": recording_id,
                "trial_id": trial_id,
                "stream_id": stream_id,
                "metric": metric,
                "value": float(value) if pd.notna(value) else np.nan,
                "threshold": (float(threshold) if pd.notna(threshold) else np.nan),
                "status": status,
                "message": message,
                "computed_at": _now_utc(),
            }
        ]
    )


def store_quality(x, report, replace_metric=False):
    """Store quality rows in the canonical quality table."""
    _assert_eye_dataset(x)
    if not isinstance(report, pd.DataFrame):
        raise TypeError("`report` must be a pandas DataFrame.")
    out = x.copy()
    incoming = report.copy()
    canonical = schema_table("quality")
    if not all(column in incoming.columns for column in canonical):
        incoming = standardize_eye_table(incoming, "quality")
    if replace_metric and not incoming.empty:
        metrics = set(incoming["metric"].dropna())
        out["quality"] = out["quality"][~out["quality"]["metric"].isin(metrics)]
    out["quality"] = standardize_eye_table(
        pd.concat([out["quality"], incoming], ignore_index=True, sort=False),
        "quality",
    )
    return add_provenance(out, "store_quality", "quality", f"{len(incoming)} rows")


def audit_sampling_rate(x, expected_hz=None, tolerance_hz=5, store=False):
    """Audit observed gaze sampling rate against expected rate."""
    _assert_eye_dataset(x)
    data = x["gaze_samples"]
    if data.empty:
        return pd.DataFrame()
    rows = []
    for recording_id, group in data.groupby("recording_id", dropna=False, sort=False):
        observed = estimate_sampling_rate(group["timestamp_seconds"])
        expected = expected_hz
        if expected is None:
            hit = x["recordings"].loc[
                x["recordings"]["recording_id"].eq(recording_id),
                "nominal_sampling_rate",
            ]
            expected = _first_nonmissing(hit, np.nan)
        expected_num = pd.to_numeric(pd.Series([expected]), errors="coerce").iloc[0]
        difference = abs(float(observed) - float(expected_num)) if np.isfinite(expected_num) else np.nan
        if not np.isfinite(expected_num):
            status = "unknown"
            message = "Expected sampling rate is unavailable."
        elif difference <= tolerance_hz:
            status = "ok"
            message = "Observed sampling rate is within tolerance."
        else:
            status = "warning"
            message = f"Observed rate differs from expected by {round(float(difference), 2)} Hz."
        stream_id = _first_nonmissing(group["stream_id"], pd.NA)
        rows.append(
            _quality_row(
                recording_id,
                stream_id=stream_id,
                metric="sampling_rate_hz",
                value=observed,
                threshold=expected_num,
                status=status,
                message=message,
            )
        )
    report = standardize_eye_table(pd.concat(rows, ignore_index=True), "quality")
    return store_quality(x, report, replace_metric=True) if store else report


def _nullable_valid_fraction(flag, finite_mask) -> float:
    valid = pd.Series(flag).astype("boolean")
    finite = pd.Series(np.asarray(finite_mask, dtype=bool), dtype="boolean")
    result = valid & finite
    return float(result.mean(skipna=True)) if result.notna().any() else np.nan


def audit_signal_quality(
    x,
    minimum_valid_gaze=0.80,
    minimum_valid_pupil=0.70,
    by_trial=True,
    store=False,
):
    """Audit valid gaze and pupil fractions."""
    _assert_eye_dataset(x)
    rows = []

    gaze = x["gaze_samples"]
    if not gaze.empty:
        keys = ["recording_id", "trial_id"] if by_trial and gaze["trial_id"].notna().any() else ["recording_id"]
        for _, group in gaze.groupby(keys, dropna=False, sort=False):
            gx = _finite_numeric(group["gaze_x"])
            gy = _finite_numeric(group["gaze_y"])
            fraction = _nullable_valid_fraction(group["valid"], np.isfinite(gx) & np.isfinite(gy))
            trial_id = group["trial_id"].iloc[0] if "trial_id" in keys else pd.NA
            rows.append(
                _quality_row(
                    group["recording_id"].iloc[0],
                    trial_id=trial_id,
                    stream_id=_first_nonmissing(group["stream_id"], pd.NA),
                    metric="valid_gaze_fraction",
                    value=fraction,
                    threshold=minimum_valid_gaze,
                    status=("ok" if np.isfinite(fraction) and fraction >= minimum_valid_gaze else "warning"),
                    message=f"{round(100 * fraction, 1)}% valid gaze samples.",
                )
            )

    eyes = x["eye_samples"]
    if not eyes.empty:
        keys = (
            ["recording_id", "trial_id", "eye"]
            if by_trial and eyes["trial_id"].notna().any()
            else ["recording_id", "eye"]
        )
        for _, group in eyes.groupby(keys, dropna=False, sort=False):
            pupil = _finite_numeric(group["pupil_diameter"])
            fraction = _nullable_valid_fraction(group["pupil_valid"], np.isfinite(pupil))
            eye = group["eye"].iloc[0]
            recording_id = group["recording_id"].iloc[0]
            trial_id = group["trial_id"].iloc[0] if "trial_id" in keys else pd.NA
            rows.append(
                _quality_row(
                    recording_id,
                    trial_id=trial_id,
                    stream_id=f"{recording_id}_pupil_{eye}",
                    metric=f"valid_pupil_fraction_{eye}",
                    value=fraction,
                    threshold=minimum_valid_pupil,
                    status=("ok" if np.isfinite(fraction) and fraction >= minimum_valid_pupil else "warning"),
                    message=(f"{round(100 * fraction, 1)}% valid pupil observations ({eye})."),
                )
            )

    report = (
        standardize_eye_table(pd.concat(rows, ignore_index=True), "quality") if rows else empty_eye_table("quality")
    )
    return store_quality(x, report, replace_metric=True) if store else report


def audit_pupil_quality(
    x,
    maximum_interpolated_fraction=0.20,
    plausible_range=None,
    store=False,
):
    """Audit interpolation burden and optional plausible pupil range."""
    _assert_eye_dataset(x)
    data = x["eye_samples"]
    if data.empty:
        return pd.DataFrame()
    rows = []
    for _, group in data.groupby(["recording_id", "eye"], dropna=False, sort=False):
        recording_id = group["recording_id"].iloc[0]
        eye = group["eye"].iloc[0]
        if "interpolated" in group:
            interpolated = pd.Series(group["interpolated"]).astype("boolean").mean(skipna=True)
            interpolated = float(interpolated) if pd.notna(interpolated) else np.nan
        else:
            interpolated = 0.0
        rows.append(
            _quality_row(
                recording_id,
                stream_id=f"{recording_id}_pupil_{eye}",
                metric=f"interpolated_pupil_fraction_{eye}",
                value=interpolated,
                threshold=maximum_interpolated_fraction,
                status=(
                    "ok" if np.isfinite(interpolated) and interpolated <= maximum_interpolated_fraction else "warning"
                ),
                message=(f"{round(100 * interpolated, 1)}% interpolated pupil observations."),
            )
        )
        if plausible_range is not None:
            pupil = pd.to_numeric(group["pupil_diameter"], errors="coerce")
            outside = (pupil < plausible_range[0]) | (pupil > plausible_range[1])
            fraction = float(outside.mean(skipna=True))
            rows.append(
                _quality_row(
                    recording_id,
                    stream_id=f"{recording_id}_pupil_{eye}",
                    metric=f"out_of_range_pupil_fraction_{eye}",
                    value=fraction,
                    threshold=0,
                    status="ok" if fraction == 0 else "warning",
                    message=(f"{round(100 * fraction, 1)}% outside declared plausible range."),
                )
            )

    report = standardize_eye_table(pd.concat(rows, ignore_index=True), "quality")
    return store_quality(x, report, replace_metric=True) if store else report


def audit_episodes(x, type=None):
    """Summarize episode counts and basic structural issues by episode type."""
    _assert_eye_dataset(x)
    data = x["episodes"]
    if type is not None:
        allowed = {type} if isinstance(type, str) else set(type)
        data = data[data["episode_type"].isin(allowed)]
    if data.empty:
        return pd.DataFrame()

    rows = []
    for episode_type in pd.unique(data["episode_type"]):
        group = data[data["episode_type"].eq(episode_type)]
        duration = pd.to_numeric(group["duration_ms"], errors="coerce")
        cx = pd.to_numeric(group["centroid_x"], errors="coerce")
        cy = pd.to_numeric(group["centroid_y"], errors="coerce")
        rows.append(
            {
                "episode_type": episode_type,
                "n": int(len(group)),
                "n_negative_duration": int((duration < 0).sum()),
                "n_missing_coordinates": int(
                    (~np.isfinite(cx.to_numpy(float)) | ~np.isfinite(cy.to_numpy(float))).sum()
                ),
                "vendor_derived": int(group["derived_by"].eq("vendor").sum()),
                "package_derived": int(group["derived_by"].eq("eyeprocess").sum()),
            }
        )
    return pd.DataFrame(rows)


def audit_event_order(x, event_type=None):
    """Audit event timestamp order within each recording."""
    _assert_eye_dataset(x)
    data = x["events"]
    if event_type is not None:
        allowed = {event_type} if isinstance(event_type, str) else set(event_type)
        data = data[data["event_type"].isin(allowed)]
    if data.empty:
        return pd.DataFrame()

    rows = []
    for recording_id, group in data.groupby("recording_id", dropna=False, sort=False):
        timestamps = _finite_numeric(group["timestamp_seconds"])
        finite = timestamps[np.isfinite(timestamps)]
        nonmonotonic = int((np.diff(finite) < 0).sum()) if finite.size > 1 else 0
        duplicate = int(pd.Series(finite).duplicated().sum())
        rows.append(
            {
                "recording_id": recording_id,
                "n_events": len(group),
                "n_nonmonotonic": nonmonotonic,
                "n_duplicate_timestamps": duplicate,
                "status": "warning" if nonmonotonic > 0 else "ok",
            }
        )
    return pd.DataFrame(rows)


def audit_trial_coverage(x):
    """Audit canonical trial intervals and attached observations."""
    _assert_eye_dataset(x)
    trials = x["intervals"][x["intervals"]["interval_type"].eq("trial")]
    if trials.empty:
        return pd.DataFrame([{"status": "error", "message": "No trials defined."}])

    rows = []
    for _, trial in trials.iterrows():
        recording_id = trial["recording_id"]
        trial_id = trial["trial_id"]

        def count_rows(component):
            data = x[component]
            if data.empty:
                return 0
            return int((data["recording_id"].eq(recording_id) & data["trial_id"].eq(trial_id)).sum())

        responses = x["responses"]
        has_response = bool((responses["recording_id"].eq(recording_id) & responses["trial_id"].eq(trial_id)).any())
        start = pd.to_numeric(pd.Series([trial["start_time"]]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([trial["end_time"]]), errors="coerce").iloc[0]
        duration = float(end - start) if np.isfinite(start) and np.isfinite(end) else np.nan
        valid = bool(trial["valid_interval"]) if pd.notna(trial["valid_interval"]) else False
        rows.append(
            {
                "recording_id": recording_id,
                "trial_id": trial_id,
                "duration_seconds": duration,
                "n_gaze_samples": count_rows("gaze_samples"),
                "n_eye_samples": count_rows("eye_samples"),
                "n_episodes": count_rows("episodes"),
                "has_response": has_response,
                "status": "ok" if valid and np.isfinite(duration) else "error",
            }
        )
    return pd.DataFrame(rows)


def audit_aois(x):
    """Audit registered AOI definitions against geometry and coordinate spaces."""
    _assert_eye_dataset(x)
    definitions = x["aoi_definitions"]
    geometry = x["aoi_geometry"]
    if definitions.empty:
        return pd.DataFrame([{"status": "unavailable", "message": "No AOIs registered."}])

    registered = set(x["coordinate_spaces"]["coordinate_space_id"].dropna())
    rows = []
    for _, definition in definitions.iterrows():
        aoi_id = definition["aoi_id"]
        has_geometry = bool(geometry["aoi_id"].eq(aoi_id).any())
        coordinate_registered = definition["coordinate_space_id"] in registered
        rows.append(
            {
                "aoi_id": aoi_id,
                "aoi_name": definition["aoi_name"],
                "shape_type": definition["shape_type"],
                "has_geometry": has_geometry,
                "n_geometry_records": int(geometry["aoi_id"].eq(aoi_id).sum()),
                "coordinate_registered": coordinate_registered,
                "status": ("ok" if has_geometry and coordinate_registered else "error"),
            }
        )
    return pd.DataFrame(rows)


def audit_missingness(
    x,
    component="gaze_samples",
    by="recording_id",
):
    """Report missing/non-finite fractions by field and grouping key."""
    _assert_eye_dataset(x)
    if component not in {"gaze_samples", "eye_samples", "biometrics"}:
        raise ValueError("Invalid `component`.")
    data = x[component]
    if data.empty:
        return pd.DataFrame()

    requested = [by] if isinstance(by, str) else list(by)
    keys = [column for column in requested if column in data.columns]
    groups = data.groupby(keys, dropna=False, sort=False) if keys else [(None, data)]

    rows = []
    for _, group in groups:
        if keys:
            label = "|".join(str(group.iloc[0][key]) for key in keys)
        else:
            label = "all"
        for column in group.columns:
            series = group[column]
            missing = series.isna().to_numpy()
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").to_numpy(float)
                missing = missing | ~np.isfinite(numeric)
            rows.append(
                {
                    "component": component,
                    "group": label,
                    "field": column,
                    "missing_fraction": float(np.mean(missing)),
                }
            )
    return pd.DataFrame(rows)


def check_process_leakage(x, response_time_tolerance=0):
    """Flag feature windows extending beyond their response timestamp."""
    _assert_eye_dataset(x)
    features = x["features"]
    responses = x["responses"]
    if features.empty or responses.empty:
        return pd.DataFrame()

    response_map = {}
    for _, response in responses.iterrows():
        key = (str(response["recording_id"]), str(response["trial_id"]))
        response_map.setdefault(key, response["response_timestamp"])

    rows = []
    for _, feature in features.iterrows():
        key = (str(feature["recording_id"]), str(feature["trial_id"]))
        response_ts = response_map.get(key, np.nan)
        window_end = pd.to_numeric(pd.Series([feature["window_end"]]), errors="coerce").iloc[0]
        response_num = pd.to_numeric(pd.Series([response_ts]), errors="coerce").iloc[0]
        leaked = bool(
            np.isfinite(window_end)
            and np.isfinite(response_num)
            and window_end > response_num + response_time_tolerance
        )
        rows.append(
            {
                "feature_id": feature["feature_id"],
                "feature_name": feature["feature_name"],
                "response_timestamp": response_num,
                "feature_window_end": window_end,
                "post_response": leaked,
                "status": "error" if leaked else "ok",
                "message": (
                    "Feature window extends beyond response time."
                    if leaked
                    else "No post-response leakage detected from recorded windows."
                ),
            }
        )
    return pd.DataFrame(rows)


def check_feature_level(x):
    """Check required identifying keys for each feature aggregation level."""
    _assert_eye_dataset(x)
    features = x["features"]
    if features.empty:
        return pd.DataFrame()
    expected = {
        "trial": ("recording_id", "trial_id"),
        "trial_aoi": ("recording_id", "trial_id", "aoi_id"),
        "trial_eye": ("recording_id", "trial_id"),
        "response": ("participant_id", "item_id"),
    }
    rows = []
    for _, feature in features.iterrows():
        required = expected.get(feature["level"], ())
        missing = []
        for column in required:
            value = feature[column]
            if pd.isna(value) or not str(value).strip():
                missing.append(column)
        rows.append(
            {
                "feature_id": feature["feature_id"],
                "feature_name": feature["feature_name"],
                "level": feature["level"],
                "missing_keys": ",".join(missing),
                "status": "warning" if missing else "ok",
            }
        )
    return pd.DataFrame(rows)


def interpretive_warnings():
    """Return the frozen R interpretation-guardrail table."""
    observations = [
        "fixation",
        "dwell time",
        "pupil dilation",
        "rapid response",
        "EDA/heart rate",
        "latent process factor",
        "gaze-derived class",
    ]
    prohibited = [
        "attention",
        "difficulty",
        "cognitive load",
        "guessing",
        "specific emotion or diagnosis",
        "effort/engagement/arousal",
        "cognitive strategy",
    ]
    guidance = [
        "Interpret fixation within task, stimulus, and measurement context.",
        ("Longer dwell may reflect difficulty, interest, confusion, rereading, or design properties."),
        ("Control luminance, baseline, blink handling, timing, and alternative arousal explanations."),
        "Use task-specific evidence and model speed-accuracy relations explicitly.",
        ("Physiological signals are nonspecific and require validated context-sensitive interpretation."),
        "Name factors neutrally until construct validity is demonstrated.",
        ("Validate classes externally and assess stability and preprocessing dependence."),
    ]
    return pd.DataFrame(
        {
            "warning_id": [f"interpretation_{i:02d}" for i in range(1, 8)],
            "observation": observations,
            "prohibited_automatic_interpretation": prohibited,
            "guidance": guidance,
        }
    )


def analysis_readiness(x):
    """Summarize readiness across schema, time, coordinates, trials, and quality."""
    _assert_eye_dataset(x)
    validation = validate_eye_dataset(x)
    gaze_quality = audit_signal_quality(x)
    coordinate = audit_coordinate_spaces(x)
    time = audit_timebase(x)
    trials = audit_trial_coverage(x)

    validation_has_error = bool(validation["severity"].eq("error").any()) if not validation.empty else False
    time_warning = bool(time["status"].eq("warning").any()) if not time.empty and "status" in time else False
    coordinate_ready = True if coordinate.empty else bool(coordinate["registered"].fillna(False).all())
    trials_ready = bool(not trials.empty and not trials["status"].eq("error").all()) if "status" in trials else False
    quality_ready = True if gaze_quality.empty else not bool(gaze_quality["status"].eq("warning").all())

    return pd.DataFrame(
        {
            "domain": [
                "schema",
                "recordings",
                "timestamps",
                "coordinates",
                "trials",
                "responses",
                "gaze_quality",
                "provenance",
            ],
            "ready": [
                not validation_has_error,
                len(x["recordings"]) > 0,
                len(time) > 0 and not time_warning,
                coordinate_ready,
                trials_ready,
                len(x["responses"]) > 0,
                quality_ready,
                len(x["provenance"]) > 0,
            ],
            "message": [
                f"{len(validation)} validation issue(s).",
                f"{len(x['recordings'])} recording(s).",
                f"{len(time)} timebase audit row(s).",
                f"{len(coordinate)} coordinate-space use row(s).",
                f"{len(trials)} trial row(s).",
                f"{len(x['responses'])} response row(s).",
                f"{len(gaze_quality)} signal-quality row(s).",
                f"{len(x['provenance'])} provenance action(s).",
            ],
        }
    )


def _normalise_dataset_inputs(xs):
    if len(xs) == 1 and isinstance(xs[0], Mapping) and not is_eye_dataset(xs[0]):
        return list(xs[0].items())
    if len(xs) == 1 and isinstance(xs[0], (list, tuple)) and not is_eye_dataset(xs[0]):
        return [(f"pipeline_{i}", value) for i, value in enumerate(xs[0], start=1)]
    return [(f"pipeline_{i}", value) for i, value in enumerate(xs, start=1)]


def compare_preprocessing(
    *xs,
    metrics=("valid_gaze_fraction", "valid_pupil_fraction", "fixation_count"),
):
    """Compare basic quality/feature summaries across preprocessing pipelines."""
    del metrics
    inputs = _normalise_dataset_inputs(xs)
    rows = []
    for label, dataset in inputs:
        if not is_eye_dataset(dataset):
            raise TypeError("All inputs must be `eye_dataset` objects.")
        quality = audit_signal_quality(dataset)
        gaze_values = quality.loc[quality["metric"].eq("valid_gaze_fraction"), "value"]
        pupil_values = quality.loc[
            quality["metric"].astype("string").str.contains("valid_pupil_fraction", na=False),
            "value",
        ]
        rows.append(
            {
                "pipeline": label,
                "valid_gaze_fraction": (
                    float(pd.to_numeric(gaze_values, errors="coerce").mean()) if len(gaze_values) else np.nan
                ),
                "valid_pupil_fraction": (
                    float(pd.to_numeric(pupil_values, errors="coerce").mean()) if len(pupil_values) else np.nan
                ),
                "fixation_count": int(dataset["episodes"]["episode_type"].eq("fixation").sum()),
                "feature_rows": len(dataset["features"]),
            }
        )
    return pd.DataFrame(rows)


def compare_aoi_definitions(*xs, source="samples"):
    """Compare AOI assignment counts across alternative definition sets."""
    inputs = _normalise_dataset_inputs(xs)
    rows = []
    for label, dataset in inputs:
        _assert_eye_dataset(dataset)
        data = dataset["gaze_samples"] if source == "samples" else dataset["episodes"]
        if "aoi_id" not in data:
            continue
        counts = data["aoi_id"].value_counts(dropna=False, sort=False)
        for aoi_id, count in counts.items():
            rows.append(
                {
                    "aoi_id": aoi_id,
                    "count": int(count),
                    "definition_set": label.replace("pipeline_", "set_"),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_process(*xs, label=None):
    """Return the frozen preprocessing-sensitivity summary contract."""
    summary = compare_preprocessing(*xs)
    if label is not None and len(label) == len(summary):
        summary = summary.copy()
        summary["pipeline"] = list(label)
    return {"summary": summary, "compared_at": _now_utc()}


__all__ = [
    "EyeAOI",
    "as_eye_dataset",
    "convert_xy",
    "synchronize_eye_biometrics",
    "audit_clock_sync",
    "build_trials",
    "build_stimulus_intervals",
    "assign_trials",
    "add_responses",
    "build_item_responses",
    "new_aoi",
    "register_aois",
    "assign_aois",
    "build_aoi_visits",
    "store_quality",
    "audit_sampling_rate",
    "audit_signal_quality",
    "audit_pupil_quality",
    "audit_episodes",
    "audit_event_order",
    "audit_trial_coverage",
    "audit_aois",
    "audit_missingness",
    "check_process_leakage",
    "check_feature_level",
    "interpretive_warnings",
    "analysis_readiness",
    "compare_preprocessing",
    "compare_aoi_definitions",
    "sensitivity_process",
]
