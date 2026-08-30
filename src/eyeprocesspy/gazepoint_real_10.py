"""Gazepoint real-export closure for frozen eyeprocess 0.11.1.

This module closes the remaining public Gazepoint contracts catalogued from
R/005-import-gazepoint.R and R/018-gazepoint-real-exports.R.  Existing
sample/fixation/folder/biometric readers remain in :mod:`eyeprocesspy.gazepoint`;
this module adds the frozen reconstruction/QC aliases plus Data Summary and
AOI-statistics import semantics.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import add_provenance, new_eye_dataset, validate_eye_dataset
from .foundation_09 import (
    assign_trials,
    audit_clock_sync,
    audit_episodes,
    audit_event_order,
    audit_pupil_quality,
    audit_sampling_rate,
    audit_signal_quality,
    build_stimulus_intervals,
    build_trials,
)
from .gazepoint import (
    _gp_filename_identity,
    _gp_id_token,
    _gp_is_summary_report,
    gp_parse_user_events,
)
from .importers import _as_character_id, _first_existing, _read_delimited, _safe_numeric
from .schema import new_coordinate_space, standardize_eye_table

__all__ = [
    "gp_align_media_ids",
    "gp_check_biometrics_sync",
    "gp_check_fixation_ids",
    "gp_check_media_timing",
    "gp_check_pupil_channels",
    "gp_check_sampling_rate",
    "gp_check_validity_fields",
    "gp_parse_markers",
    "gp_reconstruct_stimuli",
    "gp_reconstruct_trials",
    "read_gazepoint_aoi_statistics",
    "read_gazepoint_summary",
]


class GazepointSummary(dict):
    """Python list-like counterpart of the R ``gazepoint_summary`` object."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        return (
            "<gazepoint_summary "
            f"software={self.get('software')!r} "
            f"version={self.get('software_version')!r}; "
            f"aoi_summary={len(self.get('aoi_summary', []))}; "
            f"aoi_statistics={len(self.get('aoi_statistics', []))}>"
        )


def _parse_csv_block(lines: list[str], title: str) -> pd.DataFrame:
    starts = [i for i, line in enumerate(lines) if line.strip() == title]
    if not starts:
        return pd.DataFrame()
    header_index = starts[0] + 1
    if header_index >= len(lines) or not lines[header_index].strip():
        return pd.DataFrame()

    block = [lines[header_index]]
    cursor = header_index + 1
    while cursor < len(lines) and lines[cursor].strip():
        block.append(lines[cursor])
        cursor += 1

    try:
        out = pd.read_csv(io.StringIO("\n".join(block)))
    except Exception:
        reader = list(csv.reader(block))
        if not reader:
            return pd.DataFrame()
        width = len(reader[0])
        rows = [row[:width] + [""] * max(0, width - len(row)) for row in reader[1:]]
        out = pd.DataFrame(rows, columns=reader[0])

    for column in out.select_dtypes(include="object").columns:
        out[column] = out[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return out


def _column(data: pd.DataFrame, *candidates: str) -> str | None:
    return _first_existing(list(map(str, data.columns)), list(candidates))


def _character_column(
    data: pd.DataFrame,
    *candidates: str,
    default: Any = pd.NA,
) -> pd.Series:
    column = _column(data, *candidates)
    if column is None:
        return pd.Series([default] * len(data), dtype="string")
    return data[column].astype("string")


def _numeric_column(data: pd.DataFrame, *candidates: str) -> pd.Series:
    column = _column(data, *candidates)
    if column is None:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return _safe_numeric(data[column])


def _gp_aoi_id(media: Any, aoi: Any) -> pd.Series:
    media_series = pd.Series(media, dtype="string")
    aoi_series = pd.Series(aoi, dtype="string")
    out: list[Any] = []
    for media_value, aoi_value in zip(media_series, aoi_series, strict=False):
        if pd.isna(aoi_value) or not str(aoi_value).strip():
            out.append(pd.NA)
            continue
        label = re_sub_aoi_prefix(str(aoi_value))
        out.append(f"media_{_gp_id_token(media_value)}_aoi_{_gp_id_token(label)}")
    return pd.Series(out, dtype="string")


def re_sub_aoi_prefix(value: str) -> str:
    text = value.strip()
    upper = text.upper()
    if upper.startswith("AOI"):
        text = text[3:].strip()
    return text


def _recording_frame(
    participant: Any,
    recording: Any,
    session_id: str,
    *,
    source_path: str,
    software_version: Any = pd.NA,
) -> pd.DataFrame:
    participants = pd.Series(participant, dtype="string")
    recordings = pd.Series(recording, dtype="string")
    if len(participants) != len(recordings):
        raise ValueError("Participant and recording vectors must have equal length.")
    frame = pd.DataFrame(
        {
            "recording_id": recordings,
            "participant_id": participants,
            "session_id": str(session_id),
            "vendor": "Gazepoint",
            "vendor_family": "Gazepoint",
            "device_model": "Gazepoint",
            "firmware_version": pd.NA,
            "software_name": "Gazepoint Analysis",
            "software_version": software_version,
            "experiment_type": pd.NA,
            "nominal_sampling_rate": 60.0,
            "screen_width_px": np.nan,
            "screen_height_px": np.nan,
            "recording_start": pd.NA,
            "source_timezone": pd.NA,
            "source_file_set": str(Path(source_path).expanduser().resolve()),
        }
    )
    return frame.drop_duplicates(subset=["recording_id"]).reset_index(drop=True)


def read_gazepoint_summary(path: str | Path) -> GazepointSummary:
    """Parse a Gazepoint Analysis Data Summary export.

    Ports frozen R ``read_gazepoint_summary()``: the first two metadata lines
    are retained, and the ``AOI Summary`` and ``AOI Statistics (for each user)``
    CSV blocks are returned separately without collapsing vendor columns.
    """
    source = Path(path).expanduser()
    if not _gp_is_summary_report(source):
        raise ValueError(f"Not a recognized Gazepoint Data Summary export: {path}")

    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    first = next(csv.reader([lines[0]]), []) if lines else []
    software = first[0].strip() if first else pd.NA
    software_version = first[1].strip() if len(first) >= 2 else pd.NA

    processed_on: Any = pd.NA
    if len(lines) >= 2:
        second = next(csv.reader([lines[1]]), [])
        if len(second) >= 2:
            processed_on = ",".join(second[1:]).strip()

    notes = [line for line in lines if line.strip().startswith("Note:")]
    return GazepointSummary(
        path=str(source.resolve()),
        software=software,
        software_version=software_version,
        processed_on=processed_on,
        aoi_summary=_parse_csv_block(lines, "AOI Summary"),
        aoi_statistics=_parse_csv_block(lines, "AOI Statistics (for each user)"),
        notes=notes,
    )


_SUMMARY_METRICS: dict[str, tuple[tuple[str, ...], str]] = {
    "time_to_first_view": (
        ("Time to 1st View (sec) -1.0 means not viewed",),
        "seconds",
    ),
    "time_viewed": (("Time Viewed (sec)",), "seconds"),
    "time_viewed_percent": (("Time Viewed (%)",), "percent"),
    "fixation_count": (("Fixations (#)",), "count"),
    "revisit_count": (("Revisits (#)",), "count"),
    "click_count": (("Clicks (#)",), "count"),
    "mean_dial": (("Ave Dial (0-1)",), "proportion"),
    "mean_gsr_vendor_reported": (
        ("Ave GSR (kOhm)",),
        "vendor_reported_kOhm",
    ),
    "mean_heart_rate": (("Ave Heart Rate (BPM)",), "beats_per_minute"),
    "mean_interbeat_interval": (
        ("Ave Interbeat Interval (s)",),
        "seconds",
    ),
}


def _summary_features(
    summary: GazepointSummary,
    session_id: str = "S001",
) -> pd.DataFrame:
    data = summary.aoi_statistics
    if data.empty:
        return standardize_eye_table(pd.DataFrame(), "features")

    media = _character_column(data, "Media ID")
    aoi_source = _character_column(data, "AOI ID", "AOI Name")
    aoi_id = _gp_aoi_id(media, aoi_source)
    participant = _character_column(data, "User Name", "User ID")
    user_id = _character_column(data, "User ID")
    missing_participant = participant.isna() | participant.str.strip().eq("")
    participant = participant.copy()
    participant.loc[missing_participant] = "User " + user_id.loc[missing_participant].fillna("unknown")

    recording = participant.map(lambda value: f"rec_{_gp_id_token(value)}_{_gp_id_token(session_id)}")
    window_start = _numeric_column(data, "AOI Start", "AOI Start (sec)")
    duration = _numeric_column(data, "AOI Duration (sec - U=UserControlled)")
    window_end = window_start + duration

    method = f"Gazepoint Analysis {summary.software_version} Data Summary"
    parameters = f"processed_on={summary.processed_on}"
    frames: list[pd.DataFrame] = []

    for feature_name, (candidates, unit) in _SUMMARY_METRICS.items():
        column = _column(data, *candidates)
        if column is None:
            continue
        value = _safe_numeric(data[column])
        if feature_name == "time_to_first_view":
            value = value.mask(value < 0)
        feature_id = [
            "feature_"
            + "_".join(
                [
                    _gp_id_token(person),
                    _gp_id_token(stimulus),
                    _gp_id_token(aoi),
                    _gp_id_token(feature_name),
                    _gp_id_token(summary.processed_on),
                ]
            )
            for person, stimulus, aoi in zip(
                participant,
                media,
                aoi_source,
                strict=False,
            )
        ]
        if len(set(feature_id)) != len(feature_id):
            feature_id = [f"{value}_row{index:05d}" for index, value in enumerate(feature_id, start=1)]
        frames.append(
            pd.DataFrame(
                {
                    "feature_id": feature_id,
                    "recording_id": recording,
                    "participant_id": participant,
                    "trial_id": pd.NA,
                    "item_id": pd.NA,
                    "stimulus_id": media,
                    "aoi_id": aoi_id,
                    "feature_name": feature_name,
                    "value": value,
                    "unit": unit,
                    "level": "participant_aoi_summary",
                    "window_start": window_start,
                    "window_end": window_end,
                    "observed_fraction": np.nan,
                    "method": method,
                    "parameters": parameters,
                    "derived_at": summary.processed_on,
                }
            )
        )

    ttff_col = _column(
        data,
        "Time to 1st View (sec) -1.0 means not viewed",
    )
    if ttff_col is not None:
        viewed = (_safe_numeric(data[ttff_col]) >= 0).astype(float)
        feature_id = [
            "feature_"
            + "_".join(
                [
                    _gp_id_token(person),
                    _gp_id_token(stimulus),
                    _gp_id_token(aoi),
                    "aoi_viewed",
                    _gp_id_token(summary.processed_on),
                ]
            )
            for person, stimulus, aoi in zip(
                participant,
                media,
                aoi_source,
                strict=False,
            )
        ]
        if len(set(feature_id)) != len(feature_id):
            feature_id = [f"{value}_row{index:05d}" for index, value in enumerate(feature_id, start=1)]
        frames.append(
            pd.DataFrame(
                {
                    "feature_id": feature_id,
                    "recording_id": recording,
                    "participant_id": participant,
                    "trial_id": pd.NA,
                    "item_id": pd.NA,
                    "stimulus_id": media,
                    "aoi_id": aoi_id,
                    "feature_name": "aoi_viewed",
                    "value": viewed,
                    "unit": "binary",
                    "level": "participant_aoi_summary",
                    "window_start": window_start,
                    "window_end": window_end,
                    "observed_fraction": np.nan,
                    "method": method,
                    "parameters": parameters,
                    "derived_at": summary.processed_on,
                }
            )
        )

    if not frames:
        return standardize_eye_table(pd.DataFrame(), "features")
    return standardize_eye_table(
        pd.concat(frames, ignore_index=True, sort=False),
        "features",
    )


def read_gazepoint_aoi_statistics(
    path: str | Path,
    participant_id: str | None = None,
    recording_id: str | None = None,
    session_id: str = "S001",
    keep_raw: bool = True,
    quiet: bool = False,
    **kwargs: Any,
):
    """Import Gazepoint AOI statistics or a Gazepoint Data Summary export."""
    source_path = Path(path).expanduser()

    if not _gp_is_summary_report(source_path):
        data = _read_delimited(source_path, **kwargs)
        identity = _gp_filename_identity(
            source_path,
            participant_id=participant_id,
            recording_id=recording_id,
            session_id=session_id,
        )
        aoi_column = _column(data, "AOI_ID", "AOI", "AOI Name", "aoi_id")
        if aoi_column is None:
            raise ValueError("Cannot identify an AOI identifier column.")
        name_column = _column(data, "AOI_NAME", "AOI", "AOI Name", "aoi_name") or aoi_column
        stimulus_column = _column(
            data,
            "MEDIA_ID",
            "MEDIA_NAME",
            "stimulus_id",
        )
        stimulus = (
            pd.Series(pd.NA, index=data.index, dtype="string")
            if stimulus_column is None
            else _as_character_id(data[stimulus_column])
        )
        definitions = pd.DataFrame(
            {
                "aoi_id": _gp_aoi_id(stimulus, data[aoi_column]),
                "aoi_name": data[name_column].astype("string"),
                "stimulus_id": stimulus,
                "shape_type": "unknown",
                "coordinate_space_id": "coord_display_normalized_top_left",
                "parent_aoi_id": pd.NA,
                "source": "Gazepoint AOI statistics",
            }
        )
        definitions = definitions.loc[definitions["aoi_id"].notna() & ~definitions["aoi_id"].duplicated()].reset_index(
            drop=True
        )
        recordings = _recording_frame(
            [identity["participant_id"]],
            [identity["recording_id"]],
            identity["session_id"],
            source_path=str(source_path),
        )
        out = new_eye_dataset(
            recordings=recordings,
            aoi_definitions=definitions,
            coordinate_spaces=new_coordinate_space(
                "coord_display_normalized_top_left",
                "display_normalized_top_left",
            ),
            raw={"gazepoint_aoi_statistics": data.copy()} if keep_raw else {},
            vendor_metadata={"gazepoint_aoi_statistics": {"source_columns": list(map(str, data.columns))}},
            validate=False,
        )
        out = add_provenance(
            out,
            "import_gazepoint_aoi_statistics",
            "aoi_definitions",
            f"{len(definitions)} AOIs",
            source_files=str(source_path),
        )
        out.validation = validate_eye_dataset(out)
        return out

    summary = read_gazepoint_summary(source_path)
    data = summary.aoi_statistics
    summary_table = summary.aoi_summary
    definition_source = data if not data.empty else summary_table

    if definition_source.empty:
        definitions = standardize_eye_table(pd.DataFrame(), "aoi_definitions")
    else:
        media = _character_column(definition_source, "Media ID")
        aoi_source = _character_column(
            definition_source,
            "AOI ID",
            "AOI Name",
        )
        aoi_name = _character_column(
            definition_source,
            "AOI Name",
            "AOI ID",
        )
        definitions = pd.DataFrame(
            {
                "aoi_id": _gp_aoi_id(media, aoi_source),
                "aoi_name": aoi_name,
                "stimulus_id": media,
                "shape_type": "unknown",
                "coordinate_space_id": "coord_display_normalized_top_left",
                "parent_aoi_id": pd.NA,
                "source": (f"Gazepoint Analysis {summary.software_version} Data Summary"),
            }
        )
        definitions = definitions.loc[definitions["aoi_id"].notna() & ~definitions["aoi_id"].duplicated()].reset_index(
            drop=True
        )

    if not data.empty:
        user_name = _character_column(data, "User Name", "User ID")
        user_id = _character_column(data, "User ID")
        missing = user_name.isna() | user_name.str.strip().eq("")
        user_name = user_name.copy()
        user_name.loc[missing] = "User " + user_id.loc[missing].fillna("unknown")
        users = list(pd.unique(user_name.dropna()))
    else:
        users = [participant_id or "P001"]

    participants = [participant_id] if participant_id is not None and len(users) == 1 else users
    recordings_ids = (
        [recording_id]
        if recording_id is not None and len(users) == 1
        else [f"rec_{_gp_id_token(user)}_{_gp_id_token(session_id)}" for user in users]
    )
    recordings = _recording_frame(
        participants,
        recordings_ids,
        session_id,
        source_path=str(source_path),
        software_version=summary.software_version,
    )

    features = _summary_features(summary, session_id=session_id)
    if participant_id is not None and len(users) == 1 and not features.empty:
        features = features.copy()
        features["participant_id"] = participant_id
        features["recording_id"] = recordings_ids[0]
    elif recording_id is not None and len(users) == 1 and not features.empty:
        features = features.copy()
        features["recording_id"] = recording_id

    out = new_eye_dataset(
        recordings=recordings,
        aoi_definitions=definitions,
        features=features,
        coordinate_spaces=new_coordinate_space(
            "coord_display_normalized_top_left",
            "display_normalized_top_left",
        ),
        raw={
            "gazepoint_data_summary": {
                "aoi_summary": summary_table.copy(),
                "aoi_statistics": data.copy(),
            }
        }
        if keep_raw
        else {},
        vendor_metadata={"gazepoint_data_summary": dict(summary)},
        validate=False,
    )
    out = add_provenance(
        out,
        "import_gazepoint_data_summary",
        "aoi_definitions|features",
        f"{len(definitions)} AOIs; {len(features)} summary features",
        source_files=str(source_path),
    )
    out.validation = validate_eye_dataset(out)
    return out


# Frozen R/005 public compatibility contracts -----------------------------


def gp_reconstruct_trials(x, **kwargs: Any):
    return build_trials(x, **kwargs)


def gp_reconstruct_stimuli(x, **kwargs: Any):
    return build_stimulus_intervals(x, **kwargs)


def gp_align_media_ids(x):
    return assign_trials(x)


def gp_parse_markers(x):
    return gp_parse_user_events(x)


def gp_check_sampling_rate(x, **kwargs: Any):
    return audit_sampling_rate(x, expected_hz=60, **kwargs)


def gp_check_validity_fields(x, **kwargs: Any):
    return audit_signal_quality(x, **kwargs)


def gp_check_fixation_ids(x):
    return audit_episodes(x, type="fixation")


def gp_check_media_timing(x):
    return audit_event_order(x, event_type="media_change")


def gp_check_pupil_channels(x, **kwargs: Any):
    return audit_pupil_quality(x, **kwargs)


def gp_check_biometrics_sync(x, **kwargs: Any):
    return audit_clock_sync(x, **kwargs)
