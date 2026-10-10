#!/usr/bin/env python3
"""Map one SRL participant × learning-slide × eye branch into EyeDataset.

Research-only adapter for:
Juřík et al. (2025), Experimental Dataset on Eye-tracking Activity During
Self-Regulated Learning, Figshare DOI 10.6084/m9.figshare.28304069.

The adapter performs structural mapping only. It does not:
- average the two eyes;
- use released AOI labels to reconstruct geometry;
- interpolate or smooth gaze;
- choose an event detector;
- apply the historical 80% tracking-ratio exclusion;
- repair out-of-bounds or non-monotonic samples.

Those are separate, declared analysis decisions.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import eyeprocesspy as ep

SCREEN_WIDTH_PX = 1600
SCREEN_HEIGHT_PX = 900
NOMINAL_SAMPLING_RATE_HZ = 250.0

COMMON_REQUIRED = {
    "RecordingTime [ms]",
    "Trial",
    "Stimulus",
    "Participant",
    "Tracking Ratio [%]",
}
EYE_FIELDS = {
    "left": {
        "x": "Point of Regard Left X [px]",
        "y": "Point of Regard Left Y [px]",
        "aoi": "AOI Name Left",
        "pupil": "Pupil Diameter Left [mm]",
    },
    "right": {
        "x": "Point of Regard Right X [px]",
        "y": "Point of Regard Right Y [px]",
        "aoi": "AOI Name Right",
        "pupil": "Pupil Diameter Right [mm]",
    },
}
PARTICIPANT_REQUIRED = {"part_ID", "experiment_condition"}
STIMULI_REQUIRED = {
    "part_ID",
    "stimulus_name",
    "stimulus_type",
    "stimulus_time",
    "tracking_ratio",
}


@lru_cache(maxsize=4)
def _read_table_cached(
    resolved_path: str,
    mtime_ns: int,
    size_bytes: int,
) -> pd.DataFrame:
    del mtime_ns, size_bytes
    path = Path(resolved_path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=None, engine="python")


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path).resolve()
    stat = path.stat()
    return _read_table_cached(
        str(path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


def _require(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}.")


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _stimulus_stem(value: object) -> str:
    text = _clean_id(value)
    return Path(text).stem if text else ""


def _single_value(frame: pd.DataFrame, column: str, label: str) -> str:
    values = sorted({_clean_id(value) for value in frame[column] if _clean_id(value)})
    if len(values) != 1:
        raise ValueError(
            f"{label} must contain exactly one non-missing {column}; found {values}."
        )
    return values[0]


def load_srl_trial(
    raw_file: Path,
    participants_csv: Path,
    stimuli_csv: Path,
    *,
    stimulus_name: str,
    eye: str,
) -> ep.EyeDataset:
    """Load one participant × Task_n × eye branch without analytical preprocessing."""
    raw_file = Path(raw_file)
    participants_csv = Path(participants_csv)
    stimuli_csv = Path(stimuli_csv)

    if eye not in EYE_FIELDS:
        raise ValueError("eye must be 'left' or 'right'.")

    raw = _read_table(raw_file)
    fields = EYE_FIELDS[eye]
    _require(raw, COMMON_REQUIRED | set(fields.values()), str(raw_file))
    if raw.empty:
        raise ValueError("Raw eye-tracking file contains no rows.")

    participants = _read_table(participants_csv)
    stimuli = _read_table(stimuli_csv)
    _require(participants, PARTICIPANT_REQUIRED, str(participants_csv))
    _require(stimuli, STIMULI_REQUIRED, str(stimuli_csv))

    participant_id = _single_value(raw, "Participant", str(raw_file))
    requested_stimulus = _stimulus_stem(stimulus_name)
    if not requested_stimulus:
        raise ValueError("stimulus_name must be non-empty.")

    raw_stimulus = raw["Stimulus"].map(_stimulus_stem)
    trial = raw.loc[raw_stimulus.eq(requested_stimulus)].copy()
    if trial.empty:
        available = sorted(
            value for value in raw_stimulus.dropna().unique().tolist() if value
        )
        raise ValueError(
            f"Stimulus {requested_stimulus!r} was not found in {raw_file}; "
            f"available={available}."
        )

    trial_participant = _single_value(trial, "Participant", "selected raw trial")
    if trial_participant != participant_id:
        raise ValueError("Selected raw trial contains inconsistent participant IDs.")

    participant_rows = participants.loc[
        participants["part_ID"].map(_clean_id).eq(participant_id)
    ].copy()
    if len(participant_rows) != 1:
        raise ValueError(
            f"Expected one participants.csv row for {participant_id!r}; "
            f"found {len(participant_rows)}."
        )
    experiment_condition = _clean_id(
        participant_rows.iloc[0]["experiment_condition"]
    )
    if experiment_condition not in {"Prompt", "Non-prompt"}:
        raise ValueError(
            "experiment_condition must be exactly 'Prompt' or 'Non-prompt'; "
            f"found {experiment_condition!r}."
        )

    stimulus_rows = stimuli.loc[
        stimuli["part_ID"].map(_clean_id).eq(participant_id)
        & stimuli["stimulus_name"].map(_stimulus_stem).eq(requested_stimulus)
    ].copy()
    if len(stimulus_rows) != 1:
        raise ValueError(
            f"Expected one stimuli.csv row for participant={participant_id!r}, "
            f"stimulus={requested_stimulus!r}; found {len(stimulus_rows)}."
        )
    stimulus_type = _clean_id(stimulus_rows.iloc[0]["stimulus_type"])
    if stimulus_type not in {"Text", "Multimedia"}:
        raise ValueError(
            "stimulus_type must be exactly 'Text' or 'Multimedia'; "
            f"found {stimulus_type!r}."
        )

    recording_time_ms = pd.to_numeric(
        trial["RecordingTime [ms]"],
        errors="coerce",
    )
    if not np.isfinite(recording_time_ms).all():
        raise ValueError(
            "RecordingTime [ms] contains non-finite values; "
            "the adapter will not drop them."
        )
    native_time = recording_time_ms.to_numpy(dtype=float)
    if len(native_time) < 2:
        raise ValueError("Selected raw trial must contain at least two samples.")
    if np.any(np.diff(native_time) <= 0):
        raise ValueError(
            "RecordingTime [ms] is not strictly increasing within the trial; "
            "the adapter will not reorder or repair samples."
        )
    timestamp_seconds = (native_time - native_time[0]) / 1000.0
    positive_step_ms = np.diff(native_time)
    positive_step_ms = positive_step_ms[
        np.isfinite(positive_step_ms) & (positive_step_ms > 0)
    ]
    empirical_sampling_rate_hz = (
        float(1000.0 / np.median(positive_step_ms))
        if positive_step_ms.size
        else np.nan
    )

    gaze_x = pd.to_numeric(trial[fields["x"]], errors="coerce")
    gaze_y = pd.to_numeric(trial[fields["y"]], errors="coerce")
    finite_xy = np.isfinite(gaze_x) & np.isfinite(gaze_y)
    in_display = (
        gaze_x.ge(0)
        & gaze_x.lt(SCREEN_WIDTH_PX)
        & gaze_y.ge(0)
        & gaze_y.lt(SCREEN_HEIGHT_PX)
    )
    syntactic_valid = finite_xy
    out_of_bounds_finite = finite_xy & ~in_display

    tracking_ratio_raw = pd.to_numeric(
        trial["Tracking Ratio [%]"],
        errors="coerce",
    )
    tracking_ratio_values = tracking_ratio_raw[np.isfinite(tracking_ratio_raw)]
    if tracking_ratio_values.empty:
        trial_tracking_ratio = np.nan
    else:
        unique_tracking = np.unique(tracking_ratio_values.to_numpy(dtype=float))
        if len(unique_tracking) > 1:
            raise ValueError(
                "Tracking Ratio [%] varies within the selected trial; "
                "the adapter will not collapse inconsistent values."
            )
        trial_tracking_ratio = float(unique_tracking[0])

    source_trial_values = sorted(
        {
            _clean_id(value)
            for value in trial["Trial"]
            if _clean_id(value)
        }
    )
    if len(source_trial_values) != 1:
        raise ValueError(
            "Selected stimulus maps to multiple raw Trial values; "
            f"found {source_trial_values}."
        )
    source_trial = source_trial_values[0]

    recording_id = f"srl_{participant_id}_{requested_stimulus}_{eye}"
    stream_id = f"{recording_id}_gaze"
    coordinate_space_id = "srl_display_pixels_top_left"
    trial_id = f"{participant_id}_{requested_stimulus}"

    recordings = pd.DataFrame(
        [
            {
                "recording_id": recording_id,
                "participant_id": participant_id,
                "vendor": "SMI",
                "vendor_family": "SMI RED",
                "device_model": "SMI RED 250",
                "software_name": "Experiment Center / BeGaze",
                "experiment_type": "SRL 2x2 mixed factorial",
                "nominal_sampling_rate": NOMINAL_SAMPLING_RATE_HZ,
                "empirical_sampling_rate_hz": empirical_sampling_rate_hz,
                "screen_width_px": SCREEN_WIDTH_PX,
                "screen_height_px": SCREEN_HEIGHT_PX,
                "source_file_set": str(raw_file),
            }
        ]
    )
    streams = pd.DataFrame(
        [
            {
                "stream_id": stream_id,
                "recording_id": recording_id,
                "stream_type": "gaze",
                "source_device": "SMI RED 250",
                "source_clock": "RecordingTime [ms]",
                "sampling_type": "continuous",
                "nominal_rate_hz": NOMINAL_SAMPLING_RATE_HZ,
                "empirical_rate_hz": empirical_sampling_rate_hz,
                "timestamp_unit": "milliseconds",
                "value_unit": "pixels",
                "coordinate_space_id": coordinate_space_id,
                "processing_level": "sample_level_export_no_new_preprocessing",
            }
        ]
    )
    gaze_samples = pd.DataFrame(
        {
            "recording_id": recording_id,
            "stream_id": stream_id,
            "sample_id": [
                f"{recording_id}_S{i:09d}" for i in range(1, len(trial) + 1)
            ],
            "timestamp_native": native_time,
            "timestamp_seconds": timestamp_seconds,
            "gaze_x": gaze_x.to_numpy(),
            "gaze_y": gaze_y.to_numpy(),
            "valid": syntactic_valid.to_numpy(dtype=bool),
            "confidence": np.nan,
            "trial_id": trial_id,
            "stimulus_id": requested_stimulus,
            "coordinate_space_id": coordinate_space_id,
            "source_eye": eye,
            "source_trial": source_trial,
            "source_aoi_name": trial[fields["aoi"]].to_numpy(),
            "source_tracking_ratio_percent": tracking_ratio_raw.to_numpy(),
            "finite_xy": finite_xy.to_numpy(dtype=bool),
            "within_display_bounds": in_display.to_numpy(dtype=bool),
            "out_of_bounds_finite": out_of_bounds_finite.to_numpy(dtype=bool),
        }
    )

    pupil = pd.to_numeric(trial[fields["pupil"]], errors="coerce")
    eye_samples = pd.DataFrame(
        {
            "recording_id": recording_id,
            "sample_id": gaze_samples["sample_id"],
            "timestamp_native": native_time,
            "timestamp_seconds": timestamp_seconds,
            "eye": eye,
            "pupil_diameter": pupil.to_numpy(),
            "pupil_unit": "mm",
            "pupil_valid": np.isfinite(pupil).to_numpy(dtype=bool),
            "trial_id": trial_id,
            "stimulus_id": requested_stimulus,
        }
    )

    intervals = pd.DataFrame(
        [
            {
                "interval_id": f"I_{trial_id}_{eye}",
                "recording_id": recording_id,
                "interval_type": "trial",
                "start_time": 0.0,
                "end_time": float(timestamp_seconds[-1]),
                "trial_id": trial_id,
                "participant_id": participant_id,
                "item_id": requested_stimulus,
                "stimulus_id": requested_stimulus,
                "condition_id": f"{experiment_condition}|{stimulus_type}",
                "valid_interval": True,
                "experiment_condition": experiment_condition,
                "stimulus_type": stimulus_type,
                "source_tracking_ratio_percent": trial_tracking_ratio,
                "metadata_stimulus_time_seconds": pd.to_numeric(
                    pd.Series([stimulus_rows.iloc[0]["stimulus_time"]]),
                    errors="coerce",
                ).iloc[0],
                "metadata_tracking_ratio_percent": pd.to_numeric(
                    pd.Series([stimulus_rows.iloc[0]["tracking_ratio"]]),
                    errors="coerce",
                ).iloc[0],
            }
        ]
    )
    spaces = ep.new_coordinate_space(
        coordinate_space_id,
        "display_pixels_top_left",
        width=SCREEN_WIDTH_PX,
        height=SCREEN_HEIGHT_PX,
    )

    dataset = ep.new_eye_dataset(
        recordings=recordings,
        streams=streams,
        gaze_samples=gaze_samples,
        eye_samples=eye_samples,
        intervals=intervals,
        coordinate_spaces=spaces,
        raw=[trial.copy()],
        vendor_metadata={
            "source_dataset": "Juřík et al. SRL eye-tracking dataset",
            "dataset_doi": "10.6084/m9.figshare.28304069",
            "paper_doi": "10.1038/s41597-025-05304-1",
            "source_export": "ET_data_raw",
            "source_eye": eye,
            "source_aoi_labels_used_for_geometry": False,
            "source_nominal_sampling_rate_hz": NOMINAL_SAMPLING_RATE_HZ,
            "empirical_sampling_rate_hz": empirical_sampling_rate_hz,
            "binocular_fusion": "none",
            "interpolation": "none",
            "smoothing": "none",
            "quality_exclusion_applied": False,
            "coordinate_validity_rule": "finite_xy_only",
        },
        validate=True,
    )
    return ep.add_provenance(
        dataset,
        "map_srl_raw_trial",
        "gaze_samples",
        (
            f"participant={participant_id};stimulus={requested_stimulus};"
            f"eye={eye};condition={experiment_condition};"
            f"stimulus_type={stimulus_type};"
            f"empirical_sampling_rate_hz={empirical_sampling_rate_hz:.12g};"
            "timestamp_ms_to_seconds=true;"
            "coordinate_validity=finite_xy_only;"
            "binocular_fusion=none;interpolation=none;smoothing=none;"
            "quality_exclusion=false;source_aoi_geometry=false"
        ),
        source_files=[str(raw_file), str(participants_csv), str(stimuli_csv)],
        software="eyeprocesspy research adapter",
        reversible=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file", type=Path)
    parser.add_argument("participants_csv", type=Path)
    parser.add_argument("stimuli_csv", type=Path)
    parser.add_argument("--stimulus", required=True)
    parser.add_argument("--eye", required=True, choices=["left", "right"])
    args = parser.parse_args()

    dataset = load_srl_trial(
        args.raw_file,
        args.participants_csv,
        args.stimuli_csv,
        stimulus_name=args.stimulus,
        eye=args.eye,
    )
    samples = dataset["gaze_samples"]
    interval = dataset["intervals"].iloc[0]
    issues = ep.validate_eye_dataset(dataset)

    print(
        f"participant={interval['participant_id']} "
        f"stimulus={interval['stimulus_id']} "
        f"condition={interval['experiment_condition']} "
        f"type={interval['stimulus_type']} "
        f"eye={args.eye} "
        f"samples={len(samples)} "
        f"finite_xy={samples['finite_xy'].mean():.4f} "
        f"out_of_bounds_finite={samples['out_of_bounds_finite'].sum()}"
    )
    if len(issues):
        print(issues.to_string(index=False))
        if issues["severity"].astype(str).eq("error").any():
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
