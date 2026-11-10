#!/usr/bin/env python3
"""Map one MCFW-Gaze trial into the canonical eyeprocesspy data contract.

This research adapter deliberately requires explicit choices that are often
hidden in preprocessing:

- the eye used for detector input (left or right);
- the conversion from the published device timestamp's native unit to seconds.

It does not average eyes, interpolate missing samples, repair timestamps,
clip normalized coordinates, or choose an event detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import eyeprocesspy as ep

REQUIRED_COMMON = {
    "device_time_stamp",
    "system_time_stamp",
}
EYE_FIELDS = {
    "left": {
        "x": "left_gaze_point_on_display_area_x",
        "y": "left_gaze_point_on_display_area_y",
        "valid": "left_gaze_point_valid",
        "available": "left_gaze_point_available",
    },
    "right": {
        "x": "right_gaze_point_on_display_area_x",
        "y": "right_gaze_point_on_display_area_y",
        "valid": "right_gaze_point_valid",
        "available": "right_gaze_point_available",
    },
}


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "1.0"})


def _participant_id(path: Path) -> str:
    parent = path.parent.name
    if not parent.startswith("participant_"):
        raise ValueError(
            "MCFW trial path must be inside a participant_<id> directory; "
            f"found {parent!r}."
        )
    return parent


def load_mcfw_trial(
    path: Path,
    *,
    eye: str,
    timestamp_scale_seconds: float,
    nominal_sampling_rate: float = 120.0,
) -> ep.EyeDataset:
    """Load one untouched MCFW trial into an EyeDataset for a declared eye."""
    path = Path(path)
    if eye not in EYE_FIELDS:
        raise ValueError("eye must be 'left' or 'right'.")
    scale = float(timestamp_scale_seconds)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("timestamp_scale_seconds must be finite and > 0.")
    rate = float(nominal_sampling_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("nominal_sampling_rate must be finite and > 0.")

    frame = pd.read_csv(path, sep="\t")
    fields = EYE_FIELDS[eye]
    required = REQUIRED_COMMON | set(fields.values())
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MCFW trial is missing required columns: {', '.join(missing)}.")
    if frame.empty:
        raise ValueError("MCFW trial contains no samples.")

    native = pd.to_numeric(frame["device_time_stamp"], errors="coerce")
    if not np.isfinite(native).all():
        raise ValueError(
            "device_time_stamp contains non-finite values; the adapter will not drop them."
        )
    native_values = native.to_numpy(dtype=float)
    if np.any(np.diff(native_values) <= 0):
        raise ValueError(
            "device_time_stamp is not strictly increasing; "
            "the adapter will not reorder or repair it."
        )
    timestamp_seconds = (native_values - native_values[0]) * scale

    gaze_x = pd.to_numeric(frame[fields["x"]], errors="coerce")
    gaze_y = pd.to_numeric(frame[fields["y"]], errors="coerce")
    valid_flag = _bool_series(frame, fields["valid"])
    available_flag = _bool_series(frame, fields["available"])
    finite_xy = np.isfinite(gaze_x) & np.isfinite(gaze_y)
    usable = valid_flag & available_flag & finite_xy

    participant_id = _participant_id(path)
    trial_id = path.stem
    recording_id = f"{participant_id}_{trial_id}_{eye}"
    stream_id = f"{recording_id}_gaze"
    coordinate_space_id = "mcfw_display_normalized_top_left"

    recordings = pd.DataFrame(
        [
            {
                "recording_id": recording_id,
                "participant_id": participant_id,
                "vendor": "Tobii",
                "vendor_family": "Tobii Pro Fusion",
                "device_model": "Tobii Pro Fusion",
                "software_name": "Titta/PsychoPy",
                "experiment_type": "MCFW-Gaze",
                "nominal_sampling_rate": rate,
                "screen_width_px": 1920,
                "screen_height_px": 1080,
                "source_file_set": str(path),
            }
        ]
    )
    streams = pd.DataFrame(
        [
            {
                "stream_id": stream_id,
                "recording_id": recording_id,
                "stream_type": "gaze",
                "source_device": "Tobii Pro Fusion",
                "source_clock": "device_time_stamp",
                "sampling_type": "continuous",
                "nominal_rate_hz": rate,
                "timestamp_unit": "seconds_after_explicit_native_scale",
                "value_unit": "normalized_display",
                "coordinate_space_id": coordinate_space_id,
                "processing_level": "raw_mapped_no_interpolation",
            }
        ]
    )
    gaze_samples = pd.DataFrame(
        {
            "recording_id": recording_id,
            "stream_id": stream_id,
            "sample_id": [
                f"{recording_id}_S{i:08d}" for i in range(1, len(frame) + 1)
            ],
            "timestamp_native": native_values,
            "timestamp_seconds": timestamp_seconds,
            "gaze_x": gaze_x.to_numpy(),
            "gaze_y": gaze_y.to_numpy(),
            "valid": usable.to_numpy(dtype=bool),
            "confidence": np.nan,
            "trial_id": trial_id,
            "stimulus_id": trial_id,
            "coordinate_space_id": coordinate_space_id,
            "mcfw_eye": eye,
            "mcfw_valid_flag": valid_flag.to_numpy(dtype=bool),
            "mcfw_available_flag": available_flag.to_numpy(dtype=bool),
            "mcfw_system_time_stamp": frame["system_time_stamp"].to_numpy(),
        }
    )
    intervals = pd.DataFrame(
        [
            {
                "interval_id": f"I_{recording_id}",
                "recording_id": recording_id,
                "interval_type": "trial",
                "start_time": 0.0,
                "end_time": float(timestamp_seconds[-1]),
                "trial_id": trial_id,
                "participant_id": participant_id,
                "item_id": trial_id,
                "stimulus_id": trial_id,
                "condition_id": pd.NA,
                "valid_interval": True,
            }
        ]
    )
    spaces = ep.new_coordinate_space(
        coordinate_space_id,
        "display_normalized_top_left",
    )
    dataset = ep.new_eye_dataset(
        recordings=recordings,
        streams=streams,
        gaze_samples=gaze_samples,
        intervals=intervals,
        coordinate_spaces=spaces,
        raw=[frame],
        vendor_metadata={
            "source_dataset": "MCFW-Gaze",
            "source_version": "v3/Zenodo 20300972",
            "detector_input_eye": eye,
            "timestamp_scale_seconds": scale,
            "binocular_fusion": "none",
            "interpolation": "none",
        },
        validate=True,
    )
    return ep.add_provenance(
        dataset,
        "map_mcfw_trial",
        "gaze_samples",
        (
            f"eye={eye};timestamp_scale_seconds={scale:g};"
            f"nominal_sampling_rate={rate:g};"
            "binocular_fusion=none;interpolation=none"
        ),
        source_files=str(path),
        software="eyeprocesspy research adapter",
        reversible=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_tsv", type=Path)
    parser.add_argument("--eye", required=True, choices=["left", "right"])
    parser.add_argument("--timestamp-scale-seconds", required=True, type=float)
    parser.add_argument("--sampling-rate", type=float, default=120.0)
    args = parser.parse_args()

    dataset = load_mcfw_trial(
        args.trial_tsv,
        eye=args.eye,
        timestamp_scale_seconds=args.timestamp_scale_seconds,
        nominal_sampling_rate=args.sampling_rate,
    )
    samples = dataset["gaze_samples"]
    issues = ep.validate_eye_dataset(dataset)
    print(
        f"Mapped {len(samples)} samples; "
        f"valid fraction={samples['valid'].astype(bool).mean():.4f}; "
        f"eye={args.eye}."
    )
    if len(issues):
        print(issues.to_string(index=False))
        if issues["severity"].astype(str).eq("error").any():
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
