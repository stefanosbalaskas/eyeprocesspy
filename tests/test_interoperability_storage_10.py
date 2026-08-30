from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "eye_storage_spec",
    "write_eye_storage",
    "open_eye_storage",
    "collect_eye_storage",
    "export_eye_bids",
    "import_eye_bids",
    "as_eyeprocess_eyetools",
    "as_eyeprocess_eyetrackingr",
    "as_eyeprocess_gazer",
    "as_eyeprocess_eyeris",
    "as_eyeprocess_pupillometryr",
]


def _dataset():
    recordings = pd.DataFrame(
        [
            {
                "recording_id": "R1",
                "participant_id": "P-01",
                "session_id": "S1",
                "vendor": "Gazepoint",
                "vendor_family": "Gazepoint",
                "device_model": "GP3",
                "software_name": "Gazepoint Analysis",
                "software_version": "7.2",
                "nominal_sampling_rate": 60.0,
                "screen_width_px": 1920,
                "screen_height_px": 1080,
                "source_file_set": "synthetic.csv",
            }
        ]
    )
    streams = pd.DataFrame(
        [
            {
                "stream_id": "R1_gaze",
                "recording_id": "R1",
                "stream_type": "gaze_combined",
                "source_device": "GP3",
                "source_clock": "native",
                "sampling_type": "sampled",
                "nominal_rate_hz": 60.0,
                "observed_rate_hz": 60.0,
                "timestamp_unit": "seconds",
                "coordinate_space_id": "coord_display",
                "processing_level": "raw",
            }
        ]
    )
    gaze = pd.DataFrame(
        {
            "recording_id": ["R1", "R1", "R1"],
            "stream_id": ["R1_gaze"] * 3,
            "sample_id": ["s1", "s2", "s3"],
            "timestamp_native": [0.0, 1 / 60, 2 / 60],
            "timestamp_seconds": [0.0, 1 / 60, 2 / 60],
            "gaze_x": [0.2, 0.3, 0.4],
            "gaze_y": [0.4, 0.5, 0.6],
            "valid": [True, True, True],
            "coordinate_space_id": ["coord_display"] * 3,
        }
    )
    eye_rows = []
    for eye, offset in [("left", 0.0), ("right", 0.1)]:
        for i, timestamp in enumerate([0.0, 1 / 60, 2 / 60], start=1):
            eye_rows.append(
                {
                    "recording_id": "R1",
                    "sample_id": f"{eye}_{i}",
                    "timestamp_native": timestamp,
                    "timestamp_seconds": timestamp,
                    "eye": eye,
                    "pupil_diameter": 3.0 + offset + i / 100,
                    "pupil_unit": "millimetres",
                    "pupil_valid": True,
                }
            )
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "recording_id": "R1",
                "timestamp_native": 0.0,
                "timestamp_seconds": 0.0,
                "event_type": "trial_start",
                "event_name": "TRIAL_START",
                "event_value": "trial1",
                "duration": 0.0,
                "source": "synthetic",
                "trial_id": "trial1",
            }
        ]
    )
    coords = ep.new_coordinate_space(
        "coord_display",
        "display_normalized_top_left",
    )
    return ep.new_eye_dataset(
        recordings=recordings,
        streams=streams,
        gaze_samples=gaze,
        eye_samples=pd.DataFrame(eye_rows),
        events=events,
        coordinate_spaces=coords,
        raw={"fixture": pd.DataFrame({"x": [1, 2]})},
        vendor_metadata={"fixture": {"version": "1"}},
    )


def test_public_r020_targets_are_exported():
    assert len(TARGETS) == 11
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_storage_spec_and_rds_boundary(tmp_path):
    spec = ep.eye_storage_spec(
        tmp_path / "store",
        format="parquet",
        tables=["recordings", "gaze_samples", "not_a_table"],
    )
    assert spec.format == "parquet"
    assert spec.tables == ("recordings", "gaze_samples")

    with pytest.raises(ep.EyeProcessBackendError, match="RDS"):
        ep.write_eye_storage(
            _dataset(),
            tmp_path / "native.rds",
            format="rds",
        )


def test_parquet_storage_roundtrip_retains_metadata(tmp_path):
    pytest.importorskip("pyarrow")
    x = _dataset()
    path = tmp_path / "parquet"
    handle = ep.write_eye_storage(
        x,
        path,
        format="parquet",
        overwrite=True,
        retain_metadata=True,
    )
    assert handle.spec.format == "parquet"
    assert (path / "storage-manifest.csv").is_file()
    assert (path / "storage-metadata.pkl").is_file()

    opened = ep.open_eye_storage(path)
    assert opened.spec.format == "parquet"
    restored = ep.collect_eye_storage(opened)
    assert ep.is_eye_dataset(restored)
    assert len(restored["gaze_samples"]) == len(x["gaze_samples"])
    assert restored.vendor_metadata == x.vendor_metadata
    assert isinstance(restored.raw, dict)
    assert "fixture" in restored.raw


def test_arrow_dataset_storage_and_subset_collection(tmp_path):
    pytest.importorskip("pyarrow")
    x = _dataset()
    path = tmp_path / "arrow"
    handle = ep.write_eye_storage(
        x,
        path,
        format="arrow_dataset",
        tables=["recordings", "gaze_samples"],
        partitioning=["recording_id"],
        overwrite=True,
    )
    assert handle.spec.format == "arrow_dataset"
    restored = ep.collect_eye_storage(
        handle,
        tables=["recordings", "gaze_samples"],
    )
    assert len(restored["recordings"]) == 1
    assert len(restored["gaze_samples"]) == 3
    assert set(restored["gaze_samples"].recording_id.astype(str)) == {"R1"}
    assert restored["eye_samples"].empty


def test_bids_export_writes_headerless_physio_and_sidecars(tmp_path):
    x = _dataset()
    root = tmp_path / "bids"
    manifest = ep.export_eye_bids(
        x,
        root,
        task="reading-task",
        overwrite=True,
        screen_distance_m=0.6,
        screen_size_m=[0.5, 0.3],
    )
    assert len(manifest) == 2
    assert (root / "dataset_description.json").is_file()
    assert (root / "participants.tsv").is_file()
    assert (root / "eyeprocess-bids-manifest.csv").is_file()

    first_tsv = Path(manifest.iloc[0].tsv)
    first_json = Path(manifest.iloc[0].json)
    with gzip.open(first_tsv, "rt", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    assert "timestamp" not in first_line.lower()

    sidecar = json.loads(first_json.read_text(encoding="utf-8"))
    assert sidecar["PhysioType"] == "eyetrack"
    assert sidecar["Columns"][:3] == [
        "timestamp",
        "x_coordinate",
        "y_coordinate",
    ]
    assert sidecar["RecordedEye"] in {"left", "right"}


def test_bids_roundtrip_reconstructs_gaze_eyes_and_events(tmp_path):
    x = _dataset()
    root = tmp_path / "bids"
    ep.export_eye_bids(x, root, task="reading", overwrite=True)
    restored = ep.import_eye_bids(root)
    assert ep.is_eye_dataset(restored)
    assert len(restored["recordings"]) == 1
    assert len(restored["streams"]) == 2
    assert len(restored["gaze_samples"]) == 3
    assert len(restored["eye_samples"]) == 6
    assert list(restored["gaze_samples"].sample_id.astype(str)) == [
        "bids_sample_1",
        "bids_sample_2",
        "bids_sample_3",
    ]
    assert set(restored["eye_samples"].sample_id.astype(str)) == {
        "bids_sample_1",
        "bids_sample_2",
        "bids_sample_3",
    }
    assert set(restored["eye_samples"].eye.astype(str)) == {"left", "right"}
    assert len(restored["events"]) == 1
    assert restored.vendor_metadata["bids_root"] == str(root.resolve())
    issues = ep.validate_eye_dataset(restored)
    assert not (issues["severity"] == "error").any()


@pytest.mark.parametrize(
    ("name", "vendor"),
    [
        ("as_eyeprocess_eyetools", "eyetools"),
        ("as_eyeprocess_eyetrackingr", "eyetrackingR"),
        ("as_eyeprocess_gazer", "gazeR"),
        ("as_eyeprocess_eyeris", "eyeris"),
        ("as_eyeprocess_pupillometryr", "PupillometryR"),
    ],
)
def test_external_adapters_reuse_generic_mapping(name, vendor):
    data = pd.DataFrame(
        {
            "participant_id": ["P1", "P1"],
            "recording_id": ["R1", "R1"],
            "timestamp": [0.0, 0.1],
            "x": [0.2, 0.3],
            "y": [0.4, 0.5],
            "pupil_left": [3.0, 3.1],
            "pupil_right": [3.2, 3.3],
        }
    )
    out = getattr(ep, name)(data, keep_raw=False)
    assert ep.is_eye_dataset(out)
    assert len(out["gaze_samples"]) == 2
    assert set(out["recordings"].vendor.astype(str)) == {vendor}
    assert set(out["recordings"].vendor_family.astype(str)) == {vendor}
    assert "external_adapter" in set(out["provenance"].action.astype(str))
