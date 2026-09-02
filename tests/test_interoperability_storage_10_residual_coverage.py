from __future__ import annotations

import builtins
import gzip
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.interoperability_storage_10 as mod


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
                "value_unit": pd.NA,
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
            "confidence": [1.0, 1.0, 1.0],
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
                    "detector_method": "synthetic",
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
                "native_record": pd.NA,
                "trial_id": "trial1",
                "stimulus_id": pd.NA,
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


def _write_physio(
    root: Path,
    *,
    name="sub-P01_task-read_run-01_recording-eye1_physio.tsv.gz",
    rows=None,
    meta=None,
):
    path = root / "sub-P01" / "beh" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = ["0\t0.1\t0.2\n"]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(rows)
    if meta is not False:
        payload = {
            "Columns": ["timestamp", "x_coordinate", "y_coordinate"],
            "PhysioType": "eyetrack",
            "StartTime": 0.0,
            "RecordedEye": "left",
            "SamplingFrequency": 60.0,
            "timestamp": {"Units": "seconds"},
        }
        if isinstance(meta, dict):
            payload.update(meta)
        sidecar = Path(str(path)[:-7] + ".json")
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_storage_reprs_and_spec_guard_branches(tmp_path):
    spec = mod.eye_storage_spec(
        tmp_path / "a", format=("parquet", "rds"), partitioning="recording_id"
    )
    assert "eye_storage_spec" in repr(spec)
    assert spec.partitioning == ("recording_id",)

    handle = mod.EyeStorage(
        spec=spec,
        manifest=pd.DataFrame({"table": ["recordings"], "path": ["x"], "rows": [1]}),
    )
    assert handle.spec is spec
    assert list(handle.manifest["table"]) == ["recordings"]
    assert "recordings" in repr(handle)

    with pytest.raises(ValueError, match="format"):
        mod.eye_storage_spec(tmp_path / "x", format="csv")
    with pytest.raises(ValueError, match="No canonical tables"):
        mod.eye_storage_spec(tmp_path / "x", format="parquet", tables=["not_canonical"])

    all_tables = mod.eye_storage_spec(tmp_path / "x", format="parquet")
    assert "recordings" in all_tables.tables


def test_require_pyarrow_backend_guard(monkeypatch):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ep.EyeProcessBackendError, match="PyArrow"):
        mod._require_pyarrow()


def test_resolve_compression_all_fallbacks(monkeypatch):
    class Codec:
        mode = "normal"

        @staticmethod
        def is_available(name):
            if Codec.mode == "raises":
                raise RuntimeError("codec probe failed")
            if Codec.mode == "snappy":
                return name == "snappy"
            if Codec.mode == "gzip":
                return name == "gzip"
            return False

    fake_pa = SimpleNamespace(Codec=Codec)
    monkeypatch.setattr(mod, "_require_pyarrow", lambda: (fake_pa, None, None))

    with pytest.raises(ValueError, match="compression"):
        mod._resolve_compression("", allow_fallback=True)
    assert mod._resolve_compression("uncompressed", allow_fallback=False) == "NONE"

    Codec.mode = "gzip"
    assert mod._resolve_compression("gzip", allow_fallback=False) == "gzip"

    Codec.mode = "none"
    with pytest.raises(ep.EyeProcessBackendError, match="unavailable"):
        mod._resolve_compression("madeup", allow_fallback=False)

    Codec.mode = "snappy"
    assert mod._resolve_compression("zstd", allow_fallback=True) == "snappy"

    Codec.mode = "none"
    assert mod._resolve_compression("zstd", allow_fallback=True) == "NONE"

    Codec.mode = "raises"
    assert mod._resolve_compression("zstd", allow_fallback=True) == "NONE"


def test_metadata_missing_nondict_and_retain_false(tmp_path):
    assert mod._read_metadata(tmp_path) == {}

    path = mod._metadata_path(tmp_path)
    with path.open("wb") as handle:
        pickle.dump(["not", "a", "dict"], handle)
    assert mod._read_metadata(tmp_path) == {}

    x = _dataset()
    mod._write_metadata(x, tmp_path, retain_metadata=False)
    value = mod._read_metadata(tmp_path)
    assert value["raw"] == []
    assert value["vendor_metadata"] == {}
    assert value["schema_version"] == x.schema_version


def test_arrow_table_conversion_guard(monkeypatch):
    class Table:
        @staticmethod
        def from_pandas(*args, **kwargs):
            raise TypeError("cannot convert")

    monkeypatch.setattr(
        mod, "_require_pyarrow", lambda: (SimpleNamespace(Table=Table), None, None)
    )
    with pytest.raises(ep.EyeProcessBackendError, match="could not be converted"):
        mod._arrow_table(pd.DataFrame({"x": [object()]}))


def test_write_storage_existing_target_and_empty_arrow_dataset(tmp_path):
    pytest.importorskip("pyarrow")
    x = _dataset()

    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    (existing_dir / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        mod.write_eye_storage(
            x, existing_dir, format="parquet", tables=["recordings"], overwrite=False
        )

    out = mod.write_eye_storage(
        x,
        existing_dir,
        format=("parquet", "rds"),
        tables=["recordings"],
        overwrite=True,
        compression="uncompressed",
        retain_metadata=False,
    )
    assert out.spec.compression == "uncompressed"
    assert not (existing_dir / "old.txt").exists()

    existing_file = tmp_path / "as_file"
    existing_file.write_text("old", encoding="utf-8")
    out2 = mod.write_eye_storage(
        x,
        existing_file,
        format="parquet",
        tables=["recordings"],
        overwrite=True,
    )
    assert Path(out2.spec.path).is_dir()

    empty_path = tmp_path / "empty_arrow"
    empty = mod.write_eye_storage(
        x,
        empty_path,
        format="arrow_dataset",
        tables=["episodes"],
        partitioning=["recording_id", "not_a_column"],
        overwrite=True,
    )
    assert (empty_path / "episodes" / "part-0.parquet").is_file()
    assert empty.manifest.rows.iloc[0] == 0


def test_open_storage_guards_and_rds_handle(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mod.open_eye_storage(tmp_path / "missing")

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="manifest"):
        mod.open_eye_storage(bad_dir)

    native = tmp_path / "native.rds"
    native.write_bytes(b"x")
    handle = mod.open_eye_storage(native)
    assert handle.spec.format == "rds"
    assert handle.manifest.table.iloc[0] == "eye_dataset"

    with pytest.raises(ep.EyeProcessBackendError, match="RDS collection"):
        mod.collect_eye_storage(handle)

    with pytest.raises(TypeError, match="EyeStorage"):
        mod.collect_eye_storage(object())

    with pytest.raises(ValueError, match="manifest"):
        mod.open_eye_storage(bad_dir, format=("parquet", "rds"))


def test_collect_storage_from_path_and_subset(tmp_path):
    pytest.importorskip("pyarrow")
    x = _dataset()
    root = tmp_path / "store"
    mod.write_eye_storage(
        x,
        root,
        format="parquet",
        tables=["recordings", "gaze_samples"],
        overwrite=True,
    )
    out = mod.collect_eye_storage(
        root, tables=["recordings", "gaze_samples", "absent"]
    )
    assert len(out["recordings"]) == 1
    assert len(out["gaze_samples"]) == 3


def test_small_helpers_cover_empty_fallback_and_coordinate_systems():
    assert mod._sanitize_id(None, "fallback") == "fallback"
    assert mod._sanitize_id("!!!", "fallback") == "fallback"
    assert mod._sanitize_id(" A-1 ") == "A1"

    assert mod._first_text([pd.NA, "", "  ", "ok"], "fallback") == "ok"
    assert mod._first_text([pd.NA, ""], "fallback") == "fallback"

    x = _dataset()
    x["streams"].loc[:, "observed_rate_hz"] = np.nan
    x["streams"].loc[:, "nominal_rate_hz"] = 120.0
    assert mod._sampling_rate(x, "R1") == 120.0
    assert np.isnan(mod._sampling_rate(x, "missing"))
    assert mod._timestamp_unit(x, "missing") == "unknown"

    x["coordinate_spaces"].loc[:, "space_type"] = "world_coordinates"
    assert mod._coordinate_metadata(x, "R1")["system"] == "gaze-in-world"
    x["coordinate_spaces"].loc[:, "space_type"] = "head_direction"
    assert mod._coordinate_metadata(x, "R1")["system"] == "eye-in-head"
    x["coordinate_spaces"].loc[:, "space_type"] = "arbitrary"
    assert mod._coordinate_metadata(x, "R1")["system"] == "custom"
    x["gaze_samples"].loc[:, "coordinate_space_id"] = "missing"
    assert mod._coordinate_metadata(x, "R1") == {
        "system": "custom",
        "x_unit": "unknown",
        "y_unit": "unknown",
    }


def test_json_safe_and_recorded_eye_helpers():
    value = mod._json_safe(
        {
            "a": [np.int64(2), pd.NA, np.nan],
            "b": (np.float64(1.5),),
        }
    )
    assert value == {"a": [2, None, None], "b": [1.5]}
    assert mod._recorded_eye("OS") == "left"
    assert mod._recorded_eye("OD") == "right"
    assert mod._recorded_eye("both") == "cyclopean"


def test_export_bids_output_guards_and_collision(tmp_path):
    x = _dataset()

    target_file = tmp_path / "file"
    target_file.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="exists as a file"):
        mod.export_eye_bids(x, target_file)

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "x").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        mod.export_eye_bids(x, nonempty)

    rec2 = x["recordings"].iloc[0].copy()
    rec2["recording_id"] = "R2"
    rec2["participant_id"] = "P01"
    x["recordings"] = pd.concat(
        [x["recordings"], pd.DataFrame([rec2])], ignore_index=True
    )
    with pytest.raises(ValueError, match="collide"):
        mod.export_eye_bids(x, tmp_path / "collision", overwrite=True)


def test_export_bids_no_eye_samples_rate_fallback_and_overwrite(tmp_path):
    x = _dataset()
    x["eye_samples"] = x["eye_samples"].iloc[0:0].copy()
    x["events"] = x["events"].iloc[0:0].copy()
    x["streams"].loc[:, ["observed_rate_hz", "nominal_rate_hz"]] = np.nan
    x["recordings"].loc[:, "nominal_sampling_rate"] = 0.0
    x["gaze_samples"].loc[:, "coordinate_space_id"] = "missing"

    root = tmp_path / "bids"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    manifest = mod.export_eye_bids(x, root, task="read task", overwrite=True)
    assert len(manifest) == 1
    meta = json.loads(Path(manifest.json.iloc[0]).read_text(encoding="utf-8"))
    assert meta["RecordedEye"] == "cyclopean"
    assert meta["SamplingFrequency"] == 1.0
    assert meta["SampleCoordinateSystem"] == "custom"
    assert "pupil_size" not in meta
    assert not (root / "old.txt").exists()


def test_export_bids_pupil_native_timestamp_fallback(tmp_path):
    x = _dataset()
    x["eye_samples"].loc[:, "timestamp_seconds"] = np.nan
    root = tmp_path / "bids"
    manifest = mod.export_eye_bids(x, root, overwrite=True)
    with gzip.open(Path(manifest.tsv.iloc[0]), "rt", encoding="utf-8") as handle:
        rows = [line.strip().split("\t") for line in handle if line.strip()]
    assert len(rows[0]) == 4
    assert float(rows[0][3]) > 0


def test_timestamp_conversion_branches():
    ms = mod._timestamp_to_seconds([0, 1000], "ms")
    assert np.allclose(ms, [0.0, 1.0])
    rate = mod._timestamp_to_seconds([100, 200, 300], "ticks", sampling_rate=2.0)
    assert np.allclose(rate, [0.0, 0.5, 1.0])
    raw = mod._timestamp_to_seconds([5, 6], "ticks", sampling_rate=np.nan)
    assert list(raw) == [5, 6]


def test_import_bids_root_and_sidecar_guards(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mod.import_eye_bids(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No Eye-Tracking-BIDS"):
        mod.import_eye_bids(empty)

    root = tmp_path / "nosidecar"
    _write_physio(root, meta=False)
    with pytest.raises(ValueError, match="Missing BIDS sidecar"):
        mod.import_eye_bids(root)


@pytest.mark.parametrize(
    ("meta", "message"),
    [
        ({"Columns": None}, "missing required fields"),
        ({"PhysioType": "other"}, "PhysioType"),
        ({"RecordedEye": "middle"}, "RecordedEye"),
        ({"StartTime": float("inf")}, "StartTime"),
        ({"SamplingFrequency": 0.0}, "SamplingFrequency"),
        ({"Columns": ["timestamp", "x_coordinate"]}, "Columns must include"),
    ],
)
def test_import_bids_sidecar_validation_branches(tmp_path, meta, message):
    root = tmp_path / message.replace(" ", "_").replace("`", "")
    if "Columns" in meta and meta["Columns"] is None:
        path = _write_physio(root)
        sidecar = Path(str(path)[:-7] + ".json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload.pop("Columns")
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
    else:
        _write_physio(root, meta=meta)
    with pytest.raises(ValueError, match=message):
        mod.import_eye_bids(root)


def test_import_bids_column_length_guard_via_reader(monkeypatch, tmp_path):
    root = tmp_path / "shape"
    _write_physio(root)
    real_read_csv = mod.pd.read_csv

    def fake_read_csv(*args, **kwargs):
        if kwargs.get("header") is None and "names" in kwargs:
            return pd.DataFrame([[0, 0.1, 0.2, 99]], columns=["a", "b", "c", "d"])
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(mod.pd, "read_csv", fake_read_csv)
    with pytest.raises(ValueError, match="Columns length"):
        mod.import_eye_bids(root)


def test_import_bids_unparseable_recording_entity(tmp_path):
    root = tmp_path / "parse"
    _write_physio(
        root,
        name="sub-P01_task-read_run-01_recording-eye_left_physio.tsv.gz",
    )
    with pytest.raises(ValueError, match="Cannot parse"):
        mod.import_eye_bids(root)


def test_import_bids_empty_gaze_rows(tmp_path):
    root = tmp_path / "empty_rows"
    _write_physio(root, rows=[])
    with pytest.raises(ValueError, match="no gaze rows"):
        mod.import_eye_bids(root)


def test_import_bids_without_pupil_and_event_validation(tmp_path):
    root = tmp_path / "events_bad"
    physio = _write_physio(root)
    event = physio.parent / "sub-P01_task-read_run-01_events.tsv"
    pd.DataFrame({"onset": [0.0], "trial_type": ["start"]}).to_csv(
        event, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="lacks onset/duration/trial_type"):
        mod.import_eye_bids(root)


def test_import_bids_skips_unmatched_events_and_accepts_no_value(tmp_path):
    root = tmp_path / "events"
    physio = _write_physio(root)
    unmatched = physio.parent / "sub-P01_task-other_run-99_events.tsv"
    pd.DataFrame(
        {"onset": [0.0], "duration": [0.0], "trial_type": ["ignored"]}
    ).to_csv(unmatched, sep="\t", index=False)

    matching = physio.parent / "sub-P01_task-read_run-01_events.tsv"
    pd.DataFrame(
        {"onset": [0.0], "duration": [0.0], "trial_type": ["start"]}
    ).to_csv(matching, sep="\t", index=False)

    out = mod.import_eye_bids(root)
    assert len(out["eye_samples"]) == 0
    assert len(out["events"]) == 1
    assert pd.isna(out["events"].event_value.iloc[0])


def test_external_adapter_non_dataframe_mapping_and_type_guard():
    data = [
        {
            "participant_id": "P1",
            "recording_id": "R1",
            "timestamp": 0.0,
            "x": 0.2,
            "y": 0.4,
        },
        {
            "participant_id": "P1",
            "recording_id": "R1",
            "timestamp": 0.1,
            "x": 0.3,
            "y": 0.5,
        },
    ]
    frame = pd.DataFrame(data)
    mapping = mod.infer_eye_mapping(frame)
    out = mod._as_external(data, mapping=mapping, vendor="custom", keep_raw=False)
    assert set(out["recordings"].vendor.astype(str)) == {"custom"}

    with pytest.raises(TypeError, match="coercible"):
        mod._as_external(object(), vendor="bad")


def test_external_adapter_empty_recordings_branch(monkeypatch):
    x = _dataset()
    x["recordings"] = x["recordings"].iloc[0:0].copy()
    monkeypatch.setattr(mod, "read_eye_generic", lambda *args, **kwargs: x)
    monkeypatch.setattr(mod, "infer_eye_mapping", lambda data: {"timestamp": "timestamp"})
    out = mod._as_external(pd.DataFrame({"timestamp": [0.0]}), vendor="custom")
    assert out["recordings"].empty
    assert "external_adapter" in set(out["provenance"].action.astype(str))
