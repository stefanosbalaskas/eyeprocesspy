from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.io_validation_10 as io
import eyeprocesspy.vendor_importers_10 as vi


class _AttrRejectingDict(dict):
    @property
    def vendor_metadata(self):
        return "not-a-dict"

    @vendor_metadata.setter
    def vendor_metadata(self, value):
        raise RuntimeError("reject metadata attribute")

    @property
    def raw(self):
        return "not-a-dict"

    @raw.setter
    def raw(self, value):
        raise RuntimeError("reject raw attribute")


def test_vendor_private_fallbacks_and_detectors(monkeypatch, tmp_path):
    obj = _AttrRejectingDict(vendor_metadata="bad", raw="bad")
    vi._set_metadata(obj, "x", {"a": 1})
    vi._set_raw(obj, "x", {"b": 2})
    assert obj["vendor_metadata"]["x"] == {"a": 1}
    assert obj["raw"]["x"] == {"b": 2}

    ds = {"coordinate_spaces": pd.DataFrame()}
    assert pd.isna(vi._coordinate_id(ds))

    p = tmp_path / "unknown.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(vi, "_safe_read_head", lambda *a, **k: None)
    assert vi.is_tobii_export(p) == 0.0
    assert vi.is_pupil_labs_export(p) == 0.0
    assert vi.is_eyelink_export(p) == 0.0
    assert vi.is_smi_export(p) == 0.0


def test_vendor_tobii_missing_binocular_y_and_dispatch(monkeypatch, tmp_path):
    p = tmp_path / "tobii.csv"
    pd.DataFrame(
        {
            "Recording timestamp": [0, 1000],
            "Gaze point left X": [0.1, 0.2],
            "Gaze point right X": [0.3, 0.4],
        }
    ).to_csv(p, index=False)
    out = vi.read_tobii(
        p,
        participant_id="P1",
        recording_id="R1",
        time_unit="microseconds",
        coordinate_space="normalized",
        keep_raw=False,
    )
    assert len(out["gaze_samples"]) == 2
    assert out["gaze_samples"]["gaze_y"].isna().all()

    sentinel = object()
    monkeypatch.setattr(vi, "read_pupil_core", lambda *a, **k: sentinel)
    assert vi.read_pupillabs(p, format="core") is sentinel


def test_vendor_edf_missing_converter_and_temp_cleanup(monkeypatch, tmp_path):
    edf = tmp_path / "x.edf"
    edf.write_bytes(b"EDF")
    monkeypatch.setattr(vi.shutil, "which", lambda *a, **k: None)
    with pytest.raises(ep.EyeProcessBackendError, match="edf2asc"):
        vi.read_eyelink_edf(edf)

    monkeypatch.setattr(vi.shutil, "which", lambda *a, **k: "/fake/edf2asc")
    made = {}

    def fake_run(args, **kwargs):
        destination = args[-1]
        open(destination, "w", encoding="utf-8").write("MSG 0 start\n")
        made["path"] = destination
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vi.subprocess, "run", fake_run)
    monkeypatch.setattr(vi, "read_eyelink_asc", lambda path, **kwargs: ep.new_eye_dataset(validate=False))
    out = vi.read_eyelink_edf(edf, keep_asc=False)
    assert ep.is_eye_dataset(out)
    assert not vi.Path(made["path"]).exists()


def test_io_private_json_polygon_and_empty_folder(tmp_path):
    idx = pd.Index([1, 2])
    converted = io._jsonable(idx)
    assert converted["class"] == "Index"

    polygon = [[0, 0], [1, 2]]
    text = io._polygon_to_text(polygon)
    assert text == "0,0;1,2"

    folder = tmp_path / "empty-canonical"
    folder.mkdir()
    ds = io.read_eye_dataset(folder, validate=False)
    assert ep.is_eye_dataset(ds)
    assert all(name in ds for name in ep.canonical_table_names())

    provenance_path = tmp_path / "prov.csv"
    io.write_provenance(ds, provenance_path, format=("csv", "json"))
    assert provenance_path.exists()


def test_io_schema_inspection_sort_smi_and_manifest(monkeypatch, tmp_path):
    x = ep.new_eye_dataset(validate=False)
    rec = ep.standardize_eye_table(pd.DataFrame({"recording_id": [pd.NA]}), "recordings")
    x["recordings"] = rec
    coverage = io.schema_coverage(x, require_gaze=False)
    critical = coverage[(coverage.table == "recordings") & (coverage.field == "recording_id")]
    assert critical.status.iloc[0] == "fail"

    assert io._sorted_table(pd.DataFrame(), "recordings", []).empty

    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(io, "detect_eye_format", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("detect")))
    inspected = io.inspect_eye_source(p)
    assert len(inspected) == 1

    original_read_text = io.Path.read_text

    def boom(self, *args, **kwargs):
        if self == p:
            raise OSError("blocked")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(io.Path, "read_text", boom)
    assert io._smi_confidence(p) == 0.0

    with pytest.raises(ep.EyeProcessValidationError, match="Manifest does not exist"):
        io.read_validation_manifest(tmp_path / "missing.csv")


def test_validate_eye_source_auto_detection_selection(monkeypatch, tmp_path):
    p = tmp_path / "source.csv"
    p.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(io, "inspect_eye_source", lambda *a, **k: pd.DataFrame({"source_path": [str(p)]}))
    monkeypatch.setattr(io, "detect_eye_format", lambda *a, **k: pd.DataFrame({"format": ["generic"], "confidence": [0.75], "priority": [1]}))
    monkeypatch.setattr(io, "read_eye_export", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("intentional import boundary")))
    result = io.validate_eye_source(p, vendor="auto")
    assert result.vendor == "generic"
    assert float(result.detection.iloc[0]["confidence"]) == pytest.approx(0.75)
