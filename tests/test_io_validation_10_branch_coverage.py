from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep


def _mini_dataset():
    recordings = pd.DataFrame(
        [{"recording_id": "R1", "participant_id": "P1", "vendor": "Generic"}]
    )
    spaces = ep.new_coordinate_space("C1")
    gaze = pd.DataFrame(
        {
            "recording_id": ["R1", "R1"],
            "stream_id": ["G1", "G1"],
            "sample_id": ["S1", "S2"],
            "timestamp_native": [0.0, 1.0],
            "timestamp_seconds": [0.0, 1.0],
            "gaze_x": [0.1, 0.2],
            "gaze_y": [0.2, 0.3],
            "valid": [True, True],
            "coordinate_space_id": ["C1", "C1"],
        }
    )
    x = ep.new_eye_dataset(
        recordings=recordings,
        coordinate_spaces=spaces,
        gaze_samples=gaze,
        validate=False,
        raw=[{"source_path": "source.csv"}],
        vendor_metadata={"device": "demo"},
    )
    return ep.add_provenance(
        x,
        "import",
        "dataset",
        "fixture",
        source_files="source.csv",
        file_hashes="abc123",
    )


def test_write_eye_dataset_rejects_invalid_format_file_path_and_nonempty_directory(tmp_path):
    x = _mini_dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="format"):
        ep.write_eye_dataset(x, tmp_path / "bad", format="zip")

    existing_file = tmp_path / "existing"
    existing_file.write_text("x", encoding="utf-8")
    with pytest.raises(ep.EyeProcessValidationError, match="exists as a file"):
        ep.write_eye_dataset(x, existing_file)

    outdir = tmp_path / "nonempty"
    outdir.mkdir()
    (outdir / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ep.EyeProcessValidationError, match="not empty"):
        ep.write_eye_dataset(x, outdir)


def test_write_eye_dataset_overwrite_replaces_managed_files_but_preserves_unmanaged(tmp_path):
    x = _mini_dataset()
    outdir = tmp_path / "canonical"
    ep.write_eye_dataset(x, outdir)
    (outdir / "recordings.csv").write_text("corrupted", encoding="utf-8")
    (outdir / "unmanaged.txt").write_text("keep", encoding="utf-8")
    ep.write_eye_dataset(x, outdir, overwrite=True)
    assert "recording_id" in (outdir / "recordings.csv").read_text(encoding="utf-8")
    assert (outdir / "unmanaged.txt").read_text(encoding="utf-8") == "keep"


def test_read_eye_dataset_rejects_missing_folder_and_rds(tmp_path):
    with pytest.raises(ep.EyeProcessValidationError, match="does not exist"):
        ep.read_eye_dataset(tmp_path / "missing")
    rds = tmp_path / "x.rds"
    rds.write_text("not-rds", encoding="utf-8")
    with pytest.raises(ep.EyeProcessBackendError):
        ep.read_eye_dataset(rds)


def test_read_eye_dataset_tolerates_invalid_optional_json_sidecars(tmp_path):
    x = _mini_dataset()
    outdir = Path(ep.write_eye_dataset(x, tmp_path / "canonical", include_raw=True))
    (outdir / ".eyeprocess-serialization.json").write_text("{bad", encoding="utf-8")
    (outdir / "vendor_metadata.json").write_text("{bad", encoding="utf-8")
    (outdir / "raw.json").write_text("{bad", encoding="utf-8")
    restored = ep.read_eye_dataset(outdir, validate=False)
    assert restored.vendor_metadata == {}
    assert restored.raw == []
    assert len(restored["recordings"]) == 1


def test_read_eye_dataset_rejects_invalid_serialized_polygon(tmp_path):
    x = _mini_dataset()
    aoi = ep.new_aoi(
        "A1",
        shape="polygon",
        polygon=np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]]),
        coordinate_space_id="C1",
    )
    x = ep.register_aois(x, aoi)
    outdir = Path(ep.write_eye_dataset(x, tmp_path / "canonical"))
    p = outdir / "aoi_geometry.csv"
    frame = pd.read_csv(p, dtype=str, keep_default_na=False)
    frame.loc[0, "polygon"] = "0,0;broken"
    frame.to_csv(p, index=False)
    with pytest.raises(ep.EyeProcessValidationError, match="polygon"):
        ep.read_eye_dataset(outdir, validate=False)


def test_write_provenance_rejects_unknown_format(tmp_path):
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid provenance"):
        ep.write_provenance(_mini_dataset(), tmp_path / "x.bad", format="yaml")


@pytest.mark.parametrize(
    "field",
    [
        "require_gaze",
        "require_native_time",
        "require_coordinate_space",
        "require_provenance",
        "require_raw_retention",
        "run_roundtrip",
        "strict",
    ],
)
def test_format_validation_spec_rejects_non_boolean_flags(field):
    with pytest.raises(ep.EyeProcessValidationError, match="boolean"):
        ep.format_validation_spec(**{field: 1})


def test_format_compatibility_matrix_without_corpus_has_zero_empirical_counts():
    matrix = ep.format_compatibility_matrix()
    assert matrix[
        ["empirical_cases", "empirical_passes", "empirical_warnings", "empirical_failures"]
    ].to_numpy().sum() == 0


def test_inspect_eye_source_guards_and_no_hash_mode(tmp_path):
    with pytest.raises(ep.EyeProcessValidationError, match="does not exist"):
        ep.inspect_eye_source(tmp_path / "missing")
    f = tmp_path / "x.csv"
    f.write_text("time,x,y\n0,0.1,0.2\n", encoding="utf-8")
    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        ep.inspect_eye_source(f, inspect_rows=0)
    with pytest.raises(ep.EyeProcessValidationError, match="boolean"):
        ep.inspect_eye_source(f, recursive="yes")
    with pytest.raises(ep.EyeProcessValidationError, match="boolean"):
        ep.inspect_eye_source(f, include_hash="yes")
    report = ep.inspect_eye_source(f, include_hash=False)
    assert pd.isna(report.loc[0, "md5"])


def test_schema_coverage_require_gaze_flag_changes_empty_gaze_semantics():
    x = _mini_dataset()
    x["gaze_samples"] = x["gaze_samples"].iloc[0:0].copy()
    strict = ep.schema_coverage(x, require_gaze=True)
    relaxed = ep.schema_coverage(x, require_gaze=False)
    strict_rows = strict[(strict["table"] == "gaze_samples") & strict["critical"]]
    relaxed_rows = relaxed[(relaxed["table"] == "gaze_samples") & relaxed["critical"]]
    assert strict_rows["status"].eq("fail").any()
    assert not relaxed_rows["status"].eq("fail").any()
    with pytest.raises(ep.EyeProcessValidationError, match="boolean"):
        ep.schema_coverage(x, require_gaze=1)


def test_schema_coverage_summary_rejects_wrong_input_type():
    with pytest.raises(TypeError):
        ep.schema_coverage_summary(object())


def test_source_preservation_raw_requirement_and_flag_guard():
    x = _mini_dataset()
    no_raw = x.copy()
    no_raw.raw = []
    audit = ep.source_preservation_audit(no_raw, require_raw=True)
    assert audit.loc[audit["check"].eq("raw_retention"), "status"].iloc[0] == "fail"
    with pytest.raises(ep.EyeProcessValidationError, match="boolean"):
        ep.source_preservation_audit(x, require_raw=1)


def test_fingerprint_filters_unknown_tables_and_can_include_volatile_columns():
    x = _mini_dataset()
    filtered = ep.fingerprint_eye_dataset(x, tables=["recordings", "not_a_table"])
    assert filtered["table"].tolist() == ["recordings"]
    all_columns = ep.fingerprint_eye_dataset(x, tables=["provenance"], ignore_volatile=False)
    stable_columns = ep.fingerprint_eye_dataset(x, tables=["provenance"], ignore_volatile=True)
    assert all_columns.loc[0, "compared_columns"] >= stable_columns.loc[0, "compared_columns"]


def test_compare_eye_datasets_guards_tolerance_and_detects_row_count_changes():
    x = _mini_dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="numeric_tolerance"):
        ep.compare_eye_datasets(x, x, numeric_tolerance=-1)
    y = x.copy()
    y["gaze_samples"] = y["gaze_samples"].iloc[:1].copy()
    cmp = ep.compare_eye_datasets(x, y, tables=["gaze_samples"])
    assert cmp.loc[0, "status"] == "fail"
    assert cmp.loc[0, "differing_cells"] >= 1


def test_compare_eye_datasets_respects_numeric_tolerance():
    x = _mini_dataset()
    y = x.copy()
    y["gaze_samples"].loc[0, "gaze_x"] += 1e-9
    loose = ep.compare_eye_datasets(x, y, tables=["gaze_samples"], numeric_tolerance=1e-8)
    tight = ep.compare_eye_datasets(x, y, tables=["gaze_samples"], numeric_tolerance=1e-12)
    assert loose.loc[0, "status"] == "pass"
    assert tight.loc[0, "status"] == "fail"


@pytest.mark.parametrize(("field", "value"), [("include_raw", 1), ("cleanup", 1)])
def test_roundtrip_rejects_non_boolean_flags(field, value):
    with pytest.raises(ep.EyeProcessValidationError, match="boolean"):
        ep.roundtrip_eye_dataset(_mini_dataset(), **{field: value})


def test_roundtrip_cleanup_removes_requested_output(tmp_path):
    target = tmp_path / "roundtrip"
    result = ep.roundtrip_eye_dataset(_mini_dataset(), path=target, cleanup=True)
    assert result.status == "pass"
    assert not target.exists()
