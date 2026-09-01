from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.io_validation_10 as io
from eyeprocesspy.exceptions import EyeProcessValidationError


def test_jsonable_handles_tabular_numpy_mappings_sequences_missing_and_repr():
    frame = pd.DataFrame({"a": [1, pd.NA]})
    assert io._jsonable(pd.NA) is None
    assert io._jsonable(frame) == [{"a": 1}, {"a": None}]
    assert io._jsonable(pd.Series([1, pd.NA])) == [1, None]
    assert io._jsonable(np.array([1, 2])) == [1, 2]
    assert io._jsonable(np.int64(3)) == 3
    assert io._jsonable({"x": np.float64(1.5)}) == {"x": 1.5}
    assert io._jsonable((1, pd.NA)) == [1, None]
    assert io._jsonable(float("nan")) is None

    class Example:
        def __repr__(self):
            return "Example(1)"

    encoded = io._jsonable(Example())
    assert encoded == {"class": "Example", "repr": "Example(1)"}


def test_dtype_family_covers_native_and_object_fallbacks():
    assert io._dtype_family(pd.Series([True, False], dtype="bool")) == "boolean"
    assert io._dtype_family(pd.Series([1, 2], dtype="int64")) == "numeric"
    assert io._dtype_family(pd.Series(pd.to_datetime(["2026-01-01"], utc=True))) == "datetime"
    assert io._dtype_family(pd.Series(["a", "b"], dtype="string")) == "string"
    assert io._dtype_family(pd.Series([None, None], dtype="object")) == "object"
    assert io._dtype_family(pd.Series([True, False], dtype="object")) == "boolean"
    assert io._dtype_family(pd.Series([1, 2.5], dtype="object")) == "numeric"
    assert io._dtype_family(pd.Series([{"a": 1}], dtype="object")) == "object"


def test_restore_column_numeric_boolean_datetime_string_and_object():
    numeric = io._restore_column(pd.Series(["1", "bad"]), "numeric")
    assert numeric.iloc[0] == 1
    assert pd.isna(numeric.iloc[1])

    boolean = io._restore_column(pd.Series(["yes", "N", "unknown", None]), "boolean")
    assert boolean.iloc[0] == True
    assert boolean.iloc[1] == False
    assert pd.isna(boolean.iloc[2])
    assert pd.isna(boolean.iloc[3])

    dt = io._restore_column(pd.Series(["2026-01-01", "bad"]), "datetime")
    assert str(dt.dtype).startswith("datetime64")
    assert pd.isna(dt.iloc[1])
    string = io._restore_column(pd.Series([1, None]), "string")
    assert str(string.dtype).startswith("string")
    original = pd.Series([{"a": 1}], dtype="object")
    assert io._restore_column(original, "object") is original


def test_polygon_serialization_empty_missing_and_roundtrip():
    assert pd.isna(io._polygon_to_text(None))
    assert pd.isna(io._polygon_to_text(pd.NA))
    assert pd.isna(io._polygon_to_text(np.empty((0, 2))))
    polygon = np.array([[0.0, 1.0], [2.0, 3.0]])
    text = io._polygon_to_text(polygon)
    np.testing.assert_allclose(io._polygon_from_text(text), polygon)
    assert io._polygon_from_text(None) is None
    assert io._polygon_from_text("") is None
    with pytest.raises(EyeProcessValidationError, match="Invalid serialized AOI polygon"):
        io._polygon_from_text("0,1,2")


def test_write_json_serializes_non_native_values(tmp_path):
    path = tmp_path / "payload.json"
    io._write_json(path, {"missing": pd.NA, "array": np.array([1, 2])})
    text = path.read_text(encoding="utf-8")
    assert '"missing": null' in text
    assert '"array"' in text


def test_markdown_table_empty_escape_and_row_limit():
    assert io._markdown_table(pd.DataFrame()) == ""
    assert io._markdown_table([1, 2]) == ""
    table = io._markdown_table(pd.DataFrame({"a": ["x|y", None, "z"]}), max_rows=2)
    assert "x\\|y" in table
    assert "z" not in table
    assert table.count("\n") == 3


def test_save_plot_accepts_axes_tuple_and_current_figure(tmp_path):
    p1 = tmp_path / "axis.png"
    io._save_plot(p1, lambda: plt.subplots()[1])
    assert p1.exists() and p1.stat().st_size > 0
    plt.close("all")

    p2 = tmp_path / "tuple.png"
    io._save_plot(p2, lambda: plt.subplots())
    assert p2.exists() and p2.stat().st_size > 0
    plt.close("all")

    def no_figure_object():
        plt.figure()
        return object()

    p3 = tmp_path / "current.png"
    io._save_plot(p3, no_figure_object)
    assert p3.exists() and p3.stat().st_size > 0
    plt.close("all")


def test_format_validation_spec_numeric_guards():
    with pytest.raises(EyeProcessValidationError, match="min_detection_confidence"):
        io.format_validation_spec(min_detection_confidence=-0.1)
    with pytest.raises(EyeProcessValidationError, match="min_detection_confidence"):
        io.format_validation_spec(min_detection_confidence=np.inf)
    with pytest.raises(EyeProcessValidationError, match="numeric_tolerance"):
        io.format_validation_spec(numeric_tolerance=-1)
    with pytest.raises(EyeProcessValidationError, match="numeric_tolerance"):
        io.format_validation_spec(numeric_tolerance=np.nan)


def test_eye_format_profiles_shape_and_normalize_key():
    profiles = io.eye_format_profiles()
    assert len(profiles) == 12
    assert {"format_id", "adapter", "validation_level"}.issubset(profiles.columns)
    assert io._normalize_key("Pupil-Labs / Neon") == "pupillabsneon"


def test_format_compatibility_matrix_empirical_format_vendor_and_unmatched_paths():
    summary = pd.DataFrame(
        [
            {"format_family": "Tobii Pro Lab", "vendor": "tobii", "status": "pass"},
            {"format_family": pd.NA, "vendor": "gazepoint", "status": "warning"},
            {"format_family": "generic_delimited", "vendor": "generic", "status": "fail"},
            {"format_family": "unknown", "vendor": "missing", "status": "pass"},
        ]
    )
    corpus = io.EyeCorpusValidation(
        manifest=pd.DataFrame(), results={}, summary=summary, status="warning", completed="now"
    )
    matrix = io.format_compatibility_matrix(corpus)
    assert matrix["empirical_cases"].sum() == 3
    assert matrix["empirical_passes"].sum() == 1
    assert matrix["empirical_warnings"].sum() == 1
    assert matrix["empirical_failures"].sum() == 1


def test_source_delimiter_recognizes_common_delimiters_and_plain_text(tmp_path):
    for name, content, expected in [
        ("comma.csv", "a,b,c\n1,2,3\n", ","),
        ("tab.tsv", "a\tb\tc\n1\t2\t3\n", "\t"),
        ("semi.txt", "a;b;c\n1;2;3\n", ";"),
        ("pipe.txt", "a|b|c\n1|2|3\n", "|"),
    ]:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        assert io._source_delimiter(p) == expected
    plain = tmp_path / "plain.txt"
    plain.write_text("abc\n", encoding="utf-8")
    assert pd.isna(io._source_delimiter(plain))
    assert pd.isna(io._source_delimiter(tmp_path / "missing.txt"))


def test_inspect_eye_source_nonrecursive_directory_and_unreadable_tabular(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "top.csv").write_text("time,x,y\n0,1,2\n", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "nested.csv").write_text("time,x,y\n0,1,2\n", encoding="utf-8")
    report = io.inspect_eye_source(root, recursive=False, inspect_rows=1)
    assert report["file_name"].tolist() == ["top.csv"]
    assert report.loc[0, "inspected_rows"] == 1
    assert report.loc[0, "n_columns"] == 3

    bad = tmp_path / "bad.csv"
    bad.write_text("not really csv", encoding="utf-8")
    monkeypatch.setattr(io, "_read_delimited", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    report = io.inspect_eye_source(bad)
    assert not bool(report.loc[0, "readable"])
    assert report.loc[0, "inspected_rows"] == 0


def test_nonmissing_and_safe_numeric_helpers():
    mask = io._nonmissing(pd.Series([None, pd.NA, np.nan, "", " x ", 0], dtype="object"))
    assert mask.tolist() == [False, False, False, False, True, True]
    numeric = io._safe_numeric(["1", "bad", None])
    assert numeric[0] == 1
    assert np.isnan(numeric[1]) and np.isnan(numeric[2])


def test_vendor_issue_helpers_and_pupil_labs_format(tmp_path):
    issue = io._vendor_issue("warning", "code", tmp_path / "x", "message")
    assert issue.loc[0, "severity"] == "warning"
    assert io._empty_vendor_issues().empty

    neon = tmp_path / "neon"
    neon.mkdir()
    (neon / "gaze.csv").write_text("x\n", encoding="utf-8")
    assert io._pupil_labs_format(neon) == "neon"
    core = tmp_path / "core"
    core.mkdir()
    (core / "gaze_positions.csv").write_text("x\n", encoding="utf-8")
    assert io._pupil_labs_format(core) == "core"
    assert io._pupil_labs_format(neon / "gaze.csv") == "neon"
    assert io._pupil_labs_format(core / "gaze_positions.csv") == "core"
    assert io._pupil_labs_format(tmp_path / "other") == "unknown"


def test_smi_confidence_missing_directory_low_and_high(tmp_path):
    assert io._smi_confidence(tmp_path / "missing") == 0.0
    assert io._smi_confidence(tmp_path) == 0.0
    low = tmp_path / "low.txt"
    low.write_text("ordinary text", encoding="utf-8")
    assert io._smi_confidence(low) == 0.0
    high = tmp_path / "high.txt"
    high.write_text("BeGaze SMI POR X POR Y pupil diameter tracking ratio", encoding="utf-8")
    assert io._smi_confidence(high) == 1.0


def test_validate_tobii_export_directory_missing_fields_and_valid_file(tmp_path):
    directory_issue = io.validate_tobii_export(tmp_path)
    assert directory_issue.loc[0, "code"] == "expected_file"

    weak = tmp_path / "weak.csv"
    weak.write_text("a,b\n1,2\n", encoding="utf-8")
    issues = io.validate_tobii_export(weak)
    assert {"missing_timestamp", "missing_gaze_coordinates", "missing_validity"}.issubset(set(issues["code"]))

    valid = tmp_path / "tobii.csv"
    valid.write_text(
        "Recording timestamp,Gaze point X,Gaze point Y,Validity\n100,0.1,0.2,1\n",
        encoding="utf-8",
    )
    assert io.validate_tobii_export(valid).empty


def test_validate_pupillabs_unknown_neon_core_and_missing_required(tmp_path, monkeypatch):
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    assert io.validate_pupillabs_export(unknown).loc[0, "code"] == "unknown_pupil_format"

    neon_file = tmp_path / "gaze.csv"
    neon_file.write_text("x\n", encoding="utf-8")
    assert io.validate_pupillabs_export(neon_file).empty

    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    monkeypatch.setattr(io, "_pupil_labs_format", lambda p: "neon")
    assert io.validate_pupillabs_export(fake_dir).loc[0, "code"] == "missing_gaze_file"


def test_validate_eyelink_directory_edf_empty_unknown_and_known_asc(tmp_path):
    assert io.validate_eyelink_export(tmp_path).loc[0, "code"] == "expected_file"
    edf = tmp_path / "sample.edf"
    edf.write_bytes(b"binary")
    assert io.validate_eyelink_export(edf).loc[0, "code"] == "external_converter_required"
    empty = tmp_path / "empty.asc"
    empty.write_text("", encoding="utf-8")
    assert io.validate_eyelink_export(empty).loc[0, "code"] == "unreadable_export"
    unknown = tmp_path / "unknown.asc"
    unknown.write_text("HELLO WORLD\n", encoding="utf-8")
    assert io.validate_eyelink_export(unknown).loc[0, "code"] == "unknown_asc_records"
    known = tmp_path / "known.asc"
    known.write_text("MSG 1 TRIAL_START\n100 1.0 2.0\n", encoding="utf-8")
    assert io.validate_eyelink_export(known).empty


def test_validate_smi_directory_idf_low_and_high_confidence(tmp_path):
    assert io.validate_smi_export(tmp_path).loc[0, "code"] == "expected_file"
    idf = tmp_path / "x.idf"
    idf.write_text("x", encoding="utf-8")
    assert io.validate_smi_export(idf).loc[0, "code"] == "proprietary_idf"
    low = tmp_path / "low_smi.txt"
    low.write_text("ordinary text", encoding="utf-8")
    assert io.validate_smi_export(low).loc[0, "code"] == "low_smi_confidence"
    high = tmp_path / "high_smi.txt"
    high.write_text("BeGaze SMI POR X POR Y pupil diameter event info tracking ratio", encoding="utf-8")
    assert io.validate_smi_export(high).empty


def test_validate_generic_directory_mapping_required_and_inferred(tmp_path, monkeypatch):
    assert io.validate_generic_export(tmp_path).loc[0, "code"] == "expected_file"
    f = tmp_path / "generic.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(io, "infer_eye_mapping", lambda d: {})
    issue = io.validate_generic_export(f)
    assert issue.loc[0, "code"] == "mapping_required"
    monkeypatch.setattr(io, "infer_eye_mapping", lambda d: {"timestamp": "a", "x": "b", "y": "b"})
    assert io.validate_generic_export(f).empty


def test_format_validation_summary_handles_absent_severity_and_roundtrip():
    spec = io.format_validation_spec(run_roundtrip=False)
    result = io.EyeFormatValidation(
        case_id="c1",
        path="p",
        vendor="generic",
        status="pass",
        started="s",
        completed="c",
        spec=spec,
        source=pd.DataFrame([{"file": "x"}]),
        detection=pd.DataFrame(),
        adapter_issues=pd.DataFrame(),
        checks=pd.DataFrame([{"check": "import", "status": "pass"}]),
        validation=pd.DataFrame([{"code": "ok"}]),
        coverage=pd.DataFrame(),
        preservation=pd.DataFrame(),
        audits={},
        roundtrip=None,
        import_error=pd.NA,
        dataset=None,
    )
    summary = io._format_validation_summary(result)
    assert summary.loc[0, "detection_confidence"] == 0.0
    assert bool(summary.loc[0, "imported"])
    assert pd.isna(summary.loc[0, "validation_errors"])
    assert summary.loc[0, "roundtrip"] == "not_run"
