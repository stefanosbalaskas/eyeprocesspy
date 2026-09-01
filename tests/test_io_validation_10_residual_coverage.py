from __future__ import annotations

import copy
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.io_validation_10 as io

FIX = Path(__file__).parent / "fixtures" / "gazepoint"


def _dataset():
    x = ep.read_gazepoint_folder(FIX, recording_id="REC01", quiet=True)
    x = x.copy()
    x["recordings"] = x["recordings"].copy()
    x["recordings"].loc[:, "source_file_set"] = "C:/private/source.csv"
    x["recordings"].loc[:, "experiment_type"] = "private-task"
    x["events"] = io.standardize_eye_table(
        pd.DataFrame([{
            "event_id": "EV1", "recording_id": "REC01",
            "timestamp_native": 0.1, "timestamp_seconds": 0.1,
            "event_type": "message", "event_name": "PRIVATE EVENT",
            "event_value": "SECRET", "native_record": "raw private text",
            "trial_id": "TR1", "stimulus_id": "ST1",
        }]), "events"
    )
    x["intervals"] = io.standardize_eye_table(
        pd.DataFrame([{
            "interval_id": "IN1", "recording_id": "REC01",
            "interval_type": "trial", "start_time": 0.0, "end_time": 0.4,
            "trial_id": "TR1", "participant_id": x["recordings"]["participant_id"].iloc[0],
            "item_id": "IT1", "stimulus_id": "ST1", "condition_id": "PRIVATE_CONDITION",
            "valid_interval": True,
        }]), "intervals"
    )
    x["responses"] = io.standardize_eye_table(
        pd.DataFrame([{
            "response_id": "RS1", "recording_id": "REC01",
            "participant_id": x["recordings"]["participant_id"].iloc[0],
            "trial_id": "TR1", "item_id": "IT1", "response": "SECRET RESPONSE",
            "score": 1.0, "response_time": 0.3, "response_timestamp": 0.3,
            "response_type": "observed", "valid_response": True,
        }]), "responses"
    )
    x["aoi_definitions"] = io.standardize_eye_table(
        pd.DataFrame([{
            "aoi_id": "AOI_PRIVATE", "aoi_name": "Sensitive label",
            "stimulus_id": "ST1", "shape_type": "rectangle",
            "coordinate_space_id": x["coordinate_spaces"]["coordinate_space_id"].iloc[0],
            "source": "manual",
        }]), "aoi_definitions"
    )
    x["aoi_geometry"] = io.standardize_eye_table(
        pd.DataFrame([{
            "aoi_id": "AOI_PRIVATE", "valid_from": 0.0, "valid_to": 1.0,
            "frame_id": "F1", "polygon": np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]),
            "visible": True,
            "coordinate_space_id": x["coordinate_spaces"]["coordinate_space_id"].iloc[0],
        }]), "aoi_geometry"
    )
    x.raw = [{"source_path": "C:/private/source.csv"}]
    x.vendor_metadata = {"source_path": "C:/private/source.csv", "device": "demo"}
    return ep.add_provenance(
        x, "import", "dataset", "private source",
        source_files="C:/private/source.csv"
    )


def _format_result(case_id="case", status="pass", imported=True, dataset=None,
                   adapter_issues=None, roundtrip=None):
    checks = pd.DataFrame([
        {"check": "format_detection", "status": "pass", "value": 1.0, "message": "ok"},
        {"check": "import", "status": "pass" if imported else "fail",
         "value": 1 if imported else 0, "message": "ok" if imported else "failed"},
    ])
    source = pd.DataFrame([{
        "source_path": "/private/source.csv", "relative_path": "source.csv",
        "file_name": "source.csv", "extension": "csv",
        "md5": "abc", "modified": "now",
    }])
    return io.EyeFormatValidation(
        case_id=case_id, path="/private/source.csv", vendor="generic",
        status=status, started="start", completed="done",
        spec=io.format_validation_spec(run_roundtrip=False),
        source=source,
        detection=pd.DataFrame([{"format": "generic", "confidence": 1.0}]),
        adapter_issues=io._empty_vendor_issues() if adapter_issues is None else adapter_issues,
        checks=checks, validation=pd.DataFrame(), coverage=pd.DataFrame(),
        preservation=pd.DataFrame(), audits={}, roundtrip=roundtrip,
        import_error=pd.NA if imported else "failed", dataset=dataset,
    )


def test_report_plot_branches_and_biometrics_guards(tmp_path, monkeypatch):
    x = _dataset()
    calls = []
    monkeypatch.setattr(io, "_save_plot", lambda path, fn: calls.append(Path(path).name))
    monkeypatch.setattr(ep, "plot_eye_overview", lambda data: object(), raising=False)
    monkeypatch.setattr(ep, "plot_signal_quality", None, raising=False)
    monkeypatch.setattr(ep, "plot_sampling_rate", lambda data: object(), raising=False)
    monkeypatch.setattr(ep, "plot_pupil_timeseries", lambda data: object(), raising=False)

    io.report_eye_dataset(x, tmp_path / "r1.md", include_plots=True)
    assert {"overview.png", "sampling-rate.png", "pupil.png"} <= set(calls)

    calls.clear()
    io.report_eye_dataset(
        x, tmp_path / "r2.md", include_plots=True,
        plot_directory=tmp_path / "explicit-plots",
    )
    assert calls

    with pytest.raises(TypeError, match="EyeDataset"):
        io.as_eye_biometrics(object())
    with pytest.raises(io.EyeProcessValidationError, match="mapping"):
        io.as_eye_biometrics(pd.DataFrame({"time": [0.0], "eda": [1.0]}))

    captured = {}
    def fake_read_eye_generic(frame, mapping, time_unit, **kwargs):
        captured["columns"] = set(frame.columns)
        captured["mapping"] = dict(mapping)
        captured["time_unit"] = time_unit
        return _dataset()
    monkeypatch.setattr(io, "read_eye_generic", fake_read_eye_generic)
    result = io.as_eye_biometrics(
        pd.DataFrame({"time": [0.0], "eda": [1.0]}),
        mapping={"timestamp": "time", "biometric_channels": {"eda": "eda"}},
        time_unit="milliseconds",
    )
    assert len(result) >= 0
    assert {".eye_x", ".eye_y"} <= captured["columns"]
    assert captured["mapping"]["x"] == ".eye_x"
    assert captured["mapping"]["y"] == ".eye_y"
    assert captured["time_unit"] == "milliseconds"


def test_read_corrupt_optional_json_and_schema_edge_paths(tmp_path):
    x = _dataset()
    folder = tmp_path / "canonical"
    io.write_eye_dataset(x, folder, include_raw=True)
    (folder / io._VENDOR_METADATA_JSON).write_text("{bad", encoding="utf-8")
    (folder / io._RAW_JSON).write_text("{bad", encoding="utf-8")
    restored = io.read_eye_dataset(folder, validate=False)
    assert restored.vendor_metadata == {}
    assert restored.raw == []

    no_meta = tmp_path / "no-meta"
    io.write_eye_dataset(x, no_meta)
    (no_meta / io._SERIALIZATION_JSON).unlink()
    assert io.read_eye_dataset(no_meta, validate=False).schema_version == "0.1.0"

    broken = x.copy()
    broken["gaze_samples"] = broken["gaze_samples"].drop(columns=["gaze_x"])
    coverage = io.schema_coverage(broken)
    row = coverage[(coverage["table"] == "gaze_samples") & (coverage["field"] == "gaze_x")].iloc[0]
    assert row["status"] == "fail"

    empty = x.copy()
    empty["recordings"] = io.empty_eye_table("recordings")
    empty["gaze_samples"] = io.empty_eye_table("gaze_samples")
    coverage = io.schema_coverage(empty, require_gaze=True)
    assert coverage[(coverage["table"] == "recordings") & coverage["critical"]]["status"].eq("fail").any()
    assert coverage[(coverage["table"] == "gaze_samples") & coverage["critical"]]["status"].eq("fail").any()

    partial = x.copy()
    partial["recordings"] = pd.concat([x["recordings"], x["recordings"]], ignore_index=True)
    partial["recordings"].loc[1, "vendor"] = pd.NA
    coverage = io.schema_coverage(partial)
    vendor = coverage[(coverage["table"] == "recordings") & (coverage["field"] == "vendor")].iloc[0]
    assert vendor["status"] == "warning"


def test_stability_safe_audit_and_compare_edges():
    class Ambiguous:
        def __repr__(self):
            return "Ambiguous()"
    assert io._jsonable(Ambiguous())["class"] == "Ambiguous"
    assert io._stable_cell(float("inf")) == "Inf"
    assert io._stable_cell(float("-inf")) == "-Inf"
    assert io._stable_cell(np.int64(3)) == "3"
    assert io._stable_cell(np.bool_(True)) == "TRUE"
    assert io._stable_cell([1, pd.NA]) == "[1,null]"
    assert io._stable_table(pd.DataFrame({"a": [1]}), ignore_columns=("a",)).empty
    assert io._unsorted_stable_keys(pd.DataFrame(index=[0, 1]), []).tolist() == ["", ""]
    duplicated = pd.DataFrame({"sample_id": ["x", "x"], "value": [2, 1]})
    assert io._sorted_table(duplicated, "gaze_samples", ["sample_id", "value"])["value"].tolist() == [1, 2]
    failed = io._safe_audit(lambda data: (_ for _ in ()).throw(RuntimeError("boom")), object())
    assert failed.iloc[0]["code"] == "audit_failed"

    x = _dataset()
    y = x.copy()
    with pytest.raises(io.EyeProcessValidationError, match="numeric_tolerance"):
        io.compare_eye_datasets(x, y, numeric_tolerance=-1)


def test_validate_eye_source_failure_adapter_and_roundtrip_paths(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("time,x,y\n0,0.1,0.2\n", encoding="utf-8")
    with pytest.raises(io.EyeProcessValidationError, match="spec"):
        io.validate_eye_source(source, spec={})
    with pytest.raises(io.EyeProcessValidationError, match="import_args"):
        io.validate_eye_source(source, import_args=[])
    with pytest.raises(io.EyeProcessValidationError, match="does not exist"):
        io.validate_eye_source(tmp_path / "missing.csv")

    monkeypatch.setattr(io, "inspect_eye_source", lambda p: pd.DataFrame([{"file_name": Path(p).name}]))
    monkeypatch.setattr(io, "detect_eye_format", lambda p: (_ for _ in ()).throw(RuntimeError("detect")))
    monkeypatch.setattr(io, "read_eye_export", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("import")))
    failed = io.validate_eye_source(source, spec=io.format_validation_spec(run_roundtrip=False))
    assert failed.status == "fail"
    assert failed.detection.empty
    assert "import" in str(failed.import_error)

    x = _dataset()
    monkeypatch.setattr(io, "detect_eye_format", lambda p: pd.DataFrame([{
        "format": "fake", "confidence": 0.1, "priority": 1
    }]))
    monkeypatch.setattr(io, "read_eye_export", lambda *a, **k: x)
    monkeypatch.setattr(io, "validate_eye_dataset", lambda *a, **k: pd.DataFrame([
        {"severity": "warning", "code": "demo", "message": "warning"}
    ]))
    monkeypatch.setattr(io, "schema_coverage", lambda *a, **k: pd.DataFrame([
        {"critical": True, "status": "warning"},
        {"critical": True, "status": "pass"},
    ]))
    monkeypatch.setattr(io, "source_preservation_audit", lambda *a, **k: pd.DataFrame([
        {"check": "native_timestamps", "status": "pass"},
        {"check": "coordinate_registry", "status": "pass"},
        {"check": "source_provenance", "status": "pass"},
    ]))
    for name in ["audit_timebase", "audit_coordinate_spaces", "audit_sampling_rate",
                 "audit_signal_quality", "audit_event_order"]:
        monkeypatch.setattr(io, name, lambda data: pd.DataFrame())
    monkeypatch.setattr(io, "roundtrip_eye_dataset", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rt")))

    adapters = copy.deepcopy(io._ADAPTERS)
    adapters["fake"] = {"validate": lambda p: (_ for _ in ()).throw(RuntimeError("adapter"))}
    monkeypatch.setattr(io, "_ADAPTERS", adapters)
    result = io.validate_eye_source(
        source, vendor="fake",
        spec=io.format_validation_spec(
            require_gaze=False, require_native_time=False,
            require_coordinate_space=False, require_provenance=False,
            require_raw_retention=False, run_roundtrip=True,
        ),
        retain_dataset=True, case_id="explicit",
    )
    assert result.case_id == "explicit"
    assert result.dataset is x
    assert result.roundtrip.status == "fail"
    assert result.checks.loc[result.checks["check"] == "format_detection", "status"].iloc[0] == "warning"
    assert "adapter_specific_validation" in set(result.checks["check"])

    adapters["fake"]["validate"] = lambda p: io._vendor_issue("warning", "demo", p, "warn")
    monkeypatch.setattr(io, "_ADAPTERS", adapters)
    warning = io.validate_eye_source(
        source, vendor="fake",
        spec=io.format_validation_spec(run_roundtrip=False, require_gaze=False),
    )
    assert warning.roundtrip is None
    assert warning.checks.loc[
        warning.checks["check"] == "adapter_specific_validation", "status"
    ].iloc[0] == "warning"


def test_manifest_recycle_flags_validation_and_discovery(tmp_path):
    assert io._recycle("x", 3, "x") == ["x"] * 3
    assert io._recycle([1], 2, "x") == [1, 1]
    assert io._recycle([1, 2], 2, "x") == [1, 2]
    with pytest.raises(io.EyeProcessValidationError, match="length one"):
        io._recycle([1, 2], 3, "x")

    assert io._coerce_manifest_flag(pd.Series([True, False], dtype="bool"), "flag").tolist() == [True, False]
    flags = io._coerce_manifest_flag(
        pd.Series([pd.NA, "", "na", True, np.bool_(False), 1, 0.0, "yes", "n"], dtype=object), "flag"
    )
    assert flags.iloc[3:].tolist() == [True, False, True, False, True, False]
    with pytest.raises(io.EyeProcessValidationError, match="invalid logical"):
        io._coerce_manifest_flag(pd.Series(["maybe"]), "flag")

    with pytest.raises(TypeError):
        io._validate_manifest_rows([])
    with pytest.raises(io.EyeProcessValidationError, match="missing"):
        io._validate_manifest_rows(pd.DataFrame({"case_id": ["x"]}))
    with pytest.raises(io.EyeProcessValidationError, match="non-empty"):
        io._validate_manifest_rows(pd.DataFrame({"case_id": [""], "path": ["p"], "vendor": ["auto"]}))
    with pytest.raises(io.EyeProcessValidationError, match="unique"):
        io._validate_manifest_rows(pd.DataFrame({
            "case_id": ["dup", "dup"], "path": ["p1", "p2"], "vendor": ["auto", "auto"]
        }))

    with pytest.raises(io.EyeProcessValidationError, match="At least one"):
        io.validation_manifest([])

    a = tmp_path / "a" / "same.csv"
    b = tmp_path / "b" / "same.csv"
    a.parent.mkdir(); b.parent.mkdir()
    a.write_text("x\n1\n", encoding="utf-8"); b.write_text("x\n2\n", encoding="utf-8")
    mf = io.validation_manifest([a, b], vendor="generic", run_roundtrip=False)
    assert mf["case_id"].tolist() == ["same.csv", "same.csv.1"]
    with pytest.raises(io.EyeProcessValidationError, match="case_id"):
        io.validation_manifest([a, b], case_id=["a", "b", "c"])

    relative = tmp_path / "relative.csv"
    pd.DataFrame([{
        "case_id": "rel", "path": "a/same.csv", "vendor": "generic", "run_roundtrip": "false"
    }]).to_csv(relative, index=False)
    loaded = io.read_validation_manifest(relative)
    assert Path(loaded.iloc[0]["path"]).is_absolute()

    corpus = tmp_path / "corpus"
    io.init_validation_corpus(corpus)
    readme = corpus / "README.txt"
    readme.write_text("keep", encoding="utf-8")
    io.init_validation_corpus(corpus, overwrite=False)
    assert readme.read_text(encoding="utf-8") == "keep"
    io.init_validation_corpus(corpus, overwrite=True)
    assert "validation corpus" in readme.read_text(encoding="utf-8")

    empty = tmp_path / "empty"
    empty.mkdir()
    assert io.discover_validation_cases(empty).empty
    excluded = empty / "manifest.csv"
    excluded.mkdir()
    (excluded / "nested.csv").write_text("x\n1\n", encoding="utf-8")
    assert len(io.discover_validation_cases(empty, recursive=True)) == 1
    with pytest.raises(io.EyeProcessValidationError, match="does not exist"):
        io.discover_validation_cases(tmp_path / "missing")


def test_validate_eye_corpus_expectations_per_case_and_stop(tmp_path, monkeypatch):
    p1 = tmp_path / "ok.csv"; p2 = tmp_path / "fail.csv"
    p1.write_text("x\n1\n", encoding="utf-8"); p2.write_text("x\n2\n", encoding="utf-8")
    manifest = io.validation_manifest(
        [p1, p2], vendor="generic", expected_import=[True, False],
        require_gaze=[False, True], run_roundtrip=False, case_id=["ok", "expected-fail"]
    )
    calls = []
    def fake_validate(path, vendor, spec, import_args, retain_dataset, case_id):
        calls.append((case_id, dict(import_args), spec.require_gaze, retain_dataset))
        return _format_result(
            case_id=case_id,
            status="fail" if case_id == "expected-fail" else "warning",
            imported=case_id != "expected-fail",
        )
    monkeypatch.setattr(io, "validate_eye_source", fake_validate)
    corpus = io.validate_eye_corpus(
        manifest, import_args={"ok": {"alpha": 1}, "global": 2}, retain_datasets=True
    )
    assert corpus.status == "warning"
    assert corpus.summary.loc[corpus.summary["case_id"] == "expected-fail", "status"].iloc[0] == "pass"
    assert calls[0] == ("ok", {"alpha": 1}, False, True)
    assert calls[1][2] is True

    bad = io.validation_manifest(p1, vendor="generic", run_roundtrip=False, case_id="bad")
    monkeypatch.setattr(io, "validate_eye_source",
                        lambda *a, **k: _format_result(case_id="bad", status="fail", imported=False))
    with pytest.raises(io.EyeProcessValidationError, match="failed case"):
        io.validate_eye_corpus(bad, stop_on_failure=True)
    with pytest.raises(io.EyeProcessValidationError, match="no cases"):
        io.validate_eye_corpus(pd.DataFrame(columns=["case_id", "path", "vendor"]))

    corpus_dir = tmp_path / "directory"
    io.init_validation_corpus(corpus_dir)
    case = corpus_dir / "case.csv"; case.write_text("x\n1\n", encoding="utf-8")
    path = corpus_dir / "validation-manifest.csv"
    io.write_validation_manifest(
        io.validation_manifest(case, vendor="generic", run_roundtrip=False, case_id="dircase"), path
    )
    monkeypatch.setattr(io, "validate_eye_source",
                        lambda *a, **k: _format_result(case_id=k["case_id"], status="pass", imported=True))
    assert io.validate_eye_corpus(corpus_dir).status == "pass"
    assert io.validate_eye_corpus(path).status == "pass"


def test_anonymization_reports_redaction_and_bundle_variants(tmp_path):
    x = _dataset()
    preserved = io.anonymize_eye_dataset(
        x, drop_raw=False, strip_source_paths=False, redact_free_text=False,
        anonymize_aois=False, retain_map=True
    )
    assert preserved.raw
    assert preserved["events"]["event_value"].iloc[0] == "SECRET"
    assert preserved["responses"]["response"].iloc[0] == "SECRET RESPONSE"
    assert preserved["aoi_definitions"]["aoi_id"].iloc[0] == "AOI_PRIVATE"
    assert hasattr(preserved, "anonymization_map")
    assert "aois" not in preserved.anonymization_map

    redacted_ds = io.anonymize_eye_dataset(
        x, drop_raw=True, strip_source_paths=True, redact_free_text=True,
        anonymize_aois=True, retain_map=False
    )
    assert redacted_ds.raw == []
    assert redacted_ds.vendor_metadata == {"redacted": True}
    assert pd.isna(redacted_ds["events"]["event_value"].iloc[0])
    assert redacted_ds["responses"]["response"].iloc[0].startswith("response_")
    assert redacted_ds["aoi_definitions"]["aoi_id"].iloc[0].startswith("AO")
    assert not hasattr(redacted_ds, "anonymization_map")

    rt = io.EyeRoundtripValidation(
        status="pass",
        comparison=pd.DataFrame([{"table": "gaze_samples", "status": "pass", "content_equal": True}]),
        original_fingerprint=pd.DataFrame(), restored_fingerprint=pd.DataFrame(),
        path="/private/roundtrip", numeric_tolerance=1e-8,
    )
    issues = io._vendor_issue("warning", "demo", tmp_path / "private.csv", "warn")
    result = _format_result(dataset=x, adapter_issues=issues, roundtrip=rt)

    assert Path(io.write_format_validation_report(result, tmp_path / "single.md")).is_file()
    empty = _format_result(imported=False)
    empty.source = pd.DataFrame(); empty.detection = pd.DataFrame()
    empty.adapter_issues = pd.DataFrame(); empty.roundtrip = None
    text = Path(io.write_format_validation_report(empty, tmp_path / "empty.md")).read_text()
    assert "No source files were inspected" in text
    assert "Round-trip validation was not run" in text

    corpus = io.EyeCorpusValidation(
        manifest=pd.DataFrame(), results={"case": result},
        summary=pd.DataFrame([{
            "case_id": "case", "vendor": "generic",
            "format_family": "generic_delimited", "status": "pass"
        }]), status="pass", completed="done"
    )
    assert "## case" in Path(io.write_format_validation_report(corpus, tmp_path / "corpus.md")).read_text()
    with pytest.raises(io.EyeProcessValidationError):
        io.write_format_validation_report(object(), tmp_path / "bad.md")

    safe = io._redact_validation_result(result)
    assert safe.path == "<redacted>"
    assert safe.source.iloc[0]["source_path"] == "<redacted>"
    assert safe.adapter_issues.iloc[0]["file"] == "<redacted>"
    assert safe.roundtrip.path == "<redacted>"
    assert safe.dataset is None

    with pytest.raises(io.EyeProcessValidationError):
        io.create_validation_bundle(object(), tmp_path / "bad.zip")
    with pytest.raises(io.EyeProcessValidationError, match="did not retain"):
        io.create_validation_bundle(empty, tmp_path / "nodata.zip", include_dataset=True)

    bundle = Path(io.create_validation_bundle(
        result, tmp_path / "bundle.zip", include_dataset=False, anonymize=False
    ))
    with zipfile.ZipFile(bundle) as z:
        assert "roundtrip-comparison.csv" in z.namelist()
        assert "canonical-dataset/recordings.csv" not in z.namelist()
    with pytest.raises(io.EyeProcessValidationError, match="exists"):
        io.create_validation_bundle(result, bundle, include_dataset=False)

    io.create_validation_bundle(
        result, bundle, include_dataset=True, anonymize=True, overwrite=True
    )
    with zipfile.ZipFile(bundle) as z:
        assert "canonical-dataset/recordings.csv" in z.namelist()
        assert "Anonymized: TRUE" in z.read("BUNDLE").decode()


def test_remaining_unreadable_vendor_and_roundtrip_own_path(tmp_path, monkeypatch):
    tobii = tmp_path / "bad-tobii.csv"; tobii.write_text("x\n1\n", encoding="utf-8")
    generic = tmp_path / "bad-generic.csv"; generic.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(io, "_read_delimited", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    assert io.validate_tobii_export(tobii).iloc[0]["code"] == "unreadable_export"
    assert io.validate_generic_export(generic).iloc[0]["code"] == "unreadable_export"

    eyelink = tmp_path / "cannot-read.asc"; eyelink.write_text("MSG 1 X\n", encoding="utf-8")
    original = Path.read_text
    def guarded(self, *args, **kwargs):
        if self == eyelink:
            raise OSError("no")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    assert io.validate_eyelink_export(eyelink).iloc[0]["code"] == "unreadable_export"
