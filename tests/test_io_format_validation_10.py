from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGET_EXPORTS = {
    "anonymize_eye_dataset",
    "as_eye_biometrics",
    "compare_eye_datasets",
    "create_validation_bundle",
    "discover_validation_cases",
    "export_canonical",
    "eye_format_profiles",
    "fingerprint_eye_dataset",
    "format_compatibility_matrix",
    "format_validation_spec",
    "import_canonical",
    "init_validation_corpus",
    "inspect_eye_source",
    "read_eye_dataset",
    "read_validation_manifest",
    "report_eye_dataset",
    "report_processirt",
    "roundtrip_eye_dataset",
    "schema_coverage",
    "schema_coverage_summary",
    "source_preservation_audit",
    "validate_eye_corpus",
    "validate_eye_source",
    "validate_eyelink_export",
    "validate_generic_export",
    "validate_pupillabs_export",
    "validate_smi_export",
    "validate_tobii_export",
    "validation_manifest",
    "write_eye_dataset",
    "write_format_validation_report",
    "write_provenance",
    "write_validation_manifest",
}


def _dataset():
    recordings = pd.DataFrame(
        [
            {
                "recording_id": "REC01",
                "participant_id": "PERSON01",
                "session_id": "SESSION01",
                "vendor": "Generic",
                "vendor_family": "Generic",
                "experiment_type": "private-task",
                "nominal_sampling_rate": 10.0,
                "source_file_set": "C:/private/source.csv",
            }
        ]
    )
    streams = pd.DataFrame(
        [
            {
                "stream_id": "G1",
                "recording_id": "REC01",
                "stream_type": "gaze_combined",
                "source_clock": "native",
                "sampling_type": "sampled",
                "nominal_rate_hz": 10.0,
                "observed_rate_hz": 10.0,
                "timestamp_unit": "seconds",
                "value_unit": "normalized",
                "coordinate_space_id": "C1",
                "processing_level": "raw_imported",
            },
            {
                "stream_id": "B1",
                "recording_id": "REC01",
                "stream_type": "eda",
                "source_clock": "native",
                "sampling_type": "sampled",
                "nominal_rate_hz": 10.0,
                "observed_rate_hz": 10.0,
                "timestamp_unit": "seconds",
                "value_unit": "microsiemens",
                "coordinate_space_id": pd.NA,
                "processing_level": "raw_imported",
            },
        ]
    )
    spaces = ep.new_coordinate_space("C1")
    t = np.arange(0.0, 0.5, 0.1)
    gaze = pd.DataFrame(
        {
            "recording_id": "REC01",
            "stream_id": "G1",
            "sample_id": [f"G{i}" for i in range(len(t))],
            "timestamp_native": t,
            "timestamp_seconds": t,
            "gaze_x": np.linspace(0.1, 0.5, len(t)),
            "gaze_y": np.linspace(0.2, 0.6, len(t)),
            "valid": [True] * len(t),
            "trial_id": "TRIAL01",
            "stimulus_id": "STIM01",
            "coordinate_space_id": "C1",
        }
    )
    eyes = pd.DataFrame(
        {
            "recording_id": "REC01",
            "sample_id": [f"E{i}" for i in range(len(t))],
            "timestamp_native": t,
            "timestamp_seconds": t,
            "eye": "left",
            "pupil_diameter": np.linspace(3.0, 3.4, len(t)),
            "pupil_unit": "mm",
            "pupil_valid": [True] * len(t),
            "trial_id": "TRIAL01",
            "stimulus_id": "STIM01",
        }
    )
    intervals = pd.DataFrame(
        [
            {
                "interval_id": "INT01",
                "recording_id": "REC01",
                "interval_type": "trial",
                "start_time": 0.0,
                "end_time": 0.4,
                "trial_id": "TRIAL01",
                "participant_id": "PERSON01",
                "item_id": "ITEM01",
                "stimulus_id": "STIM01",
                "condition_id": "PRIVATE_CONDITION",
                "valid_interval": True,
            }
        ]
    )
    responses = pd.DataFrame(
        [
            {
                "response_id": "RESP01",
                "recording_id": "REC01",
                "participant_id": "PERSON01",
                "trial_id": "TRIAL01",
                "item_id": "ITEM01",
                "response": "SECRET RESPONSE",
                "score": 1.0,
                "response_time": 0.3,
                "response_timestamp": 0.3,
                "response_type": "observed",
                "valid_response": True,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "EV01",
                "recording_id": "REC01",
                "timestamp_native": 0.1,
                "timestamp_seconds": 0.1,
                "event_type": "message",
                "event_name": "PRIVATE EVENT",
                "event_value": "SECRET",
                "native_record": "raw private text",
                "trial_id": "TRIAL01",
                "stimulus_id": "STIM01",
            }
        ]
    )
    biometrics = pd.DataFrame(
        {
            "recording_id": "REC01",
            "stream_id": "B1",
            "timestamp_native": t,
            "timestamp_seconds": t,
            "channel": "eda",
            "value": np.linspace(1.0, 1.4, len(t)),
            "unit": "microsiemens",
            "valid": [True] * len(t),
            "processing_level": "raw_imported",
            "source_device": "device",
            "trial_id": "TRIAL01",
            "stimulus_id": "STIM01",
        }
    )
    aoi_definitions = pd.DataFrame(
        [
            {
                "aoi_id": "AOI_PRIVATE",
                "aoi_name": "Sensitive label",
                "stimulus_id": "STIM01",
                "shape_type": "rectangle",
                "coordinate_space_id": "C1",
                "source": "manual",
            }
        ]
    )
    aoi_geometry = pd.DataFrame(
        [
            {
                "aoi_id": "AOI_PRIVATE",
                "valid_from": 0.0,
                "valid_to": 1.0,
                "frame_id": "F1",
                "polygon": np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]),
                "visible": True,
                "coordinate_space_id": "C1",
            }
        ]
    )
    x = ep.new_eye_dataset(
        recordings=recordings,
        streams=streams,
        gaze_samples=gaze,
        eye_samples=eyes,
        events=events,
        intervals=intervals,
        responses=responses,
        coordinate_spaces=spaces,
        aoi_definitions=aoi_definitions,
        aoi_geometry=aoi_geometry,
        biometrics=biometrics,
        raw=[{"source_path": "C:/private/source.csv"}],
        vendor_metadata={"source_path": "C:/private/source.csv", "device": "demo"},
        validate=False,
    )
    return ep.add_provenance(
        x,
        "import",
        "dataset",
        "private source",
        source_files="C:/private/source.csv",
    )


def test_all_33_targets_are_public_callables():
    assert len(TARGET_EXPORTS) == 33
    assert all(callable(getattr(ep, name, None)) for name in TARGET_EXPORTS)


def test_format_validation_spec_contract():
    spec = ep.format_validation_spec(min_detection_confidence=0.7, run_roundtrip=False)
    assert spec.min_detection_confidence == pytest.approx(0.7)
    assert spec.run_roundtrip is False
    with pytest.raises(ep.EyeProcessValidationError):
        ep.format_validation_spec(min_detection_confidence=1.1)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.format_validation_spec(numeric_tolerance=-1)


def test_eye_format_profiles_exact_shape():
    profiles = ep.eye_format_profiles()
    assert len(profiles) == 12
    assert profiles.iloc[0]["format_id"] == "gazepoint_analysis"
    assert profiles.iloc[-1]["format_id"] == "generic_delimited"
    assert {"gaze", "pupil", "events", "biometrics", "validation_level"} <= set(profiles.columns)


def test_canonical_folder_write_read_and_aliases(tmp_path):
    x = _dataset()
    folder = tmp_path / "canonical"
    out = ep.write_eye_dataset(x, folder, include_raw=True)
    assert Path(out).is_dir()
    assert (folder / "gaze_samples.csv").is_file()
    assert (folder / ".eyeprocess-serialization.json").is_file()
    assert "__EYEPROCESS_MISSING_6E7A4D2F__" in (folder / "recordings.csv").read_text(encoding="utf-8")

    restored = ep.read_eye_dataset(folder)
    cmp = ep.compare_eye_datasets(x, restored)
    assert cmp["status"].eq("pass").all()

    folder2 = tmp_path / "canonical2"
    ep.export_canonical(x, folder2)
    assert ep.import_canonical(folder2)["recordings"].iloc[0]["recording_id"] == "REC01"


def test_rds_routes_are_explicitly_gated(tmp_path):
    x = _dataset()
    with pytest.raises(ep.EyeProcessBackendError):
        ep.write_eye_dataset(x, tmp_path / "x.rds")
    with pytest.raises(ep.EyeProcessBackendError):
        ep.write_provenance(x, tmp_path / "p.rds", format="rds")


def test_write_provenance_csv_and_json(tmp_path):
    x = _dataset()
    csv_path = Path(ep.write_provenance(x, tmp_path / "prov.csv", format="csv"))
    json_path = Path(ep.write_provenance(x, tmp_path / "prov.json", format="json"))
    assert csv_path.is_file() and json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "actions" in payload


def test_report_eye_dataset_and_alias(tmp_path):
    x = _dataset()
    p = Path(ep.report_eye_dataset(x, tmp_path / "report.md"))
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "# eyeprocess data and analysis report" in text
    assert "## Provenance" in text
    p2 = Path(ep.report_processirt(x, tmp_path / "report2.md"))
    assert p2.is_file()


def test_as_eye_biometrics_eye_dataset_and_dataframe():
    x = _dataset()
    assert len(ep.as_eye_biometrics(x)) == len(x["biometrics"])

    source = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "eda": [1.0, 2.0],
        }
    )
    mapping = ep.eye_mapping(
        timestamp="time",
        biometric_channels={"eda": "eda"},
    )
    out = ep.as_eye_biometrics(source, mapping=mapping)
    assert len(out) == 2
    assert set(out["channel"]) == {"eda"}


def test_schema_coverage_and_summary():
    coverage = ep.schema_coverage(_dataset())
    assert {"table", "field", "critical", "status"} <= set(coverage.columns)
    hit = coverage[(coverage["table"] == "gaze_samples") & (coverage["field"] == "timestamp_native")]
    assert hit.iloc[0]["status"] == "pass"
    summary = ep.schema_coverage_summary(coverage)
    assert "gaze_samples" in set(summary["table"])


def test_source_preservation_audit_contract():
    audit = ep.source_preservation_audit(_dataset(), require_raw=True)
    assert set(audit["check"]) == {
        "source_provenance",
        "source_file_reference",
        "source_file_hash",
        "native_timestamps",
        "normalized_timestamps",
        "coordinate_registry",
        "stream_units",
        "pupil_units",
        "vendor_metadata",
        "raw_retention",
    }
    assert audit.loc[audit["check"] == "raw_retention", "status"].iloc[0] == "pass"


def test_fingerprint_is_deterministic_and_ignores_volatile_ids():
    x = _dataset()
    a = ep.fingerprint_eye_dataset(x)
    b = ep.fingerprint_eye_dataset(x.copy())
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == len(ep.canonical_table_names())


def test_compare_detects_value_change():
    x = _dataset()
    y = x.copy()
    y["gaze_samples"].loc[0, "gaze_x"] = 99.0
    cmp = ep.compare_eye_datasets(x, y)
    gaze = cmp[cmp["table"] == "gaze_samples"].iloc[0]
    assert gaze["status"] == "fail"
    assert gaze["differing_cells"] >= 1


def test_roundtrip_eye_dataset(tmp_path):
    result = ep.roundtrip_eye_dataset(
        _dataset(),
        path=tmp_path / "roundtrip",
        cleanup=False,
    )
    assert result.status == "pass"
    assert result.comparison["status"].eq("pass").all()
    assert Path(result.path).is_dir()


def test_manifest_write_read_init_and_discovery(tmp_path):
    a = tmp_path / "case-a.csv"
    a.write_text("timestamp,x,y\n0,0.1,0.2\n", encoding="utf-8")
    manifest = ep.validation_manifest(a, vendor="generic", run_roundtrip=False)
    path = Path(ep.write_validation_manifest(manifest, tmp_path / "manifest.csv"))
    restored = ep.read_validation_manifest(path)
    assert restored.iloc[0]["vendor"] == "generic"
    assert Path(restored.iloc[0]["path"]).is_absolute()

    corpus = Path(ep.init_validation_corpus(tmp_path / "corpus"))
    assert (corpus / "validation-manifest.csv").is_file()
    (corpus / "case1").mkdir()
    discovered = ep.discover_validation_cases(corpus)
    assert len(discovered) == 1


def test_inspect_eye_source_and_generic_validator(tmp_path):
    f = tmp_path / "generic.csv"
    f.write_text("timestamp,x,y\n0,0.1,0.2\n1,0.2,0.3\n", encoding="utf-8")
    inspected = ep.inspect_eye_source(f)
    assert inspected.iloc[0]["extension"] == "csv"
    assert inspected.iloc[0]["n_columns"] == 3
    assert ep.validate_generic_export(f).empty


def test_vendor_source_validators(tmp_path):
    tobii = tmp_path / "tobii.tsv"
    tobii.write_text(
        "Recording timestamp\tGaze point X\tGaze point Y\tValidity left\n1\t10\t20\tValid\n",
        encoding="utf-8",
    )
    assert ep.validate_tobii_export(tobii).empty

    neon = tmp_path / "neon"
    neon.mkdir()
    (neon / "gaze.csv").write_text("timestamp [ns],gaze x,gaze y\n1,1,1\n", encoding="utf-8")
    assert ep.validate_pupillabs_export(neon).empty

    asc = tmp_path / "sample.asc"
    asc.write_text("MSG 100 TRIAL_START\n100 20 30 1000\n", encoding="utf-8")
    assert ep.validate_eyelink_export(asc).empty

    edf = tmp_path / "sample.edf"
    edf.write_bytes(b"EDF")
    assert ep.validate_eyelink_export(edf).iloc[0]["code"] == "external_converter_required"

    smi = tmp_path / "smi.txt"
    smi.write_text("SMI BeGaze\tPOR X\tPOR Y\tPupil Diameter\n", encoding="utf-8")
    assert ep.validate_smi_export(smi).empty


def test_validate_eye_source_generic_explicit_vendor(tmp_path):
    f = tmp_path / "generic.csv"
    f.write_text(
        "participant,recording,time,x,y\nP1,R1,0,0.1,0.2\nP1,R1,1,0.2,0.3\n",
        encoding="utf-8",
    )
    mapping = ep.eye_mapping(
        participant="participant",
        recording="recording",
        timestamp="time",
        x="x",
        y="y",
    )
    spec = ep.format_validation_spec(
        require_native_time=True,
        require_provenance=True,
        run_roundtrip=False,
    )
    result = ep.validate_eye_source(
        f,
        vendor="generic",
        spec=spec,
        import_args={"mapping": mapping},
        retain_dataset=True,
    )
    assert result.vendor == "generic"
    assert result.dataset is not None
    assert result.status in {"pass", "warning"}
    assert (result.checks["check"] == "import").any()


def test_validate_eye_corpus_and_compatibility_matrix(tmp_path):
    f = tmp_path / "generic.csv"
    f.write_text("time,x,y\n0,0.1,0.2\n1,0.2,0.3\n", encoding="utf-8")
    manifest = ep.validation_manifest(
        f,
        vendor="generic",
        format_family="generic_delimited",
        run_roundtrip=False,
    )
    mapping = ep.eye_mapping(timestamp="time", x="x", y="y")
    result = ep.validate_eye_corpus(
        manifest,
        spec=ep.format_validation_spec(run_roundtrip=False),
        import_args={"mapping": mapping},
    )
    assert len(result.summary) == 1
    matrix = ep.format_compatibility_matrix(result)
    generic = matrix[matrix["format_id"] == "generic_delimited"].iloc[0]
    assert generic["empirical_cases"] == 1


def test_anonymize_eye_dataset_remaps_linked_ids_and_redacts():
    out = ep.anonymize_eye_dataset(_dataset(), retain_map=True)
    assert out["recordings"].iloc[0]["participant_id"].startswith("P")
    assert out["recordings"].iloc[0]["recording_id"].startswith("R")
    assert out["responses"].iloc[0]["recording_id"] == out["recordings"].iloc[0]["recording_id"]
    assert out["responses"].iloc[0]["response"].startswith("response_")
    assert out["events"].iloc[0]["event_value"] is pd.NA or pd.isna(out["events"].iloc[0]["event_value"])
    assert out.raw == []
    assert out.vendor_metadata == {"redacted": True}
    assert hasattr(out, "anonymization_map")


def test_format_validation_report_and_bundle(tmp_path):
    f = tmp_path / "generic.csv"
    f.write_text("time,x,y\n0,0.1,0.2\n1,0.2,0.3\n", encoding="utf-8")
    mapping = ep.eye_mapping(timestamp="time", x="x", y="y")
    result = ep.validate_eye_source(
        f,
        vendor="generic",
        spec=ep.format_validation_spec(run_roundtrip=False),
        import_args={"mapping": mapping},
        retain_dataset=True,
    )
    report = Path(ep.write_format_validation_report(result, tmp_path / "validation.md"))
    assert report.is_file()

    bundle = Path(
        ep.create_validation_bundle(
            result,
            tmp_path / "bundle.zip",
            include_dataset=True,
            anonymize=True,
        )
    )
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
    assert "validation-report.md" in names
    assert "checks.csv" in names
    assert any(name.startswith("canonical-dataset/") for name in names)
