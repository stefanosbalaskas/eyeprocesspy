from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.vendor_importers_10 as vi
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessValidationError


def _write(frame: pd.DataFrame, path: Path, sep: str = ",") -> Path:
    frame.to_csv(path, index=False, sep=sep)
    return path


def test_vendor_private_helpers_cover_failure_empty_append_and_boolean_routes(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    assert vi._safe_read_head(missing, 0) is None
    assert vi._pick(["A", "B"], "B", "C") == "B"
    assert vi._pick(["A"], "Z") is None

    mean = vi._row_mean(pd.Series([1.0, np.nan]), pd.Series([3.0, np.nan]))
    assert mean.iloc[0] == pytest.approx(2.0)
    assert np.isnan(mean.iloc[1])

    dataset = {"events": pd.DataFrame({"x": [1]})}
    vi._append_table(dataset, "events", None)
    vi._append_table(dataset, "events", pd.DataFrame())
    vi._append_table(dataset, "events", pd.DataFrame({"y": [2]}))
    assert len(dataset["events"]) == 2
    vi._append_table(dataset, "episodes", pd.DataFrame({"z": [3]}))
    assert dataset["episodes"].z.tolist() == [3]

    assert vi._tobii_valid(pd.Series([True, False], dtype=bool)).astype(bool).tolist() == [True, False]
    parsed = vi._tobii_valid(pd.Series(["Valid", "Invalid", "0", "1", "unknown"]))
    assert parsed.iloc[:4].astype(bool).tolist() == [True, False, True, False]
    assert pd.isna(parsed.iloc[4])
    assert np.isnan(vi._parse_float("bad"))
    assert vi._parse_float("1.25") == pytest.approx(1.25)


def test_vendor_detectors_cover_negative_unknown_and_weak_signature_paths(tmp_path: Path):
    folder = tmp_path / "folder"
    folder.mkdir()
    assert ep.is_tobii_export(folder) == 0
    assert ep.is_eyelink_export(folder) == 0
    assert ep.is_smi_export(folder) == 0
    assert ep.is_pupil_labs_export(folder) == 0
    assert ep.pupil_labs_format(folder) == "unknown"

    binary = tmp_path / "x.bin"
    binary.write_bytes(b"x")
    assert ep.is_tobii_export(binary) == 0
    assert ep.is_pupil_labs_export(binary) == 0
    assert ep.is_eyelink_export(binary) == 0
    assert ep.is_smi_export(binary) == 0

    named = tmp_path / "gaze.csv"
    _write(pd.DataFrame({"other": [1]}), named)
    assert ep.is_pupil_labs_export(named) == pytest.approx(0.85)
    assert ep.pupil_labs_format(named) == "neon"

    weak_tobii = tmp_path / "weak_tobii.csv"
    _write(pd.DataFrame({"Pupil diameter left": [3.0], "other": [1]}), weak_tobii)
    assert 0 < ep.is_tobii_export(weak_tobii) < 0.85

    weak_pupil = tmp_path / "weak_pupil.txt"
    _write(pd.DataFrame({"gaze_timestamp": [1], "other": [2]}), weak_pupil, sep="\t")
    assert 0 < ep.is_pupil_labs_export(weak_pupil) <= 0.8

    weak_eye = tmp_path / "weak_eye.tsv"
    _write(pd.DataFrame({"CURRENT_FIX_INDEX": [1]}), weak_eye, sep="\t")
    assert 0 < ep.is_eyelink_export(weak_eye) <= 0.8

    high_smi = tmp_path / "high_smi.tsv"
    _write(pd.DataFrame({"BeGaze version": [1], "POR X [px]": [10]}), high_smi, sep="\t")
    assert ep.is_smi_export(high_smi) >= 0.85


def test_tobii_binocular_fallback_validity_and_missing_contracts(tmp_path: Path):
    missing_timestamp = tmp_path / "missing_timestamp.tsv"
    _write(pd.DataFrame({"Gaze point X": [1], "Gaze point Y": [2]}), missing_timestamp, sep="\t")
    with pytest.raises(EyeProcessValidationError, match="timestamp"):
        ep.read_tobii(missing_timestamp)

    binocular = tmp_path / "binocular.tsv"
    _write(
        pd.DataFrame(
            {
                "Recording timestamp": [1_000_000, 1_016_000, 1_032_000],
                "Gaze point left X": [100.0, 110.0, np.nan],
                "Gaze point right X": [120.0, 130.0, 140.0],
                "Gaze point left Y": [200.0, 210.0, 220.0],
                "Gaze point right Y": [220.0, 230.0, 240.0],
                "Validity left": ["Valid", "Invalid", "Invalid"],
                "Participant": ["P1"] * 3,
                "Recording": ["R1"] * 3,
            }
        ),
        binocular,
        sep="\t",
    )
    out = ep.read_tobii(binocular, keep_raw=False, quiet=True)
    np.testing.assert_allclose(out["gaze_samples"].gaze_x.iloc[:2], [110.0, 120.0])
    assert out["gaze_samples"].valid.tolist() == [True, False, False]
    assert "tobii" not in getattr(out, "raw", {})

    no_xy = tmp_path / "no_xy.tsv"
    _write(pd.DataFrame({"Recording timestamp": [1_000_000, 1_016_000]}), no_xy, sep="\t")
    out2 = ep.read_tobii(no_xy, recording_id="NOXY", quiet=True)
    assert out2["gaze_samples"].gaze_x.isna().all()
    assert out2["gaze_samples"].gaze_y.isna().all()


def test_pupillabs_dispatch_errors_and_neon_alternate_companions(tmp_path: Path):
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    with pytest.raises(EyeProcessValidationError, match="format"):
        ep.read_pupillabs(unknown, format="bad")
    with pytest.raises(EyeProcessValidationError, match="Could not determine"):
        ep.read_pupillabs(unknown, format=[])

    with pytest.raises(EyeProcessValidationError, match="gaze.csv"):
        ep.read_pupil_neon(unknown)

    malformed = tmp_path / "neon_bad.csv"
    _write(pd.DataFrame({"timestamp [ns]": [1], "gaze x [px]": [1]}), malformed)
    with pytest.raises(EyeProcessValidationError, match="timestamp/x/y"):
        ep.read_pupil_neon(malformed)

    folder = tmp_path / "neon"
    folder.mkdir()
    _write(
        pd.DataFrame(
            {
                "timestamp [ns]": [1_000_000_000, 1_010_000_000, 1_020_000_000],
                "gaze x [px]": [100, 110, 120],
                "gaze y [px]": [200, 210, 220],
                "azimuth [deg]": [1.0, 2.0, 3.0],
                "elevation [deg]": [4.0, 5.0, 6.0],
                "recording id": ["R"] * 3,
            }
        ),
        folder / "gaze.csv",
    )
    # No explicit end/duration/x/y columns: exercise fallback derivations.
    _write(
        pd.DataFrame(
            {
                "start timestamp [ns]": [1_000_000_000],
                "end_timestamp_ns": [1_020_000_000],
            }
        ),
        folder / "fixations.csv",
    )
    # No conventional event name column: importer deliberately falls back to column 2.
    _write(
        pd.DataFrame(
            {
                "timestamp [ns]": [1_005_000_000],
                "label": ["marker"],
            }
        ),
        folder / "events.csv",
    )
    # Only one eye diameter is present; the absent eye is skipped.
    _write(
        pd.DataFrame(
            {
                "timestamp [ns]": [1_000_000_000, 1_010_000_000],
                "pupil diameter left [mm]": [3.1, np.nan],
            }
        ),
        folder / "3d_eye_states.csv",
    )
    out = ep.read_pupil_neon(folder, recording_id="NALT", keep_raw=False, quiet=True)
    assert {"azimuth_deg", "elevation_deg"}.issubset(out["gaze_samples"].columns)
    assert len(out["episodes"]) >= 1
    assert (out["events"].event_name.astype(str) == "marker").any()
    assert set(out["eye_samples"].eye.astype(str)) == {"left"}
    assert "neon_fixations" not in getattr(out, "raw", {})


def test_pupil_core_missing_and_companion_fallback_routes(tmp_path: Path):
    missing = tmp_path / "missing_core"
    missing.mkdir()
    with pytest.raises(EyeProcessValidationError, match="gaze_positions"):
        ep.read_pupil_core(missing)

    malformed = tmp_path / "core_bad.csv"
    _write(pd.DataFrame({"gaze_timestamp": [1.0], "norm_pos_x": [0.2]}), malformed)
    with pytest.raises(EyeProcessValidationError, match="timestamp/x/y"):
        ep.read_pupil_core(malformed)

    folder = tmp_path / "core"
    folder.mkdir()
    _write(
        pd.DataFrame(
            {
                "gaze_timestamp": [1.0, 1.01, 1.02],
                "norm_pos_x": [0.2, 0.3, 0.4],
                "norm_pos_y": [0.8, 0.7, 0.6],
            }
        ),
        folder / "gaze_positions.csv",
    )
    # Omit eye id, confidence and method; use a 2D diameter to select image-pixel units.
    _write(
        pd.DataFrame(
            {
                "pupil_timestamp": [1.0, 1.01],
                "diameter": [40.0, np.nan],
            }
        ),
        folder / "pupil_positions.csv",
    )
    # Small durations are seconds and should be converted to milliseconds.
    _write(
        pd.DataFrame(
            {
                "start_timestamp": [1.0, 1.2],
                "duration": [0.05, 0.08],
                "norm_pos_x": [0.2, 0.3],
                "norm_pos_y": [0.8, 0.7],
                "dispersion": [0.1, 0.2],
            }
        ),
        folder / "fixations.csv",
    )
    out = ep.read_pupil_core(folder, recording_id="CALT", keep_raw=False, quiet=True)
    assert set(out["eye_samples"].eye.astype(str)) == {"unknown"}
    assert set(out["eye_samples"].pupil_unit.astype(str)) == {"image_pixels"}
    assert out["episodes"].duration_ms.tolist() == pytest.approx([50.0, 80.0])


def test_eyelink_detector_parser_and_asc_error_routes(tmp_path: Path):
    missing = tmp_path / "missing.asc"
    assert ep.is_eyelink_export(missing) == 0
    with pytest.raises(EyeProcessValidationError, match="does not exist"):
        ep.read_eyelink_asc(missing)

    empty = tmp_path / "empty.asc"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(EyeProcessValidationError, match="empty"):
        ep.read_eyelink_asc(empty)

    lines = [
        "",                              # ignored blank
        "1000 1 2",                     # short sample
        "1001 . 2 3",                   # invalid x
        "1002 10 20 3000",              # valid sample
        "EFIX R 1000 1030",             # short event
        "EFIX R 1000 1030 30 11 21 3000",
        "ESACC R 1030 1060 30 11 21 31 41 2.5 150",
        "EBLINK R 1060 1080 20",
        "MSG 1000",                     # short MSG
        "MSG 1001 CALIBRATION GOOD 0.4 0.8",
        "MSG 1002 VALIDATION OK 0.5",
        "MSG 1003 DRIFT 0.2",
        "BUTTON 1010 1",
        "INPUT 1020 2",
        "START 900",
        "END 1100",
    ]
    parsed = vi._parse_eyelink_asc(lines, "EALT")
    assert len(parsed["gaze_samples"]) == 1
    assert set(parsed["episodes"].episode_type) == {"fixation", "saccade", "blink"}
    assert len(parsed["calibrations"]) == 3
    assert set(parsed["calibrations"].calibration_type) == {"calibration", "validation", "drift"}
    assert parsed["record_types"]["SAMPLE"] == 3

    path = tmp_path / "rich.asc"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = ep.read_eyelink_asc(path, recording_id="EALT", keep_raw=False)
    assert len(out["gaze_samples"]) == 1
    assert len(out["calibrations"]) == 3

    parsed_empty = vi._parse_eyelink_asc(["NOTHING here"], "EMPTY")
    assert parsed_empty["gaze_samples"].empty
    assert parsed_empty["episodes"].empty
    assert parsed_empty["events"].empty


def test_eyelink_report_explicit_mapping_and_edf_execution_contract(tmp_path: Path, monkeypatch):
    report = tmp_path / "report.tsv"
    _write(
        pd.DataFrame({"t": [1000, 1010, 1020], "gx": [1, 2, 3], "gy": [4, 5, 6]}),
        report,
        sep="\t",
    )
    mapping = ep.eye_mapping(timestamp="t", x="gx", y="gy")
    out = ep.read_eyelink_report(report, mapping=mapping, recording_id="ER2")
    assert len(out["gaze_samples"]) == 3

    missing = tmp_path / "missing.edf"
    with pytest.raises(EyeProcessValidationError, match="does not exist"):
        ep.read_eyelink_edf(missing)

    edf = tmp_path / "demo.edf"
    edf.write_bytes(b"edf")
    converter = tmp_path / "edf2asc"
    converter.write_text("fake", encoding="utf-8")

    def successful_run(cmd, capture_output, text, check):
        destination = Path(cmd[-1])
        destination.write_text(
            "START 1000\n1000 100 200 3000\n1016 110 210 3010\nEND 1032\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="converted", stderr="")

    monkeypatch.setattr(vi.subprocess, "run", successful_run)
    converted = ep.read_eyelink_edf(edf, edf2asc=str(converter), recording_id="EDF1")
    assert len(converted["gaze_samples"]) == 2
    assert any(converted["provenance"].action.astype(str).str.contains("convert_edf2asc"))

    explicit_asc = tmp_path / "kept.asc"
    kept = ep.read_eyelink_edf(
        edf,
        edf2asc=str(converter),
        output=explicit_asc,
        keep_asc=True,
        recording_id="EDF2",
    )
    assert explicit_asc.exists() and len(kept["gaze_samples"]) == 2

    def failed_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 2, stdout="bad", stderr="conversion error")

    monkeypatch.setattr(vi.subprocess, "run", failed_run)
    failed_dest = tmp_path / "failed.asc"
    with pytest.raises(EyeProcessBackendError, match="conversion failed"):
        ep.read_eyelink_edf(edf, edf2asc=str(converter), output=failed_dest)
    assert not failed_dest.exists()


def test_smi_detector_and_reader_validation_routes(tmp_path: Path):
    malformed = tmp_path / "malformed.tsv"
    _write(pd.DataFrame({"Time": [1], "Point of Regard X [px]": [10]}), malformed, sep="\t")
    with pytest.raises(EyeProcessValidationError, match="timestamp/x/y"):
        ep.read_smi(malformed)

    valid = tmp_path / "smi_full.tsv"
    _write(
        pd.DataFrame(
            {
                "Time": [1_000_000, 1_010_000, 1_020_000],
                "Point of Regard X [px]": [100, 110, 120],
                "Point of Regard Y [px]": [200, 210, 220],
                "Participant": ["P1"] * 3,
                "Recording": ["R1"] * 3,
                "L POR X [px]": [99, 109, 119],
                "L POR Y [px]": [199, 209, 219],
                "R POR X [px]": [101, 111, 121],
                "R POR Y [px]": [201, 211, 221],
                "L Pupil Diameter [mm]": [3.0, 3.1, 3.2],
                "R Pupil Diameter [mm]": [3.1, 3.2, 3.3],
                "Fixation Index": [1, 1, 2],
                "Trial": [1, 1, 1],
                "Stimulus": ["A"] * 3,
                "Event Type": ["start", "", "end"],
            }
        ),
        valid,
        sep="\t",
    )
    out = ep.read_smi(valid, keep_raw=False, quiet=True)
    assert len(out["gaze_samples"]) == 3
    assert out["recordings"].software_name.iloc[0] == "SMI BeGaze"
    assert getattr(out, "vendor_metadata", {})["smi"]["legacy"] is True
