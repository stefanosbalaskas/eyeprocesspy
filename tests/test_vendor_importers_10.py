from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "is_eyelink_export",
    "is_pupil_labs_export",
    "is_smi_export",
    "is_tobii_export",
    "pupil_labs_format",
    "read_eyelink_asc",
    "read_eyelink_edf",
    "read_eyelink_report",
    "read_pupil_core",
    "read_pupil_neon",
    "read_pupillabs",
    "read_smi",
    "read_smi_aoi_export",
    "read_smi_event_export",
    "read_smi_raw_export",
    "read_tobii",
]


def _write_tobii(path: Path):
    frame = pd.DataFrame(
        {
            "Recording timestamp": [1_000_000, 1_016_667, 1_033_334],
            "Gaze point X": [100.0, 110.0, 120.0],
            "Gaze point Y": [200.0, 210.0, 220.0],
            "Pupil diameter left": [3.1, 3.2, 3.3],
            "Pupil diameter right": [3.0, 3.1, 3.2],
            "Validity left": ["Valid", "Invalid", "Valid"],
            "Validity right": ["Valid", "Valid", "Valid"],
            "Presented Stimulus name": ["item", "item", "item"],
        }
    )
    frame.to_csv(path, sep="\t", index=False)


def _write_neon(folder: Path):
    folder.mkdir()
    pd.DataFrame(
        {
            "timestamp [ns]": [1_000_000_000, 1_016_000_000, 1_032_000_000],
            "gaze x [px]": [100.0, 110.0, 120.0],
            "gaze y [px]": [200.0, 210.0, 220.0],
            "worn": [1, 1, 1],
        }
    ).to_csv(folder / "gaze.csv", index=False)
    pd.DataFrame(
        {
            "timestamp [ns]": [1_000_000_000, 1_016_000_000, 1_032_000_000],
            "pupil diameter left [mm]": [3.1, 3.2, 3.3],
            "pupil diameter right [mm]": [3.0, 3.1, 3.2],
        }
    ).to_csv(folder / "3d_eye_states.csv", index=False)
    pd.DataFrame(
        {
            "start timestamp [ns]": [1_000_000_000],
            "end timestamp [ns]": [1_032_000_000],
            "duration [ms]": [32.0],
            "fixation x [px]": [110.0],
            "fixation y [px]": [210.0],
        }
    ).to_csv(folder / "fixations.csv", index=False)


def _write_core(folder: Path):
    folder.mkdir()
    pd.DataFrame(
        {
            "gaze_timestamp": [1.0, 1.016, 1.032],
            "norm_pos_x": [0.2, 0.3, 0.4],
            "norm_pos_y": [0.8, 0.7, 0.6],
            "confidence": [0.9, 0.8, 0.95],
        }
    ).to_csv(folder / "gaze_positions.csv", index=False)
    pd.DataFrame(
        {
            "pupil_timestamp": [1.0, 1.016, 1.032],
            "eye_id": [0, 1, 0],
            "diameter_3d": [3.1, 3.0, 3.2],
            "confidence": [0.9, 0.8, 0.95],
            "method": ["3d", "3d", "3d"],
        }
    ).to_csv(folder / "pupil_positions.csv", index=False)


def _write_eyelink(path: Path):
    path.write_text(
        "\n".join(
            [
                "START 1000",
                "1000 100 200 3000",
                "1016 110 210 3010",
                "1032 120 220 3020",
                "EFIX R 1000 1032 32 110 210 3010",
                "MSG 1000 TRIALID T1",
                "END 1032",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_smi(path: Path):
    pd.DataFrame(
        {
            "Time": [1_000_000, 1_016_000, 1_032_000],
            "Point of Regard X [px]": [100.0, 110.0, 120.0],
            "Point of Regard Y [px]": [200.0, 210.0, 220.0],
            "L Pupil Diameter [mm]": [3.1, 3.2, 3.3],
            "R Pupil Diameter [mm]": [3.0, 3.1, 3.2],
            "Stimulus Name": ["item", "item", "item"],
        }
    ).to_csv(path, sep="\t", index=False)


def test_public_r006_exports_are_callable():
    assert len(TARGETS) == 16
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_vendor_detectors_and_pupil_format(tmp_path: Path):
    tobii = tmp_path / "tobii.tsv"
    neon = tmp_path / "neon"
    core = tmp_path / "core"
    eyelink = tmp_path / "demo.asc"
    smi = tmp_path / "smi.txt"
    _write_tobii(tobii)
    _write_neon(neon)
    _write_core(core)
    _write_eyelink(eyelink)
    _write_smi(smi)

    assert ep.is_tobii_export(tobii) >= 0.85
    assert ep.is_pupil_labs_export(neon) == pytest.approx(0.95)
    assert ep.is_pupil_labs_export(core) == pytest.approx(0.95)
    assert ep.pupil_labs_format(neon) == "neon"
    assert ep.pupil_labs_format(core) == "core"
    assert ep.is_eyelink_export(eyelink) == pytest.approx(0.95)
    assert ep.is_smi_export(smi) > 0


def test_tobii_fixture_matches_frozen_structural_contract(tmp_path: Path):
    path = tmp_path / "tobii.tsv"
    _write_tobii(path)
    out = ep.read_tobii(
        path,
        recording_id="T1",
        quiet=True,
    )
    assert len(out["gaze_samples"]) == 3
    assert out["recordings"]["vendor"].iloc[0] == "Tobii"
    assert out["recordings"]["software_name"].iloc[0] == "Tobii Pro Lab"
    assert out["gaze_samples"]["valid"].tolist() == [True, True, True]


def test_neon_fixture_matches_frozen_structural_contract(tmp_path: Path):
    folder = tmp_path / "neon"
    _write_neon(folder)
    out = ep.read_pupil_neon(
        folder,
        recording_id="N1",
        quiet=True,
    )
    assert len(out["gaze_samples"]) == 3
    assert len(out["eye_samples"]) >= 6
    assert (out["episodes"]["episode_type"] == "fixation").any()
    assert out["recordings"]["device_model"].iloc[0] == "Neon"

    dispatched = ep.read_pupillabs(
        folder,
        recording_id="N2",
        quiet=True,
    )
    assert len(dispatched["gaze_samples"]) == 3


def test_core_fixture_matches_frozen_structural_contract(tmp_path: Path):
    folder = tmp_path / "core"
    _write_core(folder)
    out = ep.read_pupil_core(
        folder,
        recording_id="C1",
        quiet=True,
    )
    assert len(out["gaze_samples"]) == 3
    assert out["coordinate_spaces"]["origin"].iloc[0] == "bottom_left"
    assert out["recordings"]["device_model"].iloc[0] == "Pupil Core"


def test_eyelink_asc_fixture_preserves_samples_episodes_events(tmp_path: Path):
    path = tmp_path / "demo.asc"
    _write_eyelink(path)
    out = ep.read_eyelink_asc(
        path,
        recording_id="E1",
        quiet=True,
    )
    assert len(out["gaze_samples"]) == 3
    assert (out["episodes"]["episode_type"] == "fixation").any()
    assert (out["events"]["event_type"] == "message").any()
    assert out["recordings"]["vendor"].iloc[0] == "SR Research"
    assert out["recordings"]["vendor_family"].iloc[0] == "EyeLink"


def test_eyelink_edf_route_is_explicitly_backend_gated(tmp_path: Path):
    path = tmp_path / "demo.edf"
    path.write_bytes(b"edf")
    assert ep.is_eyelink_export(path) == pytest.approx(0.98)
    with pytest.raises(ep.EyeProcessBackendError, match="edf2asc"):
        ep.read_eyelink_edf(
            path,
            edf2asc=str(tmp_path / "missing-edf2asc"),
        )


def test_eyelink_report_uses_generic_mapping(tmp_path: Path):
    path = tmp_path / "report.tsv"
    pd.DataFrame(
        {
            "timestamp": [1000, 1010, 1020],
            "gaze_x": [1.0, 2.0, 3.0],
            "gaze_y": [4.0, 5.0, 6.0],
        }
    ).to_csv(path, sep="\t", index=False)
    out = ep.read_eyelink_report(
        path,
        recording_id="ER1",
    )
    assert len(out["gaze_samples"]) == 3
    assert out["recordings"]["vendor"].iloc[0] == "EyeLink Data Viewer"


def test_smi_fixture_and_aliases_match_frozen_contract(tmp_path: Path):
    path = tmp_path / "smi.txt"
    _write_smi(path)
    out = ep.read_smi(path, recording_id="S1", quiet=True)
    assert len(out["gaze_samples"]) == 3
    assert out["recordings"]["software_name"].iloc[0] == "SMI BeGaze"

    for reader in (
        ep.read_smi_raw_export,
        ep.read_smi_event_export,
        ep.read_smi_aoi_export,
    ):
        alias = reader(path, recording_id="S2", quiet=True)
        assert len(alias["gaze_samples"]) == 3


def test_smi_idf_route_is_not_faked(tmp_path: Path):
    path = tmp_path / "demo.idf"
    path.write_bytes(b"idf")
    assert ep.is_smi_export(path) == pytest.approx(0.5)
    with pytest.raises(ep.EyeProcessBackendError, match="IDF"):
        ep.read_smi(path)
