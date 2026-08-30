from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.gazepoint_real_10 as gp10

TARGETS = [
    "gp_align_media_ids",
    "gp_check_biometrics_sync",
    "gp_check_fixation_ids",
    "gp_check_media_timing",
    "gp_check_pupil_channels",
    "gp_check_sampling_rate",
    "gp_check_validity_fields",
    "gp_parse_markers",
    "gp_reconstruct_stimuli",
    "gp_reconstruct_trials",
    "read_gazepoint_aoi_statistics",
    "read_gazepoint_summary",
]


def _summary_text() -> str:
    return """Gazepoint Analysis,7.2.0
Processed On,2026-08-29 12:34:56

AOI Summary
Media ID,AOI ID,AOI Name
M01,AOI 1,Headline

AOI Statistics (for each user)
Media ID,AOI ID,AOI Name,User Name,User ID,AOI Start,AOI Duration (sec - U=UserControlled),Time to 1st View (sec) -1.0 means not viewed,Time Viewed (sec),Time Viewed (%),Fixations (#),Revisits (#),Clicks (#),Ave Dial (0-1),Ave GSR (kOhm),Ave Heart Rate (BPM),Ave Interbeat Interval (s)
M01,AOI 1,Headline,Alice,1,0,5,0.25,1.5,30,4,1,2,0.7,420,72,0.83

Note: synthetic source fixture
"""


def test_public_api_targets_are_exported():
    missing = [name for name in TARGETS if not callable(getattr(ep, name, None))]
    assert not missing
    assert len(TARGETS) == 12


def test_summary_reader_preserves_metadata_and_blocks(tmp_path: Path):
    path = tmp_path / "Data_Summary_export.csv"
    path.write_text(_summary_text(), encoding="utf-8")

    out = ep.read_gazepoint_summary(path)
    assert out.software == "Gazepoint Analysis"
    assert out.software_version == "7.2.0"
    assert out.processed_on == "2026-08-29 12:34:56"
    assert len(out.aoi_summary) == 1
    assert len(out.aoi_statistics) == 1
    assert out.aoi_statistics.loc[0, "User Name"] == "Alice"
    assert out.notes == ["Note: synthetic source fixture"]


def test_summary_reader_rejects_non_summary(tmp_path: Path):
    path = tmp_path / "plain.csv"
    path.write_text("TIME,BPOGX,BPOGY\n0,0.2,0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Not a recognized"):
        ep.read_gazepoint_summary(path)


def test_plain_aoi_statistics_import(tmp_path: Path):
    path = tmp_path / "CurrentAOIStatistics.csv"
    pd.DataFrame(
        {
            "MEDIA_ID": ["M1", "M1"],
            "AOI_ID": ["AOI 1", "AOI 2"],
            "AOI_NAME": ["Header", "Button"],
        }
    ).to_csv(path, index=False)

    out = ep.read_gazepoint_aoi_statistics(
        path,
        participant_id="P1",
        recording_id="R1",
        quiet=True,
    )
    assert ep.is_eye_dataset(out)
    assert list(out["recordings"].recording_id) == ["R1"]
    assert len(out["aoi_definitions"]) == 2
    assert set(out["aoi_definitions"].aoi_name) == {"Header", "Button"}
    assert out["features"].empty
    assert "gazepoint_aoi_statistics" in out.raw


def test_data_summary_import_builds_aoi_features(tmp_path: Path):
    path = tmp_path / "Data_Summary_export.csv"
    path.write_text(_summary_text(), encoding="utf-8")

    out = ep.read_gazepoint_aoi_statistics(path, quiet=True)
    assert ep.is_eye_dataset(out)
    assert len(out["aoi_definitions"]) == 1
    assert not out["features"].empty
    names = set(out["features"].feature_name)
    assert {
        "time_to_first_view",
        "time_viewed",
        "time_viewed_percent",
        "fixation_count",
        "aoi_viewed",
    } <= names
    viewed = out["features"].loc[
        out["features"].feature_name.eq("aoi_viewed"),
        "value",
    ]
    assert list(viewed) == [1.0]
    assert not (out.validation["severity"].eq("error") if len(out.validation) else pd.Series(dtype=bool)).any()


def test_reconstruction_aliases_delegate(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(gp10, "build_trials", lambda x, **kw: (sentinel, x, kw))
    monkeypatch.setattr(
        gp10,
        "build_stimulus_intervals",
        lambda x, **kw: (sentinel, x, kw),
    )
    monkeypatch.setattr(gp10, "assign_trials", lambda x: (sentinel, x))

    assert gp10.gp_reconstruct_trials("x", overwrite=True) == (
        sentinel,
        "x",
        {"overwrite": True},
    )
    assert gp10.gp_reconstruct_stimuli("x", source="events") == (
        sentinel,
        "x",
        {"source": "events"},
    )
    assert gp10.gp_align_media_ids("x") == (sentinel, "x")


def test_marker_alias_delegates(monkeypatch):
    monkeypatch.setattr(gp10, "gp_parse_user_events", lambda x: ("events", x))
    assert gp10.gp_parse_markers("x") == ("events", "x")


def test_sampling_and_validity_audit_aliases(monkeypatch):
    monkeypatch.setattr(
        gp10,
        "audit_sampling_rate",
        lambda x, **kw: ("sampling", x, kw),
    )
    monkeypatch.setattr(
        gp10,
        "audit_signal_quality",
        lambda x, **kw: ("validity", x, kw),
    )
    assert gp10.gp_check_sampling_rate("x", tolerance_hz=3) == (
        "sampling",
        "x",
        {"expected_hz": 60, "tolerance_hz": 3},
    )
    assert gp10.gp_check_validity_fields("x", minimum_valid_gaze=0.5) == (
        "validity",
        "x",
        {"minimum_valid_gaze": 0.5},
    )


def test_episode_and_media_audit_aliases(monkeypatch):
    monkeypatch.setattr(
        gp10,
        "audit_episodes",
        lambda x, **kw: ("episodes", x, kw),
    )
    monkeypatch.setattr(
        gp10,
        "audit_event_order",
        lambda x, **kw: ("events", x, kw),
    )
    assert gp10.gp_check_fixation_ids("x") == (
        "episodes",
        "x",
        {"type": "fixation"},
    )
    assert gp10.gp_check_media_timing("x") == (
        "events",
        "x",
        {"event_type": "media_change"},
    )


def test_pupil_and_biometric_audit_aliases(monkeypatch):
    monkeypatch.setattr(
        gp10,
        "audit_pupil_quality",
        lambda x, **kw: ("pupil", x, kw),
    )
    monkeypatch.setattr(
        gp10,
        "audit_clock_sync",
        lambda x, **kw: ("clock", x, kw),
    )
    assert gp10.gp_check_pupil_channels("x", store=True) == (
        "pupil",
        "x",
        {"store": True},
    )
    assert gp10.gp_check_biometrics_sync("x", channel="eda") == (
        "clock",
        "x",
        {"channel": "eda"},
    )
