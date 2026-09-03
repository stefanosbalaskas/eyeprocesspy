from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.gazepoint as gp

FIX = Path(__file__).parent / "fixtures" / "gazepoint"


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_identity_time_helpers_and_tick_origins(tmp_path):
    assert gp._gp_id_token(None) == "unknown"
    assert gp._gp_id_token(np.nan) == "unknown"
    assert gp._gp_id_token("   ") == "unknown"
    assert gp._gp_id_token("A / B__C") == "A_B_C"

    user = gp._gp_filename_identity(tmp_path / "User 12_all_gaze.csv")
    assert user["participant_id"] == "User 12"
    assert user["recording_id"].startswith("rec_User_12_")

    summary = gp._gp_filename_identity(tmp_path / "Data_Summary_export.csv")
    assert summary["participant_id"] == "P001"

    explicit = gp._gp_filename_identity(
        tmp_path / "whatever.csv",
        participant_id="P X",
        recording_id="REC",
        session_id="S X",
    )
    assert explicit["participant_id"] == "P X"
    assert explicit["recording_id"] == "REC"
    assert explicit["session_id"] == "S X"

    d = pd.DataFrame(
        {
            "TIME(2026-01-01T00:00:00)": [0.0, 0.5],
            "TIMETICK(F=1000)": [100.0, 600.0],
        }
    )
    info = gp._gp_time_info(d)
    assert info["tick_frequency"] == 1000.0
    assert info["recording_start"] == "2026-01-01T00:00:00"
    assert gp._gp_tick_origin(d, info) == 100.0

    no_media = pd.DataFrame({"TIMETICK(F=1000)": [250.0, 300.0]})
    info2 = gp._gp_time_info(no_media)
    assert gp._gp_tick_origin(no_media, info2) == 250.0

    no_finite = pd.DataFrame({"TIMETICK(F=1000)": [np.nan]})
    assert np.isnan(gp._gp_tick_origin(no_finite, gp._gp_time_info(no_finite)))

    assert np.isnan(
        gp._gp_tick_origin(
            pd.DataFrame({"TIME": [0.0]}),
            {"tick_col": None, "tick_frequency": np.nan, "media_time_col": "TIME"},
        )
    )


def test_summary_detector_format_confidence_and_read_failure_paths(tmp_path, monkeypatch):
    assert gp._gp_is_summary_report(tmp_path) is False

    named = tmp_path / "Data_Summary_export_2026.csv"
    named.write_text("not even csv", encoding="utf-8")
    assert gp._gp_is_summary_report(named) is True
    assert gp.is_gazepoint_export(named) == 0.99

    text_summary = tmp_path / "summary.csv"
    text_summary.write_text("Gazepoint Analysis 7\nx\nAOI Summary\n", encoding="utf-8")
    assert gp._gp_is_summary_report(text_summary) is True

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert gp.is_gazepoint_export(empty_dir) == 0.0

    wrong = tmp_path / "data.xlsx"
    wrong.write_text("x", encoding="utf-8")
    assert gp.is_gazepoint_export(wrong) == 0.0

    original = gp._read_delimited

    def broken(*args, **kwargs):
        raise RuntimeError("reader failed")

    monkeypatch.setattr(gp, "_read_delimited", broken)
    signed = tmp_path / "subject_all_gaze.csv"
    signed.write_text("broken", encoding="utf-8")
    plain = tmp_path / "plain.csv"
    plain.write_text("broken", encoding="utf-8")
    assert gp.is_gazepoint_export(signed) == 0.75
    assert gp.is_gazepoint_export(plain) == 0.0
    assert gp.gp_identify_export_type(plain) == "unknown"

    monkeypatch.setattr(gp, "_read_delimited", original)
    gaze = _write_csv(
        tmp_path / "gaze.csv",
        pd.DataFrame({"TIME": [0.0], "BPOGX": [0.2], "BPOGY": [0.3]}),
    )
    assert gp.is_gazepoint_export(gaze) >= 0.0
    assert gp.gp_identify_export_type(gaze) == "gaze"

    unknown = _write_csv(tmp_path / "unknown.csv", pd.DataFrame({"foo": [1]}))
    assert gp.gp_identify_export_type(unknown) == "unknown"


def test_identify_fixations_by_content_and_aoi_statistics(tmp_path):
    fix = _write_csv(
        tmp_path / "records.csv",
        pd.DataFrame(
            {
                "FPOGID": [1, 2],
                "FPOGS": [0.0, 1.0],
                "FPOGD": [0.2, 0.3],
            }
        ),
    )
    assert gp.gp_identify_export_type(fix) == "fixations"

    all_gaze = _write_csv(
        tmp_path / "records_all_gaze.csv",
        pd.DataFrame(
            {
                "FPOGID": [1, 2],
                "FPOGS": [0.0, 1.0],
                "FPOGD": [0.2, 0.3],
                "FPOGX": [0.1, 0.2],
            }
        ),
    )
    assert gp.gp_identify_export_type(all_gaze) == "gaze"

    current = _write_csv(
        tmp_path / "CurrentAOIStatistics.csv", pd.DataFrame({"foo": [1]})
    )
    assert gp.gp_identify_export_type(current) == "aoi_statistics"


def test_profile_list_fields_and_validation_failure_paths(tmp_path, monkeypatch):
    d = tmp_path / "files"
    d.mkdir()
    low = _write_csv(d / "plain.csv", pd.DataFrame({"foo": [1], "bar": [2]}))
    good = _write_csv(
        d / "sample.csv",
        pd.DataFrame({"TIME": [0], "BPOGX": [0.2], "BPOGY": [0.3]}),
    )
    (d / "skip.bin").write_text("x", encoding="utf-8")

    fields = gp.gp_list_export_fields(d)
    assert "foo" in fields and "BPOGX" in fields

    audit = gp.gp_validate_export(low)
    assert "low_format_confidence" in set(audit["code"])

    empty = tmp_path / "empty"
    empty.mkdir()
    no_files = gp.gp_validate_export(empty)
    assert no_files.loc[0, "code"] == "no_files"

    original = gp._read_delimited

    def selective(path, *args, **kwargs):
        if Path(path).name == "broken.csv":
            raise RuntimeError("bad csv")
        return original(path, *args, **kwargs)

    broken = d / "broken.csv"
    broken.write_text("x", encoding="utf-8")
    monkeypatch.setattr(gp, "_read_delimited", selective)
    profile = gp.gp_profile_export(broken)
    assert pd.isna(profile.loc[0, "columns"])
    assert pd.isna(profile.loc[0, "n_columns"])


def test_timebase_application_covers_recording_stream_and_table_branches():
    x = ep.read_gazepoint(FIX / "demo-user.csv", recording_id="R-time", quiet=True)
    x["recordings"].loc[:, "nominal_sampling_rate"] = np.nan
    data = pd.DataFrame(
        {
            "TIME(start)": np.arange(12, dtype=float) / 60.0,
            "TIMETICK(F=1000)": 100.0 + np.arange(12, dtype=float) * 10.0,
        }
    )
    info = gp._gp_time_info(data)
    out = gp._gp_apply_timebase(x, data, info, origin_tick=100.0)

    assert out.vendor_metadata["gazepoint_timebase"]["tick_frequency"] == 1000.0
    assert out["recordings"]["nominal_sampling_rate"].iloc[0] == 60
    assert set(out["streams"]["timestamp_unit"]) == {"ticks"}
    assert set(out["streams"]["source_clock"]) == {"Gazepoint TIMETICK"}
    assert np.isfinite(
        pd.to_numeric(out["gaze_samples"]["timestamp_seconds"], errors="coerce")
    ).any()

    unchanged = gp._gp_apply_timebase(
        out,
        data,
        {"tick_col": None, "tick_frequency": np.nan},
    )
    assert unchanged is out

    no_origin = gp._gp_apply_timebase(
        out,
        pd.DataFrame({"TIMETICK(F=1000)": [np.nan]}),
        {
            "tick_col": "TIMETICK(F=1000)",
            "tick_frequency": 1000.0,
            "media_time_col": None,
            "recording_start": pd.NA,
        },
    )
    assert no_origin is out


def test_biometric_validity_units_and_empty_short_circuit():
    x = ep.read_gazepoint(FIX / "demo-user.csv", recording_id="R-bio", quiet=True)
    assert not x["biometrics"].empty

    ticks = pd.to_numeric(x["biometrics"]["timestamp_native"], errors="coerce")
    unique_ticks = pd.Series(pd.unique(ticks.dropna()))
    raw = pd.DataFrame(
        {
            "TIMETICK(F=1)": unique_ticks,
            "HRV": [1] * len(unique_ticks),
            "GSRV": [0] * len(unique_ticks),
        }
    )
    out = gp._gp_apply_biometric_validity(x, raw)
    units = out["biometrics"].groupby("channel")["unit"].first().to_dict()
    assert units["heart_rate"] == "beats_per_minute"
    assert units["gsr_raw"] == "vendor_raw"

    empty = x.copy()
    empty["biometrics"] = gp.empty_eye_table("biometrics")
    assert gp._gp_apply_biometric_validity(empty, raw) is empty


def test_user_event_parser_guards_nan_time_and_duplicate_filter():
    with pytest.raises(TypeError, match="eye_dataset"):
        gp.gp_parse_user_events({})

    x = ep.read_gazepoint(FIX / "demo-user.csv", recording_id="R-events", quiet=True)

    no_raw = x.copy()
    no_raw.raw = {}
    assert gp.gp_parse_user_events(no_raw) is no_raw

    missing_cols = x.copy()
    missing_cols.raw = {"gazepoint": pd.DataFrame({"X": [1]})}
    assert gp.gp_parse_user_events(missing_cols) is missing_cols

    blank = x.copy()
    blank.raw = {
        "gazepoint": pd.DataFrame({"TIME": [0.0], "USER_DATA": [""]})
    }
    assert gp.gp_parse_user_events(blank) is blank

    multi = x.copy()
    multi["recordings"] = pd.concat(
        [multi["recordings"], multi["recordings"].assign(recording_id="R-events-2")],
        ignore_index=True,
    )
    multi.raw = {
        "gazepoint": pd.DataFrame({"TIME": [0.0], "USER_DATA": ["marker"]})
    }
    assert gp.gp_parse_user_events(multi) is multi

    nan_time = x.copy()
    nan_time["events"] = gp.empty_eye_table("events")
    nan_time.raw = {
        "gazepoint": pd.DataFrame({"TIME": [np.nan], "USER_DATA": ["marker"]})
    }
    parsed = gp.gp_parse_user_events(nan_time)
    assert parsed["events"]["event_name"].tolist() == ["marker"]
    assert np.isnan(parsed["events"]["timestamp_seconds"].iloc[0])

    duplicated = parsed.copy()
    n_before = len(duplicated["events"])
    duplicated.raw = {
        "gazepoint": pd.DataFrame({"TIME": [np.nan], "USER_DATA": ["marker"]})
    }
    again = gp.gp_parse_user_events(duplicated)
    assert len(again["events"]) >= n_before


def test_media_event_parser_guards_and_changes():
    with pytest.raises(TypeError, match="eye_dataset"):
        gp.gp_parse_media_events({})

    x = ep.read_gazepoint(FIX / "demo-user.csv", recording_id="R-media", quiet=True)

    empty = x.copy()
    empty["gaze_samples"] = gp.empty_eye_table("gaze_samples")
    assert gp.gp_parse_media_events(empty) is empty

    none = x.copy()
    none["gaze_samples"] = none["gaze_samples"].copy()
    none["gaze_samples"].loc[:, "stimulus_id"] = pd.NA
    assert gp.gp_parse_media_events(none) is none

    y = x.copy()
    y["events"] = gp.empty_eye_table("events")
    parsed = gp.gp_parse_media_events(y)
    assert "media_change" in set(parsed["events"]["event_type"])


def test_read_gazepoint_dispatch_missing_mapping_and_gaze_alias(tmp_path):
    bad = _write_csv(tmp_path / "bad.csv", pd.DataFrame({"TIME": [0.0], "foo": [1]}))
    with pytest.raises(ValueError, match="missing identifiable"):
        gp.read_gazepoint(bad, quiet=True)

    via_alias = gp.read_gazepoint_gaze(
        FIX / "demo-user.csv", recording_id="R-alias", quiet=True
    )
    assert len(via_alias["gaze_samples"]) == 12

    folder = gp.read_gazepoint(FIX, recording_id="R-folder", quiet=True)
    assert ep.is_eye_dataset(folder)

    fix = gp.read_gazepoint(
        FIX / "demo-user-fix.csv", recording_id="R-fix-dispatch", quiet=True
    )
    assert not fix["episodes"].empty


def test_aoi_ids_and_fixation_alternate_clock_missing_ids_and_duplicates(tmp_path):
    ids = gp._gp_aoi_id(["stim 1", "stim2", "stim3"], ["AOI 2", "", pd.NA])
    assert ids.iloc[0] == "media_stim_1_aoi_2"
    assert pd.isna(ids.iloc[1]) and pd.isna(ids.iloc[2])

    frame = pd.DataFrame(
        {
            "FPOGID": ["1", "1", ""],
            "FPOGS": [0.0, 1.0, 2.0],
            "FPOGD": [0.2, 0.3, 0.4],
            "FPOGX": [0.1, 0.2, 0.3],
            "FPOGY": [0.2, 0.3, 0.4],
            "MEDIA_ID": ["stim", "stim", "stim2"],
            "AOI": ["AOI 1", "AOI 1", ""],
        }
    )
    path = _write_csv(tmp_path / "subject_fixations.csv", frame)
    out = gp.read_gazepoint_fixations(
        path, recording_id="R-alt", keep_raw=False, quiet=True
    )
    assert len(out["episodes"]) == 3
    assert out["episodes"]["episode_id"].is_unique
    assert out["episodes"]["duration_ms"].iloc[0] == pytest.approx(200.0)
    assert out.raw == {}

    tick = pd.DataFrame(
        {
            "FPOGID": ["1"],
            "FPOGD": [0.2],
            "FPOGX": [0.1],
            "FPOGY": [0.2],
            "MEDIA_ID": ["stim"],
            "TIMETICK(F=1000)": [1200.0],
            "TIME(start)": [1.0],
        }
    )
    tick_path = _write_csv(tmp_path / "tick_fixations.csv", tick)
    tick_out = gp.read_gazepoint_fixations(
        tick_path, recording_id="R-tick-fix", origin_tick=1000.0, quiet=True
    )
    assert tick_out["episodes"]["end_time"].iloc[0] == pytest.approx(0.2)


def test_pair_match_audit_and_summary_group_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        gp.gp_pair_exports(tmp_path / "missing")

    d = tmp_path / "pairs"
    d.mkdir()
    _write_csv(
        d / "alpha_all_gaze.csv",
        pd.DataFrame({"TIME": [0.0], "BPOGX": [0.2], "BPOGY": [0.3]}),
    )
    _write_csv(
        d / "alpha_fixations.csv",
        pd.DataFrame({"FPOGID": [1], "FPOGS": [0.0], "FPOGD": [0.2]}),
    )
    _write_csv(
        d / "CurrentAOIStatistics.csv",
        pd.DataFrame({"foo": [1]}),
    )
    _write_csv(
        d / "beta_fixations.csv",
        pd.DataFrame({"FPOGID": [1], "FPOGS": [0.0], "FPOGD": [0.2]}),
    )

    pairs = gp.gp_match_recordings(d)
    assert "gazepoint_data_summary" in set(pairs["group"])
    bio = gp.gp_match_biometrics(d)
    assert set(bio["export_type"]) <= {"gaze", "combined_biometrics"}

    audit = gp.gp_audit_file_pairs(d)
    assert audit.attrs["summary_reports"] == 1
    assert "incomplete" in set(audit["status"])
    assert "usable" in set(audit["status"])


def test_folder_error_multi_group_fixation_only_and_include_guards(tmp_path):
    with pytest.raises(FileNotFoundError):
        gp.read_gazepoint_folder(tmp_path / "missing", quiet=True)

    empty = tmp_path / "empty-folder"
    empty.mkdir()
    with pytest.raises(ValueError, match="No delimited"):
        gp.read_gazepoint_folder(empty, quiet=True)

    multi = tmp_path / "multi"
    multi.mkdir()
    for name in ("one_all_gaze.csv", "two_all_gaze.csv"):
        _write_csv(
            multi / name,
            pd.DataFrame({"TIME": [0.0], "BPOGX": [0.2], "BPOGY": [0.3]}),
        )
    with pytest.raises(ValueError, match="recording_id can only"):
        gp.read_gazepoint_folder(multi, recording_id="R", quiet=True)

    with pytest.raises(ValueError, match="No requested"):
        gp.read_gazepoint_folder(multi, include=(), quiet=True)

    fix_only = tmp_path / "fix-only"
    fix_only.mkdir()
    _write_csv(
        fix_only / "subject_fixations.csv",
        pd.DataFrame(
            {
                "FPOGID": [1],
                "FPOGS": [0.0],
                "FPOGD": [0.2],
                "FPOGX": [0.2],
                "FPOGY": [0.3],
                "MEDIA_ID": ["stim"],
            }
        ),
    )
    out = gp.read_gazepoint_folder(fix_only, include=("fixations",), quiet=True)
    assert not out["episodes"].empty


def test_biometrics_only_import_missing_channel_and_missing_time_paths(tmp_path):
    no_channels = _write_csv(
        tmp_path / "no-bio.csv",
        pd.DataFrame({"TIME": [0.0], "foo": [1]}),
    )
    with pytest.raises(ValueError, match="No recognized"):
        gp.read_gazepoint_biometrics(no_channels, quiet=True)

    no_time = _write_csv(
        tmp_path / "bio-no-time.csv",
        pd.DataFrame({"HR": [70.0], "GSR": [1.1]}),
    )
    with pytest.raises(ValueError, match="Cannot identify"):
        gp.read_gazepoint_biometrics(no_time, quiet=True)

    pure = _write_csv(
        tmp_path / "bio.csv",
        pd.DataFrame(
            {
                "TIME": [0.0, 1.0],
                "HR": [70.0, 71.0],
                "GSR": [1.0, 1.1],
                "LPMM": [3.0, 3.1],
                "RPMM": [3.2, 3.3],
            }
        ),
    )
    out = gp.read_gazepoint_biometrics(
        pure, recording_id="R-pure-bio", keep_raw=False, quiet=True
    )
    assert out["gaze_samples"].empty
    assert {"heart_rate", "gsr_raw"} <= set(out["biometrics"]["channel"])
    assert "gaze_combined" not in set(out["streams"]["stream_type"])


def test_combined_and_events_public_views():
    combined = gp.read_gazepoint_combined(
        FIX / "demo-user.csv",
        fixations=FIX / "demo-user-fix.csv",
        recording_id="R-combined",
        quiet=True,
    )
    assert not combined["gaze_samples"].empty
    assert not combined["episodes"].empty

    events = gp.read_gazepoint_events(
        FIX / "demo-user.csv", recording_id="R-events-only", quiet=True
    )
    assert events["gaze_samples"].empty
    assert events["eye_samples"].empty
    assert events["biometrics"].empty
    assert not events["events"].empty
