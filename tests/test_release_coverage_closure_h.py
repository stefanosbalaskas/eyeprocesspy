from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.gazepoint_workflow_10 as gw
import eyeprocesspy.partitioned_storage_10 as ps


def _tables():
    return {"responses": pd.DataFrame({"participant_id": ["P1", "P2"], "score": [1.0, 0.0]})}


def test_partition_source_class_and_existing_file_backup_cleanup(tmp_path):
    assert ps._source_class(ep.new_eye_dataset(validate=False))[0] == "eye_dataset"
    spec = ep.partition_eye_storage(by=["participant_id"], format="csv", max_rows=1)
    target = tmp_path / "file-target"
    target.write_text("old", encoding="utf-8")
    store = ps.write_partitioned_eye_storage(_tables(), target, spec, overwrite=True)
    assert target.is_dir() and Path(store.path).resolve() == target.resolve()


def test_partition_preexisting_backup_file_and_directory(monkeypatch, tmp_path):
    spec = ep.partition_eye_storage(by=["participant_id"], format="csv", max_rows=1)

    class U:
        def __init__(self, value):
            self.hex = value

    target = tmp_path / "store-file-backup"
    target.write_text("old", encoding="utf-8")
    values = iter([U("s1"), U("b1"), *[U(f"x{i}") for i in range(50)]])
    monkeypatch.setattr(ps.uuid, "uuid4", lambda: next(values))
    backup = target.with_name(f"{target.name}.backup-{os.getpid()}-b1")
    backup.write_text("stale", encoding="utf-8")
    ps.write_partitioned_eye_storage(_tables(), target, spec, overwrite=True)
    assert target.is_dir() and not backup.exists()

    target2 = tmp_path / "store-dir-backup"
    target2.write_text("old", encoding="utf-8")
    values = iter([U("s2"), U("b2"), *[U(f"y{i}") for i in range(50)]])
    monkeypatch.setattr(ps.uuid, "uuid4", lambda: next(values))
    backup2 = target2.with_name(f"{target2.name}.backup-{os.getpid()}-b2")
    backup2.mkdir()
    (backup2 / "stale").write_text("x", encoding="utf-8")
    ps.write_partitioned_eye_storage(_tables(), target2, spec, overwrite=True)
    assert target2.is_dir() and not backup2.exists()


def test_partition_outer_restore_when_target_rename_raises_after_move(monkeypatch, tmp_path):
    spec = ep.partition_eye_storage(by=["participant_id"], format="csv", max_rows=1)
    target = tmp_path / "restore-outer"
    target.write_text("original", encoding="utf-8")
    original = ps.Path.rename

    def move_then_raise(self, destination):
        if self == target:
            original(self, destination)
            raise OSError("after backup move")
        return original(self, destination)

    monkeypatch.setattr(ps.Path, "rename", move_then_raise)
    with pytest.raises(OSError, match="after backup"):
        ps.write_partitioned_eye_storage(_tables(), target, spec, overwrite=True)
    assert target.is_file() and target.read_text(encoding="utf-8") == "original"


def test_workflow_biometric_empty_duplicate_lookup_and_single_key(monkeypatch):
    x = ep.new_eye_dataset(validate=False)
    biometrics = ep.standardize_eye_table(
        pd.DataFrame(
            {
                "recording_id": ["R1"],
                "trial_id": ["T1"],
                "stimulus_id": ["S1"],
                "channel": ["eda"],
                "value": [1.0],
                "unit": ["uS"],
            }
        ),
        "biometrics",
    )
    x["biometrics"] = biometrics

    real_std = gw.standardize_eye_table
    monkeypatch.setattr(
        gw,
        "standardize_eye_table",
        lambda d, name: ep.empty_eye_table(name) if name == "features" else real_std(d, name),
    )
    assert gw._workflow_biometric_features(x).empty
    monkeypatch.setattr(gw, "standardize_eye_table", real_std)

    duplicate_trials = pd.DataFrame(
        {
            "recording_id": ["R1", "R1"],
            "trial_id": ["T1", "T1"],
            "participant_id": ["P1", "P1"],
            "item_id": ["I1", "I1"],
            "stimulus_id": ["S1", "S1"],
        }
    )
    monkeypatch.setattr(gw, "trial_table", lambda x: duplicate_trials)
    features = gw._workflow_biometric_features(x)
    assert not features.empty and features.participant_id.eq("P1").all()

    episodes = ep.standardize_eye_table(
        pd.DataFrame(
            {
                "recording_id": ["R1", "R1"],
                "episode_type": ["fixation", "fixation"],
                "derived_by": ["vendor", "vendor"],
                "duration_ms": [100.0, 120.0],
            }
        ),
        "episodes",
    )
    x["episodes"] = episodes
    summary = gw._summarize_fixations(x, by=("recording_id",), source="vendor")
    assert summary.recording_id.tolist() == ["R1"]


def test_workflow_analysis_aoi_merge_readiness_and_report(monkeypatch, tmp_path):
    x = ep.new_eye_dataset(validate=False)
    trial = ep.standardize_eye_table(
        pd.DataFrame(
            {
                "interval_id": ["int1"],
                "recording_id": ["R1"],
                "participant_id": ["P1"],
                "trial_id": ["T1"],
                "item_id": ["I1"],
                "stimulus_id": ["S1"],
                "condition_id": ["C1"],
                "start_time": [0.0],
                "end_time": [1.0],
                "interval_type": ["trial"],
            }
        ),
        "intervals",
    )
    x["intervals"] = trial
    feat = ep.standardize_eye_table(
        pd.DataFrame(
            {
                "feature_id": ["f1"],
                "recording_id": ["R1"],
                "participant_id": ["P1"],
                "trial_id": ["T1"],
                "item_id": ["I1"],
                "stimulus_id": ["S1"],
                "aoi_id": ["A1"],
                "feature_name": ["dwell"],
                "value": [1.0],
                "unit": ["s"],
            }
        ),
        "features",
    )
    x["features"] = feat
    x["aoi_definitions"] = ep.standardize_eye_table(
        pd.DataFrame({"aoi_id": ["A1"], "aoi_name": ["Target"]}), "aoi_definitions"
    )
    tables = gw.gazepoint_analysis_tables(x)
    assert tables["aoi_summary"].aoi_name.iloc[0] == "Target"

    persons = [f"P{i:03d}" for i in range(100)]
    items = [f"I{j}" for j in range(5)]
    rows = [(p, i) for p in persons for i in items]
    trials = pd.DataFrame(
        {
            "recording_id": [f"R{k}" for k in range(len(rows))],
            "participant_id": [p for p, _ in rows],
            "trial_id": [f"T{k}" for k in range(len(rows))],
            "item_id": [i for _, i in rows],
            "stimulus_id": ["S"] * len(rows),
            "condition_id": ["C"] * len(rows),
        }
    )
    monkeypatch.setattr(gw, "trial_table", lambda obj: trials)
    responses = ep.standardize_eye_table(
        pd.DataFrame(
            {
                "recording_id": trials.recording_id,
                "participant_id": trials.participant_id,
                "trial_id": trials.trial_id,
                "item_id": trials.item_id,
                "response": ["1"] * len(rows),
                "score": [1.0] * len(rows),
                "response_time": [1.0] * len(rows),
            }
        ),
        "responses",
    )
    x["responses"] = responses
    process = trials[["recording_id", "participant_id", "trial_id", "item_id", "stimulus_id"]].copy()
    process["process_metric"] = 1.0
    readiness = gw.gazepoint_irt_tables(x, process_table=process)
    assert readiness["status"] == "model_ready_subject_to_diagnostics"

    ax_path = tmp_path / "fallback.png"
    manifest = gw._save_workflow_plot(ax_path, lambda: None)
    assert manifest["status"] == "created" and ax_path.exists()

    empty = pd.DataFrame()
    assert gw._markdown_table(empty) == "_No rows._"
    workflow = gw.GazepointWorkflow(
        status="ok",
        output_dir=str(tmp_path),
        source_path="source.csv",
        dataset=ep.new_eye_dataset(validate=False),
        responses_supplied=False,
        spec=gw.gazepoint_workflow_spec(create_html_report=False),
        workflow_checks=empty,
        tables={
            "trials": empty,
            "process": empty,
            "fixation_summary": empty,
            "aoi_fixation_summary": empty,
            "aoi_summary": empty,
            "pupil_summary": empty,
            "biometric_summary": empty,
        },
        irt={"status": "pending", "readiness": empty},
        paths={},
    )
    report = gw.write_gazepoint_workflow_report(workflow)
    assert report.endswith("gazepoint-workflow-report.md")
