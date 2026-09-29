from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "gaze_claim_robustness"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


srl_adapter = _load_module("srl_adapter_research", RESEARCH / "srl_adapter.py")
srl_manifest = _load_module(
    "srl_manifest_research",
    RESEARCH / "build_srl_design_manifest.py",
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    participants = pd.DataFrame(
        {
            "part_ID": ["P001"],
            "experiment_condition": ["Prompt"],
        }
    )
    stimuli = pd.DataFrame(
        {
            "part_ID": ["P001"] * 8,
            "stimulus_name": [f"Task_{i}" for i in range(1, 9)],
            "stimulus_type": ["Text"] * 4 + ["Multimedia"] * 4,
            "stimulus_time": [10.0 + i for i in range(8)],
            "tracking_ratio": [95.0] * 8,
        }
    )
    rows = []
    clock = 1000
    for task in range(1, 9):
        for sample in range(3):
            rows.append(
                {
                    "RecordingTime [ms]": clock,
                    "Trial": f"Trial_{task}",
                    "Stimulus": f"Task_{task}.jpg",
                    "Participant": "P001",
                    "Tracking Ratio [%]": 95.0,
                    "Point of Regard Right X [px]": 100.0 + sample,
                    "Point of Regard Right Y [px]": 200.0 + sample,
                    "Point of Regard Left X [px]": 110.0 + sample,
                    "Point of Regard Left Y [px]": 210.0 + sample,
                    "AOI Name Right": "vendor_aoi_right",
                    "AOI Name Left": "vendor_aoi_left",
                    "Pupil Diameter Right [mm]": 3.2,
                    "Pupil Diameter Left [mm]": 3.1,
                }
            )
            clock += 4

    participants_path = tmp_path / "participants.csv"
    stimuli_path = tmp_path / "stimuli.csv"
    raw_dir = tmp_path / "ET_data_raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "P001.csv"

    participants.to_csv(participants_path, index=False)
    stimuli.to_csv(stimuli_path, index=False)
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    return participants_path, stimuli_path, raw_dir, raw_path


def test_srl_adapter_preserves_declared_mapping_contract(tmp_path: Path):
    participants_path, stimuli_path, _raw_dir, raw_path = _write_fixture(tmp_path)

    dataset = srl_adapter.load_srl_trial(
        raw_path,
        participants_path,
        stimuli_path,
        stimulus_name="Task_5",
        eye="left",
    )

    samples = dataset["gaze_samples"]
    interval = dataset["intervals"].iloc[0]

    assert len(samples) == 3
    assert samples["timestamp_seconds"].tolist() == pytest.approx([0.0, 0.004, 0.008])
    assert samples["gaze_x"].tolist() == pytest.approx([110.0, 111.0, 112.0])
    assert samples["source_eye"].eq("left").all()
    assert samples["source_aoi_name"].eq("vendor_aoi_left").all()
    assert samples["finite_xy"].all()
    assert samples["within_display_bounds"].all()
    assert dataset["aoi_definitions"].empty
    assert dataset["aoi_geometry"].empty

    assert interval["experiment_condition"] == "Prompt"
    assert interval["stimulus_type"] == "Multimedia"
    assert interval["condition_id"] == "Prompt|Multimedia"

    assert dataset.vendor_metadata["binocular_fusion"] == "none"
    assert dataset.vendor_metadata["interpolation"] == "none"
    assert dataset.vendor_metadata["source_aoi_labels_used_for_geometry"] is False


def test_srl_design_manifest_is_pre_results_and_deterministic(tmp_path: Path):
    participants_path, stimuli_path, raw_dir, _raw_path = _write_fixture(tmp_path)

    manifest, issues = srl_manifest.build_manifest(
        participants_path,
        stimuli_path,
        raw_dir,
    )

    assert len(manifest) == 16
    assert set(manifest["eye"]) == {"left", "right"}
    assert manifest["outcome_frozen"].all()
    assert not manifest["event_detector_frozen"].any()
    assert not manifest["aoi_geometry_frozen"].any()
    assert not manifest["quality_rule_frozen"].any()
    assert not manifest["model_frozen"].any()
    assert set(manifest["primary_estimand_family"]) == {
        "between_aoi_transition_count"
    }
    assert set(manifest["focal_contrast"]) == {"Multimedia_minus_Text"}

    errors = issues.loc[issues["severity"].eq("error")]
    warnings = issues.loc[issues["severity"].eq("warning")]
    assert errors.empty
    assert len(warnings) == 1
    assert "84 complete recordings" in warnings.iloc[0]["message"]


def test_srl_adapter_refuses_nonmonotonic_trial_time(tmp_path: Path):
    participants_path, stimuli_path, _raw_dir, raw_path = _write_fixture(tmp_path)
    raw = pd.read_csv(raw_path)
    task_one = raw["Stimulus"].eq("Task_1.jpg")
    indices = raw.index[task_one].tolist()
    raw.loc[indices[2], "RecordingTime [ms]"] = raw.loc[
        indices[1], "RecordingTime [ms]"
    ]
    raw.to_csv(raw_path, index=False)

    with pytest.raises(ValueError, match="strictly increasing"):
        srl_adapter.load_srl_trial(
            raw_path,
            participants_path,
            stimuli_path,
            stimulus_name="Task_1",
            eye="right",
        )
