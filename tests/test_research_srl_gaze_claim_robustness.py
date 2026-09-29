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
srl_coordinates = _load_module(
    "srl_coordinates_research",
    RESEARCH / "srl_coordinate_branches.py",
)
srl_outcome = _load_module(
    "srl_outcome_research",
    RESEARCH / "srl_transition_outcome.py",
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


def test_srl_coordinate_branches_make_viewing_distance_explicit(tmp_path: Path):
    participants_path, stimuli_path, _raw_dir, raw_path = _write_fixture(tmp_path)
    dataset = srl_adapter.load_srl_trial(
        raw_path,
        participants_path,
        stimuli_path,
        stimulus_name="Task_5",
        eye="left",
    )

    angular = srl_coordinates.convert_srl_pixels_to_degrees(
        dataset,
        viewing_distance_cm=65.0,
    )
    samples = angular["gaze_samples"]

    assert samples["source_gaze_x_px"].tolist() == pytest.approx(
        [110.0, 111.0, 112.0]
    )
    assert samples["source_gaze_y_px"].tolist() == pytest.approx(
        [210.0, 211.0, 212.0]
    )
    assert samples["valid"].tolist() == dataset["gaze_samples"]["valid"].tolist()
    assert samples["coordinate_space_id"].nunique() == 1
    assert "65cm" in samples["coordinate_space_id"].iloc[0]

    ppd = srl_coordinates.pixels_per_degree(65.0)
    assert ppd == pytest.approx(37.27007802017664)
    assert samples["gaze_x"].iloc[1] - samples["gaze_x"].iloc[0] == pytest.approx(
        1.0 / ppd
    )

    branch = angular.vendor_metadata["angular_coordinate_branch"]
    assert branch["viewing_distance_cm"] == pytest.approx(65.0)
    assert branch["source_pixel_coordinates_retained"] is True

    manifest = srl_coordinates.coordinate_branch_manifest()
    assert manifest["viewing_distance_cm"].tolist() == [60.0, 65.0, 70.0]
    assert manifest["pixels_per_degree"].is_monotonic_increasing


def test_episode_aoi_assignment_respects_stimulus_identity():
    recordings = pd.DataFrame(
        [{"recording_id": "R1", "participant_id": "P1"}]
    )
    spaces = ep.new_coordinate_space(
        "px",
        "display_pixels_top_left",
        width=1600,
        height=900,
    )
    episodes = pd.DataFrame(
        [
            {
                "episode_id": "E1",
                "recording_id": "R1",
                "episode_type": "fixation",
                "start_time": 0.1,
                "end_time": 0.2,
                "duration_ms": 100.0,
                "centroid_x": 100.0,
                "centroid_y": 100.0,
                "coordinate_space_id": "px",
                "derived_by": "eyeprocess",
                "trial_id": "T1",
                "stimulus_id": "Task_1",
            },
            {
                "episode_id": "E2",
                "recording_id": "R1",
                "episode_type": "fixation",
                "start_time": 0.3,
                "end_time": 0.4,
                "duration_ms": 100.0,
                "centroid_x": 100.0,
                "centroid_y": 100.0,
                "coordinate_space_id": "px",
                "derived_by": "eyeprocess",
                "trial_id": "T2",
                "stimulus_id": "Task_2",
            },
        ]
    )
    dataset = ep.new_eye_dataset(
        recordings=recordings,
        episodes=episodes,
        coordinate_spaces=spaces,
    )
    dataset = ep.register_aois(
        dataset,
        ep.new_aoi(
            "task1_quarter",
            stimulus_id="Task_1",
            x=0,
            y=0,
            width=800,
            height=450,
            coordinate_space_id="px",
        ),
        ep.new_aoi(
            "task2_quarter",
            stimulus_id="Task_2",
            x=0,
            y=0,
            width=800,
            height=450,
            coordinate_space_id="px",
        ),
        ep.new_aoi(
            "global_reference",
            x=0,
            y=0,
            width=800,
            height=450,
            coordinate_space_id="px",
        ),
    )

    assigned = ep.assign_aois(dataset, component="episodes")
    result = assigned["episodes"].set_index("episode_id")["aoi_id"]

    assert result["E1"] == "task1_quarter"
    assert result["E2"] == "task2_quarter"


def test_srl_transition_outcome_preserves_unassigned_breaks_and_missing_trials():
    recordings = pd.DataFrame(
        [{"recording_id": "R1", "participant_id": "P1"}]
    )
    intervals = pd.DataFrame(
        [
            {
                "interval_id": "I1",
                "recording_id": "R1",
                "interval_type": "trial",
                "start_time": 0.0,
                "end_time": 1.0,
                "trial_id": "T1",
                "participant_id": "P1",
                "stimulus_id": "Task_1",
                "condition_id": "Prompt|Text",
                "valid_interval": True,
                "experiment_condition": "Prompt",
                "stimulus_type": "Text",
            },
            {
                "interval_id": "I2",
                "recording_id": "R1",
                "interval_type": "trial",
                "start_time": 2.0,
                "end_time": 3.0,
                "trial_id": "T2",
                "participant_id": "P1",
                "stimulus_id": "Task_2",
                "condition_id": "Prompt|Multimedia",
                "valid_interval": True,
                "experiment_condition": "Prompt",
                "stimulus_type": "Multimedia",
            },
            {
                "interval_id": "I3",
                "recording_id": "R1",
                "interval_type": "trial",
                "start_time": 4.0,
                "end_time": 5.0,
                "trial_id": "T3",
                "participant_id": "P1",
                "stimulus_id": "Task_3",
                "condition_id": "Prompt|Text",
                "valid_interval": True,
                "experiment_condition": "Prompt",
                "stimulus_type": "Text",
            },
        ]
    )
    aois = ["A", "A", "B", pd.NA, "C", "C", "D"]
    episodes = []
    for i, aoi in enumerate(aois):
        episodes.append(
            {
                "episode_id": f"E{i}",
                "recording_id": "R1",
                "episode_type": "fixation",
                "start_time": 0.1 * i,
                "end_time": 0.1 * i + 0.05,
                "duration_ms": 50.0,
                "centroid_x": 10.0,
                "centroid_y": 10.0,
                "coordinate_space_id": "px",
                "derived_by": "eyeprocess",
                "trial_id": "T1",
                "stimulus_id": "Task_1",
                "aoi_id": aoi,
            }
        )
    episodes.append(
        {
            "episode_id": "E_T3",
            "recording_id": "R1",
            "episode_type": "fixation",
            "start_time": 4.1,
            "end_time": 4.2,
            "duration_ms": 100.0,
            "centroid_x": 10.0,
            "centroid_y": 10.0,
            "coordinate_space_id": "px",
            "derived_by": "eyeprocess",
            "trial_id": "T3",
            "stimulus_id": "Task_3",
            "aoi_id": pd.NA,
        }
    )
    dataset = ep.new_eye_dataset(
        recordings=recordings,
        intervals=intervals,
        episodes=pd.DataFrame(episodes),
    )

    outcome = srl_outcome.derive_transition_counts(dataset).set_index("trial_id")
    assert outcome.loc["T1", "status"] == "ok"
    assert outcome.loc["T1", "transition_count"] == pytest.approx(2.0)
    assert outcome.loc["T1", "n_fixations_total"] == 7
    assert outcome.loc["T1", "n_fixations_assigned"] == 6
    assert outcome.loc["T1", "n_evaluable_adjacent_pairs"] == 4

    assert outcome.loc["T2", "status"] == "no_fixations"
    assert pd.isna(outcome.loc["T2", "transition_count"])

    assert outcome.loc["T3", "status"] == "no_aoi_assigned_fixations"
    assert pd.isna(outcome.loc["T3", "transition_count"])

    adjacency = srl_outcome.transition_sequence_audit(dataset)
    t1 = adjacency.loc[adjacency["trial_id"].eq("T1")]
    assert int(t1["between_aoi_transition"].sum()) == 2
    assert int(t1["evaluable"].sum()) == 4
