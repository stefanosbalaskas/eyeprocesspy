from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

import eyeprocesspy as ep

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
srl_identifiability = _load_module(
    "srl_identifiability_research",
    RESEARCH / "audit_srl_identifiability.py",
)
srl_glmm_bridge = _load_module(
    "srl_glmm_bridge_research",
    RESEARCH / "srl_glmm_bridge.py",
)
srl_model_table = _load_module(
    "srl_model_table_research",
    RESEARCH / "prepare_srl_model_table.py",
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
        "between_aoi_transition_rate"
    }
    assert set(manifest["focal_contrast"]) == {"Prompt_minus_Non-prompt"}

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


def test_srl_identifiability_audit_blocks_nested_modality_claim():
    participants = pd.DataFrame(
        {
            "part_ID": ["P1", "P2", "P3", "P4"],
            "experiment_condition": [
                "Prompt",
                "Prompt",
                "Non-prompt",
                "Non-prompt",
            ],
        }
    )
    rows = []
    for participant in participants["part_ID"]:
        for task in range(1, 9):
            rows.append(
                {
                    "part_ID": participant,
                    "stimulus_name": f"Task_{task}",
                    "stimulus_type": (
                        "Text" if task <= 4 else "Multimedia"
                    ),
                }
            )
    stimuli = pd.DataFrame(rows)

    task_structure, estimands, issues = (
        srl_identifiability.audit_identifiability(
            participants,
            stimuli,
        )
    )
    assert task_structure["prompt_varies_within_task"].all()
    assert not task_structure["modality_varies_within_task"].any()

    status = estimands.set_index("estimand")
    assert bool(status.loc["Prompt_vs_Non-prompt", "primary_eligible"])
    assert (
        status.loc["Prompt_vs_Non-prompt", "design_status"]
        == "randomized_between_participants_and_crossed_with_tasks"
    )
    assert not bool(status.loc["Multimedia_vs_Text", "primary_eligible"])
    assert (
        status.loc["Multimedia_vs_Text", "design_status"]
        == "nested_in_task_identity"
    )
    assert issues["severity"].eq("warning").any()
    assert issues["message"].str.contains("nested in task identity").any()


def test_srl_identifiability_audit_allows_crossed_modality_design():
    participants = pd.DataFrame(
        {
            "part_ID": ["P1", "P2", "P3", "P4"],
            "experiment_condition": [
                "Prompt",
                "Prompt",
                "Non-prompt",
                "Non-prompt",
            ],
        }
    )
    rows = []
    for p_index, participant in enumerate(participants["part_ID"]):
        for task in range(1, 9):
            rows.append(
                {
                    "part_ID": participant,
                    "stimulus_name": f"Task_{task}",
                    "stimulus_type": (
                        "Text" if (task + p_index) % 2 == 0 else "Multimedia"
                    ),
                }
            )
    stimuli = pd.DataFrame(rows)

    task_structure, estimands, issues = (
        srl_identifiability.audit_identifiability(
            participants,
            stimuli,
        )
    )
    assert task_structure["prompt_varies_within_task"].all()
    assert task_structure["modality_varies_within_task"].all()
    status = estimands.set_index("estimand")
    assert bool(status.loc["Multimedia_vs_Text", "primary_eligible"])
    assert (
        status.loc["Multimedia_vs_Text", "design_status"]
        == "crossed_with_task_identity"
    )
    assert issues.empty


def test_srl_model_table_keeps_non_evaluable_rows_visible():
    outcome = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2"],
            "stimulus_id": ["Task_1", "Task_2", "Task_1", "Task_2"],
            "experiment_condition": [
                "Prompt",
                "Prompt",
                "Non-prompt",
                "Non-prompt",
            ],
            "transition_count": [5.0, pd.NA, 3.0, 4.0],
            "status": [
                "ok",
                "no_fixations",
                "ok",
                "ok",
            ],
        }
    )
    stimuli = pd.DataFrame(
        {
            "part_ID": ["P1", "P1", "P2", "P2"],
            "stimulus_name": ["Task_1", "Task_2", "Task_1", "Task_2"],
            "stimulus_type": ["Text", "Multimedia", "Text", "Multimedia"],
            "stimulus_time": [10.0, 12.0, 0.0, 8.0],
        }
    )

    table, audit = srl_model_table.prepare_model_table(outcome, stimuli)

    assert len(table) == 4
    assert int(table["model_evaluable"].sum()) == 2
    p1_t1 = table.loc[
        table["participant_id"].eq("P1")
        & table["stimulus_id"].eq("Task_1")
    ].iloc[0]
    assert p1_t1["prompt_indicator"] == pytest.approx(1.0)
    assert p1_t1["log_exposure"] == pytest.approx(__import__("math").log(10.0))

    p1_t2 = table.loc[
        table["participant_id"].eq("P1")
        & table["stimulus_id"].eq("Task_2")
    ].iloc[0]
    assert "outcome_status=no_fixations" in p1_t2["model_status"]

    p2_t1 = table.loc[
        table["participant_id"].eq("P2")
        & table["stimulus_id"].eq("Task_1")
    ].iloc[0]
    assert "invalid_or_nonpositive_exposure" in p2_t1["model_status"]

    analysis = srl_model_table.primary_analysis_rows(table)
    assert len(analysis) == 2
    assert analysis["model_status"].eq("ok").all()
    assert int(audit["rows"].sum()) == 4


def test_srl_glmm_bridge_preserves_log_rate_ratio_contract(tmp_path: Path):
    result_path = tmp_path / "primary_nb2_glmm_prompt_coefficient.csv"
    pd.DataFrame(
        [
            {
                "model_id": "primary_nb2_glmm",
                "term": "prompt_indicator",
                "estimand_id": "prompt_transition_rate_ratio",
                "estimate_log_rate_ratio": 0.2,
                "SE": 0.05,
                "CI_lower_log": 0.1,
                "CI_upper_log": 0.3,
                "converged": True,
                "n_rows": 640,
            }
        ]
    ).to_csv(result_path, index=False)

    result = srl_glmm_bridge.read_primary_glmm_result(result_path)

    assert result["estimate"] == pytest.approx(0.2)
    assert result["SE"] == pytest.approx(0.05)
    assert result["CI_lower"] == pytest.approx(0.1)
    assert result["CI_upper"] == pytest.approx(0.3)
    assert result["N"] == 640
    assert result["converged"] is True
    assert result["effect_scale"] == "log_rate_ratio"
    assert result["rate_ratio"] == pytest.approx(1.2214027581601699)


def test_srl_glmm_bridge_rejects_wrong_estimand_and_nonfinite_values(
    tmp_path: Path,
):
    result_path = tmp_path / "bad.csv"
    base = {
        "model_id": "primary_nb2_glmm",
        "term": "prompt_indicator",
        "estimand_id": "wrong",
        "estimate_log_rate_ratio": 0.2,
        "SE": 0.05,
        "CI_lower_log": 0.1,
        "CI_upper_log": 0.3,
        "converged": True,
        "n_rows": 640,
    }
    pd.DataFrame([base]).to_csv(result_path, index=False)
    with pytest.raises(ValueError, match="estimand_id"):
        srl_glmm_bridge.read_primary_glmm_result(result_path)

    base["estimand_id"] = "prompt_transition_rate_ratio"
    base["estimate_log_rate_ratio"] = float("nan")
    pd.DataFrame([base]).to_csv(result_path, index=False)
    with pytest.raises(ValueError, match="estimate_log_rate_ratio"):
        srl_glmm_bridge.read_primary_glmm_result(result_path)
