from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

import eyeprocesspy as ep

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "gaze_claim_robustness"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))


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
srl_aoi_geometry = _load_module(
    "srl_aoi_geometry_research",
    RESEARCH / "srl_aoi_geometry.py",
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
srl_readiness = _load_module(
    "srl_readiness_research",
    RESEARCH / "check_srl_execution_readiness.py",
)
srl_universe = _load_module(
    "srl_universe_research",
    RESEARCH / "build_srl_primary_universe.py",
)
srl_model_table = _load_module(
    "srl_model_table_research",
    RESEARCH / "prepare_srl_model_table.py",
)
srl_measurement = _load_module(
    "srl_measurement_research",
    RESEARCH / "run_srl_measurement_universe.py",
)
srl_model_universe = _load_module(
    "srl_model_universe_research",
    RESEARCH / "run_srl_model_universe.py",
)
srl_summary = _load_module(
    "srl_summary_research",
    RESEARCH / "summarise_srl_multiverse.py",
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
    assert manifest["event_detector_frozen"].all()
    assert manifest["aoi_geometry_frozen"].all()
    assert manifest["quality_rule_frozen"].all()
    assert manifest["model_frozen"].all()
    assert set(manifest["primary_estimand_family"]) == {
        "between_aoi_transition_rate"
    }
    assert set(manifest["focal_contrast"]) == {"Prompt_minus_Non-prompt"}

    errors = issues.loc[issues["severity"].eq("error")]
    warnings = issues.loc[issues["severity"].eq("warning")]
    assert errors.empty
    assert len(warnings) == 1
    assert "84 complete ET recordings" in warnings.iloc[0]["message"]


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
    assert t1["temporal_order_valid"].all()
    assert t1["inter_fixation_gap_ms"].tolist() == pytest.approx([50.0] * 6)
    assert outcome.loc["T1", "n_negative_gap_pairs"] == 0
    assert outcome.loc["T1", "median_inter_fixation_gap_ms"] == pytest.approx(50.0)
    assert outcome.loc["T1", "max_inter_fixation_gap_ms"] == pytest.approx(50.0)
    assert (
        outcome.loc["T1", "outcome_operationalization"]
        == "adjacent_aoi_assigned_fixation_change"
    )


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


def test_srl_execution_readiness_blocks_pre_results_execution():
    status = srl_readiness.evaluate_readiness(RESEARCH)
    blockers = status.loc[
        status["severity"].eq("BLOCKER") & ~status["ready"].astype(bool)
    ]

    assert not blockers.empty
    assert "actual_archive_audit" in set(blockers["gate_id"])
    assert "prompt_identifiability" in set(blockers["gate_id"])
    assert "aoi_exact_boundary" not in set(blockers["gate_id"])
    assert "quality_recovery_calibration" not in set(blockers["gate_id"])

    model = status.loc[status["gate_id"].eq("primary_model_engine")]
    assert len(model) == 1
    assert bool(model.iloc[0]["ready"])

    info = status.loc[status["gate_id"].eq("material_type_secondary_scope")]
    assert len(info) == 1
    assert bool(info.iloc[0]["ready"])


def test_srl_aoi_geometry_branches_are_pre_results_and_area_explicit(
    tmp_path: Path,
):
    participants_path, stimuli_path, _raw_dir, raw_path = _write_fixture(tmp_path)
    dataset = srl_adapter.load_srl_trial(
        raw_path,
        participants_path,
        stimuli_path,
        stimulus_name="Task_1",
        eye="left",
    )

    manifest = srl_aoi_geometry.geometry_manifest()
    exact = manifest.loc[manifest["convention"].eq("exact_quarters")]
    reported = manifest.loc[
        manifest["convention"].eq("reported_area_center_seam")
    ]
    assert len(exact) == 4
    assert len(reported) == 4
    assert exact["area_px"].eq(360000.0).all()
    assert reported["area_px"].eq(359550.0).all()

    registered = srl_aoi_geometry.register_srl_quartile_aois(
        dataset,
        stimulus_id="Task_1",
        convention="reported_area_center_seam",
    )
    definitions = registered["aoi_definitions"]
    geometry = registered["aoi_geometry"]

    assert len(definitions) == 4
    assert definitions["stimulus_id"].eq("Task_1").all()
    assert geometry["width"].eq(799.0).all()
    assert geometry["height"].eq(450.0).all()

    angular = srl_coordinates.convert_srl_pixels_to_degrees(
        dataset,
        viewing_distance_cm=65.0,
    )
    angular_registered = srl_aoi_geometry.register_srl_quartile_aois(
        angular,
        stimulus_id="Task_1",
        convention="exact_quarters",
    )
    angular_geometry = angular_registered["aoi_geometry"]
    ppd = srl_coordinates.pixels_per_degree(65.0)
    assert angular_geometry["width"].tolist() == pytest.approx(
        [800.0 / ppd] * 4
    )
    assert angular_geometry["height"].tolist() == pytest.approx(
        [450.0 / ppd] * 4
    )


def test_srl_primary_universe_is_deterministic_and_structurally_compatible():
    a = srl_universe.build_primary_universe(RESEARCH)
    b = srl_universe.build_primary_universe(RESEARCH)

    assert len(a) == 144
    assert a.equals(b)
    assert a["specification_hash"].nunique() == 144
    assert set(a["eye"]) == {"left", "right"}
    assert set(a["viewing_distance_cm"]) == {60.0, 65.0, 70.0}
    assert set(a["aoi_convention"]) == {
        "exact_quarters",
        "reported_area_center_seam",
    }
    assert set(a["quality_rule"]) == {
        "released_sample",
        "trial_80_sensitivity",
    }
    assert set(a["cohort_id"]) == {
        "exact_raw82",
        "nominal250_77",
    }
    assert set(a["detector_id"]) == {
        "ivt_30_100_simple",
        "ivt_40_50_simple",
        "idt_1_100",
    }
    assert not a["requires_dense_regular_timebase"].any()

    refs = srl_universe.historical_reference_manifest()
    assert set(refs["reference_id"]) == {
        "remodnav_1_1_2",
        "vendor_begaze_released",
        "adaptive_mad_eyeprocesspy",
    }
    assert refs["role"].str.contains("outside_primary_denominator").all()


def test_srl_transition_outcome_rejects_overlapping_fixation_sequence():
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
            }
        ]
    )
    episodes = pd.DataFrame(
        [
            {
                "episode_id": "E1",
                "recording_id": "R1",
                "episode_type": "fixation",
                "start_time": 0.10,
                "end_time": 0.30,
                "duration_ms": 200.0,
                "centroid_x": 10.0,
                "centroid_y": 10.0,
                "coordinate_space_id": "px",
                "derived_by": "eyeprocess",
                "trial_id": "T1",
                "stimulus_id": "Task_1",
                "aoi_id": "A",
            },
            {
                "episode_id": "E2",
                "recording_id": "R1",
                "episode_type": "fixation",
                "start_time": 0.25,
                "end_time": 0.40,
                "duration_ms": 150.0,
                "centroid_x": 20.0,
                "centroid_y": 20.0,
                "coordinate_space_id": "px",
                "derived_by": "eyeprocess",
                "trial_id": "T1",
                "stimulus_id": "Task_1",
                "aoi_id": "B",
            },
        ]
    )
    dataset = ep.new_eye_dataset(
        recordings=recordings,
        intervals=intervals,
        episodes=episodes,
    )

    outcome = srl_outcome.derive_transition_counts(dataset).iloc[0]
    assert outcome["status"] == "overlapping_fixations"
    assert pd.isna(outcome["transition_count"])
    assert outcome["n_negative_gap_pairs"] == 1
    assert outcome["median_inter_fixation_gap_ms"] == pytest.approx(-50.0)

    adjacency = srl_outcome.transition_sequence_audit(dataset)
    assert len(adjacency) == 1
    assert adjacency.iloc[0]["inter_fixation_gap_ms"] == pytest.approx(-50.0)
    assert not bool(adjacency.iloc[0]["temporal_order_valid"])
    assert not bool(adjacency.iloc[0]["evaluable"])
    assert not bool(adjacency.iloc[0]["between_aoi_transition"])


def test_srl_measurement_runner_materializes_all_base_branches(tmp_path: Path):
    participants_path, stimuli_path, _raw_dir, raw_path = _write_fixture(tmp_path)
    participants = pd.read_csv(participants_path)
    stimuli = pd.read_csv(stimuli_path)

    results, statuses, failures, detector_warnings = srl_measurement.run_participant(
        "P001",
        raw_file=raw_path,
        participants_csv=participants_path,
        stimuli_csv=stimuli_path,
        participants=participants,
        stimuli=stimuli,
    )

    assert len(results) == 288
    assert len(statuses) == 18
    assert failures.empty
    assert detector_warnings.empty
    assert set(results["detector_id"]) == {
        "ivt_30_100_simple",
        "ivt_40_50_simple",
        "idt_1_100",
    }
    assert set(results["eye"]) == {"left", "right"}
    assert set(results["viewing_distance_cm"]) == {60.0, 65.0, 70.0}
    assert set(results["aoi_convention"]) == {
        "exact_quarters",
        "reported_area_center_seam",
    }
    assert set(results["stimulus_id"]) == {
        f"Task_{i}" for i in range(1, 9)
    }
    assert results["status"].isin(
        {
            "ok",
            "no_fixations",
            "no_aoi_assigned_fixations",
            "overlapping_fixations",
        }
    ).all()
    key = [
        "participant_id",
        "stimulus_id",
        "detector_id",
        "eye",
        "viewing_distance_cm",
        "aoi_convention",
    ]
    assert not results.duplicated(key).any()


def test_srl_model_universe_cohort_quality_and_branch_selection():
    identity = pd.DataFrame(
        {
            "participant_id": [str(i) for i in range(1, 83)],
            "exact_cross_source_raw_eligible": [True] * 82,
        }
    )
    timebase = pd.DataFrame(
        {
            "participant_id": [str(i) for i in range(1, 83)],
            "all_8_tasks_nominal_250hz": [True] * 77 + [False] * 5,
        }
    )
    cohorts = srl_model_universe.cohort_sets(identity, timebase)
    assert len(cohorts["exact_raw82"]) == 82
    assert len(cohorts["nominal250_77"]) == 77

    rows = []
    for participant in ("1", "2"):
        for task_number in range(1, 9):
            task = f"Task_{task_number}"
            rows.append(
                {
                    "participant_id": participant,
                    "stimulus_id": task,
                    "detector_id": "ivt_30_100_simple",
                    "eye": "left",
                    "viewing_distance_cm": 60.0,
                    "aoi_convention": "exact_quarters",
                    "transition_count": 2.0,
                    "status": "ok",
                    "experiment_condition": (
                        "Prompt" if participant == "1" else "Non-prompt"
                    ),
                    "metadata_tracking_ratio_percent": (
                        79.0 if task == "Task_1" else 90.0
                    ),
                }
            )
    measurement = pd.DataFrame(rows)
    spec = pd.Series(
        {
            "universe_id": "srl_u001",
            "detector_id": "ivt_30_100_simple",
            "eye": "left",
            "viewing_distance_cm": 60.0,
            "aoi_convention": "exact_quarters",
        }
    )
    selected = srl_model_universe.select_measurement_branch(
        measurement,
        spec,
        {"1", "2"},
    )
    assert len(selected) == 16

    released = srl_model_universe.apply_quality_rule(
        selected,
        "released_sample",
    )
    assert released["status"].eq("ok").all()

    thresholded = srl_model_universe.apply_quality_rule(
        selected,
        "trial_80_sensitivity",
    )
    assert thresholded.loc[
        thresholded["stimulus_id"].eq("Task_1"),
        "status",
    ].str.contains("quality_excluded_tracking_ratio_lt80").all()
    assert thresholded.loc[
        thresholded["stimulus_id"].eq("Task_2"),
        "status",
    ].eq("ok").all()


def test_srl_summary_reports_direction_without_automatic_verdict():
    results = pd.DataFrame(
        [
            {
                "universe_id": "u1",
                "status": "ok",
                "estimate": 0.2,
                "SE": 0.05,
                "CI_lower": 0.1,
                "CI_upper": 0.3,
                "rate_ratio": 1.22,
                "detector_id": "ivt30",
                "eye": "left",
                "viewing_distance_cm": 60.0,
                "aoi_convention": "exact",
                "quality_rule": "released",
                "cohort_id": "all",
            },
            {
                "universe_id": "u2",
                "status": "ok",
                "estimate": -0.1,
                "SE": 0.04,
                "CI_lower": -0.2,
                "CI_upper": 0.0,
                "rate_ratio": 0.90,
                "detector_id": "idt",
                "eye": "right",
                "viewing_distance_cm": 70.0,
                "aoi_convention": "seam",
                "quality_rule": "80",
                "cohort_id": "250",
            },
            {
                "universe_id": "u3",
                "status": "fit_failed",
                "estimate": float("nan"),
                "SE": float("nan"),
                "CI_lower": float("nan"),
                "CI_upper": float("nan"),
                "rate_ratio": float("nan"),
                "detector_id": "idt",
                "eye": "left",
                "viewing_distance_cm": 65.0,
                "aoi_convention": "exact",
                "quality_rule": "released",
                "cohort_id": "all",
            },
        ]
    )

    summary, decision = srl_summary.summarize_results(results)
    row = summary.iloc[0]
    assert row["planned_specifications"] == 3
    assert row["successful_specifications"] == 2
    assert row["non_ok_specifications"] == 1
    assert row["positive_direction_proportion"] == pytest.approx(0.5)
    assert row["median_log_rate_ratio"] == pytest.approx(0.05)
    assert set(decision["decision"]) == {
        "detector_id",
        "eye",
        "viewing_distance_cm",
        "aoi_convention",
        "quality_rule",
        "cohort_id",
    }
