from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import eyeprocesspy as ep
import eyeprocesspy.gazepoint_workflow_10 as gpw

FIX = Path(__file__).parent / "fixtures" / "gazepoint"

TARGETS = [
    "build_gazepoint_media_trials",
    "derive_gazepoint_workflow_features",
    "gazepoint_analysis_tables",
    "gazepoint_irt_tables",
    "gazepoint_workflow_spec",
    "plot_gazepoint_workflow",
    "run_gazepoint_workflow",
    "validate_gazepoint_workflow",
    "write_gazepoint_workflow_report",
]


def test_r019_public_api_targets_are_exported():
    assert len(TARGETS) == 9
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_workflow_spec_preserves_frozen_defaults():
    spec = ep.gazepoint_workflow_spec()
    assert spec.expected_sampling_rate == 60
    assert spec.minimum_valid_gaze == 0.80
    assert spec.minimum_valid_pupil == 0.70
    assert spec.pupil_interpolation == "linear"
    assert spec.pupil_filter == "median"
    assert spec.pupil_baseline == "none"
    assert spec.detect_blinks is True


def test_media_trials_are_reconstructed_from_contiguous_stimuli():
    x = ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)
    out = ep.build_gazepoint_media_trials(x)
    trials = ep.trial_table(out)
    assert len(trials) == 2
    assert list(trials.item_id.astype(str)) == ["item01", "item02"]
    assert list(trials.stimulus_id.astype(str)) == ["item01", "item02"]
    assert set(out["gaze_samples"].trial_id.dropna()) == set(trials.trial_id)
    assert set(out["eye_samples"].trial_id.dropna()) == set(trials.trial_id)


def test_workflow_wide_features_keep_heterogeneous_trial_schemas():
    features = pd.DataFrame(
        {
            "recording_id": ["R1", "R2"],
            "participant_id": ["P1", "P2"],
            "trial_id": ["T1", "T2"],
            "item_id": ["I1", "I2"],
            "stimulus_id": ["S1", "S2"],
            "feature_name": ["fixation_count", "heart_rate_mean"],
            "value": [4, 72],
        }
    )
    result = gpw._wide_features(
        features,
        [
            "recording_id",
            "participant_id",
            "trial_id",
            "item_id",
            "stimulus_id",
        ],
    )
    assert len(result) == 2
    assert {"fixation_count", "heart_rate_mean"} <= set(result.columns)
    t1 = result[result.trial_id.eq("T1")].iloc[0]
    t2 = result[result.trial_id.eq("T2")].iloc[0]
    assert t1.fixation_count == 4
    assert pd.isna(t1.heart_rate_mean)
    assert pd.isna(t2.fixation_count)
    assert t2.heart_rate_mean == 72


def test_analysis_tables_accept_empty_aoi_feature_cases():
    x = ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)
    x = ep.build_gazepoint_media_trials(x)
    x = gpw._prepare_biometrics(x)
    x = ep.derive_gazepoint_workflow_features(x)
    x["features"].loc[:, "aoi_id"] = pd.NA

    tables = ep.gazepoint_analysis_tables(x)
    assert len(tables["trials"]) == 2
    assert len(tables["process"]) == 2
    assert isinstance(tables["aoi_summary"], pd.DataFrame)
    assert tables["aoi_summary"].empty
    assert not tables["fixation_summary"].empty
    assert not tables["pupil_summary"].empty
    assert not tables["biometric_summary"].empty


def test_integrated_workflow_creates_complete_downstream_outputs(tmp_path):
    output = tmp_path / "workflow"
    spec = ep.gazepoint_workflow_spec(
        create_plots=False,
        create_html_report=False,
        retain_raw=False,
        pupil_baseline="none",
    )
    result = ep.run_gazepoint_workflow(
        FIX,
        output_dir=output,
        spec=spec,
        overwrite=True,
        quiet=True,
    )

    assert isinstance(result, gpw.GazepointWorkflow)
    assert result.status == "pass"
    assert len(result.dataset["recordings"]) == 1
    assert len(result.tables["trials"]) == 2
    assert len(result.tables["process"]) == 2
    assert result.irt["status"] == "process_ready_response_pending"
    assert len(result.tables["fixation_summary"]) > 0
    assert len(result.tables["pupil_summary"]) > 0
    assert len(result.tables["biometric_summary"]) > 0
    assert Path(result.paths["report"]).is_file()
    assert Path(result.paths["canonical_dataset"]).is_dir()
    assert (output / "irt" / "response-template.csv").is_file()
    assert (output / "tables" / "process.csv").is_file()
    assert (output / "rerun-workflow.R").is_file()
    assert (output / "rerun-workflow.py").is_file()
    assert (output / "workflow-spec.json").is_file()
    assert (output / "workflow-result.json").is_file()

    checks = ep.validate_gazepoint_workflow(result)
    assert checks.passed.all()


def test_workflow_accepts_observed_responses_without_inventing_scores(tmp_path):
    output = tmp_path / "workflow-responses"
    responses = pd.DataFrame(
        {
            "participant_id": ["demo", "demo"],
            "item_id": ["item01", "item02"],
            "response": ["yes", "no"],
            "response_time": [0.8, 0.7],
        }
    )
    result = ep.run_gazepoint_workflow(
        FIX,
        output_dir=output,
        responses=responses,
        score_key={"item01": "yes", "item02": "no"},
        spec=ep.gazepoint_workflow_spec(
            create_plots=False,
            create_html_report=False,
            retain_raw=False,
        ),
        overwrite=True,
        quiet=True,
    )
    assert result.responses_supplied
    assert np.isclose(
        pd.to_numeric(result.dataset["responses"].score).sum(),
        2.0,
    )
    assert result.irt["status"] == "structurally_ready_validation_only"
    assert result.irt["response_matrix"] is not None
    assert result.irt["response_matrix"].shape == (1, 2)


def test_workflow_does_not_fabricate_response_scores(tmp_path):
    result = ep.run_gazepoint_workflow(
        FIX,
        output_dir=tmp_path / "workflow-no-responses",
        spec=ep.gazepoint_workflow_spec(
            create_plots=False,
            create_html_report=False,
            retain_raw=False,
        ),
        overwrite=True,
        quiet=True,
    )
    assert result.dataset["responses"].empty
    assert result.irt["response_matrix"] is None
    assert result.irt["status"] == "process_ready_response_pending"


def test_workflow_plot_manifest_closes_figures(tmp_path):
    import matplotlib.pyplot as plt

    x = ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)
    x = ep.build_gazepoint_media_trials(x)
    x = gpw._prepare_biometrics(x)
    x = ep.derive_gazepoint_workflow_features(x)

    before = len(plt.get_fignums())
    manifest = ep.plot_gazepoint_workflow(
        x,
        tmp_path / "plots",
        channels=["heart_rate"],
    )
    after = len(plt.get_fignums())
    assert not manifest.empty
    assert (manifest.status == "created").any()
    assert after == before
