from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.gazepoint_workflow_10 as gpw

FIX = Path(__file__).parent / "fixtures" / "gazepoint"


def _base_x():
    x = ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)
    return ep.build_gazepoint_media_trials(x)


def test_spec_tokens_first_finite_and_clean_output_guards(tmp_path):
    spec = gpw.gazepoint_workflow_spec()
    assert "eye_gazepoint_workflow_spec" in repr(spec)

    with pytest.raises(ValueError, match="pupil_interpolation"):
        gpw.gazepoint_workflow_spec(pupil_interpolation="spline")
    with pytest.raises(ValueError, match="pupil_filter"):
        gpw.gazepoint_workflow_spec(pupil_filter="butter")
    with pytest.raises(ValueError, match="pupil_baseline"):
        gpw.gazepoint_workflow_spec(pupil_baseline="center")
    with pytest.raises(ValueError, match="baseline_window"):
        gpw.gazepoint_workflow_spec(pupil_baseline_window=(1, 0))

    assert gpw._workflow_token(None) == "missing"
    assert gpw._workflow_token("   ") == "missing"
    assert gpw._workflow_token("a b///c") == "a_b_c"
    assert gpw._first_nonmissing([pd.NA, "", "ok"], "fallback") == "ok"
    assert gpw._first_nonmissing([pd.NA, ""], "fallback") == "fallback"
    finite = gpw._finite(["1", "bad", 2])
    assert np.isclose(finite[[0, 2]], [1.0, 2.0]).all()
    assert np.isnan(finite[1])

    file_path = tmp_path / "as-file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="exists as a file"):
        gpw._clean_output(file_path, overwrite=False)

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        gpw._clean_output(nonempty, overwrite=False)
    cleaned = gpw._clean_output(nonempty, overwrite=True)
    assert cleaned.is_dir()
    assert not (cleaned / "old.txt").exists()


def test_item_map_validation_csv_completion_and_duplicates(tmp_path):
    stimuli = pd.Series(["s1", "s2", "s2", pd.NA, ""])
    default = gpw._item_map(None, stimuli)
    assert default["stimulus_id"].tolist() == ["s1", "s2"]

    with pytest.raises(ValueError, match="does not exist"):
        gpw._item_map(tmp_path / "missing.csv", stimuli)
    with pytest.raises(TypeError, match="DataFrame"):
        gpw._item_map(123, stimuli)
    with pytest.raises(ValueError, match="missing required"):
        gpw._item_map(pd.DataFrame({"stimulus_id": ["s1"]}), stimuli)
    with pytest.raises(ValueError, match="must be unique"):
        gpw._item_map(
            pd.DataFrame({"stimulus_id": ["s1", "s1"], "item_id": ["i1", "i2"]}),
            stimuli,
        )

    csv_path = tmp_path / "map.csv"
    pd.DataFrame({"stimulus_id": ["s1"], "item_id": ["i1"]}).to_csv(csv_path, index=False)
    mapped = gpw._item_map(csv_path, stimuli)
    assert set(mapped["stimulus_id"].astype(str)) == {"s1", "s2"}
    assert mapped.loc[mapped["stimulus_id"].astype(str).eq("s1"), "item_id"].iloc[0] == "i1"
    assert "condition_id" in mapped


def test_media_trial_failure_and_no_sample_id_paths():
    raw = ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)
    empty = raw.copy()
    empty["gaze_samples"] = empty["gaze_samples"].iloc[0:0].copy()
    with pytest.raises(ValueError, match="media/stimulus"):
        gpw.build_gazepoint_media_trials(empty)

    invalid_time = raw.copy()
    invalid_time["gaze_samples"] = invalid_time["gaze_samples"].copy()
    invalid_time["gaze_samples"].loc[:, "timestamp_seconds"] = np.nan
    with pytest.raises(ValueError, match="No Gazepoint media trials"):
        gpw.build_gazepoint_media_trials(invalid_time)

    no_sample_id = raw.copy()
    no_sample_id["gaze_samples"] = no_sample_id["gaze_samples"].drop(columns=["sample_id"])
    out = gpw.build_gazepoint_media_trials(no_sample_id)
    assert len(ep.trial_table(out)) == 2


def test_feature_linkage_by_recording_participant_and_unmatched_paths():
    x = _base_x()
    assert gpw._link_features_to_trials(x.copy()) is not None

    template = gpw._workflow_trial_features(x).iloc[[0]].copy()
    rows = []
    direct = template.copy()
    direct.loc[:, "feature_id"] = "direct"
    direct.loc[:, "trial_id"] = pd.NA
    direct.loc[:, "item_id"] = pd.NA
    direct.loc[:, "participant_id"] = pd.NA
    rows.append(direct)

    fallback = template.copy()
    fallback.loc[:, "feature_id"] = "fallback"
    fallback.loc[:, "trial_id"] = pd.NA
    fallback.loc[:, "item_id"] = pd.NA
    fallback.loc[:, "recording_id"] = pd.NA
    rows.append(fallback)

    unmatched = template.copy()
    unmatched.loc[:, "feature_id"] = "unmatched"
    unmatched.loc[:, "trial_id"] = pd.NA
    unmatched.loc[:, "item_id"] = pd.NA
    unmatched.loc[:, "recording_id"] = "missing-recording"
    unmatched.loc[:, "participant_id"] = "missing-participant"
    rows.append(unmatched)

    y = x.copy()
    y["features"] = gpw.standardize_eye_table(pd.concat(rows, ignore_index=True), "features")
    linked = gpw._link_features_to_trials(y)
    by_id = linked["features"].set_index("feature_id")
    assert pd.notna(by_id.loc["direct", "trial_id"])
    assert pd.notna(by_id.loc["fallback", "trial_id"])
    assert pd.notna(by_id.loc["fallback", "recording_id"])
    assert pd.isna(by_id.loc["unmatched", "trial_id"])

    no_trials = y.copy()
    no_trials["intervals"] = gpw.empty_eye_table("intervals")
    untouched = gpw._link_features_to_trials(no_trials)
    assert untouched["features"].equals(no_trials["features"])


def test_response_input_validation_matching_series_key_and_defaults(tmp_path):
    x = _base_x()
    trial = ep.trial_table(x).iloc[0]

    with pytest.raises(ValueError, match="does not exist"):
        gpw._prepare_responses(x, tmp_path / "missing.csv")
    with pytest.raises(TypeError, match="DataFrame"):
        gpw._prepare_responses(x, 123)
    with pytest.raises(ValueError, match="missing required"):
        gpw._prepare_responses(x, pd.DataFrame({"participant_id": [trial["participant_id"]]}))
    with pytest.raises(ValueError, match="exactly one trial"):
        gpw._prepare_responses(
            x,
            pd.DataFrame({"participant_id": ["nobody"], "item_id": [trial["item_id"]]}),
        )
    with pytest.raises(TypeError, match="score_key"):
        gpw._prepare_responses(
            x,
            pd.DataFrame(
                {
                    "participant_id": [trial["participant_id"]],
                    "item_id": [trial["item_id"]],
                    "response": ["yes"],
                }
            ),
            score_key=["yes"],
        )

    responses = pd.DataFrame(
        {
            "participant_id": [trial["participant_id"]],
            "item_id": [trial["item_id"]],
            "trial_id": [trial["trial_id"]],
            "response": ["yes"],
        }
    )
    csv_path = tmp_path / "responses.csv"
    responses.to_csv(csv_path, index=False)
    prepared = gpw._prepare_responses(x, csv_path, pd.Series({str(trial["item_id"]): "yes"}))
    assert prepared["supplied"] is True
    assert np.isclose(pd.to_numeric(prepared["dataset"]["responses"]["score"]).iloc[0], 1.0)

    no_response = gpw._prepare_responses(
        x,
        pd.DataFrame({"participant_id": [trial["participant_id"]], "item_id": [trial["item_id"]]}),
        score_key={str(trial["item_id"]): "yes"},
    )
    assert pd.isna(no_response["dataset"]["responses"]["response"].iloc[0])
    assert np.isnan(pd.to_numeric(no_response["dataset"]["responses"]["score"]).iloc[0])


def test_empty_and_alternate_pupil_biometric_feature_paths():
    x = _base_x()

    no_eye = x.copy()
    no_eye["eye_samples"] = gpw.empty_eye_table("eye_samples")
    assert gpw._preprocess_pupil(no_eye, gpw.gazepoint_workflow_spec()) is no_eye

    minimal_spec = gpw.gazepoint_workflow_spec(
        detect_blinks=False,
        pupil_interpolation="none",
        pupil_filter="none",
        pupil_baseline="subtract",
        pupil_baseline_window=(0, 0.5),
    )
    pupil = gpw._preprocess_pupil(x.copy(), minimal_spec)
    assert not pupil["eye_samples"].empty

    no_bio = x.copy()
    no_bio["biometrics"] = gpw.empty_eye_table("biometrics")
    assert gpw._prepare_biometrics(no_bio) is no_bio
    assert gpw._workflow_biometric_features(no_bio).empty

    no_trial_bio = x.copy()
    no_trial_bio["biometrics"] = no_trial_bio["biometrics"].copy()
    no_trial_bio["biometrics"].loc[:, "trial_id"] = pd.NA
    assert gpw._workflow_biometric_features(no_trial_bio).empty

    raw = ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)
    assert gpw._workflow_trial_features(raw).empty
    with pytest.raises(ValueError, match="Media/trial intervals"):
        gpw.derive_gazepoint_workflow_features(raw)

    samples_only = x.copy()
    samples_only["episodes"] = gpw.empty_eye_table("episodes")
    derived = gpw.derive_gazepoint_workflow_features(samples_only)
    assert not derived["features"].empty


def test_summary_wide_analysis_and_response_matrix_empty_paths():
    x = _base_x()
    trials = ep.trial_table(x)
    trial_base = trials[
        ["recording_id", "participant_id", "trial_id", "item_id", "stimulus_id", "condition_id", "start_time", "end_time"]
    ].copy()

    assert gpw._wide_features(pd.DataFrame(), ["trial_id"]).empty
    assert gpw._wide_features(pd.DataFrame({"feature_name": ["x"], "value": [1]}), ["missing"]).empty
    assert gpw._wide_features(
        pd.DataFrame({"trial_id": ["t"], "feature_name": [pd.NA], "value": [1]}), ["trial_id"]
    ).empty

    no_ep = x.copy()
    no_ep["episodes"] = gpw.empty_eye_table("episodes")
    assert gpw._summarize_fixations(no_ep).empty
    assert gpw._summarize_fixations(x, source="not-a-source").empty
    assert gpw._summarize_fixations(x, by=("not_a_column",), source="all").empty

    no_eye = x.copy()
    no_eye["eye_samples"] = gpw.empty_eye_table("eye_samples")
    assert gpw._pupil_summary(no_eye, trial_base).empty
    no_eye_trial = x.copy()
    no_eye_trial["eye_samples"] = no_eye_trial["eye_samples"].copy()
    no_eye_trial["eye_samples"].loc[:, "trial_id"] = pd.NA
    assert gpw._pupil_summary(no_eye_trial, trial_base).empty

    no_bio = x.copy()
    no_bio["biometrics"] = gpw.empty_eye_table("biometrics")
    assert gpw._biometric_summary(no_bio, trial_base).empty
    no_bio_trial = x.copy()
    no_bio_trial["biometrics"] = no_bio_trial["biometrics"].copy()
    no_bio_trial["biometrics"].loc[:, "trial_id"] = pd.NA
    assert gpw._biometric_summary(no_bio_trial, trial_base).empty

    no_features = x.copy()
    no_features["features"] = gpw.empty_eye_table("features")
    assert gpw._feature_dictionary(no_features).empty
    no_features["episodes"] = gpw.empty_eye_table("episodes")
    no_features["eye_samples"] = gpw.empty_eye_table("eye_samples")
    no_features["biometrics"] = gpw.empty_eye_table("biometrics")
    tables = gpw.gazepoint_analysis_tables(no_features)
    assert len(tables["process"]) == len(trials)
    assert tables["fixation_summary"].empty

    raw = ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)
    with pytest.raises(ValueError, match="No trial intervals"):
        gpw.gazepoint_analysis_tables(raw)

    assert gpw._response_matrix(pd.DataFrame(), "score") is None
    assert gpw._response_matrix(pd.DataFrame({"participant_id": ["p"], "item_id": ["i"]}), "score") is None
    assert gpw._response_matrix(
        pd.DataFrame({"participant_id": ["p"], "item_id": ["i"], "score": [np.nan]}), "score"
    ) is None
    duplicate = pd.DataFrame(
        {
            "participant_id": ["p", "p"],
            "item_id": ["i", "i"],
            "score": [0.0, 1.0],
        }
    )
    matrix = gpw._response_matrix(duplicate, "score")
    assert matrix.loc["p", "i"] == 1.0


def test_irt_default_process_report_plot_failure_and_validation_guards(tmp_path):
    x = _base_x()
    x = gpw._prepare_biometrics(x)
    x = gpw.derive_gazepoint_workflow_features(x)
    trial = ep.trial_table(x).iloc[0]
    response = pd.DataFrame(
        {
            "participant_id": [trial["participant_id"]],
            "item_id": [trial["item_id"]],
            "trial_id": [trial["trial_id"]],
            "response": ["yes"],
            "score": [1.0],
            "response_time": [0.8],
        }
    )
    x = gpw._prepare_responses(x, response)["dataset"]
    irt = gpw.gazepoint_irt_tables(x, process_table=None)
    assert irt["status"] == "structurally_ready_validation_only"
    assert irt["response_matrix"] is not None
    assert irt["response_time_matrix"] is not None

    failed = gpw._save_workflow_plot(tmp_path / "failed.png", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert failed["status"] == "failed"
    assert "boom" in failed["message"]

    with pytest.raises(TypeError, match="workflow"):
        gpw.write_gazepoint_workflow_report({})
    with pytest.raises(TypeError, match="run_gazepoint_workflow"):
        gpw.validate_gazepoint_workflow({})


def test_full_default_output_run_covers_plots_html_raw_and_reproducibility(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)
    built = ep.build_gazepoint_media_trials(raw)
    trials = ep.trial_table(built)
    responses = pd.DataFrame(
        {
            "participant_id": trials["participant_id"].tolist(),
            "item_id": trials["item_id"].tolist(),
            "response": ["yes"] * len(trials),
            "response_time": [0.8] * len(trials),
        }
    )
    score_key = pd.Series({str(item): "yes" for item in trials["item_id"]})
    result = gpw.run_gazepoint_workflow(
        FIX,
        output_dir=None,
        responses=responses,
        score_key=score_key,
        spec=gpw.gazepoint_workflow_spec(
            create_plots=True,
            create_html_report=True,
            retain_raw=True,
        ),
        quiet=True,
    )
    assert isinstance(result, gpw.GazepointWorkflow)
    assert "eye_gazepoint_workflow" in repr(result)
    assert Path(result.paths["report"]).is_file()
    assert Path(result.paths["report_html"]).is_file()
    assert Path(result.paths["plot_manifest"]).is_file()
    assert Path(result.paths["responses_supplied"]).is_file()
    assert Path(result.paths["item_map"]).is_file()
    assert result.workflow_checks["passed"].all()

    with pytest.raises(AttributeError):
        _ = result.not_a_field

    with pytest.raises(TypeError, match="spec"):
        gpw.run_gazepoint_workflow(FIX, output_dir=tmp_path / "bad-spec", spec={})
    with pytest.raises(ValueError, match="source directory"):
        gpw.run_gazepoint_workflow(tmp_path / "missing-source", output_dir=tmp_path / "bad-source")
