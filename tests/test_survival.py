import numpy as np
import pandas as pd
import pytest

from eyeprocesspy.survival import (
    CANONICAL_GAZE_SURVIVAL_COLUMNS,
    check_gaze_proportional_hazards,
    compare_gaze_survival_models,
    estimate_gaze_latency_quantiles,
    estimate_gaze_survival,
    fit_gaze_aft_model,
    fit_gaze_cox_model,
    fit_gaze_mixed_cox_model,
    prepare_gaze_survival_data,
    report_gaze_survival_model,
    simulate_gaze_survival_example,
    simulate_gaze_survival_inputs,
    summarise_gaze_censoring,
    tidy_gaze_survival_model,
    validate_gaze_survival_data,
)


def tiny_trials():
    return pd.DataFrame({
        "participant_id": ["P1", "P1", "P2", "P2"],
        "trial_id": ["T1", "T2", "T1", "T2"],
        "stimulus_id": ["S1", "S2", "S1", "S2"],
        "condition": ["control", "detail", "control", "detail"],
        "start_time": [0.0, 0.0, 0.0, 0.0],
        "end_time": [5.0, 5.0, 5.0, 5.0],
        "n_valid_samples": [300, 298, 301, 297],
        "valid_data_fraction": [0.98, 0.97, 0.99, 0.96],
    })


def tiny_events():
    return pd.DataFrame({
        "participant_id": ["P1", "P1", "P1", "P2", "P2"],
        "trial_id": ["T1", "T1", "T2", "T1", "T2"],
        "start_time": [1.2, 2.0, 1.0, 0.8, 2.7],
        "aoi_id": ["disclosure", "body", "body", "disclosure", "body"],
        "episode_type": ["fixation"] * 5,
    })


def test_prepare_retains_never_inspected_as_censored():
    d = prepare_gaze_survival_data(tiny_trials(), tiny_events(), target_aoi="disclosure")
    assert len(d) == 4
    assert d.event_observed.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert d.loc[d.event_observed.eq(0), "analysis_time"].eq(5.0).all()


def test_missing_event_not_silently_censor_without_evidence():
    tr = tiny_trials().drop(columns=[])
    with pytest.raises(ValueError, match="event_observed must be supplied explicitly"):
        prepare_gaze_survival_data(tr, events=None, target_aoi="disclosure")


def test_duplicate_trials_fail():
    tr = pd.concat([tiny_trials(), tiny_trials().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicated"):
        prepare_gaze_survival_data(tr, tiny_events(), target_aoi="disclosure")


def test_event_after_censor_is_error():
    d = simulate_gaze_survival_example(n_participants=4)
    d.loc[0, ["event_observed", "event_time", "analysis_time"]] = [1, 8.0, 8.0]
    issues = validate_gaze_survival_data(d, raise_on_error=False)
    assert "event_after_censor" in set(issues.code)


def test_zero_event_warns_but_is_not_rewritten():
    d = simulate_gaze_survival_example(n_participants=4)
    d.loc[0, ["event_observed", "event_time", "analysis_time"]] = [1, 0.0, 0.0]
    issues = validate_gaze_survival_data(d, raise_on_error=False)
    assert "event_at_time_zero" in set(issues.code)
    assert d.loc[0, "analysis_time"] == 0.0


def test_unusable_data_is_review_not_censor():
    tr = tiny_trials(); tr.loc[1, "valid_data_fraction"] = 0.2
    with pytest.warns(RuntimeWarning):
        d = prepare_gaze_survival_data(tr, tiny_events(), target_aoi="disclosure", min_valid_fraction=0.8)
    row = d.iloc[1]
    assert pd.isna(row.event_observed)
    assert row.review_required
    assert row.censor_reason == "unusable_gaze_quality"


def test_km_truth_and_censor_summary():
    d = tiny_survival_fixture()
    km = estimate_gaze_survival(d)
    first = km.iloc[0]
    assert first.n_risk == 6
    assert np.isclose(first.survival, 5/6)
    s = summarise_gaze_censoring(d).iloc[0]
    assert s.n_observed_events == 4
    assert s.n_censored == 2
    assert np.isclose(s.censoring_fraction, 2/6)


def tiny_survival_fixture():
    return pd.DataFrame({
        "participant_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
        "trial_id": ["1", "2", "1", "2", "1", "2"],
        "stimulus_id": ["S"]*6,
        "condition": ["A", "B"]*3,
        "target_aoi": ["x"]*6,
        "time_origin": ["trial_start"]*6,
        "event_time": [1.0, np.nan, 2.0, 2.5, np.nan, 4.0],
        "censor_time": [5.0]*6,
        "analysis_time": [1.0, 5.0, 2.0, 2.5, 5.0, 4.0],
        "event_observed": [1, 0, 1, 1, 0, 1],
        "event_type": ["first_fixation"]*6,
        "n_valid_samples": [300]*6,
        "valid_data_fraction": [0.98]*6,
        "trial_duration": [5.0]*6,
        "analysis_eligible": [True]*6,
    })


def test_models_fit_and_report():
    d = simulate_gaze_survival_example(n_participants=40, trials_per_participant=3)
    cox = fit_gaze_cox_model(d, "C(condition)")
    mixed = fit_gaze_mixed_cox_model(d, "C(condition)")
    weib = fit_gaze_aft_model(d, "C(condition)", distribution="weibull")
    logn = fit_gaze_aft_model(d, "C(condition)", distribution="lognormal")
    assert not tidy_gaze_survival_model(cox).empty
    assert mixed.repeated_structure == "cluster_robust:participant_id"
    assert check_gaze_proportional_hazards(cox).shape[0] == len(cox.covariate_names)
    cmp = compare_gaze_survival_models(cox, weib, logn)
    assert set(cmp.family) == {"cox", "aft_weibull", "aft_lognormal"}
    rep = report_gaze_survival_model(mixed)
    assert rep["N_participants"] == 40
    assert rep["N_trials"] == 120


def test_aft_rejects_zero_time():
    d = simulate_gaze_survival_example(n_participants=6)
    d.loc[0, ["event_time", "analysis_time", "event_observed"]] = [0.0, 0.0, 1]
    with pytest.raises(ValueError, match="strictly positive"):
        fit_gaze_aft_model(d, "C(condition)")


def test_frailty_request_fails_explicitly_in_python():
    d = simulate_gaze_survival_example(n_participants=6)
    with pytest.raises(NotImplementedError, match="latent frailty"):
        fit_gaze_mixed_cox_model(d, "C(condition)", structure="frailty")


def test_quantiles():
    d = simulate_gaze_survival_example(n_participants=20)
    q = estimate_gaze_latency_quantiles(d, probs=[0.5], group="condition")
    assert len(q) == 3
    aft = fit_gaze_aft_model(d, "C(condition)")
    q2 = estimate_gaze_latency_quantiles(aft, probs=[0.5], newdata=d.iloc[[0]])
    assert np.isfinite(q2["quantile"].iloc[0])


def test_recording_trial_join_matches_canonical_episode_contract():
    inputs = simulate_gaze_survival_inputs(
        "disclosure", seed=99, n_participants=4, trials_per_participant=2
    )
    # Canonical eyeprocess episodes do not need participant_id when recording_id
    # and trial_id jointly identify the trial.
    assert "participant_id" not in inputs["events"].columns
    d = prepare_gaze_survival_data(
        inputs["trials"],
        inputs["events"],
        target_aoi="disclosure",
        event_type="first_fixation",
        condition_col="condition_id",
    )
    assert set(d["event_join_key"]) == {"recording_trial"}
    assert len(d) == 8


def test_ambiguous_trial_only_event_join_fails():
    trials = tiny_trials().copy()
    events = tiny_events().drop(columns="participant_id")
    with pytest.raises(ValueError, match="trial-only matching would be ambiguous"):
        prepare_gaze_survival_data(trials, events, target_aoi="disclosure")


def test_custom_time_origin_is_explicit_and_changes_latency_window():
    trials = pd.DataFrame(
        {
            "participant_id": ["P1"],
            "trial_id": ["T1"],
            "start_time": [0.0],
            "stimulus_onset": [2.0],
            "end_time": [10.0],
            "valid_data_fraction": [0.99],
        }
    )
    events = pd.DataFrame(
        {
            "participant_id": ["P1"],
            "trial_id": ["T1"],
            "start_time": [5.0],
            "aoi_id": ["target"],
            "episode_type": ["fixation"],
        }
    )
    with pytest.warns(RuntimeWarning, match="Fewer than five"):
        d = prepare_gaze_survival_data(
            trials,
            events,
            target_aoi="target",
            time_origin="stimulus_onset",
            time_origin_col="stimulus_onset",
        )
    assert d.loc[0, "event_time"] == pytest.approx(3.0)
    assert d.loc[0, "censor_time"] == pytest.approx(8.0)
    assert d.loc[0, "trial_duration"] == pytest.approx(10.0)


def test_revisit_is_visit_level_not_second_consecutive_fixation():
    trials = pd.DataFrame(
        {"participant_id": ["P1"], "trial_id": ["T1"], "start_time": [0.0], "end_time": [5.0]}
    )
    events = pd.DataFrame(
        {
            "participant_id": ["P1"] * 4,
            "trial_id": ["T1"] * 4,
            "start_time": [0.5, 0.8, 1.3, 2.0],
            "aoi_id": ["target", "target", "body", "target"],
            "episode_type": ["fixation"] * 4,
        }
    )
    with pytest.warns(RuntimeWarning, match="Fewer than five"):
        d = prepare_gaze_survival_data(
            trials, events, target_aoi="target", event_type="first_revisit"
        )
    assert d.loc[0, "event_time"] == pytest.approx(2.0)


def test_transition_and_disengagement_event_semantics():
    trials = pd.DataFrame(
        {"participant_id": ["P1"], "trial_id": ["T1"], "start_time": [0.0], "end_time": [5.0]}
    )
    events = pd.DataFrame(
        {
            "participant_id": ["P1"] * 4,
            "trial_id": ["T1"] * 4,
            "start_time": [0.4, 1.0, 1.2, 2.2],
            "aoi_id": ["body", "target", "target", "footer"],
            "episode_type": ["fixation"] * 4,
        }
    )
    with pytest.warns(RuntimeWarning):
        transition = prepare_gaze_survival_data(
            trials,
            events,
            target_aoi="target",
            event_type="first_transition_into_target",
        )
    with pytest.warns(RuntimeWarning):
        disengage = prepare_gaze_survival_data(
            trials,
            events,
            target_aoi="target",
            event_type="disengagement",
        )
    assert transition.loc[0, "event_time"] == pytest.approx(1.0)
    assert disengage.loc[0, "event_time"] == pytest.approx(2.2)


def test_all_event_and_high_censoring_edge_cases_are_retained():
    all_event = tiny_survival_fixture()
    all_event["event_observed"] = 1
    all_event["event_time"] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    all_event["analysis_time"] = all_event["event_time"]
    assert summarise_gaze_censoring(all_event).loc[0, "n_censored"] == 0

    high_censor = simulate_gaze_survival_example(n_participants=10, trials_per_participant=2)
    high_censor.loc[:, "event_observed"] = 0
    high_censor.loc[:, "event_time"] = np.nan
    high_censor.loc[:, "analysis_time"] = high_censor["censor_time"]
    issues = validate_gaze_survival_data(high_censor, raise_on_error=False)
    assert "extreme_censoring" in set(issues.code)


def test_missing_observation_window_is_not_analyzable():
    d = tiny_survival_fixture()
    d.loc[0, "censor_time"] = np.nan
    issues = validate_gaze_survival_data(d, raise_on_error=False)
    assert "missing_observation_window" in set(issues.code)


def test_simulation_is_deterministic():
    a = simulate_gaze_survival_example(seed=123, n_participants=8, trials_per_participant=2)
    b = simulate_gaze_survival_example(seed=123, n_participants=8, trials_per_participant=2)
    pd.testing.assert_frame_equal(a, b)


def test_cox_matches_direct_statsmodels_backend_call():
    from patsy import dmatrix
    from statsmodels.duration.hazard_regression import PHReg

    d = simulate_gaze_survival_example(seed=11, n_participants=50, trials_per_participant=3)
    ours = fit_gaze_cox_model(d, "C(condition)")
    x = dmatrix("C(condition)", d, return_type="dataframe").drop(columns="Intercept")
    direct = PHReg(
        d["analysis_time"].to_numpy(float),
        x,
        status=d["event_observed"].to_numpy(int),
        ties="breslow",
    ).fit()
    np.testing.assert_allclose(np.asarray(ours.result.params), np.asarray(direct.params), rtol=1e-10, atol=1e-10)


def test_sensitivity_grid_is_explicit_and_preserves_branch_metadata():
    d = simulate_gaze_survival_example(seed=7, n_participants=30, trials_per_participant=3)
    alt = d.copy()
    alt["aoi_specification"] = "expanded synthetic AOI"
    from eyeprocesspy.survival import compare_gaze_survival_specifications

    out = compare_gaze_survival_specifications(
        {"primary": d, "expanded_aoi": alt},
        "C(condition)",
        model_families=["cox_cluster_robust", "aft_weibull"],
    )
    assert set(out["specification"]) == {"primary", "expanded_aoi"}
    assert set(out["model_family"]) == {"cox_repeated", "aft_weibull"}
    assert "aoi_specification" in out


def test_cross_language_contract_fixture():
    from importlib.resources import files

    fixture = files("eyeprocesspy").joinpath("resources/extdata/gaze_survival_contract.csv")
    with fixture.open("r", encoding="utf-8") as handle:
        d = pd.read_csv(handle)
    assert list(d.columns) == CANONICAL_GAZE_SURVIVAL_COLUMNS
    assert int(d["event_observed"].sum()) == 4
    assert int((d["event_observed"] == 0).sum()) == 2
    issues = validate_gaze_survival_data(d, raise_on_error=False)
    assert not (issues["severity"] == "error").any()


def test_plotting_helpers_return_matplotlib_axes():
    import matplotlib

    matplotlib.use("Agg")
    from eyeprocesspy.survival import (
        plot_gaze_cox_diagnostics,
        plot_gaze_cumulative_incidence,
        plot_gaze_hazard,
        plot_gaze_survival_curve,
    )

    d = simulate_gaze_survival_example(seed=42, n_participants=24, trials_per_participant=3)
    cox = fit_gaze_cox_model(d, "C(condition)")
    for ax in (
        plot_gaze_survival_curve(d, group="condition"),
        plot_gaze_cumulative_incidence(d, group="condition"),
        plot_gaze_hazard(d, group="condition"),
        plot_gaze_cox_diagnostics(cox),
    ):
        assert hasattr(ax, "figure")


def test_worked_example_runs_end_to_end(tmp_path):
    import importlib.util
    from pathlib import Path

    example = Path(__file__).parents[1] / "examples" / "worked_gaze_survival_analysis.py"
    spec = importlib.util.spec_from_file_location("worked_gaze_survival_analysis", example)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with pytest.warns(RuntimeWarning, match="not directly comparable"):
        result = module.run_worked_example(tmp_path)
    for filename in (
        "censoring-summary.csv",
        "cox-effects.csv",
        "weibull-aft-effects.csv",
        "cox-ph-diagnostics.csv",
        "model-comparison.csv",
        "sensitivity-specifications.csv",
        "cox-report.json",
        "km-disclosure.svg",
        "cumulative-incidence-disclosure.svg",
        "hazard-disclosure.svg",
    ):
        assert (tmp_path / filename).exists()
    assert result["report"]["N_participants"] == 36