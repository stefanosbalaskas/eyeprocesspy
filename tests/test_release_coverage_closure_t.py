from __future__ import annotations

import sys
import types
import warnings
from dataclasses import replace
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.detector_multiverse as dm
import eyeprocesspy.survival as sv


def _ivt(
    detector_id: str = "ivt30",
    threshold: float = 30.0,
    *,
    sampling_rate: float = 60.0,
    parameters=None,
):
    return ep.define_event_detector_spec(
        detector_id,
        "ivt",
        velocity_threshold=threshold,
        minimum_duration_ms=60,
        maximum_gap_ms=75,
        sampling_rate=sampling_rate,
        coordinate_unit="degrees",
        parameters=parameters,
    )


def _adaptive(
    detector_id: str = "adaptive",
    *,
    minimum_duration_ms: float = 60.0,
):
    return ep.define_event_detector_spec(
        detector_id,
        "adaptive_velocity",
        minimum_duration_ms=minimum_duration_ms,
        maximum_gap_ms=75,
        sampling_rate=60,
        coordinate_unit="degrees",
        parameters={
            "noise_factor": 4.0,
            "minimum_velocity_threshold": 20.0,
        },
    )


def _detector_data(seed: int = 1001):
    return ep.simulate_detector_multiverse_data(
        n_participants=4,
        seed=seed,
    )


def _detector_result(seed: int = 1002):
    data = _detector_data(seed)

    result = ep.run_detector_multiverse(
        data,
        [
            _ivt("ivt_a", 25),
            _ivt("ivt_b", 45),
        ],
        continue_on_error=False,
    )

    result = ep.propagate_detector_to_aoi(
        result,
        overlap="first",
        continue_on_error=False,
    )

    result = ep.propagate_detector_to_features(
        result,
        continue_on_error=False,
    )

    return data, result


def _callback_inference(result):
    def callback(data, spec):
        return pd.DataFrame(
            [
                {
                    "term": "condition",
                    "estimate": 0.25,
                    "SE": 0.10,
                    "CI_lower": 0.05,
                    "CI_upper": 0.45,
                    "p": 0.04,
                    "converged": True,
                    "N": len(data),
                }
            ]
        )

    return ep.run_detector_inference_multiverse(
        result,
        {
            "engine": "callback",
            "outcome": "dwell_time_ms",
        },
        model_callback=callback,
    )


def _survival_data(seed: int = 1101):
    return sv.simulate_gaze_survival_example(
        seed=seed,
        n_participants=12,
        trials_per_participant=2,
    )


# -----------------------------------------------------------------
# Detector specification / fingerprints
# -----------------------------------------------------------------


def test_detector_validation_residual_object_callback_and_vendor_paths():
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="EventDetectorSpec",
    ):
        dm.validate_event_detector_spec(object())

    external = replace(
        _ivt("external_seed"),
        algorithm="external",
        velocity_threshold=None,
        sampling_rate=None,
        minimum_duration_ms=None,
        callback=None,
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="callable",
    ):
        dm.validate_event_detector_spec(external)

    vendor = replace(
        external,
        detector_id="vendor_seed",
        algorithm="vendor",
        callback=lambda **kwargs: None,
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="do not use",
    ):
        dm.validate_event_detector_spec(vendor)


def test_detector_dataset_fingerprint_empty_tables():
    data = ep.new_eye_dataset(validate=False)

    value = dm._dataset_fingerprint(data)

    assert isinstance(value, str)
    assert len(value) == 64
    assert value == dm._dataset_fingerprint(data)


# -----------------------------------------------------------------
# Adaptive / REMoDNaV / IVT branches
# -----------------------------------------------------------------


def test_adaptive_velocity_sparse_and_too_short_paths():
    data = _detector_data(1003)

    sparse = data.copy()
    gaze = sparse["gaze_samples"].copy()
    gaze["valid"] = False
    sparse["gaze_samples"] = gaze

    out = dm._run_adaptive_velocity(
        sparse,
        _adaptive("sparse"),
    )

    assert out["episodes"].empty

    short = dm._run_adaptive_velocity(
        data,
        _adaptive(
            "short",
            minimum_duration_ms=1_000_000,
        ),
    )

    assert short["episodes"].empty


def test_remodnav_parameter_routes_and_short_group(monkeypatch):
    calls = {}

    class FakeClassifier:
        def __init__(
            self,
            px2deg,
            sampling_rate,
            min_fixation_duration,
            constructor_option=None,
        ):
            calls["constructor_option"] = constructor_option

        def preproc(
            self,
            data,
            preprocessing_option=None,
        ):
            calls["preprocessing_option"] = preprocessing_option
            return data

        def __call__(
            self,
            data,
            classify_isp=True,
            sort_events=True,
        ):
            return []

    fake = types.ModuleType("remodnav")
    fake.EyegazeClassifier = FakeClassifier

    monkeypatch.setitem(
        sys.modules,
        "remodnav",
        fake,
    )

    data = _detector_data(1004)

    spec = ep.define_event_detector_spec(
        "fake_remodnav_options",
        "remodnav",
        minimum_duration_ms=60,
        sampling_rate=60,
        coordinate_unit="degrees",
        parameters={
            "constructor_option": 7,
            "preprocessing_option": 9,
        },
    )

    out = dm._run_remodnav(
        data,
        spec,
    )

    assert out["episodes"].empty

    assert calls == {
        "constructor_option": 7,
        "preprocessing_option": 9,
    }

    tiny = data.copy()
    tiny["gaze_samples"] = tiny["gaze_samples"].iloc[:2].copy()

    out = dm._run_remodnav(
        tiny,
        spec,
    )

    assert out["episodes"].empty


def test_ivt_saccade_route_and_sampling_warning_capture():
    data = _detector_data(1005)

    spec = _ivt(
        "ivt_with_saccades",
        30,
        sampling_rate=120,
        parameters={
            "include_saccades": True,
            "minimum_saccade_duration_ms": 5,
        },
    )

    result = ep.run_detector_multiverse(
        data,
        [spec],
        continue_on_error=False,
    )

    assert result.status.iloc[0]["status"] == "ok"
    assert not result.warnings.empty

    assert (
        result.warnings["warning"]
        .astype(str)
        .str.contains(
            "sampling rate",
            case=False,
        )
        .any()
    )


# -----------------------------------------------------------------
# Agreement / AOI / pupil / feature branches
# -----------------------------------------------------------------


def test_detector_agreement_and_disagreement_nonempty_paths():
    _, result = _detector_result(1006)

    agreement = dm.estimate_detector_agreement(result)

    assert not agreement.empty

    disagreement = dm.summarise_detector_disagreement(result)

    assert not disagreement.empty

    assert {
        "unmatched_reference",
        "unmatched_candidate",
        "event_count_difference",
    }.issubset(disagreement.columns)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="Detector-labelled",
    ):
        dm.estimate_detector_agreement(pd.DataFrame({"episode_type": ["fixation"]}))


def test_event_iou_zero_union_paths():
    assert (
        dm._event_iou(
            1.0,
            1.0,
            1.0,
            1.0,
        )
        == 1.0
    )

    assert (
        dm._event_iou(
            1.0,
            1.0,
            2.0,
            2.0,
        )
        == 0.0
    )


def test_detector_aoi_nonfixation_and_no_hit_paths():
    data = _detector_data(1007)

    detected = dm.detect_events_with_spec(
        data,
        _ivt("aoi_route"),
    )

    assert not detected["episodes"].empty

    preserved = detected.copy()
    episodes = preserved["episodes"].copy()

    episodes.loc[
        episodes.index[0],
        "episode_type",
    ] = "saccade"

    episodes.loc[
        episodes.index[0],
        "aoi_id",
    ] = "existing"

    preserved["episodes"] = episodes

    out = dm._assign_episode_aois_explicit(
        preserved,
        "first",
    )

    assert out["episodes"].iloc[0]["aoi_id"] == "existing"

    no_hit = detected.copy()
    episodes = no_hit["episodes"].copy()

    fixation_index = episodes.index[episodes["episode_type"].eq("fixation")][0]

    episodes.loc[
        fixation_index,
        "centroid_x",
    ] = 1e12

    episodes.loc[
        fixation_index,
        "centroid_y",
    ] = 1e12

    no_hit["episodes"] = episodes

    out = dm._assign_episode_aois_explicit(
        no_hit,
        "first",
    )

    assert pd.isna(
        out["episodes"].loc[
            fixation_index,
            "aoi_id",
        ]
    )


def test_pupil_within_fixations_positive_and_missing_paths():
    data = _detector_data(1008)

    branch = dm.detect_events_with_spec(
        data,
        _ivt("pupil_route"),
    )

    first = branch["gaze_samples"].iloc[0]

    recording_id = first["recording_id"]
    trial_id = first["trial_id"]

    template = branch["eye_samples"]

    rows = []

    for timestamp, pupil, valid in (
        (0.10, 3.0, True),
        (0.20, 5.0, True),
        (0.30, 100.0, False),
        (1.00, 9.0, True),
    ):
        row = {column: pd.NA for column in template.columns}

        row.update(
            {
                "recording_id": recording_id,
                "trial_id": trial_id,
                "timestamp_seconds": timestamp,
                "pupil_diameter": pupil,
                "pupil_valid": valid,
            }
        )

        rows.append(row)

    branch["eye_samples"] = pd.DataFrame(
        rows,
        columns=template.columns,
    )

    fixations = pd.DataFrame(
        {
            "start_time": [
                0.05,
                0.18,
            ],
            "end_time": [
                0.15,
                0.25,
            ],
        }
    )

    value = dm._pupil_within_fixations(
        branch,
        fixations,
        recording_id,
        trial_id,
    )

    assert value == pytest.approx(4.0)

    missing = dm._pupil_within_fixations(
        branch,
        fixations,
        "missing_recording",
        trial_id,
    )

    assert np.isnan(missing)


def test_detector_feature_sensitivity_nonempty_paths():
    features = pd.DataFrame(
        {
            "recording_id": [
                "r1",
                "r1",
                "r2",
            ],
            "trial_id": [
                "t1",
                "t1",
                "t2",
            ],
            "aoi_id": [
                "a",
                "a",
                "a",
            ],
            "detector_id": [
                "d1",
                "d2",
                "d1",
            ],
            "dwell_time_ms": [
                100.0,
                140.0,
                50.0,
            ],
            "fixation_count": [
                1,
                3,
                1,
            ],
        }
    )

    out = dm._feature_sensitivity(features)

    assert set(out["feature"]) == {
        "dwell_time_ms",
        "fixation_count",
    }

    dwell = out[out["feature"].eq("dwell_time_ms")].iloc[0]

    assert dwell["units_with_multiple_detectors"] == 1
    assert dwell["max_detector_range"] == pytest.approx(40)


def test_feature_propagation_reraises_explicit_failure():
    data = _detector_data(1009)

    result = ep.run_detector_multiverse(
        data,
        [_ivt("broken_features")],
        continue_on_error=False,
    )

    branch = result.branches["broken_features"].copy()

    branch["aoi_definitions"] = branch["aoi_definitions"].iloc[0:0].copy()

    broken = replace(
        result,
        branches={"broken_features": branch},
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="AOI definitions",
    ):
        ep.propagate_detector_to_features(
            broken,
            continue_on_error=False,
        )


# -----------------------------------------------------------------
# Inference residuals
# -----------------------------------------------------------------


def test_mixedlm_helper_retains_nonconvergence(monkeypatch):
    import statsmodels.formula.api as smf

    calls = {}

    class FakeModel:
        def fit(
            self,
            *,
            reml,
            method,
            maxiter,
            disp,
        ):
            calls["fit"] = {
                "reml": reml,
                "method": method,
                "maxiter": maxiter,
                "disp": disp,
            }

            return SimpleNamespace(converged=False)

    def fake_mixedlm(
        formula,
        *,
        data,
        groups,
        re_formula,
        missing,
    ):
        calls["formula"] = formula
        calls["re_formula"] = re_formula
        calls["groups"] = list(groups)
        calls["missing"] = missing

        return FakeModel()

    monkeypatch.setattr(
        smf,
        "mixedlm",
        fake_mixedlm,
    )

    data = pd.DataFrame(
        {
            "y": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "x": [
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            "participant": [
                "p1",
                "p1",
                "p2",
                "p2",
            ],
        }
    )

    fit, messages = dm._fit_statsmodels(
        data,
        {
            "engine": "statsmodels_mixedlm",
            "formula": "y ~ x",
            "groups": "participant",
            "re_formula": "~x",
            "reml": True,
            "method": "powell",
            "maxiter": 7,
        },
    )

    assert fit.converged is False

    assert any("did not converge" in message for message in messages)

    assert calls["re_formula"] == "~x"
    assert calls["fit"]["maxiter"] == 7


def test_detector_inference_propagates_model_warning(monkeypatch):
    _, result = _detector_result(1010)

    class Fit:
        params = pd.Series(
            [0.2],
            index=["fixation_count"],
        )

        bse = pd.Series(
            [0.1],
            index=["fixation_count"],
        )

        pvalues = pd.Series(
            [0.04],
            index=["fixation_count"],
        )

        def conf_int(self):
            return pd.DataFrame(
                [
                    [
                        0.01,
                        0.39,
                    ]
                ],
                index=["fixation_count"],
            )

    def fake_fit(
        data,
        model_spec,
    ):
        return (
            Fit(),
            ["planned model warning"],
        )

    monkeypatch.setattr(
        dm,
        "_fit_statsmodels",
        fake_fit,
    )

    inference = ep.run_detector_inference_multiverse(
        result,
        {
            "engine": "statsmodels_ols",
            "formula": "dwell_time_ms ~ fixation_count",
            "outcome": "dwell_time_ms",
        },
    )

    assert not inference.coefficients.empty
    assert not inference.warnings.empty

    assert inference.warnings["warning"].astype(str).str.contains("planned model warning").all()


# -----------------------------------------------------------------
# Plot/report residuals
# -----------------------------------------------------------------


def test_detector_plot_and_report_happy_paths(tmp_path):
    _, result = _detector_result(1011)

    inference = _callback_inference(result)

    assert not inference.coefficients.empty

    ax = dm.plot_detector_coefficient_stability(
        inference,
        term="condition",
    )

    assert ax.get_title().startswith("Coefficient stability")

    events = result.events
    assert not events.empty

    trial_id = events["trial_id"].dropna().astype(str).iloc[0]

    timeline = dm.plot_detector_event_timeline(
        result,
        trial_id=trial_id,
    )

    assert timeline.get_xlabel() == "Time (s)"

    plots = dm.plot_detector_multiverse(
        result,
        inference=inference,
        term="condition",
        feature="dwell_time_ms",
    )

    assert set(plots) == {
        "agreement",
        "feature",
        "coefficient",
    }

    path = tmp_path / "detector-report.md"

    text = dm.report_detector_multiverse(
        result,
        inference=inference,
        term="condition",
        substantive_threshold=0.1,
        path=str(path),
    )

    assert path.exists()

    assert path.read_text(encoding="utf-8") == text

    assert "Inference stability" in text

    plt.close("all")


# -----------------------------------------------------------------
# Survival preparation / event semantics
# -----------------------------------------------------------------


def test_survival_visit_event_success_paths():
    events = pd.DataFrame(
        {
            "start_time": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "aoi_id": [
                "body",
                "target",
                "target",
                "body",
                "target",
            ],
            "episode_type": ["fixation"] * 5,
        }
    )

    assert sv._event_time_for_trial(
        events,
        "target",
        "first_transition_into_target",
        "start_time",
        "aoi_id",
        "episode_type",
    ) == pytest.approx(0.2)

    assert sv._event_time_for_trial(
        events,
        "target",
        "disengagement",
        "start_time",
        "aoi_id",
        "episode_type",
    ) == pytest.approx(0.4)

    assert sv._event_time_for_trial(
        events,
        "target",
        "first_revisit",
        "start_time",
        "aoi_id",
        "episode_type",
    ) == pytest.approx(0.5)


def test_survival_named_time_origin_inference():
    trials = pd.DataFrame(
        {
            "participant_id": [
                "P1",
                "P2",
            ],
            "trial_id": [
                "T1",
                "T2",
            ],
            "start_time": [
                0.0,
                0.0,
            ],
            "end_time": [
                5.0,
                5.0,
            ],
            "stimulus_onset": [
                0.5,
                1.0,
            ],
        }
    )

    events = pd.DataFrame(
        {
            "participant_id": ["P1"],
            "trial_id": ["T1"],
            "start_time": [1.5],
            "aoi_id": ["target"],
            "episode_type": ["fixation"],
        }
    )

    with pytest.warns(RuntimeWarning):
        out = sv.prepare_gaze_survival_data(
            trials,
            events,
            target_aoi="target",
            time_origin="stimulus_onset",
        )

    first = out[out["trial_id"].eq("T1")].iloc[0]

    assert first["analysis_time"] == pytest.approx(1.0)


def test_survival_prepare_detects_event_after_censor():
    trials = pd.DataFrame(
        {
            "participant_id": ["P1"],
            "trial_id": ["T1"],
            "start_time": [0.0],
            "end_time": [1.0],
        }
    )

    events = pd.DataFrame(
        {
            "participant_id": ["P1"],
            "trial_id": ["T1"],
            "start_time": [2.0],
            "aoi_id": ["target"],
            "episode_type": ["fixation"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Invalid gaze survival data",
    ):
        sv.prepare_gaze_survival_data(
            trials,
            events,
            target_aoi="target",
        )


# -----------------------------------------------------------------
# Cox warning contracts
# -----------------------------------------------------------------


def test_cox_convergence_warning_is_failure(monkeypatch):
    import statsmodels.duration.hazard_regression as hazard

    class FakePHReg:
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            pass

        def fit(
            self,
            groups=None,
        ):
            warnings.warn(
                "planned convergence problem",
                RuntimeWarning,
            )

            return SimpleNamespace()

    monkeypatch.setattr(
        hazard,
        "PHReg",
        FakePHReg,
    )

    data = _survival_data(1103)

    with pytest.raises(
        RuntimeError,
        match="Cox convergence failure",
    ):
        sv.fit_gaze_cox_model(
            data,
            "C(condition)",
        )


def test_cox_harmless_warning_is_propagated(monkeypatch):
    import statsmodels.duration.hazard_regression as hazard

    captured = {}

    class FakePHReg:
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            captured["constructed"] = True

        def fit(
            self,
            groups=None,
        ):
            captured["groups"] = groups

            warnings.warn(
                "harmless backend notice",
                UserWarning,
            )

            return SimpleNamespace()

    monkeypatch.setattr(
        hazard,
        "PHReg",
        FakePHReg,
    )

    data = _survival_data(1104)

    with pytest.warns(
        UserWarning,
        match="harmless backend notice",
    ):
        fit = sv.fit_gaze_cox_model(
            data,
            "C(condition)",
            cluster="participant_id",
        )

    assert fit.repeated_structure == ("cluster_robust:participant_id")

    assert captured["groups"] is not None


# -----------------------------------------------------------------
# AFT backend failure contracts
# -----------------------------------------------------------------


def _aft_params(
    *,
    location: str = "lambda_",
    value: float = 0.2,
):
    return pd.Series(
        [value],
        index=pd.MultiIndex.from_tuples(
            [
                (
                    location,
                    "C(condition)[T.detail]",
                )
            ]
        ),
    )


def test_aft_convergence_exception_is_not_silenced(monkeypatch):
    import lifelines
    from lifelines.exceptions import ConvergenceError

    class BrokenFitter:
        def fit(
            self,
            *args,
            **kwargs,
        ):
            raise ConvergenceError("planned")

    monkeypatch.setattr(
        lifelines,
        "WeibullAFTFitter",
        BrokenFitter,
    )

    with pytest.raises(
        RuntimeError,
        match="AFT convergence failure",
    ):
        sv.fit_gaze_aft_model(
            _survival_data(1105),
            "C(condition)",
            distribution="weibull",
        )


def test_aft_backend_warning_is_convergence_failure(monkeypatch):
    import lifelines

    class WarningFitter:
        def fit(
            self,
            *args,
            **kwargs,
        ):
            warnings.warn(
                "singular design",
                RuntimeWarning,
            )

            return SimpleNamespace()

    monkeypatch.setattr(
        lifelines,
        "WeibullAFTFitter",
        WarningFitter,
    )

    with pytest.raises(
        RuntimeError,
        match="AFT convergence failure",
    ):
        sv.fit_gaze_aft_model(
            _survival_data(1106),
            "C(condition)",
            distribution="weibull",
        )


def test_aft_nonfinite_loglikelihood_contract(monkeypatch):
    import lifelines

    class Fitter:
        def fit(
            self,
            *args,
            **kwargs,
        ):
            return SimpleNamespace(
                log_likelihood_=np.nan,
                params_=_aft_params(),
            )

    monkeypatch.setattr(
        lifelines,
        "WeibullAFTFitter",
        Fitter,
    )

    with pytest.raises(
        RuntimeError,
        match="non-finite log-likelihood",
    ):
        sv.fit_gaze_aft_model(
            _survival_data(1107),
            "C(condition)",
            distribution="weibull",
        )


def test_aft_nonfinite_parameter_contract(monkeypatch):
    import lifelines

    class Fitter:
        def fit(
            self,
            *args,
            **kwargs,
        ):
            return SimpleNamespace(
                log_likelihood_=-10.0,
                params_=_aft_params(value=np.inf),
            )

    monkeypatch.setattr(
        lifelines,
        "WeibullAFTFitter",
        Fitter,
    )

    with pytest.raises(
        RuntimeError,
        match="non-finite parameter",
    ):
        sv.fit_gaze_aft_model(
            _survival_data(1108),
            "C(condition)",
            distribution="weibull",
        )


def test_aft_requires_multiindex_parameter_contract(monkeypatch):
    import lifelines

    class Fitter:
        def fit(
            self,
            *args,
            **kwargs,
        ):
            return SimpleNamespace(
                log_likelihood_=-10.0,
                params_=pd.Series(
                    [0.1],
                    index=["x"],
                ),
            )

    monkeypatch.setattr(
        lifelines,
        "WeibullAFTFitter",
        Fitter,
    )

    with pytest.raises(
        RuntimeError,
        match="parameter contract",
    ):
        sv.fit_gaze_aft_model(
            _survival_data(1109),
            "C(condition)",
            distribution="weibull",
        )


def test_aft_requires_location_coefficients(monkeypatch):
    import lifelines

    class Fitter:
        def fit(
            self,
            *args,
            **kwargs,
        ):
            return SimpleNamespace(
                log_likelihood_=-10.0,
                params_=_aft_params(location="rho_"),
            )

    monkeypatch.setattr(
        lifelines,
        "WeibullAFTFitter",
        Fitter,
    )

    with pytest.raises(
        RuntimeError,
        match="no location-model coefficients",
    ):
        sv.fit_gaze_aft_model(
            _survival_data(1110),
            "C(condition)",
            distribution="weibull",
        )


def test_aft_lognormal_success_contract(monkeypatch):
    import lifelines

    class Fitter:
        def fit(
            self,
            *args,
            **kwargs,
        ):
            return SimpleNamespace(
                log_likelihood_=-10.0,
                params_=_aft_params(location="mu_"),
            )

    monkeypatch.setattr(
        lifelines,
        "LogNormalAFTFitter",
        Fitter,
    )

    fit = sv.fit_gaze_aft_model(
        _survival_data(1111),
        "C(condition)",
        distribution="log-normal",
    )

    assert fit.model_family == "aft_lognormal"

    assert fit.covariate_names == ["C(condition)[T.detail]"]


def test_tidy_aft_model_happy_path():
    index = pd.MultiIndex.from_tuples(
        [
            (
                "mu_",
                "x",
            ),
            (
                "sigma_",
                "Intercept",
            ),
        ]
    )

    summary = pd.DataFrame(
        {
            "coef": [
                0.2,
                0.1,
            ],
            "se(coef)": [
                0.05,
                0.03,
            ],
            "z": [
                4.0,
                3.0,
            ],
            "p": [
                0.001,
                0.01,
            ],
        },
        index=index,
    )

    fit = sv.GazeSurvivalFit(
        model_family="aft_lognormal",
        backend="fake",
        result=SimpleNamespace(summary=summary),
        data=_survival_data(1112),
    )

    tidy = sv.tidy_gaze_survival_model(fit)

    assert tidy.loc[0, "term"] == "x"

    assert (
        tidy.loc[
            0,
            "effect_measure",
        ]
        == "time_ratio"
    )

    assert tidy.loc[
        0,
        "time_ratio",
    ] == pytest.approx(np.exp(0.2))


# -----------------------------------------------------------------
# Survival model-family dispatch
# -----------------------------------------------------------------


def test_compare_survival_specification_dispatch_paths(monkeypatch):
    data = _survival_data(1113)

    calls = []

    def fake_cox(
        data,
        formula,
        *,
        ties="breslow",
        cluster=None,
    ):
        calls.append(
            (
                "cox",
                cluster,
            )
        )

        return sv.GazeSurvivalFit(
            model_family="cox",
            backend="fake",
            result=SimpleNamespace(),
            data=data,
            formula=formula,
        )

    def fake_mixed(
        data,
        formula,
        *,
        participant_col,
        structure,
        ties,
    ):
        calls.append(
            (
                "cox_cluster_robust",
                participant_col,
            )
        )

        return sv.GazeSurvivalFit(
            model_family="cox_repeated",
            backend="fake",
            result=SimpleNamespace(),
            data=data,
            formula=formula,
        )

    def fake_aft(
        data,
        formula,
        *,
        distribution,
        maxiter=2000,
    ):
        calls.append(
            (
                "aft",
                distribution,
            )
        )

        return sv.GazeSurvivalFit(
            model_family=f"aft_{distribution}",
            backend="fake",
            result=SimpleNamespace(),
            data=data,
            formula=formula,
        )

    def fake_tidy(
        fit,
        *,
        conf_level=0.95,
    ):
        return pd.DataFrame(
            {
                "term": ["x"],
                "estimate_log_scale": [0.1],
                "std_error": [0.02],
                "effect": [1.1],
            }
        )

    monkeypatch.setattr(
        sv,
        "fit_gaze_cox_model",
        fake_cox,
    )

    monkeypatch.setattr(
        sv,
        "fit_gaze_mixed_cox_model",
        fake_mixed,
    )

    monkeypatch.setattr(
        sv,
        "fit_gaze_aft_model",
        fake_aft,
    )

    monkeypatch.setattr(
        sv,
        "tidy_gaze_survival_model",
        fake_tidy,
    )

    out = sv.compare_gaze_survival_specifications(
        {"baseline": data},
        "C(condition)",
        model_families=[
            "cox",
            "cox_cluster_robust",
            "aft_weibull",
            "aft_lognormal",
        ],
    )

    assert len(out) == 4

    assert {
        "specification",
        "model_family",
        "event_detector",
        "aoi_specification",
        "preprocessing_specification",
        "time_origin",
    }.issubset(out.columns)

    assert calls == [
        (
            "cox",
            None,
        ),
        (
            "cox_cluster_robust",
            "participant_id",
        ),
        (
            "aft",
            "weibull",
        ),
        (
            "aft",
            "lognormal",
        ),
    ]


def test_survival_unsupported_quantile_fit_contract():
    fit = sv.GazeSurvivalFit(
        model_family="unsupported",
        backend="fake",
        result=SimpleNamespace(),
        data=_survival_data(1114),
    )

    with pytest.raises(
        TypeError,
        match="Unsupported",
    ):
        sv.estimate_gaze_latency_quantiles(
            fit,
            probs=[0.5],
        )
