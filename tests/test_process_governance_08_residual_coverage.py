from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.process_governance_08 as pg
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessValidationError
from eyeprocesspy.irt import EyeResult


def _window_data(n_samples: int = 24) -> pd.DataFrame:
    t = np.arange(n_samples, dtype=float) * 50.0
    return pd.DataFrame(
        {
            "person_id": "P1",
            "trial_id": "T1",
            "time_ms": t,
            "align_ms": 100.0,
            "pupil_bc": 3.0 + 0.2 * np.sin(t / 150.0),
            "pupil_tonic": 3.0 + 0.01 * t / 50.0,
            "pupil_phasic": 0.2 * np.sin(t / 150.0),
            "x": np.linspace(0.2, 0.8, n_samples),
            "y": np.linspace(0.8, 0.2, n_samples),
            "aoi": np.resize(["target", "text", None], n_samples),
            "valid_gaze_prop": np.resize([1.0, 0.9], n_samples),
            "valid_pupil_prop": np.resize([1.0, 0.95], n_samples),
            "blink": np.resize([False, True, False], n_samples),
            "trackloss": False,
        }
    )


def _drift_data(batch_kind: str = "numeric", include_item: bool = True) -> pd.DataFrame:
    rows = []
    batches = [1, 2, 3]
    if batch_kind == "datetime":
        batches = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
    elif batch_kind == "string":
        batches = ["b1", "b2", "b3"]
    for item_i, item in enumerate(["i1", "i2"]):
        for j, batch in enumerate(batches):
            row = {
                "deployment_batch": batch,
                "irt_difficulty": 0.1 * item_i + 0.3 * j,
                "irt_discrimination": 1.0 - 0.2 * j,
                "valid_gaze_prop": 0.95 - 0.08 * j,
                "screen_luminance": 100.0 + 20 * j,
                "dwell_ms": 700.0 + 80 * j,
                "device_id": "D1" if item_i == 0 else "D2",
                "site_id": "S1" if item_i == 0 else "S2",
                "vendor": "V1" if item_i == 0 else "V2",
                "stimulus_version": "v1" if item_i == 0 else "v2",
            }
            if include_item:
                row["item_id"] = item
            rows.append(row)
    return pd.DataFrame(rows)


def _pupil_model_data(n: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    trial = np.resize([1.0, 2.0, 3.0, 4.0], n)
    lum = np.resize([80.0, 120.0, 160.0], n)
    theta = np.linspace(-1.5, 1.5, n)
    return pd.DataFrame(
        {
            "person_id": np.resize(["P1", "P2", "P3", "P4"], n),
            "item_id": np.resize(["I1", "I2", "I3"], n),
            "pupil_peak": 2.5 + 0.003 * lum + 0.05 * trial + 0.1 * theta + rng.normal(0, 0.03, n),
            "screen_luminance": lum,
            "trial_sequence": trial,
            "theta": theta,
            "difficulty": np.resize([0.2, 0.5, 0.8], n),
        }
    )


def test_private_helpers_cover_empty_constant_group_and_ols_branches():
    with pytest.raises(EyeProcessValidationError, match="coercible"):
        pg._df(object(), "bad")
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        pg._req(pd.DataFrame({"x": [1]}), ["x", None, "y"], "frame")

    assert np.isnan(pg._mean([np.nan]))
    assert np.isnan(pg._sd([1.0]))
    np.testing.assert_array_equal(pg._z([2.0, 2.0]), np.zeros(2))
    assert np.isnan(pg._z([np.nan, np.nan])).all()

    with pytest.raises(EyeProcessValidationError, match="at least one row"):
        pg._groups(pd.DataFrame({"id": pd.Series(dtype=str)}), ["id"])
    with pytest.raises(EyeProcessValidationError, match="must not contain missing"):
        pg._groups(pd.DataFrame({"id": ["a", None]}), ["id"])
    d = pd.DataFrame({"x": [1, 2]})
    groups = pg._groups(d, [])
    assert len(groups) == 1 and groups[0].tolist() == [0, 1]
    assert pg._group_values(d, groups[0], []) == {".group": "all"}

    fit = pg._ols(np.ones(4), pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}))
    assert np.isnan(fit.r_squared)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_gaze_validity": np.nan},
        {"min_gaze_validity": 1.1},
        {"sampling_rate_tolerance": -0.1},
        {"blink_quantile": 0.0},
        {"min_rt_ms": 500, "max_rt_ms": 500},
        {"caution_flags": 0},
        {"caution_flags": 2, "review_flags": 2},
    ],
)
def test_preflight_spec_rejects_invalid_threshold_contracts(kwargs):
    with pytest.raises(EyeProcessValidationError):
        ep.process_preflight_spec(**kwargs)


def test_preflight_defaults_invalid_audit_application_and_unmatched_rows():
    d = pd.DataFrame({"person_id": ["P1", "P1", "P2", "P2"]})
    a = ep.audit_biometric_preflight(d)
    assert np.isnan(a.blink_cutoff)
    assert np.isnan(a.target_sampling_rate_hz)
    assert set(a.table.preflight_decision) == {"pass_preflight"}

    for fun in [ep.preflight_decisions, ep.preflight_failures, ep.preflight_passed, ep.preflight_exclusion_manifest]:
        with pytest.raises(EyeProcessValidationError):
            fun({})
    with pytest.raises(EyeProcessValidationError, match="spec"):
        ep.audit_biometric_preflight(d, spec={})
    with pytest.raises(EyeProcessValidationError, match="audit"):
        ep.apply_preflight_decision(d, {})

    bad_ids = d.copy()
    bad_ids.loc[0, "person_id"] = None
    with pytest.raises(EyeProcessValidationError, match="must not contain missing"):
        ep.apply_preflight_decision(bad_ids, a)

    extra = pd.DataFrame({"person_id": ["P3"]})
    with pytest.raises(EyeProcessValidationError, match="could not be matched"):
        ep.apply_preflight_decision(pd.concat([d, extra], ignore_index=True), a)

    kept = ep.apply_preflight_decision(d, a, keep_decisions=["pass_preflight"])
    assert kept.attrs["preflight_application"]["n_output"] == len(d)


def test_anomaly_and_presentation_validation_paths():
    d = pd.DataFrame(
        {
            "person_id": [f"P{i}" for i in range(12)],
            "m1": np.arange(12, dtype=float),
            "m2": np.arange(12, dtype=float) ** 2,
            "m3": np.linspace(0, 1, 12),
        }
    )
    with pytest.raises(EyeProcessValidationError, match="alpha"):
        ep.audit_process_anomalies(d, metrics=["m1", "m2"], alpha=1.0)
    with pytest.raises(EyeProcessValidationError, match="At least two"):
        ep.audit_process_anomalies(d, metrics=["m1"])
    with pytest.raises(EyeProcessValidationError):
        ep.process_anomaly_distance({})

    a = ep.audit_multivariate_process_quality(d, metrics=["m1", "m2", "m3"], aggregate=True)
    assert len(ep.process_anomaly_distance(a)) == len(a.table)

    one = pd.DataFrame({"person_id": ["P1"], "m1": [1.0], "m2": [2.0]})
    with pytest.raises(EyeProcessValidationError, match="covariance"):
        ep.audit_process_anomalies(one, metrics=["m1", "m2"], aggregate=False)

    with pytest.raises(EyeProcessValidationError, match="review_quantile"):
        ep.audit_presentation_accessibility(d, rt="m1", dwell="m2", revisits="m3", review_quantile=1.0)
    with pytest.raises(EyeProcessValidationError, match="At least three"):
        ep.audit_presentation_accessibility(d[["person_id", "m1"]], rt="m1")

    zero = pd.DataFrame(
        {
            "person_id": ["P1", "P2"],
            "rt_ms": [np.nan, np.nan],
            "dwell_ms": [np.nan, np.nan],
            "revisits": [np.nan, np.nan],
        }
    )
    with pytest.raises(EyeProcessValidationError, match="No finite"):
        ep.audit_presentation_accessibility(zero)

    with pytest.raises(EyeProcessValidationError):
        ep.simulate_presentation_variants({})

    fairness = pd.DataFrame({"variant": ["A", "A"], "outcome": [1.0, 2.0]})
    with pytest.raises(EyeProcessValidationError, match="two presentation"):
        ep.compare_presentation_fairness(fairness, "variant", "outcome")

    fairness = pd.DataFrame(
        {
            "person_id": np.repeat(["P1", "P2", "P3"], 2),
            "variant": np.tile(["A", "B"], 3),
            "outcome": [1.0, 1.2, 2.0, 2.1, 3.0, 3.2],
        }
    )
    out = ep.compare_presentation_fairness(fairness, "variant", "outcome", person="person_id")
    assert len(out.summary) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline": "bad"},
        {"difficulty_limit": -1},
        {"relative_metric_quantile": 1},
        {"min_batches": 1},
    ],
)
def test_drift_spec_rejects_invalid_contracts(kwargs):
    with pytest.raises(EyeProcessValidationError):
        ep.process_drift_spec(**kwargs)


def test_drift_reference_batch_ordering_comparison_and_grouping_paths():
    d = _drift_data("string")
    with pytest.raises(EyeProcessValidationError, match="spec"):
        ep.audit_process_drift(d, spec={})
    with pytest.raises(EyeProcessValidationError, match="No usable"):
        ep.audit_process_drift(d[["item_id", "deployment_batch"]], metrics=["missing"])

    spec = ep.process_drift_spec(baseline="reference_batch")
    with pytest.raises(EyeProcessValidationError, match="reference_batch"):
        ep.audit_process_drift(d, spec=spec, metrics=["irt_difficulty"])
    ref = ep.audit_process_drift(d, spec=spec, reference_batch="b1", metrics=["irt_difficulty", "dwell_ms"])
    assert len(ref.table) == 2

    dt = ep.audit_process_drift(_drift_data("datetime"), metrics=["irt_difficulty"])
    assert np.isfinite(dt.trajectories.batch_order).all()

    with pytest.raises(EyeProcessValidationError, match="No usable"):
        ep.compare_deployment_batches(d[["deployment_batch"]], batch_a="b1", batch_b="b2")
    with pytest.raises(EyeProcessValidationError, match="Both comparison batches"):
        ep.compare_deployment_batches(d, batch_a="missing", batch_b="b2", metrics=["dwell_ms"])

    no_item = _drift_data("numeric", include_item=False)
    cmp = ep.compare_deployment_batches(no_item, batch_a=1, batch_b=2, metrics=["dwell_ms"])
    assert cmp.metric.tolist() == ["dwell_ms"]

    all_missing = d.copy()
    all_missing["device_id"] = np.nan
    with pytest.raises(EyeProcessValidationError, match="No non-missing"):
        ep.drift_by_device(all_missing, metrics=["irt_difficulty"])

    for fun in [ep.drift_by_device, ep.drift_by_site, ep.drift_by_vendor, ep.drift_by_stimulus_version]:
        grouped = fun(d, metrics=["irt_difficulty"])
        assert not grouped.empty
    with pytest.raises(EyeProcessValidationError):
        ep.process_drift_alerts({})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"align": "bad"},
        {"width_ms": np.nan},
        {"width_ms": 0},
        {"width_ms": 4000, "start_ms": 0, "end_ms": 3000},
        {"min_samples": 1},
    ],
)
def test_window_spec_rejects_invalid_contracts(kwargs):
    with pytest.raises(EyeProcessValidationError):
        ep.process_window_spec(**kwargs)


def test_window_helpers_extraction_binding_validation_and_sensitivity_paths():
    assert np.isnan(pg._auc([1], [0]))
    assert np.isnan(pg._slope([1, 2], [0, 1]))
    assert np.isnan(pg._slope([1, 2, 3], [1, 1, 1]))
    assert np.isnan(pg._rmssd([1]))
    assert pg._entropy_numeric([1, 1, 1]) == 0.0
    assert np.isnan(pg._entropy_factor([None, ""]))
    assert pg._switch_count(["a"]) == 0
    assert np.isnan(pg._path_length([1], [1]))
    assert np.isnan(pg._velocity_mean([1], [1], [0]))
    assert np.isnan(pg._velocity_mean([1, 2], [1, 2], [1, 1]))

    d = _window_data()
    with pytest.raises(EyeProcessValidationError, match="spec"):
        ep.extract_process_windows(d, spec={})
    with pytest.raises(EyeProcessValidationError, match="align_ms_missing"):
        ep.extract_process_windows(d, align_time="align_ms_missing")

    minimal = d[["person_id", "trial_id", "time_ms", "pupil_bc"]].copy()
    x = ep.extract_process_windows(
        minimal,
        spec=ep.process_window_spec(500, 500, 0, 1000, min_samples=2),
    )
    assert not x.data.empty
    assert np.isnan(x.data.aoi_entropy).all()
    assert np.isnan(x.data.gaze_path_length).all()

    aligned = ep.extract_process_windows(
        d,
        align_time="align_ms",
        spec=ep.process_window_spec(400, 200, -100, 700, align="custom", min_samples=2),
    )
    assert not aligned.data.empty

    with pytest.raises(EyeProcessValidationError):
        ep.summarize_process_windows({})
    empty = EyeResult({"data": pd.DataFrame()}, eyeprocess_class="eye_process_windows")
    assert ep.summarize_process_windows(empty).empty
    grouped = ep.summarize_process_windows(aligned, by=["person_id"])
    assert len(grouped) == 1

    with pytest.raises(EyeProcessValidationError):
        ep.bind_process_windows()
    bound = ep.bind_process_windows([x, aligned])
    assert len(bound.data) == len(x.data) + len(aligned.data)

    with pytest.raises(EyeProcessValidationError):
        ep.validate_process_windows({})
    malformed = EyeResult({"data": pd.DataFrame({"window_start": [0.0]})}, eyeprocess_class="eye_process_windows")
    val = ep.validate_process_windows(malformed)
    assert bool(val.valid.iloc[0]) is False and "missing columns" in val.issue.iloc[0]
    bad = EyeResult(
        {
            "data": pd.DataFrame(
                {"window_start": [1.0], "window_end": [1.0], "window_mid": [1.0], "n_samples_window": [0]}
            )
        },
        eyeprocess_class="eye_process_windows",
    )
    issue = ep.validate_process_windows(bad).issue.iloc[0]
    assert "non-positive" in issue and "empty windows" in issue

    sens = ep.audit_process_window_sensitivity(
        d,
        widths_ms=[300, 500],
        steps_ms=[100],
        grid=False,
        metric="missing_metric",
        spec=ep.process_window_spec(500, 250, 0, 1000),
    )
    assert sens.table.mean_value.isna().all()


def test_aoi_trajectory_and_growth_curve_error_and_prediction_paths():
    d = _window_data()
    with pytest.raises(EyeProcessValidationError, match="at least one row"):
        ep.aoi_trajectory_features(d.iloc[0:0])
    with pytest.raises(EyeProcessValidationError, match="degree"):
        ep.aoi_trajectory_features(d, degree=0)
    with pytest.raises(EyeProcessValidationError, match="bin_ms"):
        ep.aoi_trajectory_features(d, bin_ms=0)
    none = d.copy()
    none["aoi"] = None
    with pytest.raises(EyeProcessValidationError, match="No AOI"):
        ep.aoi_trajectory_features(none)

    short = ep.aoi_trajectory_features(d.iloc[:4], degree=3, aois=["target"])
    assert short.features.filter(like="gca_degree").isna().all().all()

    with pytest.raises(EyeProcessValidationError, match="degree"):
        ep.fit_aoi_growth_curve(pd.DataFrame({"t": [1, 2, 3], "y": [1, 2, 3]}), "t", "y", degree=7)
    with pytest.raises(EyeProcessValidationError, match="Insufficient"):
        ep.fit_aoi_growth_curve(pd.DataFrame({"t": [1, 2], "y": [1, 2]}), "t", "y", degree=1)

    gd = pd.DataFrame({"t": np.arange(8.0), "y": np.arange(8.0) ** 2})
    fit = ep.fit_aoi_growth_curve(gd, "t", "y", degree=2)
    assert len(ep.predict_aoi_trajectory(fit)) == 101
    assert len(ep.predict_aoi_trajectory(fit, time=[0, 1])) == 2
    with pytest.raises(EyeProcessValidationError):
        ep.predict_aoi_trajectory({})
    with pytest.raises(EyeProcessValidationError, match="Supply at least one"):
        ep.compare_aoi_trajectories()
    with pytest.raises(EyeProcessValidationError, match="All inputs"):
        ep.compare_aoi_trajectories(fit)


def test_pupil_signal_representation_edge_paths():
    assert np.isnan(ep.pupil_band_power([1, np.nan, 2], 60, 0.1, 1.0))
    y = np.ones(12)
    assert ep.pupil_band_power(y, 60, 0.1, 1.0) == 0.0
    with pytest.raises(EyeProcessValidationError, match="sampling_rate"):
        ep.pupil_band_power(np.arange(12.0), 0, 0.1, 1.0)
    with pytest.raises(EyeProcessValidationError, match="lower_hz"):
        ep.pupil_band_power(np.arange(12.0), 60, 1.0, 0.5)
    assert np.isnan(ep.pupil_band_power(np.arange(12.0), 60, 40, 50, detrend=False))

    assert np.isnan(ep.pupil_velocity_activity([1, 2, 3], [0, 1, 2]))
    assert np.isnan(ep.pupil_velocity_activity([1, 2, 3, 4], [3, 2, 1, 0]))
    assert np.isnan(pg._deriv_power([1, 2, 3], [0, 1, 2], 200))

    assert np.isfinite(ep.pupil_activity_index(np.arange(10.0), method="velocity"))
    with pytest.raises(EyeProcessValidationError, match="sampling_rate_hz"):
        ep.pupil_activity_index(np.arange(10.0), method="frequency_contrast")
    with pytest.raises(EyeProcessValidationError, match="method"):
        ep.pupil_activity_index(np.arange(10.0), method="bad")
    assert np.isnan(ep.pupil_activity_index([1, 2, 3], [0, 1, 2], method="ripa_proxy"))

    d = _window_data(30)
    d["sr"] = 60.0
    f = ep.pupil_frequency_features(d, sampling_rate_hz="sr")
    assert f.features.sampling_rate_hz.iloc[0] == 60.0

    no_time = d.copy()
    no_time["time_ms"] = np.nan
    stab = ep.audit_pupil_frequency_stability(no_time, windows_ms=[500])
    assert stab.table.empty
    short = ep.audit_pupil_frequency_stability(d.iloc[:6], windows_ms=[100])
    assert short.table.empty

    with pytest.raises(EyeProcessValidationError, match="positive"):
        ep.pupil_response_kernel([0, 1], tmax_ms=0)
    with pytest.raises(EyeProcessValidationError, match="positive"):
        ep.pupil_response_kernel([0, 1], shape=0)
    k = ep.pupil_response_kernel([-10, 0, 100], normalize=False)
    assert k[0] == 0


def test_pupil_deconvolution_event_and_empty_fit_paths():
    d = _window_data(30)
    with pytest.raises(EyeProcessValidationError, match="named mapping"):
        ep.fit_pupil_event_deconvolution(d, events={})
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        ep.fit_pupil_event_deconvolution(d, events={"stim": "missing_event"})
    with pytest.raises(EyeProcessValidationError, match="scalar numeric"):
        ep.fit_pupil_event_deconvolution(d, events={"stim": [1, 2]})

    invalid_event = d.copy()
    invalid_event["event_ms"] = np.nan
    empty = ep.fit_pupil_event_deconvolution(invalid_event, events={"stim": "event_ms"}, min_samples=4)
    assert empty.effects.empty

    constant = d.copy()
    constant["pupil_bc"] = 3.0
    empty2 = ep.fit_pupil_event_deconvolution(constant, events={"stim": 0}, min_samples=4)
    assert empty2.effects.empty

    with pytest.raises(EyeProcessValidationError):
        ep.pupil_event_effects({})
    kernels = ep.compare_pupil_kernels(constant, tmax_values=[500, 930], events={"stim": 0}, min_samples=4)
    assert kernels.n_groups.tolist() == [0, 0]


def test_pupil_confound_fatigue_and_filter_residual_paths():
    d = _pupil_model_data()
    with pytest.raises(EyeProcessValidationError, match="engine"):
        ep.fit_pupil_confound_model(d, engine="bad")
    with pytest.raises(EyeProcessValidationError, match="30 complete"):
        ep.fit_pupil_confound_model(d.iloc[:20], engine="lm")
    one_lum = d.copy()
    one_lum["screen_luminance"] = 100.0
    with pytest.raises(EyeProcessValidationError, match="at least two unique"):
        ep.fit_pupil_confound_model(one_lum, engine="lm")

    fit = ep.fit_pupil_confound_model(d, theta="theta", engine="auto")
    assert fit.engine == "lm"
    assert {"theta", "luminance_theta", "luminance2", "trial2"}.issubset(fit.model.design_columns)
    assert not ep.compare_raw_adjusted_pupil(fit).empty
    assert not ep.adjust_pupil_confounds(fit).empty
    assert not ep.pupil_confound_effects(fit).empty
    for fun in [ep.adjust_pupil_confounds, ep.pupil_confound_effects, ep.compare_raw_adjusted_pupil]:
        with pytest.raises(EyeProcessValidationError):
            fun({})

    with pytest.raises(EyeProcessValidationError, match="engine"):
        ep.audit_pupil_fatigue_drift(d, engine="bad")
    with pytest.raises(EyeProcessValidationError, match="20 complete"):
        ep.audit_pupil_fatigue_drift(d.iloc[:10])
    with pytest.raises(EyeProcessBackendError, match="plm"):
        ep.audit_pupil_fatigue_drift(d, luminance="screen_luminance", difficulty="difficulty", engine="plm")
    fatigue = ep.audit_pupil_fatigue_drift(
        d, luminance="screen_luminance", difficulty="difficulty", engine="auto"
    )
    assert fatigue.engine == "lm_fixed_effects"

    with pytest.raises(EyeProcessValidationError, match="method"):
        ep.filter_eye_signal(np.arange(10.0), method="bad")
    with pytest.raises(EyeProcessValidationError, match="Too few finite"):
        ep.filter_eye_signal([1, np.nan, 2, np.nan, 3])
    auto = ep.filter_eye_signal(np.arange(10.0), width=4, method="auto")
    assert auto.method == "runmed" and auto.width % 2 == 1
    with pytest.raises(EyeProcessValidationError):
        ep.audit_signal_filter({})
    assert len(ep.audit_signal_filter(auto)) == 1
    assert len(ep.compare_signal_filters(np.arange(10.0), widths=[3, 5], methods=["runmed"])) == 2
    with pytest.raises(EyeProcessValidationError, match="No requested"):
        ep.compare_signal_filters(np.arange(10.0), methods=["robfilter"])
