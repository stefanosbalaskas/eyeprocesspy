from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.aoi_perturbation as ap
import eyeprocesspy.bayesian_3pl_08 as b3
import eyeprocesspy.dataset as ds
import eyeprocesspy.detector_multiverse as dm
import eyeprocesspy.dynamic_irt as di
import eyeprocesspy.irt as irt
import eyeprocesspy.irt_validation_07 as iv
import eyeprocesspy.measurement_accountability_11 as ma
import eyeprocesspy.measurement_intelligence as mi
import eyeprocesspy.pupil_missingness as pm
import eyeprocesspy.survival as surv
import eyeprocesspy.timebase as tb
from eyeprocesspy.irt import EyeResult


def _close() -> None:
    plt.close("all")


def _recovery() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "replicate": [1, 2, 3],
            "parameter": ["b", "b", "b"],
            "truth": [-0.5, 0.0, 0.5],
            "estimate": [-0.45, 0.05, 0.48],
            "converged": [True, True, True],
        }
    )


def test_pattern_mixture_explicit_metric_regression_and_missing_metric_contract():
    data = pd.DataFrame(
        {
            "metric": [1.0, np.nan, 3.0],
            "other": [9.0, 8.0, 7.0],
        }
    )

    out = pm.process_pattern_mixture(
        data,
        delta=[-1.0, 0.0, 1.0],
        metric="metric",
    )

    assert len(out.summary) == 3
    assert out.metric == "metric"

    with pytest.raises(Exception, match="Metric `missing` is unavailable"):
        pm.process_pattern_mixture(
            data,
            metric="missing",
        )


def test_pupil_latency_high_low_and_duplicate_tail_paths():
    # High-resolution case: the three estimators agree within ~10 ms.
    time = np.arange(-0.50, 1.51, 0.01)
    response = np.where(
        time >= 0.20,
        0.30 * (time - 0.20),
        0.0,
    )
    pupil = 3.0 - response

    high = ma.pupil_latency_sensitivity(
        time,
        pupil,
        simulations=0,
        seed=20260920,
    )

    assert high["latency_resolvability"] == "high"
    assert high["estimator_spread_ms"] <= 50

    # Low-resolution case: deterministic baseline structure creates
    # large estimator/simulation disagreement.
    baseline_variation = np.sin(np.arange(len(time), dtype=float) * 1.7) * 0.10

    weak_response = np.where(
        time >= 0.20,
        0.05 * (time - 0.20),
        0.0,
    )

    low = ma.pupil_latency_sensitivity(
        time,
        3.0 + baseline_variation - weak_response,
        simulations=200,
        seed=1,
    )

    assert low["latency_resolvability"] == "low"
    assert low["simulation"]["successful"] < 200

    # Trailing duplicate timestamps make one piecewise candidate
    # have a zero denominator. It must be skipped rather than failing.
    duplicate_time = [
        -0.50,
        -0.40,
        -0.30,
        -0.20,
        -0.10,
        0.00,
        0.10,
        0.20,
        0.30,
        0.30,
        0.30,
    ]

    duplicate_pupil = [
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        2.99,
        2.98,
        2.97,
        2.97,
        2.97,
    ]

    with pytest.raises(
        ValueError,
        match="increasing samples",
    ):
        ma.pupil_latency_sensitivity(
            duplicate_time,
            duplicate_pupil,
            baseline_window=(-0.5, 0.0),
            search_window=(0.0, 0.3),
            sustain_ms=10,
            simulations=0,
        )


def test_compact_dataset_true_branches():
    data = ds.new_eye_dataset(
        raw={"source": "synthetic"},
        validate=False,
    )

    out = ds.compact_eye_dataset(
        data,
        drop_raw=True,
        drop_empty=True,
    )

    assert out.raw == []
    assert len(out.empty_components) > 0


def test_3pl_missing_rt_and_ttff_do_not_create_review_flags():
    obj = EyeResult(
        {
            "item_parameters": pd.DataFrame(
                {
                    "item_id": ["I1", "I2"],
                    "lower_asymptote": [0.10, 0.20],
                    "rt_ms": [np.nan, np.nan],
                    "ttff_ms": [np.nan, np.nan],
                }
            )
        },
        eyeprocess_class="eye_gaze_anchored_3pl_audit",
    )

    out = b3.audit_3pl_process_signatures(obj)

    assert not out["fast_rt_review"].any()
    assert not out["fast_ttff_review"].any()


def test_aoi_report_with_empty_inference_stability():
    x = {
        "stability": {
            "overall": pd.DataFrame(
                {
                    "perturbation_id": ["baseline"],
                    "proportion_unchanged": [1.0],
                }
            )
        },
        "models": pd.DataFrame(),
        "failures": pd.DataFrame(
            columns=[
                "stage",
            ]
        ),
        "grid_result": {
            "audit": pd.DataFrame(
                {
                    "perturbation_id": ["baseline"],
                    "status": ["completed"],
                }
            )
        },
        "caveat": "synthetic sensitivity audit",
    }

    report = ap.report_aoi_sensitivity(x)

    assert "Model-level sensitivity was summarized" not in report
    assert "Interpretation:" in report


def test_dynamic_explicit_parallel_chain_paths():
    theory = di.theory_strategy_spec(
        {
            "s1": {"f1": 1.0},
            "s2": {"f1": -1.0},
        },
        chains=4,
        parallel_chains=1,
    )

    assert theory.parallel_chains == 1

    diffusion = di.gaze_diffusion_spec(
        chains=4,
        parallel_chains=2,
    )

    assert diffusion.parallel_chains == 2


def test_process_penalty_with_vector_quality_risk():
    out = irt.eyeprocess_irt_process_aware_selection_penalty(
        information=[2.0, 3.0, 4.0],
        burden=[0.2, 0.2, 0.2],
        burden_weight=1.0,
        quality_risk=[0.1, 0.2, 0.3],
        quality_weight=1.0,
    )

    np.testing.assert_allclose(
        out,
        [1.7, 2.6, 3.5],
    )


def test_irt_validation_ungrouped_and_nonproblematic_correlation_paths():
    recovery = _recovery()

    convergence = iv.audit_convergence(
        recovery,
        minimum=0.90,
        by=(),
    )

    assert convergence.loc[0, "convergence_rate"] == pytest.approx(1.0)

    identifiability = iv.audit_identifiability(
        recovery,
        correlation_matrix=np.array(
            [
                [1.0, 0.2],
                [0.2, 1.0],
            ]
        ),
        max_abs_correlation=0.995,
    )

    assert identifiability.attrs["correlation_issue"] is False
    assert identifiability["pass"].all()


def test_device_transfer_equipercentile_skips_linear_transfer():
    linking = {
        "paired": pd.DataFrame(
            {
                "device": ["D1", "D1", "D1"],
                "device_value": [1.0, 2.0, 3.0],
                "reference_value": [1.1, 2.1, 3.1],
                "mean_value": [1.05, 2.05, 3.05],
                "difference": [-0.1, -0.1, -0.1],
            }
        ),
        "device_col": "device",
        "method": "equipercentile",
        "reference_device": "REF",
        "models": {},
    }

    fig, ax = plt.subplots()

    returned = mi.plot_device_transfer_curve(
        linking,
        device="D1",
        ax=ax,
    )

    assert returned is ax
    _close()


def test_detector_timeline_and_agreement_existing_axes(monkeypatch):
    events = pd.DataFrame(
        {
            "episode_type": ["fixation"],
            "trial_id": ["T1"],
            "detector_id": ["d1"],
            "start_time": [0.10],
            "end_time": [0.25],
        }
    )

    fig1, ax1 = plt.subplots()

    returned = dm.plot_detector_event_timeline(
        events,
        trial_id="T1",
        ax=ax1,
    )

    assert returned is ax1

    agreement = pd.DataFrame(
        {
            "detector_a": ["d1"],
            "detector_b": ["d2"],
            "mean_event_overlap": [0.8],
        }
    )

    monkeypatch.setattr(
        dm,
        "estimate_detector_agreement",
        lambda x: agreement,
    )

    fig2, ax2 = plt.subplots()

    returned = dm.plot_detector_agreement(
        object(),
        ax=ax2,
    )

    assert returned is ax2
    _close()


def test_survival_event_filter_and_disengagement_no_target_paths():
    no_episode_filter = pd.DataFrame(
        {
            "time": [0.2, 0.4],
            "aoi": ["body", "target"],
        }
    )

    first = surv._event_time_for_trial(
        no_episode_filter,
        target_aoi="target",
        event_type="first_aoi_entry",
        time_col="time",
        aoi_col="aoi",
        episode_type_col=None,
    )

    assert first == pytest.approx(0.4)

    no_target = pd.DataFrame(
        {
            "time": [0.1, 0.2, 0.3],
            "aoi": ["body", "body", "footer"],
            "episode_type": ["saccade", "saccade", "saccade"],
        }
    )

    disengagement = surv._event_time_for_trial(
        no_target,
        target_aoi="target",
        event_type="disengagement",
        time_col="time",
        aoi_col="aoi",
        episode_type_col="episode_type",
    )

    assert disengagement is None


def test_timebase_nonfinite_origin_no_overwrite_and_empty_transform_component():
    data = dm.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260920,
    )

    gaze = data["gaze_samples"].copy()
    gaze["timestamp_native"] = np.nan
    gaze["timestamp_seconds"] = 42.0
    data["gaze_samples"] = gaze

    normalized = tb.normalize_timebase(
        data,
        component="gaze_samples",
        native_unit="seconds",
        origin="recording_start",
        overwrite=False,
    )

    assert normalized["gaze_samples"]["timestamp_seconds"].eq(42.0).all()

    transformed = tb.apply_clock_transform(
        data,
        {
            "offset": 1.0,
            "slope": 1.0,
        },
        components="biometrics",
    )

    assert transformed["biometrics"].empty
