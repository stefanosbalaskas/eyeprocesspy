from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy._aoi_assignment_core as ac
import eyeprocesspy.adapters as adapters
import eyeprocesspy.advanced_process_irt_07 as advanced
import eyeprocesspy.coordinates as coordinates
import eyeprocesspy.dataset as ds
import eyeprocesspy.detector_multiverse as dm
import eyeprocesspy.functional_pupil as fp
import eyeprocesspy.gazepoint as gp
import eyeprocesspy.governance_09 as governance
import eyeprocesspy.irt as irt
import eyeprocesspy.legacy_models as legacy
import eyeprocesspy.measurement_accountability_11 as accountability
import eyeprocesspy.measurement_intelligence as intelligence
import eyeprocesspy.plots_governance_09 as plots_governance
import eyeprocesspy.process_governance_08 as process_governance
import eyeprocesspy.requested_api_07 as requested
import eyeprocesspy.survival as survival
from eyeprocesspy.exceptions import EyeProcessBackendError
from eyeprocesspy.irt import EyeResult


def _close() -> None:
    plt.close("all")


# ---------------------------------------------------------------------
# AOI assignment scalar grouping
# ---------------------------------------------------------------------


def test_aoi_recompute_single_group_scalar_key():
    frame = pd.DataFrame(
        {
            "participant": ["P1", "P1", "P2", "P2"],
            "duration": [0.1, 0.2, 0.3, 0.4],
            "time": [0.1, 0.2, 0.1, 0.2],
        }
    )

    out = ac.recompute_aoi_features(
        frame,
        ["A", "B", "A", "B"],
        participant_col="participant",
        trial_col=None,
        duration_col="duration",
        time_col="time",
        aoi_levels=["A", "B"],
    )

    assert set(out["participant"]) == {"P1", "P2"}
    assert len(out) == 4


# ---------------------------------------------------------------------
# Dataset false branches
# ---------------------------------------------------------------------


def test_compact_dataset_preserves_raw_and_empty_metadata_when_not_requested():
    x = ds.new_eye_dataset(
        raw={"synthetic": pd.DataFrame({"x": [1]})},
        validate=False,
    )

    out = ds.compact_eye_dataset(
        x,
        drop_raw=False,
        drop_empty=False,
    )

    assert isinstance(out.raw, dict)


# ---------------------------------------------------------------------
# Adapter residual branches
# ---------------------------------------------------------------------


def test_read_eye_folder_skips_non_dataset_and_continues(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x\n2\n", encoding="utf-8")

    valid = ds.new_eye_dataset(validate=False)
    outputs = iter([object(), valid])

    monkeypatch.setattr(
        adapters,
        "read_eye_export",
        lambda *args, **kwargs: next(outputs),
    )
    monkeypatch.setattr(
        adapters,
        "combine_eye_datasets",
        lambda objects: objects[0],
    )

    out = adapters.read_eye_folder(
        tmp_path,
        combine=True,
    )

    assert ds.is_eye_dataset(out)


def test_combine_eye_datasets_handles_noncollection_raw_and_metadata():
    a = ds.new_eye_dataset(validate=False)
    b = ds.new_eye_dataset(validate=False)

    a.raw = None
    b.raw = None
    a.vendor_metadata = None
    b.vendor_metadata = None

    out = adapters.combine_eye_datasets(
        a,
        b,
        resolve_ids=False,
    )

    assert ds.is_eye_dataset(out)


# ---------------------------------------------------------------------
# Advanced process IRT fitted-probability reconstruction
# ---------------------------------------------------------------------


def test_process_person_fit_reconstructs_mismatched_fitted_vector(monkeypatch):
    frame = pd.DataFrame(
        {
            "person": ["P1", "P2"],
            "item": ["I1", "I2"],
            "response": [1, 0],
            "rt": [1.0, 1.2],
            "gaze": [2.0, 3.0],
        }
    )

    monkeypatch.setattr(
        advanced,
        "_dummy_design",
        lambda *args, **kwargs: (
            np.ones((2, 1), dtype=float),
            ["Intercept"],
        ),
    )

    obj = SimpleNamespace(
        eyeprocess_class="eye_joint_gaze_rt_irt",
        columns={
            "person": "person",
            "item": "item",
            "response": "response",
            "rt": "rt",
            "gaze": "gaze",
        },
        response_model=SimpleNamespace(
            fitted=np.array([0.5]),
            coefficients=np.array([0.0]),
        ),
        rt_model=SimpleNamespace(
            fitted=np.log(frame["rt"].to_numpy(float)),
        ),
        gaze_model=SimpleNamespace(
            fitted=np.log1p(frame["gaze"].to_numpy(float)),
        ),
    )

    out = advanced.process_person_fit(
        obj,
        data=frame,
    )

    assert len(out) == 2
    assert np.isfinite(out["combined_rms"]).all()


# ---------------------------------------------------------------------
# Coordinate-space branch with no tagged components
# ---------------------------------------------------------------------


def test_coordinate_audit_handles_components_without_space_column():
    x = dm.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260920,
    )

    for name in ("gaze_samples", "episodes", "aoi_geometry"):
        x[name] = x[name].drop(
            columns=["coordinate_space_id"],
            errors="ignore",
        )

    out = coordinates.audit_coordinate_spaces(x)

    assert out.empty


# ---------------------------------------------------------------------
# Functional pupil residual paths:
# - timestamp-like column that is already in ms
# - drop_invalid_baseline=False
# - no response_time in prepared trials
# - no response_time in trial coefficients
# ---------------------------------------------------------------------


def test_functional_pupil_ms_timestamp_and_optional_response_time_paths():
    rows = []

    for trial, item, response in (
        ("T1", "I1", 0),
        ("T2", "I2", 1),
    ):
        for j, timestamp in enumerate([0.0, 100.0, 200.0, 300.0, 400.0]):
            rows.append(
                {
                    "participant_id": "P1",
                    "item_id": item,
                    "trial_id": trial,
                    "score": response,
                    "timestamp": timestamp,
                    "pupil": 3.0 + 0.02 * j + 0.05 * response,
                }
            )

    data = pd.DataFrame(rows)

    spec = fp.functional_pupil_irt_spec(
        pupil_column="pupil",
        time_column="timestamp",
        latency_ms=0,
        baseline_window=(0, 200),
        baseline_method="subtract",
        min_baseline_samples=2,
        drop_invalid_baseline=False,
        df=2,
    )

    prepared = fp.prepare_functional_pupil_data(
        data,
        spec,
    )

    assert "response_time" not in prepared.trials.columns

    basis = pd.DataFrame(
        {
            "pupil_basis_1": np.ones(len(prepared.data)),
            "pupil_basis_2": prepared.data["time_scaled"].to_numpy(float),
        }
    )

    coefficients = fp._trial_coefficients(
        prepared,
        basis,
    )

    assert "response_time" not in coefficients.columns
    assert coefficients["basis_supported"].all()


# ---------------------------------------------------------------------
# Gazepoint optional combined biometric branch
# ---------------------------------------------------------------------


def test_gazepoint_combined_biometrics_without_fixations(monkeypatch):
    base = dm.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260920,
    )

    called = []

    monkeypatch.setattr(
        gp,
        "read_gazepoint",
        lambda *args, **kwargs: base.copy(),
    )

    def fake_biometrics(*args, **kwargs):
        called.append(True)
        return base.copy()

    monkeypatch.setattr(
        gp,
        "read_gazepoint_biometrics",
        fake_biometrics,
    )
    monkeypatch.setattr(
        gp,
        "combine_eye_datasets",
        lambda xs, resolve_ids=False: xs[0],
    )

    out = gp.read_gazepoint_combined(
        "gaze.csv",
        fixations=None,
        biometrics="biometrics.csv",
    )

    assert called == [True]
    assert ds.is_eye_dataset(out)


# ---------------------------------------------------------------------
# Governance residual paths
# ---------------------------------------------------------------------


def test_validation_summary_multigroup_toposort_and_nan_rank():
    validation = SimpleNamespace(
        estimates=pd.DataFrame(
            {
                "parameter": ["b", "b", "b", "b"],
                "scenario": ["A", "A", "B", "B"],
                "truth": [0.0, 0.1, 0.0, 0.1],
                "estimate": [0.01, 0.11, -0.01, 0.09],
            }
        )
    )

    summary = governance.validation_summary_mcse(
        validation,
        by="scenario",
    )

    assert len(summary) == 2

    steps = {
        "a": SimpleNamespace(requires=[]),
        "b": SimpleNamespace(requires=[]),
        "c": SimpleNamespace(requires=["a", "b"]),
    }

    order = governance._toposort(steps)

    assert order[-1] == "c"

    value = governance.sensitivity_rank_stability(
        [
            [1, 1, 1],
            [1, 1, 1],
        ]
    )

    assert np.isnan(value)


# ---------------------------------------------------------------------
# IRT latent-regression all-missing numeric centering branch
# ---------------------------------------------------------------------


def test_latent_regression_all_missing_numeric_center():
    out = irt.eyeprocess_irt_latent_regression_design(
        pd.DataFrame(
            {
                "x": [np.nan, np.nan, np.nan],
            }
        ),
        "~ x",
        center_numeric=True,
    )

    assert np.isnan(out.centers["x"])
    assert not out.complete.any()


# ---------------------------------------------------------------------
# Legacy model residual paths
# ---------------------------------------------------------------------


def test_legacy_model_data_no_common_feature_join_keys(monkeypatch):
    fake = {
        "responses": pd.DataFrame(
            {
                "participant_id": ["P1"],
                "item_id": ["I1"],
                "score": [1],
                "response_time": [1.0],
            }
        ),
        "features": pd.DataFrame(
            {
                "dummy": [1.0],
            }
        ),
    }

    monkeypatch.setattr(
        legacy,
        "_require_dataset",
        lambda x: x,
    )
    monkeypatch.setattr(
        legacy,
        "_features_wide",
        lambda x, aggregate=np.mean: pd.DataFrame(
            {
                "unrelated": [1.0],
            }
        ),
    )

    out = legacy.model_data(fake)

    assert "participant_id" in out


def test_joint_process_brms_backend_contract(monkeypatch):
    monkeypatch.setattr(
        legacy,
        "_require_dataset",
        lambda x: x,
    )
    monkeypatch.setattr(
        legacy,
        "model_data",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "response_time": [1.0, 1.1],
            }
        ),
    )

    with pytest.raises(EyeProcessBackendError):
        legacy.fit_joint_process_model(
            object(),
            "score ~ 1",
            "log_response_time ~ 1",
            engine="brms",
        )


def test_ez_diffusion_multicolumn_grouping():
    frame = pd.DataFrame(
        {
            "item_id": ["I1"] * 6,
            "condition": ["A"] * 6,
            "score": [1, 1, 0, 1, 0, 1],
            "response_time": [0.45, 0.50, 0.55, 0.60, 0.52, 0.48],
        }
    )

    out = legacy.estimate_ez_diffusion(
        frame,
        by=["item_id", "condition"],
    )

    assert len(out) == 1


def test_missing_process_without_complete_case_branch(monkeypatch):
    frame = pd.DataFrame(
        {
            "participant_id": ["P1", "P2", "P3"],
            "item_id": ["I1", "I1", "I2"],
            "score": [1, 0, 1],
            "feature": [1.0, np.nan, 2.0],
        }
    )

    monkeypatch.setattr(
        legacy,
        "_require_dataset",
        lambda x: x,
    )
    monkeypatch.setattr(
        legacy,
        "model_data",
        lambda *args, **kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        legacy,
        "_fit_binomial",
        lambda *args, **kwargs: SimpleNamespace(ok=True),
    )

    out = legacy.sensitivity_missing_process(
        object(),
        "feature",
        "score ~ feature",
        methods=("median_indicator",),
    )

    assert set(out.fits) == {"median_indicator"}


# ---------------------------------------------------------------------
# Measurement accountability
# ---------------------------------------------------------------------


def test_pupil_latency_rejects_too_small_analysis_windows():
    time = np.linspace(-0.5, 0.2, 8)
    pupil = np.linspace(3.0, 2.9, 8)

    with pytest.raises(
        ValueError,
        match="too few samples",
    ):
        accountability.pupil_latency_sensitivity(
            time,
            pupil,
            baseline_window=(-0.50, -0.49),
            search_window=(0.0, 0.2),
            simulations=0,
        )


# ---------------------------------------------------------------------
# Measurement intelligence:
# tuple grouping + deterministic evolutionary improvement
# ---------------------------------------------------------------------


def test_device_equivalence_tuple_group_and_evolutionary_improvement():
    linking = EyeResult(
        {
            "paired": pd.DataFrame(
                {
                    "device": ["D1", "D1", "D2", "D2"],
                    "task": ["A", "B", "A", "B"],
                    "difference": [0.01, 0.02, -0.01, -0.02],
                }
            ),
            "device_col": "device",
        },
        eyeprocess_class="eye_device_linking",
    )

    audit = intelligence.audit_device_equivalence(
        linking,
        equivalence_margin=1.0,
        by=("task",),
    )

    assert len(audit.summary) == 4

    pareto = EyeResult(
        {
            "table": pd.DataFrame(
                {
                    "item_id": ["I1", "I2", "I3"],
                    "pareto_front": [False, False, True],
                    "weighted_score": [0.0, 1.0, 100.0],
                }
            )
        },
        eyeprocess_class="eye_item_pareto",
    )

    optimized = intelligence.optimize_item_bank(
        pareto,
        n_items=1,
        objectives={},
        method="evolutionary",
        iterations=1,
        seed=1,
    )

    assert optimized.selected.iloc[0]["item_id"] == "I3"


# ---------------------------------------------------------------------
# Governance plot without confidence interval columns
# ---------------------------------------------------------------------


def test_specification_curve_without_interval_columns(
    monkeypatch,
):
    monkeypatch.setattr(
        plots_governance,
        "specification_curve_data",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "curve_order": [1, 2],
                ".effect": [0.1, -0.1],
            }
        ),
    )

    fig, ax = plt.subplots()

    returned = plots_governance.plot_eye_process_sensitivity(
        object(),
        ax=ax,
    )

    assert returned is ax
    _close()


# ---------------------------------------------------------------------
# Process-governance residual branches
# ---------------------------------------------------------------------


def test_pupil_frequency_empty_result_and_simple_confound_design(
    monkeypatch,
):
    data = pd.DataFrame(
        {
            "person_id": ["P1"] * 8,
            "trial_id": ["T1"] * 8,
            "time_ms": np.arange(8) * 100.0,
            "pupil_bc": np.linspace(3.0, 3.2, 8),
        }
    )

    monkeypatch.setattr(
        process_governance,
        "pupil_frequency_features",
        lambda *args, **kwargs: SimpleNamespace(features=pd.DataFrame()),
    )

    stability = process_governance.audit_pupil_frequency_stability(
        data,
        windows_ms=(1000,),
    )

    assert stability.table.empty

    n = 30

    confounds = pd.DataFrame(
        {
            "pupil_peak": np.linspace(3.0, 4.0, n),
            "screen_luminance": np.r_[np.zeros(15), np.ones(15)],
            "trial_sequence": np.tile([1.0, 2.0], 15),
        }
    )

    fit = process_governance.fit_pupil_confound_model(
        confounds,
        engine="lm",
    )

    assert fit.engine == "lm"
    assert "trial2" not in fit.model.design_columns


# ---------------------------------------------------------------------
# Requested API covariance-shape fallback
# ---------------------------------------------------------------------


def test_irf_uncertainty_wrong_covariance_shape_uses_zero_width():
    model = SimpleNamespace(
        theta_degree=1,
        coefficients=np.array([0.0, 1.0]),
        covariance=np.eye(1),
    )

    obj = SimpleNamespace(
        eyeprocess_class="eye_gpirt",
        engine="spline_reference",
        item_names=["I1"],
        models=[model],
    )

    fig, ax = plt.subplots()

    returned = requested.plot_irf_uncertainty(
        obj,
        item="I1",
        theta_grid=[-1.0, 0.0, 1.0],
        ax=ax,
    )

    assert returned is ax
    _close()


# ---------------------------------------------------------------------
# Detector plot false dispatch path
# ---------------------------------------------------------------------


def test_detector_multiverse_plot_without_inference(
    monkeypatch,
):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    monkeypatch.setattr(
        dm,
        "plot_detector_agreement",
        lambda *args, **kwargs: ax1,
    )
    monkeypatch.setattr(
        dm,
        "plot_detector_feature_distributions",
        lambda *args, **kwargs: ax2,
    )

    plots = dm.plot_detector_multiverse(
        object(),
        inference=None,
        term=None,
    )

    assert set(plots) == {"agreement", "feature"}
    _close()


# ---------------------------------------------------------------------
# Survival residual branches
# ---------------------------------------------------------------------


def test_survival_no_supplied_event_time_no_intercept_and_fake_models():
    trials = pd.DataFrame(
        {
            "participant_id": ["P1", "P2", "P3", "P4"],
            "trial_id": ["T1", "T2", "T3", "T4"],
            "start_time": [0.0, 0.0, 0.0, 0.0],
            "end_time": [5.0, 5.0, 5.0, 5.0],
            "event_observed": [0, 0, 0, 0],
        }
    )

    prepared = survival.prepare_gaze_survival_data(
        trials,
        events=None,
        target_aoi="target",
    )

    assert prepared["event_time"].isna().all()
    assert prepared["event_observed"].eq(0).all()

    cox_data = pd.DataFrame(
        {
            "participant_id": ["P1", "P2", "P3", "P4"],
            "analysis_time": [1.0, 2.0, 3.0, 4.0],
            "event_observed": [1, 0, 1, 0],
            "x": [0.0, 1.0, 0.5, -0.5],
        }
    )

    design, info = survival._design_matrix(
        "0 + x",
        cox_data,
    )

    assert "Intercept" not in design.columns

    result = SimpleNamespace(
        params=np.array([0.1]),
        llf=-3.0,
        schoenfeld_residuals=np.array(
            [
                [0.10],
                [-0.10],
            ]
        ),
    )

    fit = survival.GazeSurvivalFit(
        model_family="cox",
        backend="synthetic",
        result=result,
        data=cox_data,
        formula="0 + x",
        covariate_names=["x"],
        design_info=info,
    )

    ph = survival.check_gaze_proportional_hazards(fit)
    assert len(ph) == 1

    prediction = survival.predict_gaze_survival(
        fit,
        pd.DataFrame({"x": [0.25]}),
        times=[1.0, 2.0],
    )
    assert len(prediction) == 2

    quantiles = survival.estimate_gaze_latency_quantiles(
        fit,
        probs=[0.5],
        newdata=pd.DataFrame({"x": [0.25]}),
    )
    assert len(quantiles) == 1

    comparison = survival.compare_gaze_survival_models(
        fit,
        survival.GazeSurvivalFit(
            model_family="cox",
            backend="synthetic",
            result=SimpleNamespace(
                params=np.array([0.2]),
                llf=-3.5,
            ),
            data=cox_data.copy(),
            formula="0 + x",
            covariate_names=["x"],
            design_info=info,
        ),
    )

    assert comparison["information_criteria_comparable"].all()

    ax = survival.plot_gaze_cox_diagnostics(fit)
    assert ax is not None

    fig, supplied_ax = plt.subplots()
    returned_ax = survival.plot_gaze_cox_diagnostics(
        fit,
        ax=supplied_ax,
    )
    assert returned_ax is supplied_ax
    _close()

    index = pd.MultiIndex.from_tuples(
        [
            ("lambda_", "x"),
        ]
    )

    aft_summary = pd.DataFrame(
        {
            "coef": [0.1],
            "se(coef)": [0.05],
            "z": [2.0],
            "p": [0.045],
        },
        index=index,
    )

    aft = survival.GazeSurvivalFit(
        model_family="aft_weibull",
        backend="synthetic",
        result=SimpleNamespace(
            summary=aft_summary,
        ),
        data=prepared,
        formula="x",
        covariate_names=["x"],
    )

    report = survival.report_gaze_survival_model(aft)

    assert report["model_family"] == "aft_weibull"
    assert report["diagnostic_result"] == ("not applicable to AFT model")
