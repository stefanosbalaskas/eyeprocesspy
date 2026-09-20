from __future__ import annotations

import builtins
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy._aoi_assignment_core as aac
import eyeprocesspy._aoi_geometry_primitives as agp
import eyeprocesspy.aoi_perturbation as ap
import eyeprocesspy.bayesian_3pl_08 as b3
import eyeprocesspy.detector_multiverse as dm
import eyeprocesspy.dynamic_irt as di
import eyeprocesspy.measurement_intelligence as mi
import eyeprocesspy.process_governance_08 as pg
import eyeprocesspy.process_irt_07 as pi
import eyeprocesspy.pupil_missingness as pm
import eyeprocesspy.reproducibility_provenance_09 as rp
import eyeprocesspy.sensitivity_08 as sens
import eyeprocesspy.spatial_quality as sq
import eyeprocesspy.survival as surv
import eyeprocesspy.timebase as tb
from eyeprocesspy.irt import EyeResult


def _ivt_spec(detector_id: str = "coverage_ivt"):
    return ep.define_event_detector_spec(
        detector_id,
        "ivt",
        velocity_threshold=30,
        minimum_duration_ms=60,
        maximum_gap_ms=75,
        sampling_rate=60,
        coordinate_unit="degrees",
    )


def test_aoi_recompute_without_grouping_executes_ungrouped_contract():
    data = pd.DataFrame(
        {
            "duration": [0.10, 0.20, 0.15],
            "time": [0.10, 0.30, 0.50],
        }
    )

    out = aac.recompute_aoi_features(
        data,
        ["A", "B", "A"],
        duration_col="duration",
        time_col="time",
        aoi_levels=["A", "B", "C"],
        observation_level="fixation",
    )

    assert set(out["aoi"]) == {"A", "B", "C"}
    assert out.loc[out["aoi"].eq("A"), "fixation_count"].iloc[0] == 2
    assert out.loc[out["aoi"].eq("C"), "fixation_count"].iloc[0] == 0
    assert out.loc[out["aoi"].eq("C"), "inspected"].iloc[0] == False  # noqa: E712


def test_aoi_geometry_private_result_and_self_intersection_contracts():
    result = agp._result("coverage_contract", answer=42)
    assert result.eyeprocess_class == "coverage_contract"
    assert result["answer"] == 42

    bow_tie = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )
    assert agp._polygon_self_intersects(bow_tie)

    with pytest.raises(ep.EyeProcessValidationError, match="parallel adjacent edges"):
        agp._line_intersection(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([2.0, 0.0]),
        )


def test_aoi_sensitivity_rejects_invalid_observation_level_before_analysis():
    with pytest.raises(ep.EyeProcessValidationError, match="observation_level"):
        ap.run_aoi_sensitivity_analysis(
            pd.DataFrame({"x": [0.0], "y": [0.0]}),
            pd.DataFrame(),
            {"specifications": []},
            x_col="x",
            y_col="y",
            observation_level="invalid",
        )


def test_aoi_sensitivity_requires_observation_id_and_successful_baseline(monkeypatch):
    frame = pd.DataFrame({"x": [0.0], "y": [0.0]})

    monkeypatch.setattr(
        ap,
        "validate_aoi_geometry",
        lambda aois: {"geometry": pd.DataFrame()},
    )

    with pytest.raises(ep.EyeProcessValidationError, match="Observation id column"):
        ap.run_aoi_sensitivity_analysis(
            frame,
            pd.DataFrame(),
            {"specifications": []},
            x_col="x",
            y_col="y",
            observation_id_col="missing_id",
        )

    monkeypatch.setattr(
        ap,
        "apply_aoi_perturbation_grid",
        lambda geometry, grid: {
            "audit": pd.DataFrame(
                {
                    "perturbation_id": ["shift_only"],
                    "status": ["completed"],
                    "message": [""],
                }
            ),
            "geometries": {"shift_only": pd.DataFrame()},
        },
    )

    with pytest.raises(ep.EyeProcessValidationError, match="successful `baseline`"):
        ap.run_aoi_sensitivity_analysis(
            frame,
            pd.DataFrame(),
            {"specifications": []},
            x_col="x",
            y_col="y",
        )


def test_aoi_inference_stability_empty_missing_term_and_missing_baseline():
    empty = ap.assess_aoi_inference_stability({"models": pd.DataFrame()})
    assert empty.empty
    assert "convergence_proportion" in empty.columns

    models = pd.DataFrame(
        {
            "term": ["condition", "condition"],
            "model_converged": [True, False],
            "perturbation_id": ["shift", "expand"],
            "estimate": [0.30, 0.40],
            "CI_low": [0.10, 0.20],
            "CI_high": [0.50, 0.60],
            "N": [100, 100],
        }
    )

    out = ap.assess_aoi_inference_stability({"models": models})
    assert len(out) == 1
    assert np.isnan(out.loc[0, "same_sign_proportion"])

    with pytest.raises(ep.EyeProcessValidationError, match="No model rows"):
        ap.assess_aoi_inference_stability(
            {"models": models},
            term="not_present",
        )


def test_bayesian_diagnostic_flags_create_missing_diagnostic_columns():
    dashboard = EyeResult(
        {
            "posterior": pd.DataFrame(
                {
                    "parameter": ["theta"],
                    "rhat": [1.02],
                }
            )
        },
        eyeprocess_class="eye_bayesian_process_dashboard",
    )

    out = b3.bayesian_process_diagnostic_flags(dashboard)
    assert {"ess_bulk", "ess_tail", "review_required"}.issubset(out.columns)
    assert bool(out.loc[0, "rhat_review"])
    assert bool(out.loc[0, "review_required"])


def test_dataset_validation_detects_nonfinite_timestamp_and_negative_interval():
    data = ep.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260920,
    )

    bad = data.copy()
    bad["gaze_samples"] = bad["gaze_samples"].copy()
    bad["intervals"] = bad["intervals"].copy()

    bad["gaze_samples"].loc[
        bad["gaze_samples"].index[0],
        "timestamp_seconds",
    ] = np.inf

    bad["intervals"].loc[
        bad["intervals"].index[0],
        "start_time",
    ] = 10.0
    bad["intervals"].loc[
        bad["intervals"].index[0],
        "end_time",
    ] = 9.0

    issues = ep.validate_eye_dataset(bad, stop_on_error=False)

    assert "nonfinite_timestamp" in set(issues["code"])
    assert "negative_interval" in set(issues["code"])


def test_set_eye_table_validation_path_is_executed():
    data = ep.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260921,
    )

    out = ep.set_eye_table(
        data,
        "events",
        data["events"].copy(),
        validate=True,
    )

    assert hasattr(out, "validation")
    assert isinstance(out.validation, pd.DataFrame)


def test_detector_spec_defensive_validation_contracts():
    spec = _ivt_spec()

    with pytest.raises(ep.EyeProcessValidationError, match="EventDetectorSpec"):
        dm.validate_event_detector_spec(object())

    with pytest.raises(ep.EyeProcessValidationError, match="Unsupported detector algorithm"):
        dm.validate_event_detector_spec(replace(spec, algorithm="not_a_detector"))

    with pytest.raises(ep.EyeProcessValidationError, match="coordinate_unit"):
        dm.validate_event_detector_spec(replace(spec, coordinate_unit="centimeters"))

    vendor = ep.define_event_detector_spec(
        "vendor_events",
        "vendor",
    )

    with pytest.raises(ep.EyeProcessValidationError, match="do not use `callback`"):
        dm.validate_event_detector_spec(
            replace(
                vendor,
                callback=lambda **kwargs: None,
            )
        )


def test_detector_multiverse_handles_branch_without_detector_id(monkeypatch):
    data = ep.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260922,
    )
    spec = _ivt_spec()

    branch = data.copy()
    branch["episodes"] = pd.DataFrame(
        {
            "episode_type": ["fixation"],
            "start_time": [0.10],
            "end_time": [0.20],
        }
    )

    monkeypatch.setattr(
        dm,
        "detect_events_with_spec",
        lambda x, detector_spec: branch,
    )

    result = dm.run_detector_multiverse(
        data,
        [spec],
        continue_on_error=False,
    )

    assert result.status.loc[0, "status"] == "ok"
    assert result.events.empty


def test_detector_aoi_assignment_covers_stimulus_coordinate_and_all_overlap():
    data = ep.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260923,
    )

    for name in ("coverage_overlap_a", "coverage_overlap_b"):
        data = ep.register_aois(
            data,
            ep.new_aoi(
                name,
                name,
                "stim_01",
                "rectangle",
                x=4.5,
                y=1.5,
                width=3.0,
                height=2.0,
                coordinate_space_id="deg_display",
            ),
        )

    data = data.copy()
    data["episodes"] = pd.DataFrame(
        [
            {
                "episode_id": "E-overlap",
                "episode_type": "fixation",
                "centroid_x": 6.0,
                "centroid_y": 2.4,
                "start_time": 0.2,
                "stimulus_id": "stim_01",
                "coordinate_space_id": "deg_display",
                "aoi_id": pd.NA,
            },
            {
                "episode_id": "E-stimulus-mismatch",
                "episode_type": "fixation",
                "centroid_x": 6.0,
                "centroid_y": 2.4,
                "start_time": 0.3,
                "stimulus_id": "different_stimulus",
                "coordinate_space_id": "deg_display",
                "aoi_id": pd.NA,
            },
            {
                "episode_id": "E-coordinate-mismatch",
                "episode_type": "fixation",
                "centroid_x": 6.0,
                "centroid_y": 2.4,
                "start_time": 0.4,
                "stimulus_id": "stim_01",
                "coordinate_space_id": "different_space",
                "aoi_id": pd.NA,
            },
        ]
    )

    out = dm._assign_episode_aois_explicit(
        data,
        overlap="all",
    )

    assigned = out["episodes"].set_index("episode_id")["aoi_id"]

    assert "|" in str(assigned.loc["E-overlap"])
    assert pd.isna(assigned.loc["E-coordinate-mismatch"])


def test_statsmodels_backend_import_failure_is_explicit(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "statsmodels.formula.api":
            raise ImportError("blocked for coverage contract")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as ctx:
        ctx.setattr(builtins, "__import__", blocked_import)

        with pytest.raises(ep.EyeProcessBackendError, match="requires statsmodels"):
            dm._fit_statsmodels(
                pd.DataFrame(
                    {
                        "y": [1.0, 2.0, 3.0],
                        "x": [0.0, 1.0, 2.0],
                    }
                ),
                {"formula": "y ~ x"},
            )


def test_detector_coefficient_plot_rejects_unknown_term():
    multiverse = dm.create_detector_multiverse([_ivt_spec()])

    inference = dm.DetectorInferenceResult(
        multiverse=multiverse,
        coefficients=pd.DataFrame(
            {
                "term": ["condition"],
                "estimate": [0.1],
                "CI_lower": [-0.1],
                "CI_upper": [0.3],
                "detector_id": ["coverage_ivt"],
            }
        ),
        failures=pd.DataFrame(),
        warnings=pd.DataFrame(),
        model_spec={},
    )

    with pytest.raises(ep.EyeProcessValidationError, match="term is unavailable"):
        dm.plot_detector_coefficient_stability(
            inference,
            term="missing_term",
        )


def test_detector_report_surfaces_recorded_branch_failure():
    data = ep.simulate_detector_multiverse_data(
        n_participants=4,
        seed=20260924,
    )

    def broken_detector(data, spec):
        raise RuntimeError("intentional release-gate failure branch")

    spec = ep.define_event_detector_spec(
        "broken_external",
        "external",
        callback=broken_detector,
        implementation="coverage_test",
    )

    result = ep.run_detector_multiverse(
        data,
        [spec],
        continue_on_error=True,
    )

    assert not result.failures.empty

    report = ep.report_detector_multiverse(result)

    assert "## Branch failures" in report
    assert "intentional release-gate failure branch" in report


def test_dynamic_strategy_simulation_dataframe_signature_contract():
    signatures = pd.DataFrame(
        [[1.0, 0.2], [-1.0, 0.8]],
        index=["analytic", "heuristic"],
        columns=["dwell", "revisit"],
    )

    out = di.simulate_strategy_mixture_data(
        n_person=3,
        n_item=2,
        signatures=signatures,
        trials_per_item=1,
        seed=20260925,
    )

    assert set(out["true_strategy"]) <= {"analytic", "heuristic"}
    assert {"dwell", "revisit"}.issubset(out.columns)


def test_irt_classification_precision_broadcasts_scalar_standard_error():
    out = ep.eyeprocess_irt_classification_precision(
        theta_estimate=[-0.5, 0.0, 0.5],
        standard_error=0.20,
        cut_score=0.0,
    )

    assert len(out) == 3
    assert out["se"].eq(0.20).all()


def test_device_equivalence_single_group_key_contract():
    linking = EyeResult(
        {
            "paired": pd.DataFrame(
                {
                    "device": ["candidate", "candidate", "candidate"],
                    "difference": [0.05, -0.02, 0.01],
                }
            ),
            "device_col": "device",
        },
        eyeprocess_class="eye_device_linking",
    )

    result = mi.audit_device_equivalence(
        linking,
        equivalence_margin=0.50,
        by=(),
    )

    assert len(result.summary) == 1
    assert result.summary.loc[0, "device"] == "candidate"


def test_item_bank_evolutionary_all_items_hits_no_available_replacement():
    items = pd.DataFrame(
        {
            "raw": [10.0, 20.0, 30.0],
        }
    )

    spec = ep.item_objective_spec(
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
    )

    pareto = ep.item_pareto_front(items, spec)

    result = ep.optimize_item_bank(
        pareto,
        3,
        spec,
        method="evolutionary",
        iterations=3,
        seed=20260926,
    )

    assert len(result.selected) == 3


def test_process_dif_ability_path_and_evidence_decomposition_merges():
    data = pd.DataFrame(
        {
            "item": ["I1"] * 8 + ["I2"] * 8,
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"] * 2,
            "response": [0, 1, 0, 1, 0, 1, 1, 1] * 2,
            "process": np.linspace(-1.5, 1.5, 16),
            "ability": np.linspace(-1.0, 1.0, 16),
        }
    )

    fitted = mi.fit_process_dif(
        data,
        response="response",
        process="process",
        group="group",
        item="item",
        ability="ability",
    )

    assert set(fitted.summary["item_id"]) == {"I1", "I2"}

    psychometric = pd.DataFrame(
        {
            "item_id": ["I1", "I2"],
            "psychometric_p": [0.01, 0.50],
        }
    )
    process = pd.DataFrame(
        {
            "item_id": ["I1", "I2"],
            "process_p": [0.02, 0.60],
        }
    )
    design = pd.DataFrame(
        {
            "item_id": ["I1", "I2"],
            "visual_complexity": [1.0, 2.0],
        }
    )

    decomposed = mi.decompose_dif_evidence(
        psychometric,
        process=process,
        design_features=design,
    )

    assert "visual_complexity" in decomposed.table.columns
    assert (
        decomposed.table.set_index("item_id").loc["I1", "evidence_pattern"]
        == "convergent_psychometric_and_process_difference"
    )


def test_preflight_pass_failure_and_manifest_paths():
    result = EyeResult(
        {
            "table": pd.DataFrame(
                {
                    "participant_id": ["P1", "P2", "P3"],
                    "quality_flag": [False, True, True],
                    "preflight_flag_count": [0, 1, 2],
                    "preflight_decision": [
                        "pass_preflight",
                        "use_with_caution",
                        "review_or_exclude_from_biometric_models",
                    ],
                }
            ),
            "by": ["participant_id"],
            "flag_columns": ["quality_flag"],
        },
        eyeprocess_class="eye_biometric_preflight",
    )

    failures = pg.preflight_failures(result)
    passed = pg.preflight_passed(result)
    manifest = pg.preflight_exclusion_manifest(result)

    assert len(failures) == 2
    assert len(passed) == 1
    assert set(manifest["recommended_action"]) == {
        "retain",
        "retain_with_sensitivity_analysis",
        "manual_review_before_biometric_model_inclusion",
    }


def test_signal_filter_even_width_and_unavailable_robfilter_backend():
    result = pg.filter_eye_signal(
        [1.0, 1.1, 0.9, 1.2, 1.0, 1.1],
        width=4,
        method="runmed",
    )

    assert result.width % 2 == 1
    assert result.method == "runmed"

    with pytest.raises(ep.EyeProcessBackendError, match="robfilter"):
        pg.filter_eye_signal(
            [1.0, 1.1, 0.9, 1.2, 1.0, 1.1],
            method="robfilter",
        )


def test_promote_irt_model_string_lookup_branch_is_explicit():
    with pytest.raises(Exception):
        pi.promote_irt_model(
            "__coverage_missing_model__",
            evidence={},
        )


def test_mnar_tipping_plot_draws_finite_tipping_line():
    sensitivity = EyeResult(
        {
            "table": pd.DataFrame(
                {
                    "delta": [-1.0, 0.0, 1.0],
                    "estimate": [-0.5, 0.1, 0.5],
                }
            )
        },
        eyeprocess_class="eye_mnar_sensitivity",
    )

    tipping = EyeResult(
        {
            "sensitivity": sensitivity,
            "tipping_delta": 0.0,
        },
        eyeprocess_class="eye_mnar_tipping_point",
    )

    fig, ax = plt.subplots()
    try:
        returned = pm._miss_plot(
            tipping,
            "mnar_tipping_point",
            ax=ax,
        )
        assert returned is ax
        assert len(ax.lines) >= 3
    finally:
        plt.close(fig)


def test_reproducibility_canonicalization_special_float_and_fallback_paths():
    assert rp._canonicalize(float("nan")) == {"__float__": "nan"}
    assert rp._canonicalize(float("inf")) == {"__float__": "inf"}
    assert rp._canonicalize(float("-inf")) == {"__float__": "-inf"}
    assert rp._canonicalize(pd.NA) is None

    class Custom:
        def __repr__(self):
            return "custom-object"

    out = rp._canonicalize(Custom())
    assert out["repr"] == "custom-object"
    assert out["__type__"].endswith(".Custom")


def test_sensitivity_required_column_guard():
    with pytest.raises(ep.EyeProcessValidationError, match="missing required columns"):
        sens._req(
            pd.DataFrame({"present": [1]}),
            ["present", "missing"],
        )


def test_spatial_quality_flags_zero_valid_samples():
    data = pd.DataFrame(
        {
            "gaze_x": [1.0, 1.1, 1.2, 1.3],
            "gaze_y": [2.0, 2.1, 2.2, 2.3],
            "timestamp_ms": [0.0, 16.7, 33.4, 50.1],
            "valid": [False, False, False, False],
        }
    )

    report = sq.create_gaze_quality_report(
        data,
        valid="valid",
        target_x=None,
        target_y=None,
        unit="degrees",
        time_unit="ms",
        nominal_sampling_hz=60,
    )

    assert len(report) == 1
    assert "no_valid_gaze_samples" in report.loc[0, "quality_flags"]
    assert bool(report.loc[0, "review_required"])


def test_survival_unsupported_event_and_single_group_summary_paths():
    events = pd.DataFrame(
        {
            "start_time": [0.5, 1.0],
            "aoi_id": ["body", "target"],
            "episode_type": ["fixation", "fixation"],
        }
    )

    with pytest.raises(ValueError, match="Unsupported event_type"):
        surv._event_time_for_trial(
            events,
            target_aoi="target",
            event_type="unsupported_event",
            time_col="start_time",
            aoi_col="aoi_id",
            episode_type_col="episode_type",
        )

    data = ep.simulate_gaze_survival_example(
        seed=20260927,
        n_participants=8,
        trials_per_participant=2,
    )

    out = surv.summarise_gaze_censoring(
        data,
        by="condition",
    )

    assert "condition" in out.columns
    assert out["n_trials"].sum() == len(data)


def test_sampling_rate_short_duplicate_and_trimmed_paths():
    assert np.isnan(tb.estimate_sampling_rate([1.0]))
    assert np.isnan(tb.estimate_sampling_rate([1.0, 1.0]))

    timestamps = np.arange(0.0, 1.0, 1 / 60)
    rate = tb.estimate_sampling_rate(
        timestamps,
        trim=0.05,
    )

    assert rate == pytest.approx(60.0, rel=1e-6)
