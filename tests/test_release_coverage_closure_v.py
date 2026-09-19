from __future__ import annotations

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
import eyeprocesspy.dataset as ds
import eyeprocesspy.detector_multiverse as dm
import eyeprocesspy.irt_validation_07 as iv
import eyeprocesspy.measurement_intelligence as mi
import eyeprocesspy.multilevel_mediation as mm
import eyeprocesspy.process_governance_08 as pg
import eyeprocesspy.pupil_missingness as pm
import eyeprocesspy.spatial_quality as sq
import eyeprocesspy.survival as sv
from eyeprocesspy.exceptions import (
    EyeProcessBackendError,
    EyeProcessSchemaError,
    EyeProcessValidationError,
)

# =================================================================
# DATASET / SCHEMA GUARDS
# =================================================================


def test_dataset_assert_and_table_contract_guards():
    with pytest.raises(
        TypeError,
        match="EyeDataset",
    ):
        ds._assert_eye_dataset(object())

    data = ds.new_eye_dataset(validate=False)

    with pytest.raises(
        TypeError,
        match="string",
    ):
        ds.get_eye_table(
            data,
            123,
        )

    with pytest.raises(
        EyeProcessSchemaError,
        match="Unknown component",
    ):
        ds.get_eye_table(
            data,
            "does_not_exist",
        )

    with pytest.raises(
        EyeProcessSchemaError,
        match="Unknown canonical table",
    ):
        ds.set_eye_table(
            data,
            "does_not_exist",
            pd.DataFrame(),
            validate=False,
        )


def test_dataset_stop_on_validation_error():
    data = ds.new_eye_dataset(validate=False)

    recordings = data["recordings"].copy()

    if recordings.empty:
        recordings = pd.DataFrame(
            {
                "recording_id": [
                    "duplicate",
                    "duplicate",
                ]
            }
        )
    else:
        recordings = pd.concat(
            [
                recordings.iloc[[0]],
                recordings.iloc[[0]],
            ],
            ignore_index=True,
        )

    data["recordings"] = recordings

    with pytest.raises(
        EyeProcessValidationError,
        match="validation failed",
    ):
        ds.validate_eye_dataset(
            data,
            stop_on_error=True,
        )


# =================================================================
# IRT VALIDATION LONG-TAIL
# =================================================================


def _recovery_data():
    return pd.DataFrame(
        {
            "replicate": [
                1,
                2,
                3,
            ],
            "parameter": [
                "difficulty",
                "difficulty",
                "difficulty",
            ],
            "truth": [
                0.0,
                0.0,
                0.0,
            ],
            "estimate": [
                0.1,
                np.nan,
                -0.1,
            ],
        }
    )


def test_irt_recovery_default_columns_are_explicit():
    out = iv.as_irt_recovery_results(_recovery_data())

    assert "converged" in out
    assert "failure_type" in out
    assert "scenario" in out
    assert "engine" in out
    assert "error" in out

    assert out.loc[
        0,
        "converged",
    ]

    assert not out.loc[
        1,
        "converged",
    ]

    assert set(out["scenario"]) == {"baseline"}

    assert set(out["engine"]) == {"unspecified"}


def test_irt_failure_taxonomy_all_major_categories():
    messages = [
        "singular boundary fit",
        "optimizer convergence failed",
        "divergent treedepth warning",
        "rank deficient identifiability",
        "numerical overflow nan",
        "cannot allocate memory",
        "package foo not installed",
        "ordinary unrelated failure",
    ]

    out = iv.validation_failure_taxonomy(messages)

    mapping = dict(
        zip(
            out["message"],
            out["failure_type"],
        )
    )

    assert mapping["singular boundary fit"] == "singular_fit"

    assert mapping["optimizer convergence failed"] == "nonconvergence"

    assert mapping["divergent treedepth warning"] == "bayesian_sampling"

    assert mapping["rank deficient identifiability"] == "identifiability"

    assert mapping["numerical overflow nan"] == "numerical"

    assert mapping["cannot allocate memory"] == "resource"

    assert mapping["package foo not installed"] == "dependency"

    assert mapping["ordinary unrelated failure"] == "other"


def test_irt_stress_mapping_and_generated_scenario_paths():
    def runner(
        scenario,
        replicate,
    ):
        return pd.DataFrame({"value": [float(replicate)]})

    mapped = iv.stress_test_misspecification(
        {
            "normal": 1,
            "heavy": 2,
        },
        runner,
        replications=1,
        seed=3,
    )

    assert set(mapped["scenario"]) == {
        "normal",
        "heavy",
    }

    unnamed = iv.stress_test_misspecification(
        pd.DataFrame(
            {
                "value": [
                    1,
                    2,
                ]
            }
        ),
        runner,
        replications=1,
        seed=3,
    )

    assert set(unnamed["scenario"]) == {
        "scenario_1",
        "scenario_2",
    }


def test_irt_stress_failure_is_retained_not_dropped():
    def failing_runner(
        scenario,
        replicate,
    ):
        raise MemoryError("cannot allocate memory")

    out = iv.stress_test_misspecification(
        {"failure": 1},
        failing_runner,
        replications=1,
        seed=4,
    )

    assert bool(
        out.loc[
            0,
            "failed",
        ]
    )

    assert (
        out.loc[
            0,
            "failure_type",
        ]
        == "resource"
    )


def test_irt_preprocessing_variant_routes():
    def runner(
        scenario,
        replicate,
    ):
        return pd.DataFrame({"value": [replicate]})

    string_variants = iv.stress_test_preprocessing(
        runner,
        [
            "raw",
            "filtered",
        ],
        replications=1,
        seed=5,
    )

    assert set(string_variants["scenario"]) == {
        "raw",
        "filtered",
    }

    frame_variants = iv.stress_test_preprocessing(
        runner,
        pd.DataFrame(
            {
                "setting": [
                    "a",
                    "b",
                ]
            }
        ),
        replications=1,
        seed=5,
    )

    assert set(frame_variants["scenario"]) == {
        "preprocess_1",
        "preprocess_2",
    }


def test_transportability_empty_metric_contract():
    validation = pd.DataFrame(
        {
            "metric": [
                np.nan,
                np.nan,
            ]
        }
    )

    out = iv.audit_measurement_transportability(
        validation,
        metric="metric",
        max_range=1,
        minimum=0,
        maximum=1,
    )

    row = out.iloc[0]

    assert row["n_groups"] == 0

    assert np.isnan(row["mean"])

    assert not bool(row["pass"])


# =================================================================
# MEASUREMENT INTELLIGENCE HELPERS / DRIFT
# =================================================================


def test_measurement_intelligence_dataframe_and_required_guards():
    with pytest.raises(
        EyeProcessValidationError,
        match="data frame",
    ):
        mi._df(
            object(),
            "bad",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="Missing required columns",
    ):
        mi._req(
            pd.DataFrame({"a": [1]}),
            [
                "a",
                "b",
            ],
        )


def test_measurement_intelligence_z_constant_and_varying():
    constant = mi._z(
        [
            2,
            2,
            2,
        ]
    )

    assert np.allclose(
        constant,
        0,
    )

    varying = mi._z(
        [
            1,
            2,
            3,
        ]
    )

    assert np.nanmean(varying) == pytest.approx(
        0,
        abs=1e-12,
    )


def test_dif_drift_single_and_multiple_time_points():
    data = pd.DataFrame(
        {
            "time": [
                1,
                2,
                1,
            ],
            "group": [
                "A",
                "A",
                "B",
            ],
            "metric": [
                1.0,
                3.0,
                5.0,
            ],
        }
    )

    out = mi.monitor_dif_drift(
        data,
        time="time",
        group="group",
        metrics=["metric"],
    )

    slopes = out["slopes"]

    group_a = slopes[slopes["group"].eq("A")].iloc[0]

    group_b = slopes[slopes["group"].eq("B")].iloc[0]

    assert group_a["slope"] == pytest.approx(2.0)

    assert np.isnan(group_b["slope"])

    assert group_b["n_time_points"] == 1


# =================================================================
# SPATIAL QUALITY RESIDUALS
# =================================================================


def test_quality_merge_empty_contract():
    out = sq._merge_quality_parts(
        [],
        [],
    )

    assert isinstance(
        out,
        pd.DataFrame,
    )

    assert out.empty


def test_quality_threshold_validation_failure_routes():
    with pytest.raises(
        ValueError,
        match="not present",
    ):
        sq._validate_thresholds(
            {"unknown_metric": 1},
            ["known_metric"],
        )

    with pytest.raises(
        ValueError,
        match="only 'min' and/or 'max'",
    ):
        sq._validate_thresholds(
            {"metric": {"bad": 1}},
            ["metric"],
        )

    with pytest.raises(
        ValueError,
        match="only 'min' and/or 'max'",
    ):
        sq._validate_thresholds(
            {"metric": {}},
            ["metric"],
        )

    with pytest.raises(
        ValueError,
        match="finite numeric",
    ):
        sq._validate_thresholds(
            {"metric": "not-numeric"},
            ["metric"],
        )

    with pytest.raises(
        ValueError,
        match="finite numeric",
    ):
        sq._validate_thresholds(
            {"metric": np.inf},
            ["metric"],
        )


def test_quality_threshold_flags_all_routes():
    row = pd.Series(
        {
            "missing": np.nan,
            "high": 10.0,
            "low": 1.0,
            "scalar": 9.0,
        }
    )

    flags = sq._threshold_flags(
        row,
        {
            "missing": {"max": 1},
            "high": {"max": 5},
            "low": {"min": 2},
            "scalar": 5,
        },
    )

    assert "missing>max" not in flags

    assert "high>max" in flags
    assert "low<min" in flags
    assert "scalar>max" in flags

    assert (
        sq._threshold_flags(
            row,
            None,
        )
        == []
    )


def test_quality_simulator_input_guards():
    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        sq.simulate_gaze_quality_calibration(
            seed="not-a-seed",
        )

    for seed in (
        -1,
        1.5,
        np.inf,
        2_147_483_648,
    ):
        with pytest.raises(
            ValueError,
            match="seed",
        ):
            sq.simulate_gaze_quality_calibration(
                seed=seed,
            )

    for samples in (
        3,
        4.5,
        np.inf,
    ):
        with pytest.raises(
            ValueError,
            match="samples_per_target",
        ):
            sq.simulate_gaze_quality_calibration(
                samples_per_target=samples,
            )

    for sampling_rate in (
        0,
        -1,
        np.inf,
    ):
        with pytest.raises(
            ValueError,
            match="nominal_sampling_hz",
        ):
            sq.simulate_gaze_quality_calibration(
                nominal_sampling_hz=sampling_rate,
            )


# =================================================================
# PROCESS GOVERNANCE GUARDS
# =================================================================


def test_preflight_accessor_requires_preflight_object():
    with pytest.raises(
        EyeProcessValidationError,
        match="eye_biometric_preflight",
    ):
        pg.preflight_decisions({})


def test_signal_filter_guard_paths():
    signal = np.arange(
        9,
        dtype=float,
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="method",
    ):
        pg.filter_eye_signal(
            signal,
            method="mystery",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="Too few finite",
    ):
        pg.filter_eye_signal(
            [
                1.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        )

    with pytest.raises(
        EyeProcessBackendError,
        match="robfilter",
    ):
        pg.filter_eye_signal(
            signal,
            method="robfilter",
        )

    out = pg.filter_eye_signal(
        signal,
        width=4,
        method="runmed",
    )

    assert out.data["filtered"].notna().all()


# =================================================================
# PUPIL MISSINGNESS INTERNAL CONTRACTS
# =================================================================


def test_pupil_missingness_dataframe_and_required_guards():
    with pytest.raises(
        EyeProcessValidationError,
        match="data frame",
    ):
        pm._df(
            object(),
            "bad",
        )

    with pytest.raises(
        EyeProcessValidationError,
        match="Missing required columns",
    ):
        pm._req(
            pd.DataFrame({"a": [1]}),
            [
                "a",
                "b",
            ],
        )


def test_pupil_missingness_axis_passthrough():
    fig, ax = plt.subplots()

    assert pm._ax(ax) is ax

    plt.close(fig)


# =================================================================
# SURVIVAL RESIDUALS
# =================================================================


def _survival_data(
    seed: int = 1801,
):
    return sv.simulate_gaze_survival_example(
        seed=seed,
        n_participants=10,
        trials_per_participant=2,
    )


def test_survival_unsupported_event_type():
    events = pd.DataFrame(
        {
            "start_time": [0.1],
            "aoi_id": ["target"],
            "episode_type": ["fixation"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Unsupported event_type",
    ):
        sv._event_time_for_trial(
            events,
            "target",
            "unsupported_event",
            "start_time",
            "aoi_id",
            "episode_type",
        )


def test_survival_censor_summary_single_group_column():
    data = _survival_data(1802)

    out = sv.summarise_gaze_censoring(
        data,
        by="condition",
    )

    assert not out.empty
    assert "condition" in out
    assert "n_trials" in out
    assert "n_censored" in out


def _aft_params():
    return pd.Series(
        [0.1],
        index=pd.MultiIndex.from_tuples(
            [
                (
                    "lambda_",
                    "C(condition)[T.detail]",
                )
            ]
        ),
    )


def test_aft_harmless_backend_warning_is_propagated(
    monkeypatch,
):
    import lifelines

    class WarningFitter:
        def fit(
            self,
            *args,
            **kwargs,
        ):
            warnings.warn(
                "harmless AFT backend notice",
                UserWarning,
            )

            self.log_likelihood_ = -10.0
            self.params_ = _aft_params()

            return self

    monkeypatch.setattr(
        lifelines,
        "WeibullAFTFitter",
        WarningFitter,
    )

    with pytest.warns(
        UserWarning,
        match="harmless AFT backend notice",
    ):
        fit = sv.fit_gaze_aft_model(
            _survival_data(1803),
            "C(condition)",
            distribution="weibull",
        )

    assert fit.model_family == "aft_weibull"


def test_tidy_aft_rejects_non_multiindex_summary():
    fit = sv.GazeSurvivalFit(
        model_family="aft_weibull",
        backend="fake",
        result=SimpleNamespace(
            summary=pd.DataFrame(
                {
                    "coef": [0.1],
                    "se(coef)": [0.1],
                    "z": [1.0],
                    "p": [0.3],
                },
                index=["x"],
            )
        ),
        data=_survival_data(1804),
    )

    with pytest.raises(
        RuntimeError,
        match="AFT summary contract",
    ):
        sv.tidy_gaze_survival_model(fit)


# =================================================================
# DETECTOR MULTIVERSE RESIDUALS
# =================================================================


def _ivt(
    detector_id: str = "ivt30",
):
    return ep.define_event_detector_spec(
        detector_id,
        "ivt",
        velocity_threshold=30,
        minimum_duration_ms=60,
        maximum_gap_ms=75,
        sampling_rate=60,
        coordinate_unit="degrees",
    )


def test_detector_invalid_coordinate_unit_guard():
    bad = replace(
        _ivt(),
        coordinate_unit="radians",
    )

    with pytest.raises(
        EyeProcessValidationError,
        match="coordinate_unit",
    ):
        dm.validate_event_detector_spec(bad)


def test_detect_events_requires_eye_dataset():
    with pytest.raises(
        EyeProcessValidationError,
        match="EyeDataset",
    ):
        dm.detect_events_with_spec(
            object(),
            _ivt(),
        )


def test_statsmodels_helper_rejects_unknown_engine():
    data = pd.DataFrame(
        {
            "y": [
                1.0,
                2.0,
                3.0,
            ],
            "x": [
                0.0,
                1.0,
                0.0,
            ],
        }
    )

    with pytest.raises(
        (
            EyeProcessValidationError,
            ValueError,
        )
    ):
        dm._fit_statsmodels(
            data,
            {
                "engine": "not_a_model_engine",
                "formula": "y ~ x",
            },
        )


def test_detector_feature_sensitivity_empty_contract():
    out = dm._feature_sensitivity(pd.DataFrame())

    assert isinstance(
        out,
        pd.DataFrame,
    )

    assert out.empty


# =================================================================
# MULTILEVEL MEDIATION VERSION FALLBACK
# =================================================================


def test_multilevel_mediation_development_version_fallback(
    monkeypatch,
):
    def missing_version(
        name,
    ):
        raise mm.PackageNotFoundError(name)

    monkeypatch.setattr(
        mm,
        "version",
        missing_version,
    )

    assert mm._software_version() == "development"
