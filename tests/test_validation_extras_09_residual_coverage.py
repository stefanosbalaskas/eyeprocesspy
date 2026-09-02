from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.validation_extras_09 as ve


def test_private_numeric_frame_column_and_recycle_residual_paths(monkeypatch):
    np.testing.assert_allclose(ve._numeric_vector(pd.Series(["1", "bad"])), [1.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(ve._numeric_vector(np.asarray([1, 2])), [1.0, 2.0])
    np.testing.assert_allclose(ve._numeric_vector("3"), [3.0])
    assert math.isnan(ve._first_numeric([]))

    real_asarray = np.asarray
    calls = 0

    def flaky_asarray(value, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("array conversion failed")
        return real_asarray(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ve.np, "asarray", flaky_asarray)
        assert np.isnan(ve._numeric_vector(object())[0])

    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        ve._as_frame(object(), name="broken")
    with pytest.raises(ep.EyeProcessValidationError, match="Missing required"):
        ve._require_columns(pd.DataFrame({"a": [1]}), ["a", "b"])

    assert ve._recycle("Hz", 3) == ["Hz", "Hz", "Hz"]
    assert ve._recycle(7, 3) == [7, 7, 7]
    assert ve._recycle([], 3) == [None, None, None]
    assert ve._recycle(["a", "b"], 5) == ["a", "b", "a", "b", "a"]


def test_simulation_rank_missing_seed_and_tie_paths():
    assert ep.simulation_rank_statistic(np.nan, [0, 1]) is None
    assert ep.simulation_rank_statistic(0, [np.nan, np.inf]) is None
    with pytest.raises(ep.EyeProcessValidationError, match="seed"):
        ep.simulation_rank_statistic(0, [-1, 0, 1], seed=np.nan)

    tied = ep.simulation_rank_statistic(0, [-1, 0, 0, 1], seed=123)
    assert tied in {1, 2, 3}


def test_sbc_default_bins_empty_ranks_and_raw_ecdf_paths():
    default = ep.sbc_rank_diagnostics([0, 1, 2, 3], n_draws=3)
    assert default["bins"] == 4

    truncated = ep.sbc_rank_diagnostics([0, 1, 2], n_draws=3, bins=2.9)
    assert truncated["bins"] == 2

    with pytest.raises(ep.EyeProcessValidationError, match="n_draws"):
        ep.sbc_rank_diagnostics([0], n_draws=1.5)
    with pytest.raises(ep.EyeProcessValidationError, match="bins"):
        ep.sbc_rank_diagnostics([0], n_draws=2, bins=np.inf)

    empty = ep.sbc_rank_diagnostics([], n_draws=1, bins=2)
    assert empty["ranks"].size == 0
    assert np.isnan(empty["chi_square"])
    assert np.isnan(empty["chi_square_p"])
    assert np.isnan(empty["ecdf_max_deviation"])

    raw = ep.sbc_ecdf_deviation([0, 1, 2], n_draws=2)
    assert np.isfinite(raw)


def test_interval_matrix_all_shapes_and_numeric_coercion():
    series = ve._interval_matrix(pd.Series([1, 2]))
    assert series.shape == (2, 1)
    scalar = ve._interval_matrix(3)
    assert scalar.shape == (1, 1)
    mixed = ve._interval_matrix(pd.DataFrame({"a": ["1", "bad"], "b": [2, 3]}))
    assert mixed.shape == (2, 2)
    assert np.isnan(mixed[1, 0])

    with pytest.raises(ep.EyeProcessValidationError, match="two-dimensional"):
        ve._interval_matrix(np.zeros((2, 2, 2)))


def test_coverage_single_default_and_nominal_validation_paths():
    single = ep.coverage_calibration_curve([0.0], [-1.0], [1.0])
    assert single.loc[0, "nominal"] == pytest.approx(0.95)
    assert single.loc[0, "empirical"] == pytest.approx(1.0)

    for nominal in ([0.5, 0.9], [np.inf], [-0.1], [1.1]):
        with pytest.raises(ep.EyeProcessValidationError, match="nominal"):
            ep.coverage_calibration_curve([0.0], [-1.0], [1.0], nominal=nominal)


def test_measurement_budget_recycle_variants():
    scalar = ep.measurement_error_budget(units=7)
    assert scalar["unit"].tolist() == [7, 7, 7, 7, 7]
    empty = ep.measurement_error_budget(units=[])
    assert empty["unit"].isna().all()
    cycled = ep.measurement_error_budget(units=["deg", "Hz"])
    assert cycled["unit"].tolist() == ["deg", "Hz", "deg", "Hz", "deg"]


def test_resolution_guard_remaining_validation_and_failure_states():
    cases = [
        ({"event_duration_ms": 100, "effective_hz": 0}, "effective_hz"),
        ({"event_duration_ms": 100, "effective_hz": 60, "min_samples": 0}, "min_samples"),
        ({"event_duration_ms": 100, "effective_hz": 60, "max_error_fraction": -0.1}, "max_error_fraction"),
        ({"event_duration_ms": 100, "effective_hz": 60, "spatial_feature_size": 0}, "spatial_feature_size"),
        ({"event_duration_ms": 100, "effective_hz": 60, "spatial_feature_size": 1, "radial_error": -0.1}, "radial_error"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ep.EyeProcessValidationError, match=message):
            ep.analysis_resolution_guard(**kwargs)

    temporal = ep.analysis_resolution_guard(10, 60, min_samples=3)
    assert temporal["temporal_ok"] is False
    assert temporal["overall"] is False

    spatial = ep.analysis_resolution_guard(100, 60, spatial_feature_size=0.1, radial_error=0.2, max_error_fraction=0.5)
    assert spatial["spatial_ok"] is False
    assert spatial["overall"] is False


def test_preprocessing_order_input_coercion_and_pattern_guards():
    string_steps = ep.audit_pupil_preprocessing_order("baseline correction")
    assert string_steps["steps"] == ["baseline correction"]

    with pytest.raises(ep.EyeProcessValidationError, match="steps"):
        ep.audit_pupil_preprocessing_order(1)
    for steps in ([""], [pd.NA]):
        with pytest.raises(ep.EyeProcessValidationError, match="steps"):
            ep.audit_pupil_preprocessing_order(steps)

    string_clean = ep.audit_pupil_preprocessing_order(
        ["blink interpolation", "baseline correction"],
        cleaning_patterns="blink",
    )
    assert string_clean["cleaning_positions"] == [1]

    with pytest.raises(ep.EyeProcessValidationError, match="cleaning_patterns"):
        ep.audit_pupil_preprocessing_order(["baseline"], cleaning_patterns=1)
    for patterns in ([""], [pd.NA]):
        with pytest.raises(ep.EyeProcessValidationError, match="cleaning_patterns"):
            ep.audit_pupil_preprocessing_order(["baseline"], cleaning_patterns=patterns)

    list_baseline = ep.audit_pupil_preprocessing_order(["baseline correction"], baseline_pattern=["baseline", "other"])
    assert list_baseline["baseline_positions"] == [1]
    numeric_baseline = ep.audit_pupil_preprocessing_order(["step 7"], baseline_pattern=7)
    assert numeric_baseline["baseline_positions"] == [1]
    with pytest.raises(ep.EyeProcessValidationError, match="baseline_pattern"):
        ep.audit_pupil_preprocessing_order(["baseline"], baseline_pattern=[])

    duplicate = ep.audit_pupil_preprocessing_order(
        ["blink filter", "baseline"],
        cleaning_patterns=["blink", "filter"],
    )
    assert duplicate["cleaning_positions"] == [1]


def test_normalize_windows_blank_mapping_sequence_and_invalid_inputs():
    blank = ve._normalize_windows({"": [-100, 0], "also": [-50, 0]})
    assert [name for name, _ in blank] == ["W1", "W2"]

    sequence = ve._normalize_windows([[-100, 0], [-50, 0]])
    assert [name for name, _ in sequence] == ["W1", "W2"]

    with pytest.raises(ep.EyeProcessValidationError, match="windows"):
        ve._normalize_windows("bad")
    with pytest.raises(ep.EyeProcessValidationError, match="windows"):
        ve._normalize_windows(7)


def test_pupil_baseline_remaining_validation_group_and_nan_paths():
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "time_ms": [-100.0, 100.0, -100.0, 100.0],
            "pupil": [2.0, 4.0, np.nan, 5.0],
        }
    )

    with pytest.raises(ep.EyeProcessValidationError, match="correction"):
        ep.pupil_baseline_sensitivity(data, windows={"B": [-100, 0]}, correction="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        ep.pupil_baseline_sensitivity(object(), windows={"B": [-100, 0]})
    with pytest.raises(ep.EyeProcessValidationError, match="Missing required"):
        ep.pupil_baseline_sensitivity(data.drop(columns="pupil"), windows={"B": [-100, 0]})

    grouped = ep.pupil_baseline_sensitivity(data, by="group", windows={"B": [-100, 0]})
    assert set(grouped["group"]) == {"A", "B"}
    assert grouped.loc[grouped["group"] == "A", "corrected_post_mean"].iloc[0] == pytest.approx(2.0)
    assert np.isnan(grouped.loc[grouped["group"] == "B", "corrected_post_mean"].iloc[0])

    no_post = ep.pupil_baseline_sensitivity(
        pd.DataFrame({"time_ms": [-100.0, 0.0], "pupil": [2.0, 2.2]}),
        windows={"B": [-100, 0]},
    )
    assert np.isnan(no_post.loc[0, "corrected_post_mean"])

    for window in ([0], [0, np.inf]):
        with pytest.raises(ep.EyeProcessValidationError, match="two finite endpoints"):
            ep.pupil_baseline_sensitivity(data, windows={"bad": window})
