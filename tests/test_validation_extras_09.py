from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "analysis_resolution_guard",
    "audit_pupil_preprocessing_order",
    "coverage_calibration_curve",
    "measurement_error_budget",
    "pupil_baseline_sensitivity",
    "sbc_ecdf_deviation",
    "sbc_rank_diagnostics",
    "simulation_rank_statistic",
]


def test_public_r081_exports_are_callable():
    assert len(TARGETS) == 8
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_frozen_sbc_and_resolution_audits_are_descriptive_and_deterministic():
    assert (
        ep.simulation_rank_statistic(
            0,
            [-1, 1],
        )
        == 1
    )

    sbc = ep.sbc_rank_diagnostics(
        np.tile(np.arange(10), 10),
        n_draws=9,
        bins=10,
    )
    assert sbc["eyeprocess_class"] == "eye_sbc_diagnostics"
    assert np.isfinite(ep.sbc_ecdf_deviation(sbc))

    sbc_uneven = ep.sbc_rank_diagnostics(
        np.tile(np.arange(10), 6),
        n_draws=9,
        bins=6,
    )
    assert float(np.sum(sbc_uneven["expected_count"])) == pytest.approx(len(sbc_uneven["ranks"]))

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="bins",
    ):
        ep.sbc_rank_diagnostics(
            [0, 1, 2],
            n_draws=2,
            bins=1,
        )

    guard = ep.analysis_resolution_guard(
        100,
        60,
        spatial_feature_size=0.2,
        radial_error=0.04,
    )
    assert guard["eyeprocess_class"] == "eye_analysis_resolution_guard"
    assert guard["expected_samples"] == pytest.approx(6.0)
    assert guard["temporal_ok"] is True
    assert guard["spatial_error_fraction"] == pytest.approx(0.2)
    assert guard["spatial_ok"] is True
    assert guard["overall"] is True

    order = ep.audit_pupil_preprocessing_order(
        [
            "blink interpolation",
            "artifact filter",
            "baseline correction",
        ]
    )
    assert order["status"] == "pass"


def _pupil_data():
    return pd.DataFrame(
        {
            "person_id": np.repeat([1, 2], 8),
            "trial_id": np.tile(
                np.repeat([1, 2], 4),
                2,
            ),
            "time_ms": np.tile(
                [-300, -100, 100, 300],
                4,
            ),
            "pupil": np.tile(
                [3.0, 3.1, 3.2, 3.3],
                4,
            ),
        }
    )


def test_frozen_pupil_baseline_sensitivity_contract():
    data = _pupil_data()

    sensitivity = ep.pupil_baseline_sensitivity(
        data,
        by=["person_id", "trial_id"],
        windows={
            "W300": [-300, 0],
            "W150": [-150, 0],
        },
    )
    assert sensitivity.attrs["eyeprocess_class"] == "eye_pupil_baseline_sensitivity"
    assert len(sensitivity) == 8
    assert set(sensitivity["window"].unique()) == {"W300", "W150"}

    zero_baseline = data.copy()
    zero_baseline.loc[
        zero_baseline["time_ms"] < 0,
        "pupil",
    ] = 0

    divisive = ep.pupil_baseline_sensitivity(
        zero_baseline,
        by=["person_id", "trial_id"],
        windows={"W": [-300, 0]},
        correction="divisive",
    )
    assert divisive["corrected_post_mean"].isna().all()


def test_frozen_calibration_extras_handle_degenerate_evidence():
    coverage = ep.coverage_calibration_curve(
        [np.nan, np.nan],
        [0, 0],
        [1, 1],
        nominal=0.95,
    )
    assert np.isnan(coverage.loc[0, "empirical"])

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="seed",
    ):
        ep.simulation_rank_statistic(
            0,
            [0, 1],
            seed=-2,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="steps",
    ):
        ep.audit_pupil_preprocessing_order([])

    budget = ep.measurement_error_budget(
        accuracy=[1, 2],
        precision=3,
    )
    assert len(budget) == 5
    assert budget.loc[0, "value"] == pytest.approx(1.0)


def test_sbc_histogram_matches_r_right_closed_breaks():
    # n_draws=2, bins=2 -> breaks [-0.5, 1.0, 2.5].
    # R hist(right=TRUE) assigns rank 1 to the first bin.
    diagnostics = ep.sbc_rank_diagnostics(
        [0, 1, 2],
        n_draws=2,
        bins=2,
    )
    np.testing.assert_array_equal(
        diagnostics["counts"],
        [2, 1],
    )
    np.testing.assert_array_equal(
        diagnostics["expected_count"],
        [2.0, 1.0],
    )


def test_coverage_treats_infinity_as_comparable_not_missing():
    coverage = ep.coverage_calibration_curve(
        [np.inf],
        [0.0],
        [np.inf],
        nominal=0.95,
    )
    assert coverage.loc[0, "empirical"] == pytest.approx(1.0)


def test_coverage_defaults_and_multicolumn_semantics():
    truth = [0.0, 1.0]
    lower = np.asarray(
        [
            [-1.0, -0.5],
            [0.5, 0.5],
        ]
    )
    upper = np.asarray(
        [
            [1.0, 0.2],
            [1.5, 0.8],
        ]
    )

    coverage = ep.coverage_calibration_curve(
        truth,
        lower,
        upper,
    )
    np.testing.assert_allclose(
        coverage["nominal"].to_numpy(dtype=float),
        [0.5, 0.95],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        coverage["empirical"].to_numpy(dtype=float),
        [1.0, 0.5],
        atol=1e-12,
        rtol=0,
    )


def test_measurement_error_budget_preserves_noncollapsed_components_and_units():
    budget = ep.measurement_error_budget(
        accuracy=0.2,
        precision=0.1,
        data_loss=0.05,
        effective_hz=58.0,
        calibration_drift=0.03,
        units=["deg", "deg", "prop", "Hz", "deg"],
    )

    assert list(budget["component"]) == [
        "accuracy_error",
        "precision_error",
        "data_loss",
        "effective_sampling_hz",
        "calibration_drift",
    ]
    assert list(budget["direction"]) == [
        "lower_better",
        "lower_better",
        "lower_better",
        "context_dependent",
        "lower_better",
    ]
    assert list(budget["unit"]) == [
        "deg",
        "deg",
        "prop",
        "Hz",
        "deg",
    ]
    assert "should not be collapsed" in budget.attrs["interpretation"]


def test_resolution_guard_missing_spatial_evidence_keeps_temporal_decision():
    guard = ep.analysis_resolution_guard(
        50,
        100,
    )
    assert guard["expected_samples"] == pytest.approx(5.0)
    assert guard["temporal_ok"] is True
    assert guard["spatial_ok"] is None
    assert np.isnan(guard["spatial_error_fraction"])
    assert guard["overall"] is True


def test_preprocessing_order_detects_late_cleaning_and_missing_baseline():
    late = ep.audit_pupil_preprocessing_order(
        [
            "blink interpolation",
            "baseline correction",
            "artifact filter",
        ]
    )
    assert late["baseline_positions"] == [2]
    assert late["cleaning_positions"] == [1, 3]
    assert late["cleaning_after_baseline"] == ["artifact filter"]
    assert late["status"] == "review"

    no_baseline = ep.audit_pupil_preprocessing_order(["blink interpolation", "artifact filter"])
    assert no_baseline["status"] == "baseline_not_declared"


def test_pupil_baseline_uses_inclusive_baseline_and_strictly_post_window():
    data = pd.DataFrame(
        {
            "time_ms": [-100, 0, 100],
            "pupil": [2.0, 4.0, 8.0],
        }
    )
    result = ep.pupil_baseline_sensitivity(
        data,
        windows={"B": [-100, 0]},
    )
    assert result.loc[0, "baseline"] == pytest.approx(3.0)
    assert result.loc[0, "corrected_post_mean"] == pytest.approx(5.0)

    divisive = ep.pupil_baseline_sensitivity(
        data,
        windows={"B": [-100, 0]},
        correction="divisive",
    )
    assert divisive.loc[
        0,
        "corrected_post_mean",
    ] == pytest.approx(8.0 / 3.0)


def test_validation_boundaries_are_explicit():
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="n_draws",
    ):
        ep.sbc_rank_diagnostics([0], 0)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="integer-valued",
    ):
        ep.sbc_rank_diagnostics([0.5], 5)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="between 0 and n_draws",
    ):
        ep.sbc_rank_diagnostics([6], 5)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="same dimensions",
    ):
        ep.coverage_calibration_curve(
            [0, 1],
            [0, 0],
            np.zeros((2, 2)),
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="truth length",
    ):
        ep.coverage_calibration_curve(
            [0],
            [0, 0],
            [1, 1],
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="positive",
    ):
        ep.analysis_resolution_guard(0, 60)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="cleaning_patterns",
    ):
        ep.audit_pupil_preprocessing_order(
            ["baseline"],
            cleaning_patterns=[],
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="baseline_pattern",
    ):
        ep.audit_pupil_preprocessing_order(
            ["baseline"],
            baseline_pattern="",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="windows",
    ):
        ep.pupil_baseline_sensitivity(
            pd.DataFrame(
                {
                    "time_ms": [0],
                    "pupil": [1.0],
                }
            ),
            windows=[],
        )
