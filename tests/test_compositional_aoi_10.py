from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "aoi_balance_coordinates",
    "compare_aoi_compositions",
    "derive_aoi_composition",
    "fit_aoi_compositional_model",
    "plot_aoi_balance_biplot",
    "plot_aoi_composition_trajectory",
    "plot_aoi_ternary",
    "plot_aoi_variation_matrix",
    "plot_compositional_group_difference",
    "transform_aoi_composition",
]


def _wide_data(seed=1, n=30):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "stem": rng.exponential(size=n),
            "evidence": rng.exponential(size=n),
            "options": rng.exponential(size=n),
            "group": np.repeat(["A", "B"], n // 2),
        }
    )


def _composition(seed=1, n=30):
    data = _wide_data(seed=seed, n=n)
    return ep.derive_aoi_composition(
        data,
        ["stem", "evidence", "options"],
        id_cols=["id", "group"],
    )


def _close(axis):
    import matplotlib.pyplot as plt

    if axis is not None and hasattr(axis, "figure"):
        plt.close(axis.figure)


def test_public_r032_exports_are_callable():
    assert len(TARGETS) == 10
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_frozen_compositional_aoi_contract():
    composition = _composition()

    assert composition.eyeprocess_class == "eye_aoi_composition"
    np.testing.assert_allclose(
        composition["proportions"].sum(axis=1).to_numpy(dtype=float),
        np.ones(30),
        atol=1e-8,
        rtol=0,
    )

    ilr = ep.transform_aoi_composition(composition, "ilr")
    assert ilr.eyeprocess_class == "eye_aoi_logratio"
    assert ilr["transformed"].shape == (30, 2)

    rng = np.random.default_rng(8)
    model_data = pd.DataFrame({"outcome": rng.normal(size=30)})
    model = ep.fit_aoi_compositional_model(
        composition,
        "outcome ~ ilr_1",
        data=model_data,
    )
    assert model.eyeprocess_class == "eye_aoi_composition_model"
    assert "(Intercept)" in model["summary"].index
    assert "ilr_1" in model["summary"].index

    comparison = ep.compare_aoi_compositions(
        composition,
        "group",
        permutations=19,
        seed=20260807,
    )
    assert comparison.eyeprocess_class == "eye_aoi_composition_comparison"
    assert 0 <= comparison["p_value"] <= 1
    assert len(comparison["null"]) == 19

    axis = ep.plot_aoi_ternary(composition)
    assert hasattr(axis, "eyeprocess_plot_data")
    _close(axis)


def test_clr_alr_and_ilr_semantics():
    composition = _composition(n=30)

    clr = ep.transform_aoi_composition(composition, "clr")
    np.testing.assert_allclose(
        clr["transformed"].sum(axis=1).to_numpy(dtype=float),
        np.zeros(30),
        atol=1e-10,
        rtol=0,
    )

    alr = ep.transform_aoi_composition(
        composition,
        "alr",
        reference="options",
    )
    assert list(alr["transformed"].columns) == [
        "alr_stem_vs_options",
        "alr_evidence_vs_options",
    ]

    ilr = ep.transform_aoi_composition(composition, "ilr")
    basis = np.asarray(ilr["basis"], dtype=float)
    np.testing.assert_allclose(
        basis.T @ basis,
        np.eye(2),
        atol=1e-12,
        rtol=0,
    )


def test_wide_zero_replacement_and_trial_duration_closure():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "A": [1.0, 0.0, np.nan],
            "B": [2.0, 3.0, 1.0],
            "C": [0.0, 1.0, 4.0],
            "duration": [10.0, 10.0, 10.0],
        }
    )

    multiplicative = ep.derive_aoi_composition(
        data,
        ["A", "B", "C"],
        id_cols=["id"],
        zero_method="multiplicative",
    )
    bayesian = ep.derive_aoi_composition(
        data,
        ["A", "B", "C"],
        denominator="trial_duration",
        trial_duration_col="duration",
        zero_method="bayesian",
        id_cols=["id"],
    )

    for result in (multiplicative, bayesian):
        assert np.isfinite(result["proportions"].to_numpy(dtype=float)).all()
        assert (result["proportions"].to_numpy(dtype=float) > 0).all()
        np.testing.assert_allclose(
            result["proportions"].sum(axis=1),
            1.0,
            atol=1e-12,
            rtol=0,
        )


def test_long_format_composition_retains_identifiers():
    long = pd.DataFrame(
        {
            "person": ["P1"] * 3 + ["P2"] * 3,
            "aoi": ["stem", "evidence", "options"] * 2,
            "dwell_ms": [100, 200, 50, 80, 100, 220],
            "trial_ms": [500] * 6,
        }
    )

    composition = ep.derive_aoi_composition(
        long,
        ["stem", "evidence", "options"],
        id_cols=["person"],
        aoi_col="aoi",
        value_col="dwell_ms",
        trial_duration_col="trial_ms",
    )

    assert list(composition["table"]["person"]) == ["P1", "P2"]
    assert composition["proportions"].shape == (2, 3)


def test_balance_coordinates_support_named_and_matrix_contracts():
    composition = _composition()

    named = ep.aoi_balance_coordinates(
        composition,
        {
            "stem_vs_rest": {
                "numerator": ["stem"],
                "denominator": ["evidence", "options"],
            }
        },
    )
    assert list(named.columns) == ["stem_vs_rest"]
    assert len(named) == 30

    contrast = pd.DataFrame(
        {
            "contrast_1": [1.0, -0.5, -0.5],
            "contrast_2": [0.0, 1.0, -1.0],
        },
        index=["stem", "evidence", "options"],
    )
    matrix = ep.aoi_balance_coordinates(composition, contrast)
    assert list(matrix.columns) == ["contrast_1", "contrast_2"]


def test_composition_comparison_is_seeded_and_returns_centroids():
    composition = _composition()

    first = ep.compare_aoi_compositions(
        composition,
        "group",
        permutations=19,
        seed=11,
    )
    second = ep.compare_aoi_compositions(
        composition,
        "group",
        permutations=19,
        seed=11,
    )

    np.testing.assert_allclose(first["null"], second["null"])
    assert list(first["centroids"]["group"]) == ["A", "B"]
    assert first["summary"]["permutations"].iloc[0] == 19


def test_fixed_effects_model_records_random_without_silently_fitting_it():
    composition = _composition()
    data = _wide_data().loc[:, ["group"]].copy()
    data["outcome"] = np.linspace(-1.0, 1.0, len(data))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ep.fit_aoi_compositional_model(
            composition,
            "outcome ~ ilr_1 + group",
            random="~ 1 | group",
            data=data,
        )

    assert model["random"] == "~ 1 | group"
    assert any("fixed-effects model" in str(item.message) for item in caught)
    assert np.isfinite(model["model"].fittedvalues).all()


def test_all_frozen_public_compositional_plots_execute():
    composition = _composition()
    comparison = ep.compare_aoi_compositions(
        composition,
        "group",
        permutations=9,
    )

    axes = [
        ep.plot_aoi_ternary(composition),
        ep.plot_aoi_balance_biplot(composition),
        ep.plot_aoi_variation_matrix(composition),
        ep.plot_compositional_group_difference(comparison),
        ep.plot_aoi_composition_trajectory(composition),
    ]

    for axis in axes:
        assert hasattr(axis, "eyeprocess_plot_data")
        _close(axis)


def test_validation_boundaries_are_explicit():
    with pytest.raises(ep.EyeProcessValidationError):
        ep.derive_aoi_composition(
            pd.DataFrame({"A": [1.0]}),
            ["A"],
        )

    composition = _composition()
    with pytest.raises(ep.EyeProcessValidationError, match="reference"):
        ep.transform_aoi_composition(
            composition,
            "alr",
            reference="missing",
        )

    with pytest.raises(ep.EyeProcessValidationError, match="permutations"):
        ep.compare_aoi_compositions(
            composition,
            "group",
            permutations=0,
        )

    with pytest.raises(ep.EyeProcessValidationError):
        ep.fit_aoi_compositional_model(
            composition,
            "missing ~ ilr_1",
        )
