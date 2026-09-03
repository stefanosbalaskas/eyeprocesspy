from __future__ import annotations

import builtins
import warnings

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.compositional_aoi_10 as mod


def _wide_three(n=6):
    return pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(n)],
            "group": ["a", "a", "a", "b", "b", "b"][:n],
            "A": np.linspace(10, 20, n),
            "B": np.linspace(20, 10, n),
            "C": np.linspace(5, 15, n),
            "duration": [100.0] * n,
            "x": np.arange(n, dtype=float),
            "cat": ["u", "v"] * (n // 2) + (["u"] if n % 2 else []),
            "y": np.linspace(1, 3, n),
        }
    )


def _composition(n=6):
    data = _wide_three(n)
    return ep.derive_aoi_composition(
        data,
        ["A", "B", "C"],
        id_cols=["id", "group"],
    )


def test_private_frame_class_numeric_and_mean_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="data frame"):
        mod._require_frame([], name="x")
    with pytest.raises(ep.EyeProcessValidationError, match="at least one row"):
        mod._require_frame(pd.DataFrame(), name="x")
    with pytest.raises(ep.EyeProcessValidationError, match="eye_aoi_composition"):
        mod._require_class(object(), "eye_aoi_composition")

    with pytest.raises(ep.EyeProcessValidationError, match="two-dimensional"):
        mod._as_numeric_frame([1, 2, 3])
    with pytest.raises(ep.EyeProcessValidationError, match="at least two"):
        mod._as_numeric_frame(pd.DataFrame({"A": [1, 2]}))

    frame = mod._as_numeric_frame(
        [[1, "bad"], [2, 3]],
        columns=["left", "right"],
    )
    assert list(frame.columns) == ["left", "right"]
    assert np.isnan(frame.loc[0, "right"])
    assert np.isnan(mod._safe_mean([None, "bad", np.nan]))
    assert mod._safe_mean([1, "bad", 3]) == 2.0


def test_close_composition_zero_methods_and_ilr_guard():
    with pytest.raises(ep.EyeProcessValidationError, match="zero_method"):
        mod._close_composition([[1, 2], [2, 1]], zero_method="bad")

    raw = pd.DataFrame(
        {
            "A": [0.0, np.nan, -1.0],
            "B": [2.0, 0.0, np.nan],
            "C": [3.0, 4.0, 0.0],
        }
    )
    mult = mod._close_composition(raw, zero_method="multiplicative")
    bayes = mod._close_composition(raw, zero_method="bayesian")
    assert np.allclose(mult.sum(axis=1), 1.0)
    assert np.allclose(bayes.sum(axis=1), 1.0)
    assert (mult.to_numpy() > 0).all()
    assert (bayes.to_numpy() > 0).all()

    with pytest.raises(ep.EyeProcessValidationError, match="at least two"):
        mod._ilr_basis(1)
    basis = mod._ilr_basis(4)
    assert basis.shape == (4, 3)
    assert np.allclose(basis.T @ basis, np.eye(3))


def test_derive_argument_guards_and_empty_tuple_defaults():
    x = _wide_three()
    with pytest.raises(ep.EyeProcessValidationError, match="denominator"):
        ep.derive_aoi_composition(x, ["A", "B"], denominator="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="zero_method"):
        ep.derive_aoi_composition(x, ["A", "B"], zero_method="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="at least two"):
        ep.derive_aoi_composition(x, ["A"])

    out = ep.derive_aoi_composition(
        x,
        ["A", "B", "C"],
        denominator=[],
        zero_method=[],
        id_cols=["id", "missing"],
    )
    assert out["denominator"] == "total_aoi_dwell"
    assert out["zero_method"] == "multiplicative"
    assert out["id_cols"] == ["id"]


def test_derive_long_missing_columns_and_automatic_row_id():
    bad = pd.DataFrame({"aoi": ["A", "B"]})
    with pytest.raises(ep.EyeProcessValidationError, match="required long-format"):
        ep.derive_aoi_composition(bad, ["A", "B"])

    long = pd.DataFrame(
        {
            "aoi": ["A", "B", "A", "B"],
            "dwell_ms": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = ep.derive_aoi_composition(long, ["A", "B"])
    assert out["id_cols"] == [".composition_row_id"]
    assert len(out["raw"]) == 4


def test_derive_long_multi_id_duration_and_trial_fallbacks():
    long = pd.DataFrame(
        {
            "participant": ["p1"] * 4 + ["p2"] * 4,
            "trial": [1] * 4 + [2] * 4,
            "aoi": ["A", "B", "C", "A"] * 2,
            "dwell_ms": [10, 20, 30, 5, 15, 10, 5, 5],
            "trial_duration": [100.0] * 4 + [np.nan] * 4,
        }
    )
    out = ep.derive_aoi_composition(
        long,
        ["A", "B", "C"],
        denominator="trial_duration",
        zero_method="bayesian",
        id_cols=["participant", "trial"],
        trial_duration_col="trial_duration",
    )
    assert out["id_cols"] == ["participant", "trial"]
    assert len(out["proportions"]) == 2
    assert np.allclose(out["proportions"].sum(axis=1), 1.0)

    wide = _wide_three()
    wide.loc[0, "duration"] = 0.0
    wide.loc[1, "duration"] = np.nan
    wide_out = ep.derive_aoi_composition(
        wide,
        ["A", "B", "C"],
        denominator="trial_duration",
        id_cols=["id"],
        trial_duration_col="duration",
    )
    assert np.allclose(wide_out["proportions"].sum(axis=1), 1.0)


def test_transform_all_methods_and_reference_guards():
    comp = _composition()
    with pytest.raises(ep.EyeProcessValidationError, match="method"):
        ep.transform_aoi_composition(comp, method="bad")

    clr = ep.transform_aoi_composition(comp, method=("clr", "ilr"))
    assert clr["basis"] is None
    assert list(clr["transformed"].columns) == ["clr_A", "clr_B", "clr_C"]

    alr = ep.transform_aoi_composition(comp, method="alr")
    assert alr["basis"] == "C"
    assert alr["transformed"].shape[1] == 2

    alr_a = ep.transform_aoi_composition(comp, method="alr", reference="A")
    assert alr_a["basis"] == "A"
    with pytest.raises(ep.EyeProcessValidationError, match="not an AOI part"):
        ep.transform_aoi_composition(comp, method="alr", reference="missing")

    direct = ep.transform_aoi_composition(
        pd.DataFrame({"A": [1, 2], "B": [2, 1]}),
        method="ilr",
    )
    assert direct["transformed"].shape == (2, 1)


def test_formula_expansion_branches():
    intercept, terms = mod._expand_formula_terms("1 + x + x + C(cat)")
    assert intercept is True
    assert terms == ["x", "C(cat)"]

    intercept, terms = mod._expand_formula_terms("0 + x * C(cat)")
    assert intercept is False
    assert terms == ["x", "C(cat)", "x:C(cat)"]

    intercept2, terms2 = mod._expand_formula_terms("-1 + x")
    assert intercept2 is False and terms2 == ["x"]

    with pytest.raises(ep.EyeProcessValidationError, match="two-way"):
        mod._expand_formula_terms("x * y * z")


def test_design_for_term_numeric_categorical_interaction_and_guards():
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
            "mixed": ["1", "x", "2", "y"],
        }
    )
    numeric = mod._design_for_term(data, "x")
    assert list(numeric.columns) == ["x"]

    categorical = mod._design_for_term(data, "C(cat)")
    assert categorical.shape[1] == 1

    automatic_cat = mod._design_for_term(data, "mixed")
    assert automatic_cat.shape[1] >= 1

    interaction = mod._design_for_term(data, "x:C(cat)")
    assert all(":" in name for name in interaction.columns)

    with pytest.raises(ep.EyeProcessValidationError, match="unavailable"):
        mod._design_for_term(data, "C(missing)")
    with pytest.raises(ep.EyeProcessValidationError, match="unavailable"):
        mod._design_for_term(data, "missing")


def test_fit_formula_ols_guards_and_no_intercept_model(monkeypatch):
    data = _wide_three()
    with pytest.raises(ep.EyeProcessValidationError, match="R-style strings"):
        mod._fit_formula_ols(None, data)
    with pytest.raises(ep.EyeProcessValidationError, match="response"):
        mod._fit_formula_ols("missing ~ x", data)
    with pytest.raises(ep.EyeProcessValidationError, match="no design columns"):
        mod._fit_formula_ols("y ~ 0", data)

    tiny = pd.DataFrame({"y": [1.0, np.nan], "x": [1.0, 2.0]})
    with pytest.raises(ep.EyeProcessValidationError, match="Insufficient"):
        mod._fit_formula_ols("y ~ x", tiny)

    fit = mod._fit_formula_ols("y ~ 0 + x + C(cat)", data)
    assert "(Intercept)" not in fit.design.columns
    assert fit.df_resid > 0
    assert fit.coef().equals(fit.params)

    original = np.linalg.lstsq

    def fake_lstsq(X, y, rcond=None):
        coef, residuals, rank, s = original(X, y, rcond=rcond)
        return coef, residuals, len(y), s

    monkeypatch.setattr(mod.np.linalg, "lstsq", fake_lstsq)
    edge = mod._fit_formula_ols("y ~ 1", data)
    assert edge.df_resid == 0
    assert edge["sigma2"] if False else np.isnan(edge.sigma2)
    assert edge.summary_table["Pr(>|t|)"].isna().all()


def test_fit_compositional_model_data_and_random_paths():
    comp = _composition()
    transformed = ep.transform_aoi_composition(comp, "ilr")

    with pytest.raises(ep.EyeProcessValidationError, match="data frame"):
        ep.fit_aoi_compositional_model(transformed, "ilr_1 ~ ilr_2", data=[])
    with pytest.raises(ep.EyeProcessValidationError, match="one row"):
        ep.fit_aoi_compositional_model(
            transformed,
            "ilr_1 ~ ilr_2",
            data=pd.DataFrame({"x": [1, 2]}),
        )

    with pytest.warns(RuntimeWarning, match="random"):
        model = ep.fit_aoi_compositional_model(
            transformed,
            "ilr_1 ~ ilr_2",
            random="(1|participant)",
        )
    assert model.eyeprocess_class == "eye_aoi_composition_model"
    assert "Estimate" in model["summary"].columns

    data = pd.DataFrame({"covariate": np.arange(len(comp["proportions"]))})
    model2 = ep.fit_aoi_compositional_model(
        comp,
        "ilr_1 ~ covariate",
        data=data,
        method="ilr",
    )
    assert "covariate" in model2["model"].params.index


def test_composition_pseudo_f_and_compare_guards():
    comp = _composition()
    with pytest.raises(ep.EyeProcessValidationError, match="eye_aoi_composition"):
        ep.compare_aoi_compositions(object(), ["a", "b"])

    with pytest.raises(ep.EyeProcessValidationError, match="method"):
        ep.compare_aoi_compositions(comp, comp["table"]["group"], method="bad")

    with pytest.raises(ep.EyeProcessValidationError, match="one value"):
        ep.compare_aoi_compositions(comp, ["a", "b"])
    with pytest.raises(ep.EyeProcessValidationError, match="at least two groups"):
        ep.compare_aoi_compositions(comp, ["a"] * len(comp["proportions"]))
    with pytest.raises(ep.EyeProcessValidationError, match="one value"):
        ep.compare_aoi_compositions(comp, "literal_missing_group")

    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        ep.compare_aoi_compositions(comp, comp["table"]["group"], permutations="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="positive integer"):
        ep.compare_aoi_compositions(comp, comp["table"]["group"], permutations=0)

    out = ep.compare_aoi_compositions(
        comp,
        "group",
        method=("compositional_manova", "permanova"),
        permutations=3,
        seed=1,
    )
    assert out["method"] == "compositional_manova"
    assert len(out["null"]) == 3
    assert 0 <= out["p_value"] <= 1


def test_balance_matrix_sequence_mapping_and_guards():
    comp = _composition()

    matrix_df = pd.DataFrame(
        {"custom": [1.0, -0.5, -0.5]},
        index=["A", "B", "C"],
    )
    df_out = ep.aoi_balance_coordinates(comp, matrix_df)
    assert list(df_out.columns) == ["custom"]

    matrix = np.array([[1.0], [-0.5], [-0.5]])
    arr_out = ep.aoi_balance_coordinates(comp, matrix)
    assert list(arr_out.columns) == ["balance_1"]

    with pytest.raises(ep.EyeProcessValidationError, match="one row per AOI"):
        ep.aoi_balance_coordinates(comp, np.array([1.0, -1.0]))
    with pytest.raises(ep.EyeProcessValidationError, match="named mapping"):
        ep.aoi_balance_coordinates(comp, "A/B")
    with pytest.raises(ep.EyeProcessValidationError, match="at least one"):
        ep.aoi_balance_coordinates(comp, {})

    mapping = {
        "A_vs_BC": {"numerator": "A", "denominator": ["B", "C"]},
        "AB_vs_C": (["A", "B"], "C"),
    }
    mapped = ep.aoi_balance_coordinates(comp, mapping)
    assert list(mapped.columns) == ["A_vs_BC", "AB_vs_C"]

    seq = ep.aoi_balance_coordinates(comp, [("A", "B"), (["A", "C"], ["B"])])
    assert seq.shape[1] == 2

    with pytest.raises(ep.EyeProcessValidationError, match="numerator and denominator"):
        ep.aoi_balance_coordinates(comp, {"bad": "A"})
    with pytest.raises(ep.EyeProcessValidationError, match="valid numerator"):
        ep.aoi_balance_coordinates(
            comp,
            {"bad": {"numerator": ["missing"], "denominator": ["B"]}},
        )

    direct = ep.aoi_balance_coordinates(
        pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]}),
        {"A_vs_B": ("A", "B")},
    )
    assert direct.shape == (2, 1)


def test_matplotlib_backend_guard_and_axis_passthrough(monkeypatch):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ep.EyeProcessBackendError, match="matplotlib"):
        mod._get_plt()

    sentinel = object()
    assert mod._axis(sentinel) is sentinel


def test_plot_class_guards_and_ternary_paths():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    with pytest.raises(ep.EyeProcessValidationError, match="eye_aoi_composition"):
        ep.plot_aoi_ternary(object())

    two = ep.derive_aoi_composition(
        pd.DataFrame({"A": [1, 2], "B": [2, 1]}),
        ["A", "B"],
    )
    fig, ax = plt.subplots()
    out = ep.plot_aoi_ternary(two, ax=ax)
    assert out is ax
    assert len(out.get_xticks()) == 0
    plt.close(fig)

    comp = _composition()
    fig, ax = plt.subplots()
    out = ep.plot_aoi_ternary(comp, ax=ax)
    assert out is ax
    assert hasattr(out, "eyeprocess_plot_data")
    plt.close(fig)


def test_balance_biplot_single_dimension_and_zero_scale(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fake = mod._result(
        "eye_aoi_composition",
        proportions=pd.DataFrame({"A": [1.0, 1.0, 1.0]}),
        table=pd.DataFrame(index=range(3)),
    )

    real_svd = mod.np.linalg.svd

    def one_dim_svd(matrix, full_matrices=False):
        return (
            np.ones((len(matrix), 1)),
            np.array([0.0]),
            np.array([[1.0]]),
        )

    monkeypatch.setattr(mod.np.linalg, "svd", one_dim_svd)
    fig, ax = plt.subplots()
    out = ep.plot_aoi_balance_biplot(fake, ax=ax)
    assert out.eyeprocess_plot_data.shape == (3, 2)
    plt.close(fig)
    monkeypatch.setattr(mod.np.linalg, "svd", real_svd)


def test_variation_group_difference_and_trajectory_edge_paths():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    one_row = ep.derive_aoi_composition(
        pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}),
        ["A", "B", "C"],
    )
    fig, ax = plt.subplots()
    variation = ep.plot_aoi_variation_matrix(one_row, ax=ax)
    assert np.isnan(variation.eyeprocess_plot_matrix[0, 1])
    plt.close(fig)

    bad_comparison = mod._result(
        "eye_aoi_composition_comparison",
        centroids=pd.DataFrame({"A": [0.5], "B": [0.5]}),
    )
    with pytest.raises(ep.EyeProcessValidationError, match="group"):
        ep.plot_compositional_group_difference(bad_comparison)

    no_parts = mod._result(
        "eye_aoi_composition_comparison",
        centroids=pd.DataFrame({"group": ["a", "b"]}),
    )
    fig, ax = plt.subplots()
    out = ep.plot_compositional_group_difference(no_parts, ax=ax)
    assert out is ax
    plt.close(fig)

    comp = _composition()
    comparison = ep.compare_aoi_compositions(
        comp,
        "group",
        permutations=2,
        seed=2,
    )
    fig, ax = plt.subplots()
    out = ep.plot_compositional_group_difference(comparison, ax=ax)
    assert out is ax
    plt.close(fig)

    fig, ax = plt.subplots()
    trajectory = ep.plot_aoi_composition_trajectory(comp, ax=ax)
    assert trajectory is ax
    assert hasattr(trajectory, "eyeprocess_plot_data")
    plt.close(fig)
