from __future__ import annotations

import builtins

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.measurement_intelligence as mi


def _array_pareto():
    items = pd.DataFrame({"raw": [10.0, 20.0, 30.0]})
    spec = ep.item_objective_spec(
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
    )
    return items, spec, ep.item_pareto_front(items, spec)


def test_private_frame_and_empty_fit_residual_paths():
    out = mi._df({"a": [1, 2]})
    assert out["a"].tolist() == [1, 2]

    with pytest.raises(ep.EyeProcessValidationError, match="must be a data frame"):
        mi._df(object())

    x = np.array([[1.0, np.nan]])
    y = np.array([1.0])
    assert mi._linear_fit(x, y)["n"] == 0
    assert mi._logistic_fit(x, y)["n"] == 0


def test_device_linking_validation_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="invalid linking method"):
        ep.fit_device_linking(pd.DataFrame(), "metric", "ref", method="bad")

    with pytest.raises(ep.EyeProcessValidationError, match="pairing identifier"):
        ep.fit_device_linking(
            pd.DataFrame({"metric": [1.0], "device": ["ref"]}),
            "metric",
            "ref",
            id_cols=("person_id",),
        )

    with pytest.raises(ep.EyeProcessValidationError, match="reference_device"):
        ep.fit_device_linking(
            pd.DataFrame(
                {"person_id": [1], "metric": [1.0], "device": ["candidate"]}
            ),
            "metric",
            "ref",
            id_cols=("person_id",),
        )

    with pytest.raises(ep.EyeProcessValidationError, match="eye_device_linking"):
        ep.apply_device_linking(
            pd.DataFrame({"metric": [1.0], "device": ["candidate"]}),
            {},
        )

    with pytest.raises(ep.EyeProcessValidationError, match="positive scalar"):
        ep.audit_device_equivalence({}, 0.0)
    with pytest.raises(ep.EyeProcessValidationError, match="eye_device_linking"):
        ep.audit_device_equivalence({}, 1.0)
    with pytest.raises(ep.EyeProcessValidationError, match="eye_device_linking"):
        ep.estimate_device_specific_error({})


def test_array_objectives_constraints_and_optimizer_residual_paths():
    items, spec, pareto = _array_pareto()
    assert pareto.table["item_id"].tolist() == ["item_1", "item_2", "item_3"]

    bad = ep.item_objective_spec(
        [1.0, 2.0],
        [3.0, 2.0, 1.0],
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
    )
    with pytest.raises(ep.EyeProcessValidationError, match="one value per item"):
        ep.item_pareto_front(items, bad)
    with pytest.raises(ep.EyeProcessValidationError, match="eye_item_objective_spec"):
        ep.item_pareto_front(items, {})

    callable_mask = mi._constraint_mask(
        pareto.table,
        lambda t: np.arange(len(t)) != 0,
    )
    assert callable_mask.tolist() == [False, True, True]

    mapped = mi._constraint_mask(
        pareto.table,
        {"information": (1.5, 3.0), "missing": (0.0, 1.0)},
    )
    assert mapped.tolist() == [False, True, True]

    fallback = mi._constraint_mask(pareto.table, "ignored")
    assert fallback.tolist() == [True, True, True]

    with pytest.raises(ep.EyeProcessValidationError, match="invalid optimization method"):
        ep.optimize_item_bank(pareto, 1, spec, method="bad")
    with pytest.raises(
        ep.EyeProcessValidationError, match="outside the available item count"
    ):
        ep.optimize_item_bank(pareto, 0, spec)
    with pytest.raises(ep.EyeProcessValidationError, match="fewer items"):
        ep.optimize_item_bank(
            pareto,
            1,
            spec,
            constraints=lambda t: np.zeros(len(t), dtype=bool),
        )

    zero_iter = ep.optimize_item_bank(
        pareto, 1, spec, method="evolutionary", iterations=0, seed=1
    )
    assert len(zero_iter.selected) == 1

    exact = ep.optimize_item_bank(
        pareto,
        1,
        spec,
        constraints=lambda t: np.arange(len(t)) == 0,
        method="evolutionary",
        iterations=2,
        seed=1,
    )
    assert set(exact.selected["item_id"]) == {"item_1"}

    evolved = ep.optimize_item_bank(
        pareto, 1, spec, method="evolutionary", iterations=3, seed=2
    )
    assert len(evolved.selected) == 1

    with pytest.raises(ep.EyeProcessValidationError, match="item-bank optimization"):
        ep.audit_bank_decision_stability({})


def test_drift_and_fairness_single_item_context_residual_paths():
    data = pd.DataFrame(
        {
            "time": [1, 2, 1, 2],
            "group": ["A", "A", "B", "B"],
            "metric": [1.0, 2.0, 3.0, 4.0],
        }
    )
    drift = ep.monitor_dif_drift(data, "time", "group", ["metric"])
    assert set(drift.trajectories["item_id"]) == {"all_items"}

    transport = ep.audit_fairness_transportability(
        pd.DataFrame(
            {
                "item_id": ["I1", "I2"],
                "process_dif": [0.1, 0.2],
            }
        )
    )
    assert transport.summary.loc[0, "contexts"] == 1
    assert transport.summary.loc[
        0, "mean_cross_context_correlation"
    ] == pytest.approx(1.0)


def test_norm_guards_explicit_family_and_plot_residual_paths(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="invalid norm family"):
        ep.fit_process_norms(
            pd.DataFrame({"metric": [1.0, 2.0], "group": ["A", "B"]}),
            "metric",
            "group",
            family="bad",
        )

    with pytest.raises(ep.EyeProcessValidationError, match="complete reference data"):
        ep.fit_process_norms(
            pd.DataFrame({"metric": [1.0], "group": ["A"]}),
            "metric",
            "group",
            family="gaussian",
        )

    ref = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "metric": [1.0, 1.2, 2.0, 2.2],
        }
    )
    model = ep.fit_process_norms(ref, "metric", "group", family="gaussian")
    assert model["family"] == "gaussian"

    with pytest.raises(ep.EyeProcessValidationError, match="eye_process_norms"):
        ep.predict_process_centiles({}, pd.DataFrame({"group": ["A"]}))
    with pytest.raises(ep.EyeProcessValidationError, match="eye_process_norms"):
        ep.score_process_deviation(
            {}, pd.DataFrame({"group": ["A"], "metric": [1.0]})
        )
    with pytest.raises(ep.EyeProcessValidationError, match="invalid score type"):
        ep.score_process_deviation(model, ref.iloc[:1], "bad")

    fig, ax = plt.subplots()
    try:
        assert mi._ax(ax) is ax
        assert ep.plot_process_centiles(model, ax=ax) is ax
    finally:
        plt.close(fig)

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("blocked for coverage")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as ctx:
        ctx.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(ImportError, match="Plotting requires"):
            mi._ax()
