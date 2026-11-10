import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.gaze_claim_robustness as gr
from eyeprocesspy.exceptions import EyeProcessValidationError


def _claim(**kwargs):
    return gr.define_gaze_claim_spec(
        kwargs.pop("claim_id", "c"),
        term=kwargs.pop("term", "condition"),
        estimand_id=kwargs.pop("estimand_id", "dwell_difference"),
        **kwargs,
    )


def _one(claim=None):
    claim = claim or _claim()
    return gr.define_gaze_robustness_spec(
        claim,
        {"detector": ["ivt"]},
        primary_decisions={"detector": "ivt"},
    )


def _ok(estimate=0.3, **kwargs):
    return {
        "estimate": estimate,
        "SE": kwargs.get("SE"),
        "CI_lower": kwargs.get("CI_lower"),
        "CI_upper": kwargs.get("CI_upper"),
        "converged": kwargs.get("converged"),
        "N": kwargs.get("N"),
    }


def test_private_jsonable_paths_and_spec_fingerprints():
    assert gr._jsonable(float("nan")) is None
    assert gr._jsonable(float("inf")) == "inf"
    assert gr._jsonable(np.int64(2)) == 2
    assert gr._jsonable({"b": (np.int64(1),), "a": [2]}) == {"a": [2], "b": [1]}
    assert isinstance(gr._jsonable(object()), str)

    s = gr.define_gaze_robustness_spec(
        _claim(),
        {"b": [np.int64(2)], "a": ["x"]},
        primary_decisions={"a": "x", "b": np.int64(2)},
        label="study",
    )
    assert s.decision_dict["a"] == ("x",)
    assert s.primary_dict["a"] == "x"
    assert len(s.fingerprint) == 64


def test_spec_validation_edges():
    with pytest.raises(EyeProcessValidationError):
        gr.define_gaze_robustness_spec("not-claim", {"a": [1]}, primary_decisions={"a": 1})
    with pytest.raises(EyeProcessValidationError):
        gr.define_gaze_robustness_spec(_claim(), [], primary_decisions={})
    with pytest.raises(EyeProcessValidationError):
        gr.define_gaze_robustness_spec(_claim(), {"a": [1]}, primary_decisions=[])
    with pytest.raises(EyeProcessValidationError):
        gr.define_gaze_robustness_spec(_claim(), {"a": []}, primary_decisions={"a": 1})

    bad = gr.GazeRobustnessSpec(
        claim=_claim(),
        decision_grid=(("a", (1,)),),
        primary_decisions=(("a", 2),),
    )
    with pytest.raises(EyeProcessValidationError):
        gr.expand_gaze_robustness_spec(bad)
    with pytest.raises(EyeProcessValidationError):
        gr.expand_gaze_robustness_spec("bad")


def test_runner_dataframe_shape_type_and_finite_validation():
    s = _one()

    def no_term(choices, focal):
        return pd.DataFrame([{"estimate": 0.2}])

    assert gr.run_gaze_robustness_audit(s, no_term).results.iloc[0].status == "ok"

    def no_focal(choices, focal):
        return pd.DataFrame([{"term": "other", "estimate": 0.2}])

    assert gr.run_gaze_robustness_audit(s, no_focal).results.iloc[0].status == "failed"

    def duplicate(choices, focal):
        return pd.DataFrame(
            [{"term": focal.term, "estimate": 0.2}, {"term": focal.term, "estimate": 0.3}]
        )

    assert gr.run_gaze_robustness_audit(s, duplicate).results.iloc[0].status == "failed"
    assert gr.run_gaze_robustness_audit(s, lambda c, f: 3).results.iloc[0].status == "failed"
    assert (
        gr.run_gaze_robustness_audit(s, lambda c, f: {"estimate": np.inf})
        .results.iloc[0]
        .status
        == "failed"
    )


def test_runner_alias_defaults_and_pandas_na_convergence():
    s = _one()

    def aliases(choices, focal):
        return {
            "estimate": 0.25,
            "standard_error": 0.02,
            "ci_lower": 0.1,
            "ci_upper": 0.4,
            "model_converged": pd.NA,
            "n": 44,
        }

    row = gr.run_gaze_robustness_audit(s, aliases).results.iloc[0]
    assert row.status == "ok"
    assert row.SE == pytest.approx(0.02)
    assert row.CI_lower == pytest.approx(0.1)
    assert row.CI_upper == pytest.approx(0.4)
    assert row.N == pytest.approx(44)

    row2 = gr.run_gaze_robustness_audit(
        s, lambda c, f: {"estimate": 0.2, "converged": False}
    ).results.iloc[0]
    assert row2.status == "non_converged"


@pytest.mark.parametrize(
    ("direction", "threshold", "values", "expected"),
    [
        ("above", 0.2, [0.1, 0.3], 0.5),
        ("below", 0.2, [0.1, 0.3], 0.5),
        ("absolute", 0.2, [-0.3, 0.1], 0.5),
    ],
)
def test_threshold_support_directions(direction, threshold, values, expected):
    claim = _claim(substantive_threshold=threshold, threshold_direction=direction)
    assert gr._threshold_support(pd.Series(values), claim) == pytest.approx(expected)
    assert np.isnan(gr._threshold_support(pd.Series([], dtype=float), claim))


def test_summary_zero_primary_missing_ci_missing_n_and_invalid_input():
    s = gr.define_gaze_robustness_spec(
        _claim(substantive_threshold=None),
        {"branch": ["primary", "other"]},
        primary_decisions={"branch": "primary"},
    )

    def zero(choices, focal):
        return {"estimate": 0.0 if choices["branch"] == "primary" else 0.2}

    result = gr.run_gaze_robustness_audit(s, zero)
    summary = gr.summarise_gaze_claim_robustness(result).iloc[0]
    assert summary.primary_direction == 0
    assert summary.same_direction_proportion_evaluable == pytest.approx(0.5)
    assert pd.isna(summary.common_CI_overlap)
    assert np.isnan(summary.median_N)
    assert np.isnan(summary.substantive_support_proportion)

    with pytest.raises(EyeProcessValidationError):
        gr.summarise_gaze_claim_robustness("bad")


def test_all_failed_summary_report_and_empty_decomposition():
    s = _one()

    def fail(choices, focal):
        raise RuntimeError("failure")

    result = gr.run_gaze_robustness_audit(s, fail)
    summary = gr.summarise_gaze_claim_robustness(result).iloc[0]
    assert summary.evaluable_universes == 0
    assert np.isnan(summary.primary_estimate)
    assert np.isnan(summary.median_estimate)
    dec = gr.decompose_gaze_decision_sensitivity(result)
    assert dec.iloc[0].evaluable_levels == 0
    assert np.isnan(dec.iloc[0].marginal_mean_range)
    text = gr.report_gaze_claim_robustness(result)
    assert "Primary estimate" not in text
    assert "Estimate median" not in text

    with pytest.raises(EyeProcessValidationError):
        gr.decompose_gaze_decision_sensitivity("bad")


def test_manual_empty_audit_covers_zero_denominator_paths():
    s = _one()
    empty = pd.DataFrame(
        columns=[
            "status",
            "estimate",
            "is_primary",
            "CI_lower",
            "CI_upper",
            "N",
            "detector",
        ]
    )
    result = gr.GazeRobustnessAuditResult(
        spec=s,
        universes=pd.DataFrame(),
        results=empty,
        failures=pd.DataFrame(),
    )
    summary = gr.summarise_gaze_claim_robustness(result).iloc[0]
    assert np.isnan(summary.planned_evaluable_rate)
    assert np.isnan(summary.same_direction_proportion_planned)


def test_report_with_threshold_and_plot_all_branches():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    s = gr.define_gaze_robustness_spec(
        _claim(substantive_threshold=0.15),
        {"branch": ["primary", "missing_ci", "nonconv"]},
        primary_decisions={"branch": "primary"},
    )

    def mixed(choices, focal):
        if choices["branch"] == "primary":
            return _ok(0.3, SE=0.05, CI_lower=0.2, CI_upper=0.4, N=50, converged=True)
        if choices["branch"] == "missing_ci":
            return _ok(0.1, N=40, converged=True)
        return _ok(-0.2, converged=False)

    result = gr.run_gaze_robustness_audit(s, mixed)
    text = gr.report_gaze_claim_robustness(result)
    assert "Prespecified substantive-threshold" in text

    fig, ax = plt.subplots()
    returned = gr.plot_gaze_specification_curve(result, ax=ax)
    assert returned is ax

    failed_primary = gr.define_gaze_robustness_spec(
        _claim(),
        {"branch": ["primary", "ok"]},
        primary_decisions={"branch": "primary"},
    )

    def primary_fail(choices, focal):
        if choices["branch"] == "primary":
            return _ok(0.1, converged=False)
        return _ok(0.2, converged=True)

    result2 = gr.run_gaze_robustness_audit(failed_primary, primary_fail)
    ax2 = gr.plot_gaze_specification_curve(result2)
    assert ax2.get_title() == "Gaze claim specification curve"

    all_failed = gr.run_gaze_robustness_audit(
        _one(),
        lambda c, f: (_ for _ in ()).throw(RuntimeError("x")),
    )
    ax3 = gr.plot_gaze_specification_curve(all_failed)
    assert ax3.get_xlabel() == "Specification"
