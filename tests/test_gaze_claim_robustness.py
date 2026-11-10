import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.gaze_claim_robustness as gr
from eyeprocesspy.exceptions import EyeProcessValidationError


def claim(threshold=0.2):
    return gr.define_gaze_claim_spec(
        "interface-attention",
        term="condition",
        estimand_id="mean_dwell_difference_ms",
        substantive_threshold=threshold,
    )


def spec(threshold=0.2):
    return gr.define_gaze_robustness_spec(
        claim(threshold),
        {
            "detector": ["ivt", "idt"],
            "aoi": ["nominal", "expanded"],
            "quality": [0.70, 0.80],
        },
        primary_decisions={"detector": "ivt", "aoi": "nominal", "quality": 0.80},
    )


def runner(choices, focal):
    detector = {"ivt": 0.05, "idt": -0.02}[choices["detector"]]
    aoi = {"nominal": 0.03, "expanded": -0.01}[choices["aoi"]]
    quality = 0.02 if choices["quality"] == 0.80 else -0.01
    estimate = 0.25 + detector + aoi + quality
    return {
        "estimate": estimate,
        "SE": 0.05,
        "CI_lower": estimate - 0.10,
        "CI_upper": estimate + 0.10,
        "converged": True,
        "N": 100,
    }


def test_claim_validation_and_fingerprint():
    x = claim()
    assert x.claim_id == "interface-attention"
    assert len(x.fingerprint) == 64
    with pytest.raises(EyeProcessValidationError):
        gr.define_gaze_claim_spec("", term="x", estimand_id="e")
    with pytest.raises(EyeProcessValidationError):
        gr.define_gaze_claim_spec("c", term="x", estimand_id="e", threshold_direction="wrong")
    with pytest.raises(EyeProcessValidationError):
        gr.define_gaze_claim_spec("c", term="x", estimand_id="e", substantive_threshold=np.inf)


def test_spec_expansion_is_deterministic_and_has_one_primary():
    x = spec()
    a = gr.expand_gaze_robustness_spec(x)
    b = gr.expand_gaze_robustness_spec(x)
    assert len(a) == 8
    assert a.equals(b)
    assert a.is_primary.sum() == 1
    assert a.universe_id.tolist()[0] == "u0001"
    assert a.specification_hash.str.len().eq(64).all()


def test_spec_validation_rejects_invalid_primary():
    with pytest.raises(EyeProcessValidationError):
        gr.define_gaze_robustness_spec(
            claim(),
            {"detector": ["ivt"]},
            primary_decisions={},
        )
    with pytest.raises(EyeProcessValidationError):
        gr.define_gaze_robustness_spec(
            claim(),
            {"detector": ["ivt"]},
            primary_decisions={"detector": "idt"},
        )


def test_audit_summary_and_decision_decomposition():
    result = gr.run_gaze_robustness_audit(spec(), runner)
    assert result.failures.empty
    assert set(result.results.status) == {"ok"}
    summary = gr.summarise_gaze_claim_robustness(result).iloc[0]
    assert summary.planned_universes == 8
    assert summary.evaluable_universes == 8
    assert summary.failed_universes == 0
    assert summary.primary_estimate == pytest.approx(0.35)
    assert summary.same_direction_proportion_planned == 1
    assert 0 <= summary.substantive_support_proportion <= 1
    decomposition = gr.decompose_gaze_decision_sensitivity(result)
    assert set(decomposition.decision) == {"detector", "aoi", "quality"}
    assert decomposition.declared_levels.eq(2).all()


def test_runner_dataframe_aliases_are_normalised():
    def frame_runner(choices, focal):
        return pd.DataFrame(
            {
                "term": ["other", focal.term],
                "estimate": [9.0, 0.4],
                "se": [1.0, 0.1],
                "CI_low": [7.0, 0.2],
                "CI_high": [11.0, 0.6],
                "model_converged": [True, True],
                "model_rows_used": [5, 80],
            }
        )

    one = gr.define_gaze_robustness_spec(
        claim(),
        {"detector": ["ivt"]},
        primary_decisions={"detector": "ivt"},
    )
    result = gr.run_gaze_robustness_audit(one, frame_runner)
    row = result.results.iloc[0]
    assert row.estimate == pytest.approx(0.4)
    assert row.SE == pytest.approx(0.1)
    assert row.N == pytest.approx(80)


def test_failures_and_nonconvergence_remain_in_planned_denominator():
    def problematic(choices, focal):
        if choices["detector"] == "idt" and choices["aoi"] == "expanded":
            raise RuntimeError("planned branch failed")
        out = runner(choices, focal)
        if choices["detector"] == "idt" and choices["aoi"] == "nominal":
            out["converged"] = False
        return out

    result = gr.run_gaze_robustness_audit(spec(), problematic)
    summary = gr.summarise_gaze_claim_robustness(result).iloc[0]
    assert summary.planned_universes == 8
    assert summary.failed_universes == 2
    assert summary.nonconverged_universes == 2
    assert summary.evaluable_universes == 4
    assert len(result.failures) == 2
    assert summary.planned_evaluable_rate == pytest.approx(0.5)
    text = gr.report_gaze_claim_robustness(result)
    assert "Planned universes: 8" in text
    assert "not probabilities" in text


def test_fail_fast_and_invalid_runner_outputs():
    one = gr.define_gaze_robustness_spec(
        claim(),
        {"detector": ["ivt"]},
        primary_decisions={"detector": "ivt"},
    )

    with pytest.raises(RuntimeError):
        gr.run_gaze_robustness_audit(
            one, lambda choices, focal: (_ for _ in ()).throw(RuntimeError("boom")),
            continue_on_error=False,
        )

    result = gr.run_gaze_robustness_audit(one, lambda choices, focal: {"estimate": "bad"})
    assert result.results.iloc[0].status == "failed"

    with pytest.raises(EyeProcessValidationError):
        gr.run_gaze_robustness_audit(one, object())


def test_estimand_is_fixed_within_spec_and_threshold_can_be_absent():
    x = gr.define_gaze_claim_spec("c", term="b", estimand_id="censored_time_to_first_fixation")
    s = gr.define_gaze_robustness_spec(
        x,
        {"detector": ["ivt", "idt"]},
        primary_decisions={"detector": "ivt"},
    )
    result = gr.run_gaze_robustness_audit(
        s,
        lambda choices, focal: {
            "estimate": 0.1 if choices["detector"] == "ivt" else -0.1,
            "converged": True,
            "N": 20,
        },
    )
    summary = gr.summarise_gaze_claim_robustness(result).iloc[0]
    assert summary.estimand_id == "censored_time_to_first_fixation"
    assert np.isnan(summary.substantive_support_proportion)


def test_plot_smoke():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    result = gr.run_gaze_robustness_audit(spec(), runner)
    ax = gr.plot_gaze_specification_curve(result)
    assert ax.get_title() == "Gaze claim specification curve"
