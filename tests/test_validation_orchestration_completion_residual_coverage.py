from __future__ import annotations

import math
from pathlib import Path
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.validation_orchestration_completion_10 as voc


def _collection(**overrides):
    payload = {
        "plan": None,
        "paths": [],
        "results": [],
        "jobs": pd.DataFrame(),
        "estimates": pd.DataFrame(),
        "diagnostics": pd.DataFrame(),
        "predictions": pd.DataFrame(),
        "draws": pd.DataFrame(),
        "corrupt": [],
        "collected_utc": "test",
    }
    payload.update(overrides)
    return voc.EyeValidationCollection(**payload)


class _Tagged:
    def __init__(self, eyeprocess_class, **fields):
        self.eyeprocess_class = eyeprocess_class
        self.__dict__.update(fields)


class _VendorFrame(pd.DataFrame):
    eyeprocess_class = "eye_vendor_validation"


def test_private_container_and_grouping_residual_branches():
    item = voc.EyeValidationThresholds(answer=42)
    assert item.answer == 42
    with pytest.raises(AttributeError, match="missing"):
        _ = item.missing

    with pytest.raises(ep.EyeProcessValidationError, match="boom"):
        voc._stop("boom")

    frame = pd.DataFrame({"g": ["a", "b"], "x": [1, 2]})
    copied = voc._as_frame(frame)
    assert copied.equals(frame)
    assert copied is not frame
    assert voc._as_frame({"x": [3]}).x.iloc[0] == 3

    assert voc._by_list(None) == []
    assert voc._by_list("g") == ["g"]
    assert voc._by_list(("g", 7)) == ["g", "7"]

    assert voc._group_summary(pd.DataFrame(), ["g"], lambda x: {"n": len(x)}).empty
    ungrouped = voc._group_summary(frame, ["missing"], lambda x: {"n": len(x)})
    assert ungrouped.n.iloc[0] == 2
    grouped = voc._group_summary(frame, ["g"], lambda x: {"n": len(x)})
    assert grouped.set_index("g").loc["a", "n"] == 1

    assert voc._class_name(item) == "eye_validation_thresholds"
    assert voc._field({"x": 1}, "x") == 1
    assert voc._field(types.SimpleNamespace(x=2), "x") == 2
    assert voc._field({}, "x", 9) == 9
    assert voc._frame_field({"x": frame}, "x").equals(frame)
    assert voc._frame_field({"x": 1}, "x").empty

    finite = voc._finite_numeric(pd.Series([1, np.nan, "2", "bad"]))
    assert finite.tolist() == [1.0, 2.0]
    assert voc._safe_max(pd.Series([np.nan]), 7.0) == 7.0
    assert voc._safe_min(pd.Series([np.nan]), -7.0) == -7.0


def test_recovery_summary_guards_and_sparse_defaults():
    with pytest.raises(ep.EyeProcessValidationError, match="estimates data frame"):
        voc.validation_recovery_summary({"bad": 1})
    with pytest.raises(ep.EyeProcessValidationError, match="Recovery data require"):
        voc.validation_recovery_summary(pd.DataFrame({"parameter": ["a"]}))

    sparse = pd.DataFrame(
        {
            "parameter": ["a"],
            "estimate": [1.0],
            "truth": [0.0],
        }
    )
    out = voc.validation_recovery_summary(sparse, by=None)
    row = out.iloc[0]
    assert row.replications == 1
    assert row.successful == 1
    assert math.isnan(row.relative_bias)
    assert math.isnan(row.empirical_sd)
    assert math.isnan(row.mean_model_se)
    assert math.isnan(row.se_ratio)
    assert math.isnan(row.coverage)
    assert math.isnan(row.mean_interval_width)

    failed = pd.DataFrame(
        {
            "parameter": ["a"],
            "estimate": [1.0],
            "truth": [1.0],
            "status": ["failed"],
            "std_error": [0.1],
            "covered": [True],
            "interval_width": [0.2],
            "relative_bias": [0.0],
        }
    )
    failed_out = voc.validation_recovery_summary(failed)
    assert failed_out.successful.iloc[0] == 0
    assert failed_out.failure_rate.iloc[0] == 1.0
    assert math.isnan(failed_out.rmse.iloc[0])


def test_failure_runtime_and_weighted_mean_negative_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="job table"):
        voc.validation_failure_summary("bad")
    with pytest.raises(ep.EyeProcessValidationError, match="status"):
        voc.validation_failure_summary(pd.DataFrame({"x": [1]}))

    failures = voc.validation_failure_summary(
        pd.DataFrame(
            {
                "status": ["complete", "failed", "nonconverged", "locked"],
            }
        ),
        by=None,
    )
    row = failures.iloc[0]
    assert row.jobs == 4
    assert row.warning_rate == 0.0
    assert row.failure_rate == pytest.approx(0.75)

    with pytest.raises(ep.EyeProcessValidationError, match="Runtime data"):
        voc.validation_runtime_summary(pd.DataFrame({"x": [1]}))

    runtime = voc.validation_runtime_summary(
        pd.DataFrame({"elapsed_seconds": [np.nan, "bad"]}),
        by=None,
    ).iloc[0]
    assert runtime.total_seconds == 0.0
    assert math.isnan(runtime.mean_seconds)
    assert math.isnan(runtime.median_seconds)
    assert math.isnan(runtime.p90_seconds)
    assert math.isnan(runtime.max_seconds)

    assert math.isnan(voc._weighted_mean([1, 2], [0, 0]))


def test_calibration_guards_empty_weights_and_failed_optimizer(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="Prediction data"):
        voc.validation_calibration_summary(pd.DataFrame({"observed": [1]}))
    with pytest.raises(ep.EyeProcessValidationError, match="binary"):
        voc.validation_calibration_summary(
            pd.DataFrame({"observed": [2], "predicted": [0.5]})
        )

    empty = voc.validation_calibration_summary(
        pd.DataFrame(
            {
                "observed": [0, 1],
                "predicted": [0.2, 0.8],
                "weight": [0.0, np.nan],
            }
        ),
        by=None,
        bins=2,
    ).iloc[0]
    assert empty.n == 0
    assert math.isnan(empty.brier)

    def failed_minimize(fun, x0, method):
        del fun, method
        return types.SimpleNamespace(success=False, x=np.asarray(x0, dtype=float))

    monkeypatch.setattr(voc, "minimize", failed_minimize)
    intercept, slope = voc._calibration_coefficients(
        [0, 1], [0.25, 0.75], [1, 1]
    )
    assert math.isnan(intercept)
    assert math.isnan(slope)

    default_weight = voc.validation_calibration_summary(
        pd.DataFrame({"observed": [0, 1], "predicted": [0.25, 0.75]}),
        by=None,
        bins=2,
    )
    assert default_weight.n.iloc[0] == 2


def test_sbc_guard_empty_rank_single_rank_and_zero_histogram(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="SBC draws require"):
        voc.validation_sbc_summary(pd.DataFrame({"job_id": ["j"]}))

    bad = pd.DataFrame(
        {
            "job_id": ["j"],
            "parameter": ["a"],
            "draw": [np.nan],
            "truth": [1.0],
        }
    )
    assert voc._sbc_ranks(bad, []).empty
    assert voc.validation_sbc_summary(bad, by=None, bins=2).empty

    one = pd.DataFrame(
        {
            "job_id": ["j", "j"],
            "parameter": ["a", "a"],
            "draw": [0.5, 1.0],
            "truth": [0.75, 0.75],
        }
    )
    out = voc.validation_sbc_summary(one, by=None, bins=2)
    assert out.replications.iloc[0] == 1
    assert math.isnan(out.rank_variance.iloc[0])

    real_histogram = np.histogram

    def zero_histogram(values, bins, **kwargs):
        del values, kwargs
        edges = np.asarray(bins, dtype=float)
        return np.zeros(len(edges) - 1, dtype=int), edges

    monkeypatch.setattr(voc.np, "histogram", zero_histogram)
    zero = voc.validation_sbc_summary(one, by=None, bins=2).iloc[0]
    assert math.isnan(zero.chi_square)
    assert math.isnan(zero.p_value)
    monkeypatch.setattr(voc.np, "histogram", real_histogram)


def test_evidence_dispatcher_all_specialized_and_generic_paths():
    assert not voc._evidence_pass(None, "completion")
    assert voc._evidence_pass(True, "anything")
    assert not voc._evidence_pass(np.bool_(False), "anything")

    completion = voc.EyeValidationCompletionAudit(status="complete")
    assert voc._evidence_pass(completion, "completion")
    assert not voc._evidence_pass(
        voc.EyeValidationCompletionAudit(status="incomplete"), "completion"
    )

    grouped = _Tagged(
        "eye_grouped_cv",
        results=pd.DataFrame({"score": [0.2, 0.3]}),
    )
    assert voc._evidence_pass(grouped, "grouped_validation")
    assert not voc._evidence_pass(
        _Tagged("eye_crossed_grouped_cv", results=pd.DataFrame()),
        "grouped_validation",
    )
    assert not voc._evidence_pass(
        _Tagged("eye_grouped_cv", results=pd.DataFrame({"x": [1]})),
        "grouped_validation",
    )

    engine = _Tagged(
        "eye_engine_comparison",
        estimates=pd.DataFrame({"equivalent": [True, True]}),
    )
    assert voc._evidence_pass(engine, "engine_equivalence")
    assert not voc._evidence_pass(
        _Tagged("eye_engine_comparison", estimates=pd.DataFrame()),
        "engine_equivalence",
    )
    assert not voc._evidence_pass(
        _Tagged(
            "eye_engine_comparison",
            estimates=pd.DataFrame({"equivalent": [True, False]}),
        ),
        "engine_equivalence",
    )

    empirical = _Tagged(
        "eye_empirical_reproduction",
        comparison=pd.DataFrame({"reproduced": [True, True]}),
    )
    assert voc._evidence_pass(empirical, "empirical_reproduction")
    assert not voc._evidence_pass(
        _Tagged("eye_empirical_reproduction", comparison=pd.DataFrame()),
        "empirical_reproduction",
    )
    assert not voc._evidence_pass(
        _Tagged(
            "eye_empirical_reproduction",
            comparison=pd.DataFrame({"reproduced": [True, False]}),
        ),
        "empirical_reproduction",
    )

    assert voc._evidence_pass(
        _Tagged(
            "eye_multiverse",
            specifications=[1, 2],
            results=pd.DataFrame({"estimate": [1.0]}),
        ),
        "preprocessing_sensitivity",
    )
    assert not voc._evidence_pass(
        _Tagged(
            "eye_multiverse",
            specifications=[1],
            results=pd.DataFrame({"estimate": [1.0]}),
        ),
        "preprocessing_sensitivity",
    )

    vendor = _VendorFrame({"status": ["pass", "pass"]})
    assert voc._evidence_pass(vendor, "multi_vendor")
    vendor_bad = _VendorFrame({"status": ["pass", "fail"]})
    assert not voc._evidence_pass(vendor_bad, "multi_vendor")

    assert voc._evidence_pass(pd.DataFrame({"pass": [True, pd.NA]}), "sbc")
    assert not voc._evidence_pass(pd.DataFrame({"pass": [False]}), "sbc")
    assert voc._evidence_pass({"pass": np.bool_(True)}, "misspecification")
    assert voc._evidence_pass({"status": "SUCCESS"}, "misspecification")
    assert not voc._evidence_pass({"status": "pending"}, "misspecification")
    assert not voc._evidence_pass(object(), "misspecification")


def test_completion_audit_sparse_and_validation_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="validation_collection"):
        voc.audit_validation_completion({})
    with pytest.raises(ep.EyeProcessValidationError, match="validation_thresholds"):
        voc.audit_validation_completion(_collection(), thresholds={})

    sparse = _collection(
        plan={"jobs": pd.DataFrame({"not_job_id": [1]})},
        draws=pd.DataFrame({"bad": [1]}),
        diagnostics=pd.DataFrame(
            {
                "divergences": [np.nan],
                "max_rhat": [np.nan],
                "min_ess_bulk": [np.nan],
            }
        ),
    )
    audit = voc.audit_validation_completion(
        sparse,
        empirical_reproduction={"status": "pass"},
    )
    assert audit.status == "incomplete"
    assert audit.missing_jobs.empty
    assert audit.recovery.empty
    assert audit.failure.empty
    assert audit.sbc.empty

    jobs = pd.DataFrame(
        {
            "job_id": ["j1"],
            "status": ["complete"],
            "warning_count": [0],
        }
    )
    estimates = pd.DataFrame(
        {
            "job_id": ["j1"],
            "parameter": ["a"],
            "estimate": [np.nan],
            "truth": [1.0],
        }
    )
    no_diagnostics = _collection(jobs=jobs, estimates=estimates)
    thresholds = voc.validation_thresholds(
        required_replications=1,
        require_sbc=False,
        require_empirical_reproduction=False,
        min_coverage=0,
        max_coverage=1,
    )
    second = voc.audit_validation_completion(no_diagnostics, thresholds)
    assert second.status == "incomplete"


def test_plot_engine_and_plot_guard_residuals():
    assert voc._plot_engine(("base", "ggplot2")) == "matplotlib"
    with pytest.raises(ep.EyeProcessValidationError, match="engine"):
        voc._plot_engine("unknown")

    with pytest.raises(ep.EyeProcessValidationError, match="Recovery estimates"):
        voc.plot_parameter_recovery(pd.DataFrame({"truth": [1]}))
    with pytest.raises(ep.EyeProcessValidationError, match="No recovery rows"):
        voc.plot_parameter_recovery(
            pd.DataFrame(
                {
                    "parameter": ["a"],
                    "truth": [1.0],
                    "estimate": [1.0],
                }
            ),
            parameter="missing",
        )

    with pytest.raises(ep.EyeProcessValidationError, match="Coverage summary"):
        voc.plot_interval_coverage(pd.DataFrame({"coverage": [0.9]}))
    with pytest.raises(ep.EyeProcessValidationError, match="No finite coverage"):
        voc.plot_interval_coverage(
            pd.DataFrame({"parameter": ["a"], "coverage": [np.nan]})
        )

    with pytest.raises(ep.EyeProcessValidationError, match="SBC draws"):
        voc._plot_sbc_ranks(pd.DataFrame({"job_id": ["j"]}))
    with pytest.raises(ep.EyeProcessValidationError, match="No SBC ranks"):
        voc._plot_sbc_ranks(
            pd.DataFrame(
                {
                    "job_id": ["j"],
                    "parameter": ["a"],
                    "draw": [1.0],
                    "truth": [1.0],
                }
            ),
            parameter="missing",
        )
    with pytest.raises(ep.EyeProcessValidationError, match="SBC draws"):
        voc.plot_sbc_rank("bad")

    with pytest.raises(ep.EyeProcessValidationError, match="Failure summary"):
        voc.plot_validation_failures(pd.DataFrame({"x": [1]}))
    with pytest.raises(ep.EyeProcessValidationError, match="Runtime data"):
        voc.plot_validation_runtime(pd.DataFrame({"x": [1]}))
    with pytest.raises(ep.EyeProcessValidationError, match="No finite runtime"):
        voc.plot_validation_runtime(pd.DataFrame({"elapsed_seconds": [np.nan]}))

    fig, ax = plt.subplots()
    try:
        returned = voc.plot_validation_failures(
            pd.DataFrame({"failure_rate": [0.1, 0.2]}),
            ax=ax,
        )
        assert returned is ax
    finally:
        plt.close(fig)

    ax = voc.plot_validation_runtime(
        pd.DataFrame({"elapsed_seconds": [0.1, 0.2]})
    )
    try:
        assert ax.get_title() == "Validation runtime"
    finally:
        plt.close(ax.figure)


def test_model_promotion_spec_guards_default_and_full_dispatch():
    spec = voc.model_promotion_spec(
        "demo",
        require_multi_vendor=True,
    )
    assert spec.model_families == ["demo"]

    deduped = voc.model_promotion_spec(["a", "a", "b"])
    assert deduped.model_families == ["a", "b"]

    for bad in ([None], [""], []):
        with pytest.raises(ep.EyeProcessValidationError, match="model_families"):
            voc.model_promotion_spec(bad)

    with pytest.raises(ep.EyeProcessValidationError, match="named list"):
        voc.audit_model_promotion([])
    with pytest.raises(ep.EyeProcessValidationError, match="model_promotion_spec"):
        voc.audit_model_promotion({}, spec={})

    default = voc.audit_model_promotion({})
    assert set(default.models.status) == {"experimental"}

    evidence = {
        "demo": {
            "completion": voc.EyeValidationCompletionAudit(status="complete"),
            "sbc": {"status": "pass"},
            "misspecification": {"pass": True},
            "grouped_validation": _Tagged(
                "eye_grouped_cv",
                results=pd.DataFrame({"score": [0.1]}),
            ),
            "engine_equivalence": _Tagged(
                "eye_engine_comparison",
                estimates=pd.DataFrame({"equivalent": [True]}),
            ),
            "empirical_reproduction": _Tagged(
                "eye_empirical_reproduction",
                comparison=pd.DataFrame({"reproduced": [True]}),
            ),
            "preprocessing_sensitivity": _Tagged(
                "eye_multiverse",
                specifications=[1, 2],
                results=pd.DataFrame({"estimate": [1]}),
            ),
            "multi_vendor": _VendorFrame({"status": ["pass"]}),
        }
    }
    promoted = voc.audit_model_promotion(evidence, spec)
    assert promoted.models.status.iloc[0] == "promotable"

    nonmapping = voc.audit_model_promotion(
        {"demo": 3},
        voc.model_promotion_spec(
            "demo",
            require_completion=False,
            require_sbc=False,
            require_misspecification=False,
            require_grouped_validation=False,
            require_engine_equivalence=False,
            require_empirical_reproduction=False,
            require_preprocessing_sensitivity=False,
            require_multi_vendor=False,
        ),
    )
    assert nonmapping.models.status.iloc[0] == "promotable"


def test_markdown_and_report_guard_optional_sections(tmp_path):
    assert voc._markdown_table(None) == "_No rows available._"
    assert voc._markdown_table(pd.DataFrame()) == "_No rows available._"
    table = voc._markdown_table(
        pd.DataFrame(
            {
                "number": [1.23456, np.nan],
                "label": ["x", pd.NA],
            }
        ),
        digits=2,
    )
    assert "1.23" in table
    assert "| x |" in table

    with pytest.raises(ep.EyeProcessValidationError, match="validation_collection"):
        voc.write_validation_release_report({}, tmp_path / "bad.md")

    empty = _collection()
    with pytest.raises(ep.EyeProcessValidationError, match="completion"):
        voc.write_validation_release_report(
            empty,
            tmp_path / "bad-completion.md",
            completion={},
        )

    completion = voc.EyeValidationCompletionAudit(
        status="incomplete",
        gates=pd.DataFrame(),
    )
    report = Path(
        voc.write_validation_release_report(
            empty,
            tmp_path / "minimal.md",
            completion=completion,
            include_session=False,
        )
    ).read_text(encoding="utf-8")
    assert "## Prediction calibration" not in report
    assert "## Simulation-based calibration" not in report
    assert "## Model promotion audit" not in report
    assert "## Session" not in report
    assert "_No rows available._" in report

    malformed = _collection(
        predictions=pd.DataFrame({"x": [1]}),
        draws=pd.DataFrame({"x": [1]}),
    )
    malformed_report = Path(
        voc.write_validation_release_report(
            malformed,
            tmp_path / "malformed.md",
            completion=completion,
            include_session=False,
        )
    ).read_text(encoding="utf-8")
    assert "## Prediction calibration" not in malformed_report
    assert "## Simulation-based calibration" not in malformed_report

    with pytest.raises(ep.EyeProcessValidationError, match="model_promotion_audit"):
        voc.write_model_promotion_report({}, tmp_path / "bad-promotion.md")
