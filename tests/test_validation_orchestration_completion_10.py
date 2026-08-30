from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "validation_recovery_summary",
    "validation_failure_summary",
    "validation_runtime_summary",
    "validation_calibration_summary",
    "validation_sbc_summary",
    "validation_thresholds",
    "audit_validation_completion",
    "plot_parameter_recovery",
    "plot_interval_coverage",
    "plot_sbc_rank",
    "plot_validation_failures",
    "plot_validation_runtime",
    "model_promotion_spec",
    "audit_model_promotion",
    "write_validation_release_report",
    "write_model_promotion_report",
]


def _collection(replications=4):
    plan = ep.validation_job_plan(
        {"n": [20]},
        replications=replications,
        base_seed=4,
        model_family="demo",
    )
    jobs = plan["jobs"].copy()
    jobs["status"] = "complete"
    jobs["stage"] = "complete"
    jobs["elapsed_seconds"] = np.linspace(0.1, 0.4, replications)
    jobs["warning_count"] = 0
    jobs["message_count"] = 0
    jobs["error"] = None

    estimates = pd.DataFrame(
        {
            "job_id": jobs["job_id"],
            "model_family": "demo",
            "scenario_id": "S00001",
            "replication": np.arange(1, replications + 1),
            "parameter": "a",
            "estimate": [0.9, 1.0, 1.1, 1.05][:replications],
            "truth": 1.0,
            "std_error": 0.1,
            "lower": [0.7, 0.8, 0.9, 0.85][:replications],
            "upper": [1.1, 1.2, 1.3, 1.25][:replications],
            "status": "complete",
        }
    )

    diagnostics = pd.DataFrame(
        {
            "job_id": jobs["job_id"],
            "max_rhat": 1.0,
            "min_ess_bulk": 500,
            "divergences": 0,
            "converged": True,
        }
    )

    prediction_rows = []
    draw_rows = []
    for index, row in jobs.iterrows():
        prediction_rows.extend(
            [
                {
                    "job_id": row["job_id"],
                    "model_family": "demo",
                    "scenario_id": "S00001",
                    "observed": 0,
                    "predicted": 0.15,
                },
                {
                    "job_id": row["job_id"],
                    "model_family": "demo",
                    "scenario_id": "S00001",
                    "observed": 1,
                    "predicted": 0.85,
                },
            ]
        )
        rng = np.random.default_rng(int(row["seed"]))
        for draw in rng.normal(1.0, 0.2, 100):
            draw_rows.append(
                {
                    "job_id": row["job_id"],
                    "model_family": "demo",
                    "scenario_id": "S00001",
                    "parameter": "a",
                    "draw": draw,
                    "truth": 1.0,
                }
            )

    return ep.validation_orchestration_10.EyeValidationCollection(
        plan=plan,
        paths=[],
        results=[],
        jobs=jobs,
        estimates=estimates,
        diagnostics=diagnostics,
        predictions=pd.DataFrame(prediction_rows),
        draws=pd.DataFrame(draw_rows),
        corrupt=[],
        collected_utc="test",
    )


def test_public_final_r023_exports_are_callable():
    assert len(TARGETS) == 16
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_recovery_failure_and_runtime_summaries_follow_frozen_contracts():
    collection = _collection()
    recovery = ep.validation_recovery_summary(collection)
    assert recovery["replications"].iloc[0] == 4
    assert recovery["successful"].iloc[0] == 4
    assert recovery["rmse"].iloc[0] > 0
    assert 0 <= recovery["coverage"].iloc[0] <= 1

    failure = ep.validation_failure_summary(collection)
    assert failure["jobs"].iloc[0] == 4
    assert failure["failure_rate"].iloc[0] == 0

    runtime = ep.validation_runtime_summary(collection)
    assert runtime["jobs"].iloc[0] == 4
    assert runtime["total_seconds"].iloc[0] == pytest.approx(1.0)
    assert runtime["p90_seconds"].iloc[0] > runtime["median_seconds"].iloc[0]


def test_calibration_and_sbc_summaries_are_finite():
    collection = _collection()
    calibration = ep.validation_calibration_summary(collection, bins=5)
    assert calibration["n"].iloc[0] == 8
    assert 0 <= calibration["brier"].iloc[0] <= 1
    assert calibration["log_loss"].iloc[0] > 0
    assert 0 <= calibration["ece"].iloc[0] <= 1

    sbc = ep.validation_sbc_summary(collection, bins=5)
    assert sbc["replications"].iloc[0] == 4
    assert 0 <= sbc["mean_scaled_rank"].iloc[0] <= 1
    assert 0 <= sbc["p_value"].iloc[0] <= 1


def test_completion_audit_enforces_recovery_and_empirical_gates():
    collection = _collection()
    thresholds = ep.validation_thresholds(
        required_replications=4,
        max_failure_rate=0,
        max_absolute_bias=0.2,
        max_rmse=0.2,
        min_coverage=0.5,
        max_coverage=1.0,
        max_rhat=1.01,
        min_ess_bulk=400,
        max_divergence_rate=0,
        require_sbc=False,
        require_empirical_reproduction=True,
    )
    incomplete = ep.audit_validation_completion(
        collection,
        thresholds,
        empirical_reproduction=False,
    )
    assert incomplete.eyeprocess_class == "eye_validation_completion_audit"
    assert incomplete["status"] == "incomplete"
    empirical = incomplete["gates"].loc[incomplete["gates"]["gate"].eq("empirical_reproduction")]
    assert not bool(empirical["pass"].iloc[0])

    complete = ep.audit_validation_completion(
        collection,
        thresholds,
        empirical_reproduction=True,
    )
    assert complete["status"] == "complete"
    assert complete["gates"]["pass"].all()


def test_missing_expected_job_fails_all_jobs_present_gate():
    collection = _collection()
    collection["jobs"] = collection["jobs"].iloc[:-1].copy()
    thresholds = ep.validation_thresholds(
        required_replications=1,
        require_sbc=False,
        require_empirical_reproduction=False,
        min_coverage=0,
        max_coverage=1,
    )
    audit = ep.audit_validation_completion(collection, thresholds)
    row = audit["gates"].loc[audit["gates"]["gate"].eq("all_jobs_present")].iloc[0]
    assert not bool(row["pass"])
    assert len(audit["missing_jobs"]) == 1


def test_promotion_defaults_to_experimental_without_evidence():
    spec = ep.model_promotion_spec(
        model_families=["dynamic_irtree"],
        require_multi_vendor=True,
    )
    audit = ep.audit_model_promotion(
        {"dynamic_irtree": {}},
        spec,
    )
    assert audit.eyeprocess_class == "eye_model_promotion_audit"
    assert audit["models"]["status"].iloc[0] == "experimental"
    assert audit["models"]["passed_required_gates"].iloc[0] < audit["models"]["required_gates"].iloc[0]


def test_promotion_requires_only_declared_gates():
    spec = ep.model_promotion_spec(
        model_families=["dynamic_irtree"],
        require_completion=True,
        require_sbc=False,
        require_misspecification=False,
        require_grouped_validation=False,
        require_engine_equivalence=False,
        require_empirical_reproduction=False,
        require_preprocessing_sensitivity=False,
        require_multi_vendor=False,
    )
    audit = ep.audit_model_promotion(
        {"dynamic_irtree": {"completion": True}},
        spec,
    )
    assert audit["models"]["status"].iloc[0] == "promotable"


def test_validation_plots_render_and_do_not_leak_when_closed():
    collection = _collection()
    before = set(plt.get_fignums())
    summaries = {
        "recovery": ep.validation_recovery_summary(collection),
        "failure": ep.validation_failure_summary(collection),
    }
    functions = [
        lambda: ep.plot_parameter_recovery(collection),
        lambda: ep.plot_interval_coverage(summaries["recovery"]),
        lambda: ep.plot_sbc_rank(collection, bins=5),
        lambda: ep.plot_validation_failures(summaries["failure"]),
        lambda: ep.plot_validation_runtime(collection),
    ]
    for function in functions:
        ax = function()
        assert ax.figure is not None
        plt.close(ax.figure)
    assert set(plt.get_fignums()) == before

    with pytest.raises(ep.EyeProcessBackendError, match="ggplot2"):
        ep.plot_validation_runtime(collection, engine="ggplot2")


def test_release_and_promotion_reports_include_interpretation_boundaries(tmp_path):
    collection = _collection()
    thresholds = ep.validation_thresholds(
        required_replications=4,
        require_sbc=False,
        require_empirical_reproduction=False,
        min_coverage=0.5,
        max_coverage=1.0,
    )
    completion = ep.audit_validation_completion(collection, thresholds)
    promotion = ep.audit_model_promotion(
        {
            "dynamic_irtree": {
                "completion": completion,
            }
        },
        ep.model_promotion_spec(
            model_families=["dynamic_irtree"],
            require_sbc=False,
            require_misspecification=False,
            require_grouped_validation=False,
            require_engine_equivalence=False,
            require_empirical_reproduction=False,
            require_preprocessing_sensitivity=False,
        ),
    )

    release_path = Path(
        ep.write_validation_release_report(
            collection,
            tmp_path / "release.md",
            completion=completion,
            promotion=promotion,
        )
    )
    promotion_path = Path(
        ep.write_model_promotion_report(
            promotion,
            tmp_path / "promotion.md",
        )
    )
    release = release_path.read_text(encoding="utf-8")
    report = promotion_path.read_text(encoding="utf-8")

    assert "## Completion gates" in release
    assert "## Parameter recovery" in release
    assert "## Prediction calibration" in release
    assert "## Simulation-based calibration" in release
    assert "## Model promotion audit" in release
    assert "## Interpretation boundary" in release
    assert "## Session" in release
    assert "## Evidence gates" in report
    assert "## Interpretation boundary" in report
