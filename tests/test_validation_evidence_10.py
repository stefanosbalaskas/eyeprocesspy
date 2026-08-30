from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "advanced_model_evidence_spec",
    "audit_advanced_model_evidence",
    "write_advanced_model_evidence_report",
    "simulation_based_calibration",
    "sbc_summary",
    "raven_reproduction_spec",
    "run_raven_reproduction",
]


def _recovery(with_intervals=True):
    def simulator():
        return {"truth": {"beta": 0.5}}

    def extractor(_fit):
        row = {"parameter": ["beta"], "estimate": [0.5]}
        if with_intervals:
            row["lower"] = [0.4]
            row["upper"] = [0.6]
        return pd.DataFrame(row)

    return ep.run_model_validation(
        simulator,
        lambda sim: sim,
        extractor,
        lambda sim: sim["truth"],
        spec=ep.model_validation_spec(replications=2),
        seed=31,
    )


class _Grouped(dict):
    eyeprocess_class = "eye_grouped_cv"


def test_public_validation_evidence_exports():
    assert len(TARGETS) == 7
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_sbc_harness_and_summary_are_executable():
    def simulator():
        return {"y": np.random.normal(0.25, 1.0, 30), "truth": {"mu": 0.25}}

    def fitter(sim):
        return {
            "mu": float(np.mean(sim["y"])),
            "se": float(np.std(sim["y"], ddof=1) / np.sqrt(len(sim["y"]))),
        }

    def draws(fit):
        return pd.DataFrame({"mu": np.random.normal(fit["mu"], fit["se"], 200)})

    sbc = ep.simulation_based_calibration(
        simulator,
        fitter,
        draws,
        lambda sim: sim["truth"],
        replications=4,
        seed=12,
    )
    assert getattr(sbc, "eyeprocess_class", None) == "eye_sbc"
    summary = ep.sbc_summary(sbc)
    assert len(summary) == 1
    assert summary.parameter.iloc[0] == "mu"
    assert summary.successful.iloc[0] == 4
    assert summary.status.iloc[0] == "insufficient"


def test_sbc_retains_simulation_fit_and_draw_failures():
    sim_failure = ep.simulation_based_calibration(
        lambda: (_ for _ in ()).throw(RuntimeError("simulation failed")),
        lambda sim: sim,
        lambda fit: pd.DataFrame({"mu": [0.0]}),
        lambda sim: {"mu": 0.0},
        replications=1,
        seed=1,
    )
    assert "simulation failed" in sim_failure["ranks"].error.iloc[0]

    fit_failure = ep.simulation_based_calibration(
        lambda: {"truth": {"mu": 0.0}},
        lambda sim: (_ for _ in ()).throw(RuntimeError("fit failed")),
        lambda fit: pd.DataFrame({"mu": [0.0]}),
        lambda sim: sim["truth"],
        replications=1,
        seed=1,
    )
    assert "fit failed" in fit_failure["ranks"].error.iloc[0]

    draw_failure = ep.simulation_based_calibration(
        lambda: {"truth": {"mu": 0.0}},
        lambda sim: sim,
        lambda fit: (_ for _ in ()).throw(RuntimeError("draw failed")),
        lambda sim: sim["truth"],
        replications=1,
        seed=1,
    )
    assert "draw failed" in draw_failure["ranks"].error.iloc[0]


def test_sbc_truth_failure_is_not_silently_swallowed():
    with pytest.raises(RuntimeError, match="truth failed"):
        ep.simulation_based_calibration(
            lambda: {"truth": {"mu": 0.0}},
            lambda sim: sim,
            lambda fit: pd.DataFrame({"mu": [0.0, 0.1]}),
            lambda sim: (_ for _ in ()).throw(RuntimeError("truth failed")),
            replications=1,
            seed=1,
        )


def test_advanced_evidence_audit_enforces_independent_gates(tmp_path):
    spec = ep.advanced_model_evidence_spec(
        models=["fit_process_irt", "fit_dynamic_irtree"],
        require_calibration=False,
        require_engine_equivalence=False,
        require_empirical_reproduction=False,
        require_sensitivity=False,
    )
    empty = ep.audit_advanced_model_evidence({}, spec)
    assert empty.attrs["eyeprocess_class"] == "eye_advanced_evidence_audit"
    assert empty.status.eq("fail").all()

    grouped = _Grouped(results=pd.DataFrame({"fold": [1], "n_assessment": [10], "score": [0.7]}))
    evidence = {
        "fit_process_irt": {
            "recovery": _recovery(with_intervals=True),
            "misspecification": pd.DataFrame({"expected_failure": [True], "detected": [True]}),
            "grouped_validation": grouped,
        }
    }
    audit = ep.audit_advanced_model_evidence(evidence, spec)
    assert audit.loc[audit.model.eq("fit_process_irt"), "status"].iloc[0] == "pass"
    assert audit.loc[audit.model.eq("fit_dynamic_irtree"), "status"].iloc[0] == "fail"

    report = Path(ep.write_advanced_model_evidence_report(audit, tmp_path / "advanced-evidence.md"))
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "# Advanced-model scientific-evidence audit" in text
    assert "`fit_process_irt()`" in text


def test_advanced_recovery_requires_interval_coverage():
    spec = ep.advanced_model_evidence_spec(
        models="fit_process_irt",
        require_calibration=False,
        require_misspecification=False,
        require_grouped_validation=False,
        require_engine_equivalence=False,
        require_empirical_reproduction=False,
        require_sensitivity=False,
    )
    audit = ep.audit_advanced_model_evidence(
        {"fit_process_irt": {"recovery": _recovery(with_intervals=False)}},
        spec,
    )
    assert audit.status.iloc[0] == "fail"


def test_engine_equivalence_evidence_accepts_existing_engine_comparison():
    data = pd.DataFrame({"x": np.linspace(-1, 1, 20), "y": np.linspace(-1, 1, 20)})

    def engine(frame):
        return float(np.polyfit(frame["x"], frame["y"], 1)[0])

    comparison = ep.compare_model_engines(
        data,
        engines={"a": engine, "b": engine},
        extractors=lambda fit: {"beta": fit},
        reference="a",
        tolerance=1e-12,
    )
    spec = ep.advanced_model_evidence_spec(
        models="fit_process_irt",
        require_recovery=False,
        require_calibration=False,
        require_misspecification=False,
        require_grouped_validation=False,
        require_engine_equivalence=True,
        require_empirical_reproduction=False,
        require_sensitivity=False,
    )
    audit = ep.audit_advanced_model_evidence(
        {"fit_process_irt": {"engine_equivalence": comparison}},
        spec,
    )
    assert audit.status.iloc[0] == "pass"


def test_raven_reproduction_refuses_unreviewed_materials(tmp_path):
    spec = ep.raven_reproduction_spec(
        tmp_path / "not-reviewed.csv",
        response="score",
        strategy_features=["toggle", "latency"],
    )
    with pytest.raises(ep.EyeProcessValidationError, match="licence_reviewed"):
        ep.run_raven_reproduction(
            spec,
            lambda path: pd.DataFrame(),
            lambda data, spec: None,
            lambda fit: {"beta": 0.0},
        )


def test_raven_reproduction_requires_targets_for_empirical_promotion(tmp_path):
    data_path = tmp_path / "raven.csv"
    pd.DataFrame({"x": [0.5, 0.5, 0.5]}).to_csv(data_path, index=False)

    without_targets = ep.raven_reproduction_spec(
        data_path,
        response="score",
        strategy_features=["toggle", "latency"],
        licence_reviewed=True,
    )
    reproduction = ep.run_raven_reproduction(
        without_targets,
        pd.read_csv,
        lambda data, spec: {"beta": float(data["x"].mean())},
        lambda fit: fit,
    )
    assert "target" not in reproduction["comparison"].columns

    gate = ep.advanced_model_evidence_spec(
        models="fit_process_irt",
        require_recovery=False,
        require_calibration=False,
        require_misspecification=False,
        require_grouped_validation=False,
        require_engine_equivalence=False,
        require_empirical_reproduction=True,
        require_sensitivity=False,
    )
    audit = ep.audit_advanced_model_evidence(
        {"fit_process_irt": {"empirical_reproduction": reproduction}},
        gate,
    )
    assert audit.status.iloc[0] == "fail"


def test_raven_reproduction_compares_published_targets(tmp_path):
    data_path = tmp_path / "raven.csv"
    pd.DataFrame({"x": [0.5, 0.5, 0.5]}).to_csv(data_path, index=False)
    spec = ep.raven_reproduction_spec(
        data_path,
        response="score",
        strategy_features=["toggle", "latency"],
        published_targets={"beta": 0.5},
        licence_reviewed=True,
    )
    reproduction = ep.run_raven_reproduction(
        spec,
        pd.read_csv,
        lambda data, spec: {"beta": float(data["x"].mean())},
        lambda fit: fit,
        tolerance=1e-12,
    )
    comparison = reproduction["comparison"]
    assert comparison.target.iloc[0] == pytest.approx(0.5)
    assert comparison.absolute_difference.iloc[0] == pytest.approx(0.0)
    assert bool(comparison.reproduced.iloc[0])

    gate = ep.advanced_model_evidence_spec(
        models="fit_process_irt",
        require_recovery=False,
        require_calibration=False,
        require_misspecification=False,
        require_grouped_validation=False,
        require_engine_equivalence=False,
        require_empirical_reproduction=True,
        require_sensitivity=False,
    )
    audit = ep.audit_advanced_model_evidence(
        {"fit_process_irt": {"empirical_reproduction": reproduction}},
        gate,
    )
    assert audit.status.iloc[0] == "pass"


def test_specs_reject_invalid_model_names_targets_and_tolerance(tmp_path):
    with pytest.raises(ep.EyeProcessValidationError):
        ep.advanced_model_evidence_spec(models=[])
    with pytest.raises(ep.EyeProcessValidationError):
        ep.raven_reproduction_spec(
            tmp_path / "x.csv",
            response="score",
            strategy_features=[],
        )
    with pytest.raises(ep.EyeProcessValidationError):
        ep.raven_reproduction_spec(
            tmp_path / "x.csv",
            response="score",
            strategy_features=["toggle"],
            published_targets={"beta": np.nan},
        )

    data_path = tmp_path / "raven.csv"
    data_path.write_text("x\n1\n", encoding="utf-8")
    spec = ep.raven_reproduction_spec(
        data_path,
        response="score",
        strategy_features=["toggle"],
        licence_reviewed=True,
    )
    with pytest.raises(ep.EyeProcessValidationError):
        ep.run_raven_reproduction(
            spec,
            pd.read_csv,
            lambda data, spec: None,
            lambda fit: {"beta": 0.0},
            tolerance=-1,
        )
