from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "vendor_validation_spec",
    "audit_vendor_validation",
    "write_vendor_validation_report",
    "model_validation_spec",
    "run_model_validation",
    "model_validation_summary",
]


def test_public_validation_core_exports():
    assert len(TARGETS) == 6
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_vendor_audit_distinguishes_complete_and_incomplete_evidence(tmp_path):
    data = pd.DataFrame(
        {
            "vendor": ["gazepoint", "gazepoint", "tobii"],
            "status": ["pass", "pass", "warning"],
            "software_version": ["7.2", "7.2", "1"],
            "device_model": ["GP3", "GP3", "Pro"],
            "independent_source": [True, True, True],
            "licence_reviewed": [True, True, True],
        }
    )
    spec = ep.vendor_validation_spec(
        required_vendors=["gazepoint", "tobii"],
        min_cases_per_vendor=2,
        min_pass_rate=0.9,
    )
    audit = ep.audit_vendor_validation(data, spec)
    assert audit.attrs["eyeprocess_class"] == "eye_vendor_validation"
    assert audit.loc[audit.vendor.eq("gazepoint"), "status"].iloc[0] == "pass"
    assert audit.loc[audit.vendor.eq("tobii"), "status"].iloc[0] == "warning"

    report = Path(ep.write_vendor_validation_report(audit, tmp_path / "vendor.md"))
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "# Multi-vendor empirical validation audit" in text
    assert "gazepoint" in text
    assert "independent real-export cases" in text


def test_vendor_audit_merges_manifest_governance_flags():
    summary = pd.DataFrame(
        {
            "case_id": ["g1", "g2"],
            "vendor": ["gazepoint", "gazepoint"],
            "status": ["pass", "pass"],
            "software_version": ["7.2", "7.2"],
            "device_model": ["GP3", "GP3"],
        }
    )
    manifest = pd.DataFrame(
        {
            "case_id": ["g1", "g2"],
            "independent_source": [True, True],
            "licence_reviewed": [True, True],
        }
    )
    corpus = {"summary": summary, "manifest": manifest, "status": "pass"}
    spec = ep.vendor_validation_spec(required_vendors="gazepoint")
    audit = ep.audit_vendor_validation(corpus, spec)
    row = audit.iloc[0]
    assert row.independent_cases == 2
    assert row.licence_reviewed_cases == 2
    assert row.status == "pass"


def test_validation_specs_reject_invalid_thresholds():
    with pytest.raises(ep.EyeProcessValidationError):
        ep.vendor_validation_spec(required_vendors=[])
    with pytest.raises(ep.EyeProcessValidationError):
        ep.vendor_validation_spec(min_cases_per_vendor=0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.vendor_validation_spec(min_pass_rate=1.1)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.model_validation_spec(replications=0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.model_validation_spec(confidence=-0.1)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.model_validation_spec(max_abs_bias=-1)


def test_model_validation_computes_recovery_and_coverage():
    def simulator(n=80, beta=0.5):
        x = np.random.normal(size=int(n))
        y = beta * x + np.random.normal(size=int(n))
        return {"x": x, "y": y, "truth": {"beta": beta}}

    def fitter(sim):
        slope = np.polyfit(sim["x"], sim["y"], 1)[0]
        residual = sim["y"] - slope * sim["x"]
        se = np.std(residual, ddof=1) / np.sqrt(np.sum(sim["x"] ** 2))
        return {"beta": slope, "se": se}

    def extractor(fit):
        return pd.DataFrame(
            {
                "parameter": ["beta"],
                "estimate": [fit["beta"]],
                "std_error": [fit["se"]],
                "lower": [fit["beta"] - 1.96 * fit["se"]],
                "upper": [fit["beta"] + 1.96 * fit["se"]],
            }
        )

    result = ep.run_model_validation(
        simulator,
        fitter,
        extractor,
        lambda sim: sim["truth"],
        grid=pd.DataFrame({"n": [50], "beta": [0.5]}),
        spec=ep.model_validation_spec(
            replications=5,
            max_abs_bias=1.0,
            min_coverage=0.0,
        ),
        seed=2,
    )
    summary = ep.model_validation_summary(result)
    assert result["runs"].shape[0] == 5
    assert summary.replications.iloc[0] == 5
    assert np.isfinite(summary.rmse.iloc[0])
    assert np.isfinite(summary.coverage.iloc[0])
    assert summary.status.iloc[0] == "pass"


def test_coverage_remains_missing_without_intervals():
    def simulator():
        return {"x": np.random.normal(size=20), "truth": {"mu": 0.0}}

    result = ep.run_model_validation(
        simulator,
        lambda sim: float(np.mean(sim["x"])),
        lambda fit: {"mu": fit},
        lambda sim: sim["truth"],
        spec=ep.model_validation_spec(replications=3, max_abs_bias=10),
        seed=11,
    )
    assert result["runs"].covered.isna().all()
    summary = ep.model_validation_summary(result)
    assert pd.isna(summary.coverage.iloc[0])


def test_model_validation_retains_heterogeneous_failure_rows():
    def simulator(mode="ok"):
        if mode == "simulation_error":
            raise RuntimeError("simulation failed")
        return {
            "y": np.random.normal(size=10),
            "truth": {"mu": 0.0},
            "mode": mode,
        }

    def fitter(sim):
        if sim["mode"] == "fit_error":
            raise RuntimeError("fit failed")
        return float(np.mean(sim["y"]))

    result = ep.run_model_validation(
        simulator,
        fitter,
        lambda fit: {"mu": fit},
        lambda sim: sim["truth"],
        grid=pd.DataFrame({"mode": ["ok", "simulation_error", "fit_error"]}),
        spec=ep.model_validation_spec(replications=1),
        seed=13,
    )
    assert len(result["runs"]) == 3
    assert {"mode", "scenario", "converged", "error"} <= set(result["runs"].columns)
    assert ".simulation" in set(result["runs"].parameter.astype(str))
    assert ".fit" in set(result["runs"].parameter.astype(str))
    summary = ep.model_validation_summary(result)
    assert "fail" in set(summary.status.astype(str))


def test_model_validation_records_extractor_and_truth_failures():
    def simulator(mode="extract"):
        return {
            "y": np.random.normal(size=10),
            "truth": {"mu": 0.0},
            "mode": mode,
        }

    fit = lambda sim: float(np.mean(sim["y"]))

    extract_failure = ep.run_model_validation(
        simulator,
        fit,
        lambda value: (_ for _ in ()).throw(RuntimeError("extract failed")),
        lambda sim: sim["truth"],
        grid=pd.DataFrame({"mode": ["extract"]}),
        spec=ep.model_validation_spec(replications=1),
        seed=18,
    )
    assert extract_failure["runs"].parameter.iloc[0] == ".extract"
    assert not bool(extract_failure["runs"].converged.iloc[0])

    truth_failure = ep.run_model_validation(
        simulator,
        fit,
        lambda value: {"mu": value},
        lambda sim: (_ for _ in ()).throw(RuntimeError("truth failed")),
        grid=pd.DataFrame({"mode": ["truth"]}),
        spec=ep.model_validation_spec(replications=1),
        seed=19,
    )
    assert truth_failure["runs"].parameter.iloc[0] == ".truth"

    with pytest.raises(RuntimeError, match="truth failed"):
        ep.run_model_validation(
            simulator,
            fit,
            lambda value: {"mu": value},
            lambda sim: (_ for _ in ()).throw(RuntimeError("truth failed")),
            grid=pd.DataFrame({"mode": ["truth"]}),
            spec=ep.model_validation_spec(replications=1),
            seed=19,
            continue_on_error=False,
        )
