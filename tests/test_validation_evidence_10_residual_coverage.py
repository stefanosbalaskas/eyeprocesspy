from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.validation_evidence_10 as ve
from eyeprocesspy.exceptions import EyeProcessValidationError


class _Marked:
    eyeprocess_class = "marked"


class _Grouped(dict):
    eyeprocess_class = "eye_grouped_cv"


class _Engine(dict):
    eyeprocess_class = "eye_engine_comparison"


class _Multiverse(dict):
    eyeprocess_class = "eye_multiverse"


def test_name_class_and_mapping_helpers_cover_all_representations():
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        ve._nonempty_names(None, "values")
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        ve._nonempty_names(["ok", np.nan], "values")
    assert ve._nonempty_names("x", "values") == ("x",)
    assert ve._unique(("a", "a", "b")) == ("a", "b")

    assert ve._class_name(_Marked()) == "marked"
    frame = pd.DataFrame({"x": [1]})
    frame.attrs["eyeprocess_class"] = "frame_marker"
    assert ve._class_name(frame) == "frame_marker"
    assert ve._class_name({"eyeprocess_class": "mapping_marker"}) == "mapping_marker"
    assert ve._class_name({"_eyeprocess_class": "legacy_marker"}) == "legacy_marker"
    assert ve._class_name({}) is None

    assert ve._mapping_component({"x": 1}, "x") == 1
    assert ve._mapping_component(SimpleNamespace(x=2), "x") == 2
    assert ve._mapping_component(SimpleNamespace(), "x", 3) == 3


def test_recovery_gate_rejects_wrong_empty_missing_threshold_and_bad_coverage(monkeypatch):
    assert not ve._recovery_pass({})
    value = {"runs": pd.DataFrame({"x": [1]}), "spec": SimpleNamespace(min_coverage=0.9)}

    monkeypatch.setattr(ve, "model_validation_summary", lambda x: pd.DataFrame())
    assert not ve._recovery_pass(value)

    monkeypatch.setattr(
        ve,
        "model_validation_summary",
        lambda x: pd.DataFrame({"status": ["pass"], "coverage": [0.95]}),
    )
    value["spec"] = SimpleNamespace(min_coverage=None)
    assert not ve._recovery_pass(value)

    value["spec"] = SimpleNamespace(min_coverage=0.9)
    monkeypatch.setattr(
        ve,
        "model_validation_summary",
        lambda x: pd.DataFrame({"status": ["pass"], "coverage": [np.nan]}),
    )
    assert not ve._recovery_pass(value)

    monkeypatch.setattr(
        ve,
        "model_validation_summary",
        lambda x: pd.DataFrame({"status": ["pass"], "coverage": [0.95]}),
    )
    assert ve._recovery_pass(value)


def test_misspecification_grouped_engine_empirical_and_sensitivity_gates(monkeypatch):
    validation = {"runs": pd.DataFrame({"x": [1]}), "spec": SimpleNamespace()}
    monkeypatch.setattr(
        ve,
        "model_validation_summary",
        lambda x: pd.DataFrame({"status": ["fail"]}),
    )
    assert not ve._misspecification_pass(validation)

    monkeypatch.setattr(
        ve,
        "model_validation_summary",
        lambda x: pd.DataFrame({"status": ["pass"], "expected_failure": [False]}),
    )
    assert not ve._misspecification_pass(validation)

    monkeypatch.setattr(
        ve,
        "model_validation_summary",
        lambda x: pd.DataFrame({"status": ["fail"], "expected_failure": [True]}),
    )
    assert ve._misspecification_pass(validation)

    assert not ve._misspecification_pass(pd.DataFrame({"expected_failure": [False], "detected": [True]}))
    assert not ve._misspecification_pass(
        pd.DataFrame({"expected_failure": [True], "detected": [pd.NA]})
    )
    assert ve._misspecification_pass(
        pd.DataFrame({"expected_failure": [True], "detected": [True]})
    )
    assert not ve._misspecification_pass(object())

    assert not ve._grouped_validation_pass({})
    assert not ve._grouped_validation_pass(_Grouped(results=pd.DataFrame()))
    assert not ve._grouped_validation_pass(_Grouped(results=pd.DataFrame({"x": [1]})))
    assert not ve._grouped_validation_pass(_Grouped(results=pd.DataFrame({"score": [np.nan]})))
    assert ve._grouped_validation_pass(_Grouped(results=pd.DataFrame({"score": [1.0]})))
    assert ve._grouped_validation_pass(
        _Grouped(results=pd.DataFrame({"score": [1.0], "error": [pd.NA]}))
    )
    assert not ve._grouped_validation_pass(
        _Grouped(results=pd.DataFrame({"score": [1.0], "error": ["boom"]}))
    )

    assert not ve._engine_pass({})
    assert not ve._engine_pass(_Engine(estimates=pd.DataFrame()))
    assert not ve._engine_pass(_Engine(estimates=pd.DataFrame({"x": [1]})))
    assert not ve._engine_pass(_Engine(estimates=pd.DataFrame({"equivalent": [pd.NA]})))
    assert ve._engine_pass(_Engine(estimates=pd.DataFrame({"equivalent": [True]})))

    comparison = pd.DataFrame(
        {"target": [1.0], "absolute_difference": [0.0], "reproduced": [True]}
    )
    empirical = ve._EmpiricalReproduction(comparison=comparison)
    assert ve._empirical_pass(empirical)
    assert not ve._empirical_pass(ve._EmpiricalReproduction(comparison=pd.DataFrame()))

    assert not ve._sensitivity_pass({})
    assert not ve._sensitivity_pass(_Multiverse(specifications=None, results=pd.DataFrame()))
    assert not ve._sensitivity_pass(_Multiverse(specifications=object(), results=pd.DataFrame()))
    assert not ve._sensitivity_pass(
        _Multiverse(specifications=[1], results=pd.DataFrame({"score": [1.0]}))
    )
    assert not ve._sensitivity_pass(
        _Multiverse(specifications=[1, 2], results=pd.DataFrame({"text": ["x"]}))
    )
    assert ve._sensitivity_pass(
        _Multiverse(specifications=[1, 2], results=pd.DataFrame({"score": [1.0]}))
    )
    assert not ve._sensitivity_pass(
        _Multiverse(
            specifications=[1, 2],
            results=pd.DataFrame({"score": [1.0], "error": ["boom"]}),
        )
    )


def test_advanced_audit_argument_record_warning_and_no_required_paths():
    with pytest.raises(TypeError, match="spec"):
        ep.audit_advanced_model_evidence({}, spec={})
    with pytest.raises(EyeProcessValidationError, match="named mapping"):
        ep.audit_advanced_model_evidence([])

    no_requirements = ep.advanced_model_evidence_spec(
        models="model_a",
        require_recovery=False,
        require_calibration=False,
        require_misspecification=False,
        require_grouped_validation=False,
        require_engine_equivalence=False,
        require_empirical_reproduction=False,
        require_sensitivity=False,
    )
    out = ep.audit_advanced_model_evidence({"model_a": "not-a-record"}, no_requirements)
    assert out.status.iloc[0] == "pass"
    assert out.required.iloc[0] == 0

    partial = ep.advanced_model_evidence_spec(
        models="model_a",
        require_recovery=False,
        require_calibration=False,
        require_misspecification=True,
        require_grouped_validation=True,
        require_engine_equivalence=False,
        require_empirical_reproduction=False,
        require_sensitivity=False,
    )
    evidence = {
        "model_a": {
            "misspecification": pd.DataFrame(
                {"expected_failure": [True], "detected": [True]}
            )
        }
    }
    warning = ep.audit_advanced_model_evidence(evidence, partial)
    assert warning.status.iloc[0] == "warning"
    assert warning.completed.iloc[0] == 1


def test_write_report_guard_and_boolean_rendering(tmp_path):
    with pytest.raises(EyeProcessValidationError, match="advanced-model evidence audit"):
        ep.write_advanced_model_evidence_report(pd.DataFrame(), tmp_path / "bad.md")

    spec = ep.advanced_model_evidence_spec(
        models="m",
        require_recovery=False,
        require_calibration=False,
        require_misspecification=False,
        require_grouped_validation=False,
        require_engine_equivalence=False,
        require_empirical_reproduction=False,
        require_sensitivity=False,
    )
    audit = ep.audit_advanced_model_evidence({}, spec)
    path = ep.write_advanced_model_evidence_report(audit, tmp_path / "nested" / "report.md")
    assert "FALSE" in open(path, encoding="utf-8").read()


def test_truth_posterior_and_estimate_frame_conversion_guards():
    duplicate = pd.Series([1.0, 2.0], index=["a", "a"])
    assert ve._truth_map(duplicate) is None
    assert ve._truth_map({"a": "bad"}) is None
    assert ve._truth_map([1, 2]) is None
    assert ve._truth_map({"a": 1}) == {"a": 1.0}

    frame = pd.DataFrame({"a": [1.0]})
    assert ve._posterior_frame(frame).equals(frame)
    assert ve._posterior_frame({"a": [1.0]}).equals(frame)
    assert ve._posterior_frame({"a": 1.0}) is None
    assert ve._posterior_frame(object()) is None

    series = pd.Series([1.0, 2.0], index=["a", "b"])
    assert list(ve._estimate_frame(series).parameter) == ["a", "b"]
    assert ve._estimate_frame(duplicate) is None
    assert list(ve._estimate_frame({"a": 1.0}).parameter) == ["a"]
    assert ve._estimate_frame(object()) is None


@pytest.mark.parametrize("replications", [0, 1.5, "bad"])
def test_sbc_component_and_replication_guards(replications):
    with pytest.raises(EyeProcessValidationError):
        ep.simulation_based_calibration(
            lambda: {},
            lambda x: x,
            lambda x: pd.DataFrame(),
            lambda x: {},
            replications=replications,
        )

    with pytest.raises(EyeProcessValidationError, match="callable"):
        ep.simulation_based_calibration(
            None,
            lambda x: x,
            lambda x: pd.DataFrame(),
            lambda x: {},
            replications=1,
        )


def test_sbc_invalid_draw_truth_name_no_finite_and_tie_paths():
    bad_draws = ep.simulation_based_calibration(
        lambda: {"truth": {"mu": 0.0}},
        lambda sim: sim,
        lambda fit: {"mu": 1.0},
        lambda sim: sim["truth"],
        replications=1,
    )
    assert "matching parameter names" in bad_draws["ranks"].error.iloc[0]

    no_match = ep.simulation_based_calibration(
        lambda: {"truth": {"mu": 0.0}},
        lambda sim: sim,
        lambda fit: pd.DataFrame({"sigma": [1.0, 2.0]}),
        lambda sim: sim["truth"],
        replications=1,
    )
    assert "No matching" in no_match["ranks"].error.iloc[0]
    assert no_match["ranks"].draws.iloc[0] == 2

    no_finite = ep.simulation_based_calibration(
        lambda: {"truth": {"mu": 0.0}},
        lambda sim: sim,
        lambda fit: pd.DataFrame({"mu": [np.nan, np.inf]}),
        lambda sim: sim["truth"],
        replications=1,
    )
    assert pd.isna(no_finite["ranks"].rank.iloc[0])
    assert np.isnan(no_finite["ranks"].posterior_mean.iloc[0])

    tied = ep.simulation_based_calibration(
        lambda: {"truth": {"mu": 0.0}},
        lambda sim: sim,
        lambda fit: pd.DataFrame({"mu": [-1.0, 0.0, 0.0, 1.0]}),
        lambda sim: sim["truth"],
        replications=1,
        seed=3,
    )
    assert 1 <= tied["ranks"].rank.iloc[0] <= 3
    assert np.isfinite(tied["ranks"].normalized_rank.iloc[0])


def _manual_sbc(normalized, posterior_mean=None, truth=None, posterior_sd=None):
    n = len(normalized)
    if posterior_mean is None:
        posterior_mean = np.zeros(n)
    if truth is None:
        truth = np.zeros(n)
    if posterior_sd is None:
        posterior_sd = np.ones(n)
    return ve._SBC(
        ranks=pd.DataFrame(
            {
                "replication": np.arange(1, n + 1),
                "parameter": ["mu"] * n,
                "rank": np.arange(n),
                "draws": [100] * n,
                "normalized_rank": normalized,
                "truth": truth,
                "posterior_mean": posterior_mean,
                "posterior_sd": posterior_sd,
                "error": [pd.NA] * n,
            }
        )
    )


def test_sbc_summary_guards_zero_success_pass_and_fail_paths(monkeypatch):
    with pytest.raises(EyeProcessValidationError, match="eye_sbc"):
        ep.sbc_summary({})
    with pytest.raises(EyeProcessValidationError, match="eye_sbc"):
        ep.sbc_summary(ve._SBC(ranks=[]))

    zero = ep.sbc_summary(_manual_sbc([np.nan, np.nan]))
    assert zero.successful.iloc[0] == 0
    assert zero.status.iloc[0] == "insufficient"

    good = np.linspace(0.025, 0.975, 20)
    monkeypatch.setattr(ve.stats, "kstest", lambda *args, **kwargs: SimpleNamespace(pvalue=0.5))
    passed = ep.sbc_summary(_manual_sbc(good))
    assert passed.status.iloc[0] == "pass"
    assert np.isfinite(passed.rank_variance.iloc[0])

    bad = ep.sbc_summary(_manual_sbc(np.repeat(0.9, 20)))
    assert bad.status.iloc[0] == "fail"

    biased = ep.sbc_summary(
        _manual_sbc(
            [0.5] * 4,
            posterior_mean=[1.0] * 4,
            truth=[0.0] * 4,
            posterior_sd=[0.5] * 4,
        )
    )
    assert biased.mean_standardized_bias.iloc[0] == pytest.approx(2.0)


def test_published_target_and_raven_spec_edge_guards(tmp_path):
    with pytest.raises(EyeProcessValidationError, match="published_targets"):
        ve._published_targets([])
    with pytest.raises(EyeProcessValidationError, match="published_targets"):
        ve._published_targets({"a": np.inf})
    assert ve._published_targets(None) is None

    with pytest.raises(EyeProcessValidationError, match="data_path"):
        ep.raven_reproduction_spec("", "response", ["x"])
    with pytest.raises(EyeProcessValidationError, match="response"):
        ep.raven_reproduction_spec(tmp_path / "x.csv", "", ["x"])


def test_raven_run_remaining_validation_and_estimate_paths(tmp_path):
    with pytest.raises(EyeProcessValidationError, match="spec"):
        ep.run_raven_reproduction({}, lambda x: x, lambda x, y: x, lambda x: x)

    path = tmp_path / "missing.csv"
    spec = ep.raven_reproduction_spec(
        path,
        "response",
        ["x"],
        licence_reviewed=True,
    )
    with pytest.raises(EyeProcessValidationError, match="not found"):
        ep.run_raven_reproduction(spec, lambda x: x, lambda x, y: x, lambda x: x)

    path.write_text("x\n1\n", encoding="utf-8")
    spec = ep.raven_reproduction_spec(
        path,
        "response",
        ["x"],
        licence_reviewed=True,
    )
    with pytest.raises(EyeProcessValidationError, match="callable"):
        ep.run_raven_reproduction(spec, None, lambda x, y: x, lambda x: x)
    with pytest.raises(EyeProcessValidationError, match="parameter"):
        ep.run_raven_reproduction(
            spec,
            pd.read_csv,
            lambda data, spec: data,
            lambda fit: pd.DataFrame({"wrong": [1.0]}),
        )
    with pytest.raises(EyeProcessValidationError, match="tolerance"):
        ep.run_raven_reproduction(
            spec,
            pd.read_csv,
            lambda data, spec: data,
            lambda fit: {"beta": 1.0},
            tolerance="bad",
        )

    targeted = ep.raven_reproduction_spec(
        path,
        "response",
        ["x"],
        published_targets={"beta": 1.0, "missing": 2.0},
        licence_reviewed=True,
    )
    reproduction = ep.run_raven_reproduction(
        targeted,
        pd.read_csv,
        lambda data, spec: data,
        lambda fit: pd.Series([1.0, 3.0], index=["beta", "other"]),
    )
    assert bool(reproduction["comparison"].reproduced.iloc[0])
    assert pd.isna(reproduction["comparison"].reproduced.iloc[1])
