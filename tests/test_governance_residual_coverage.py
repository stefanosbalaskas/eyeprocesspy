from __future__ import annotations

import json
import types
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.governance_09 as gov
import eyeprocesspy.plots_governance_09 as gplots
from eyeprocesspy.exceptions import EyeProcessValidationError
from eyeprocesspy.irt import EyeResult


def _design(**overrides):
    args = dict(
        n_persons=4,
        n_trials=3,
        missingness=0,
        sampling_rate_hz=60,
        aoi_error="low",
        calibration_error=0,
        pupil_dropout=0,
        heterogeneity="low",
        model_misspecification=False,
        replications=1,
        seed=7,
    )
    args.update(overrides)
    return ep.process_validation_design(**args)


def _mutated_design(key, value):
    d = _design()
    x = EyeResult(dict(d), eyeprocess_class="eye_process_validation_design")
    x[key] = value
    return x


def _one_condition(**overrides):
    return ep.expand_process_validation_design(_design(**overrides)).iloc[[0]].copy()


def _validation_result(estimates=None, failures=None, design=None):
    if design is None:
        design = _one_condition()
    return EyeResult(
        {
            "design": design.reset_index(drop=True),
            "estimates": pd.DataFrame() if estimates is None else estimates,
            "failures": pd.DataFrame() if failures is None else failures,
            "warnings": pd.DataFrame(),
            "design_hash": "test",
        },
        eyeprocess_class="eye_process_validation_result",
    )


def _sensitivity_result(grid=None, results=None, failures=None):
    if grid is None:
        grid = ep.process_sensitivity_grid(method=("a", "b"))
    return EyeResult(
        {
            "grid": grid,
            "results": pd.DataFrame() if results is None else results,
            "failures": pd.DataFrame() if failures is None else failures,
            "warnings": pd.DataFrame(),
        },
        eyeprocess_class="eye_process_sensitivity",
    )


def test_private_governance_helpers_cover_coercion_warning_error_and_fill_paths():
    scalar_mapping = gov._df({"x": 1})
    assert scalar_mapping.to_dict("records") == [{"x": 1}]
    with pytest.raises(EyeProcessValidationError, match="coercible"):
        gov._df(object(), "bad")

    with pytest.raises(EyeProcessValidationError, match="missing required"):
        gov._req(pd.DataFrame({"x": [1]}), ["x", None, "y"], "frame")

    assert isinstance(gov._hash_object(len), str)

    joined = gov._rbind_fill([None, pd.DataFrame(), {"a": [1]}, {"b": [2]}])
    assert list(joined.columns) == ["a", "b"]
    assert gov._rbind_fill([None, pd.DataFrame()]).empty

    def warn_fun():
        warnings.warn("captured warning", RuntimeWarning)
        return 3

    cap = gov._capture(warn_fun)
    assert cap["value"] == 3 and cap["warnings"]

    def fail_fun():
        raise RuntimeError("captured failure")

    cap = gov._capture(fail_fun)
    assert cap["error"] == "captured failure"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_persons", [1], "n_persons"),
        ("n_trials", [1], "n_trials"),
        ("missingness", [1.0], "missingness"),
        ("pupil_dropout", [-0.1], "pupil_dropout"),
        ("sampling_rate_hz", [0.0], "sampling_rate_hz"),
        ("calibration_error", [-1.0], "calibration_error"),
        ("model_misspecification", [], "model_misspecification"),
        ("seed", -1, "seed"),
        ("aoi_error", [""], "aoi_error"),
        ("heterogeneity", [""], "heterogeneity"),
        ("replications", 0, "replications"),
    ],
)
def test_validation_design_rejects_each_invalid_dimension(field, value, message):
    with pytest.raises(EyeProcessValidationError, match=message):
        ep.validate_process_validation_design(_mutated_design(field, value))

    with pytest.raises(EyeProcessValidationError):
        ep.validate_process_validation_design({})


def test_validation_design_expansion_condition_and_simulation_edge_paths():
    d = _design(
        missingness=0.35,
        pupil_dropout=0.35,
        calibration_error=1.0,
        heterogeneity="high",
        aoi_error="severe",
    )
    with pytest.raises(EyeProcessValidationError, match="max_conditions"):
        ep.expand_process_validation_design(d, max_conditions=np.nan)
    with pytest.raises(EyeProcessValidationError, match="max_conditions"):
        ep.expand_process_validation_design(d, max_conditions=-1)
    with pytest.raises(EyeProcessValidationError, match="expands"):
        ep.expand_process_validation_design(d, max_conditions=0)

    grid = ep.expand_process_validation_design(d, max_conditions=np.inf)
    assert ep.validation_condition_id(d) == grid.condition_id.astype(str).tolist()
    with pytest.raises(EyeProcessValidationError, match="condition_id"):
        ep.validation_condition_id(pd.DataFrame({"x": [1]}))

    with pytest.raises(EyeProcessValidationError, match="exactly one"):
        ep.simulate_process_validation_data(pd.concat([grid, grid], ignore_index=True))
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        ep.simulate_process_validation_data(pd.DataFrame({"n_persons": [4]}))
    with pytest.raises(EyeProcessValidationError, match="seed"):
        ep.simulate_process_validation_data(grid, seed=-1)
    with pytest.raises(EyeProcessValidationError, match="beta"):
        ep.simulate_process_validation_data(grid, beta=np.nan)

    sim = ep.simulate_process_validation_data(grid, replication=2, seed=None)
    assert len(sim.data) == 12
    assert sim.data.valid_gaze.dtype == bool
    assert np.array_equal(
        sim.data.valid_gaze.to_numpy(),
        (sim.data.gaze_x.notna() & sim.data.gaze_y.notna()).to_numpy(),
    )


def test_validation_default_fit_and_runner_failure_warning_progress_paths(capsys):
    tiny = gov._result(
        "eye_process_validation_simulation",
        data=pd.DataFrame({"x": [1.0, 2.0], "process_value": [1.0, 2.0]}),
    )
    with pytest.raises(EyeProcessValidationError, match="Not enough"):
        gov._default_validation_fit(tiny)

    mapped = {
        "data": pd.DataFrame({"x": [0.0, 1.0, 2.0], "process_value": [0.0, 1.0, 2.0]})
    }
    assert np.isfinite(gov._default_validation_fit(mapped).slope)

    grid = _one_condition()
    grid.attrs["master_seed"] = 12

    with pytest.raises(EyeProcessValidationError, match="max_conditions"):
        ep.run_process_validation(grid, max_conditions=-1)
    with pytest.raises(EyeProcessValidationError, match="max_conditions"):
        ep.run_process_validation(grid, max_conditions=np.nan)

    skipped = grid.copy()
    skipped["replications"] = 0
    out = ep.run_process_validation(skipped)
    assert out.estimates.empty

    def sim_error(condition, replication, seed):
        raise RuntimeError("sim broke")

    out = ep.run_process_validation(grid, simulate_fun=sim_error)
    assert out.failures.stage.tolist() == ["simulate"]

    def sim_warn(condition, replication, seed):
        warnings.warn("sim warning", RuntimeWarning)
        return ep.simulate_process_validation_data(condition, replication, seed)

    def fit_error(simulated, condition):
        raise RuntimeError("fit broke")

    out = ep.run_process_validation(grid, simulate_fun=sim_warn, fit_fun=fit_error, progress=True)
    assert out.failures.stage.tolist() == ["fit"]
    assert not out.warnings.empty
    assert "validation" in capsys.readouterr().out

    def fit_warn(simulated, condition):
        warnings.warn("fit warning", RuntimeWarning)
        return gov._default_validation_fit(simulated, condition)

    def extract_error(fit, simulated, condition):
        raise RuntimeError("extract broke")

    out = ep.run_process_validation(grid, fit_fun=fit_warn, extract_fun=extract_error)
    assert out.failures.stage.tolist() == ["extract"]
    assert not out.warnings.empty

    def extract_warn_empty(fit, simulated, condition):
        warnings.warn("extract warning", RuntimeWarning)
        return pd.DataFrame()

    out = ep.run_process_validation(grid, fit_fun=fit_warn, extract_fun=extract_warn_empty)
    assert out.estimates.empty
    assert not out.warnings.empty


def test_validation_summary_reference_and_evidence_edge_paths(tmp_path):
    with pytest.raises(EyeProcessValidationError):
        ep.summarise_process_validation({})

    empty = _validation_result()
    assert ep.summarise_process_validation(empty).empty
    assert ep.validation_coverage_table(empty).empty
    assert ep.validation_summary_mcse(empty).empty
    assert ep.validation_condition_ranking(empty).empty
    assert np.isnan(ep.validation_robustness_score(empty))

    est = pd.DataFrame(
        {
            "parameter": ["b", "b", "b"],
            "condition_id": ["C1", "C1", "C2"],
            "estimate": [0.1, 0.2, np.nan],
            "truth": [0.0, 0.0, 0.0],
        }
    )
    x = _validation_result(estimates=est)
    s = ep.summarise_process_validation(x, by=["condition_id"])
    assert s.coverage.isna().all()
    assert np.isfinite(s.loc[s.condition_id == "C1", "convergence_rate"]).all()
    assert len(ep.validation_recovery_table(x, by="condition_id")) == 2

    with pytest.raises(EyeProcessValidationError):
        ep.validation_failure_profile({})
    failures = pd.DataFrame(
        {"condition_id": ["C1", "C1"], "replication": [1, 2], "stage": ["fit", "extract"], "error": ["x", "y"]}
    )
    xd = _validation_result(estimates=est, failures=failures)
    assert set(ep.validation_failure_profile(xd).stage) == {"fit", "extract"}

    with pytest.raises(EyeProcessValidationError, match="weights"):
        ep.validation_condition_ranking(xd, weights={"rmse": 1})

    fitted = ep.run_process_validation(_design(replications=2))
    with pytest.raises(EyeProcessValidationError, match="digits"):
        ep.freeze_validation_reference(fitted, digits=-1)
    path = tmp_path / "reference.pkl"
    ref = ep.freeze_validation_reference(fitted, path=path)
    assert path.exists()
    comparison = ep.validate_against_reference(fitted, path)
    assert comparison["pass"] is True

    with pytest.raises(EyeProcessValidationError, match="tolerance"):
        ep.validate_against_reference(fitted, ref, tolerance=-1)
    with pytest.raises(EyeProcessValidationError, match="reference"):
        ep.validate_against_reference(fitted, {})
    no_keys = gov._result(
        "eye_validation_reference",
        summary=pd.DataFrame({"other": [1.0]}),
        summary_hash="none",
    )
    with pytest.raises(EyeProcessValidationError, match="No common keys"):
        ep.validate_against_reference(fitted, no_keys)

    assert ep.validation_evidence_matrix().empty
    bundle = gov._result("eye_validation_bundle", evidence={"sensitivity": 1, "provenance": 1})
    mat = ep.validation_evidence_matrix(bundle=bundle)
    assert bool(mat.loc[0, "sensitivity"]) and bool(mat.loc[0, "provenance"])


def test_pipeline_constructor_graph_and_execution_error_paths(tmp_path):
    with pytest.raises(EyeProcessValidationError):
        ep.eye_pipeline_step("x", 4)
    with pytest.raises(EyeProcessValidationError, match="requires"):
        ep.eye_pipeline_step("x", lambda: 1, requires=[""])
    with pytest.raises(EyeProcessValidationError, match="cannot require itself"):
        ep.eye_pipeline_step("x", lambda: 1, requires=["x"])
    with pytest.raises(EyeProcessValidationError, match="At least one"):
        ep.eye_analysis_pipeline()
    with pytest.raises(EyeProcessValidationError, match="pipeline_step"):
        ep.eye_analysis_pipeline("bad")

    a = ep.eye_pipeline_step("a", lambda context, spec: 1)
    with pytest.raises(EyeProcessValidationError, match="unique"):
        ep.eye_analysis_pipeline(a, a)
    with pytest.raises(EyeProcessValidationError, match="Unknown pipeline"):
        ep.eye_analysis_pipeline(ep.eye_pipeline_step("b", lambda: 2, requires="missing"))
    ca = ep.eye_pipeline_step("a", lambda b: b, requires="b")
    cb = ep.eye_pipeline_step("b", lambda a: a, requires="a")
    with pytest.raises(EyeProcessValidationError, match="cycle"):
        ep.eye_analysis_pipeline(ca, cb)

    invalid_spec = gov._result(
        "eye_analysis_pipeline", name="bad", steps={"a": a}, spec={}, strict=False, hash="x"
    )
    with pytest.raises(EyeProcessValidationError, match="analysis_spec"):
        ep.validate_eye_pipeline(invalid_spec)

    loose_a = ep.eye_pipeline_step("a", lambda: 1)
    loose_b = ep.eye_pipeline_step("b", lambda a: a + 1)
    loose = ep.eye_analysis_pipeline(loose_a, loose_b, strict=False)
    assert ep.validate_eye_pipeline(loose) is True

    kwargs_pipe = ep.eye_analysis_pipeline(
        ep.eye_pipeline_step("allkw", lambda **kw: len(kw["context"]) + len(kw["spec"].decisions))
    )
    assert ep.pipeline_result(ep.run_eye_pipeline(kwargs_pipe, context={"x": 1}), "allkw") > 0
    with pytest.raises(EyeProcessValidationError, match="context"):
        ep.run_eye_pipeline(kwargs_pipe, context=[])

    opt = ep.eye_analysis_pipeline(
        ep.eye_pipeline_step("bad", lambda: (_ for _ in ()).throw(RuntimeError("optional")), optional=True),
        ep.eye_pipeline_step("ok", lambda: 2),
    )
    opt_run = ep.run_eye_pipeline(opt)
    assert opt_run.completed is True
    assert ep.pipeline_failures(opt_run).status.tolist() == ["optional_error"]

    hard = ep.eye_analysis_pipeline(
        ep.eye_pipeline_step("bad", lambda: (_ for _ in ()).throw(RuntimeError("hard"))),
        ep.eye_pipeline_step("down", lambda bad: bad + 1, requires="bad"),
    )
    hard_run = ep.run_eye_pipeline(hard, stop_on_error=False)
    assert hard_run.completed is False
    assert "bad" in hard_run.errors and "down" in hard_run.errors
    with pytest.raises(EyeProcessValidationError, match="Pipeline step"):
        ep.run_eye_pipeline(hard, stop_on_error=True)
    with pytest.raises(EyeProcessValidationError, match="No successful output"):
        ep.pipeline_result(hard_run, "bad")
    with pytest.raises(EyeProcessValidationError):
        ep.pipeline_result({}, "bad")

    with pytest.raises(EyeProcessValidationError, match="previous"):
        ep.run_eye_pipeline(kwargs_pipe, previous={})
    other = ep.eye_analysis_pipeline(ep.eye_pipeline_step("z", lambda: 1))
    with pytest.raises(EyeProcessValidationError, match="different pipeline"):
        ep.run_eye_pipeline(other, previous=ep.run_eye_pipeline(kwargs_pipe, context={"x": 1}))

    warn_pipe = ep.eye_analysis_pipeline(
        ep.eye_pipeline_step("warn", lambda: warnings.warn("pipeline warning", RuntimeWarning) or 1)
    )
    warned = ep.run_eye_pipeline(warn_pipe)
    assert warned.warnings["warn"]

    undec = ep.eye_analysis_pipeline(
        ep.eye_pipeline_step("a", lambda: 1, decision="not_declared", description="documented")
    )
    assert ep.audit_eye_pipeline(undec).valid is False

    empty_records = EyeResult(
        {
            "pipeline": kwargs_pipe,
            "pipeline_hash": kwargs_pipe.hash,
            "outputs": {},
            "records": pd.DataFrame(),
            "errors": {},
            "warnings": {},
            "context_hash": "x",
        },
        eyeprocess_class="eye_pipeline_run",
    )
    assert ep.pipeline_step_status(empty_records).status.tolist() == ["not_run"]

    report = tmp_path / "pipeline.md"
    export = tmp_path / "pipeline.csv"
    ep.write_eye_pipeline_report(kwargs_pipe, report)
    ep.export_eye_pipeline(kwargs_pipe, export)
    assert report.exists() and export.exists()
    assert "allkw" in ep.eye_pipeline_mermaid(kwargs_pipe)


def test_api_lifecycle_inventory_family_and_recommendation_edge_paths():
    minimal = ep.eye_api_lifecycle(pd.DataFrame({"name": ["x"], "status": ["core"]}))
    assert {"canonical", "replacement", "since", "notes"}.issubset(minimal.columns)

    for frame in [
        pd.DataFrame({"name": [""], "status": ["core"]}),
        pd.DataFrame({"name": ["x"], "status": [""]}),
        pd.DataFrame({"name": ["x", "x"], "status": ["core", "core"]}),
        pd.DataFrame({"name": ["x"], "status": ["impossible"]}),
    ]:
        with pytest.raises(EyeProcessValidationError):
            ep.eye_api_lifecycle(frame)

    fam = ep.api_family_map(
        [
            "plot_x",
            "audit_x",
            "fit_x",
            "read_x",
            "process_demo_spec",
            "compare_x",
            "simulate_x",
            "misc_x",
        ]
    )
    assert set(fam.family) == {"plot", "validation", "model", "io", "workflow", "summary", "simulation", "utility"}

    supplied = pd.DataFrame({"name": ["plot_x"], "family": ["custom"]})
    assert ep.api_family_map(supplied).family.iloc[0] == "custom"
    assert ep.api_family_map(pd.DataFrame({"name": ["fit_x"]})).family.iloc[0] == "model"

    dummy = types.SimpleNamespace(foo=1, bar=lambda: 2)
    inv = ep.eye_api_inventory(dummy, lifecycle=minimal)
    assert {"foo", "bar"}.issubset(set(inv.name))
    assert set(inv.status) == {"unreviewed"}
    assert len(ep.eye_api_inventory("eyeprocesspy")) >= 1182

    reg = ep.eye_api_lifecycle(
        pd.DataFrame(
            {
                "name": ["a", "b", "c"],
                "status": ["deprecated", "core", "compatibility"],
                "canonical": [np.nan, "b", np.nan],
                "replacement": ["b", np.nan, np.nan],
            }
        )
    )
    assert set(ep.eye_api_superseded(reg).name) == {"a", "c"}
    assert "b" in set(ep.canonical_eye_api(reg).name)
    assert ep.eye_api_status(["a", "missing"], reg).status.tolist() == ["deprecated", "unreviewed"]

    audit = ep.audit_eye_api(pd.DataFrame({"name": ["a", "b"]}), reg)
    rec = ep.eye_api_recommendation(audit).set_index("name").recommendation
    assert rec["a"] == "retain"

    unreviewed = ep.audit_eye_api(pd.DataFrame({"name": ["z"]}), reg)
    assert ep.eye_api_recommendation(unreviewed).recommendation.iloc[0] == "classify"

    badreg = ep.register_eye_api_status(reg, "a", "deprecated", replacement="missing")
    bad = ep.audit_eye_api(pd.DataFrame({"name": ["a", "b"]}), badreg)
    assert ep.eye_api_recommendation(bad).set_index("name").loc["a"] == "repair_replacement"

    with pytest.raises(EyeProcessValidationError):
        ep.eye_api_recommendation({})


def test_sensitivity_grid_runner_and_stability_error_paths(capsys):
    with pytest.raises(EyeProcessValidationError):
        ep.process_sensitivity_grid("bad", method=("a", "b"))
    with pytest.raises(EyeProcessValidationError):
        ep.process_sensitivity_grid()
    with pytest.raises(EyeProcessValidationError):
        ep.process_sensitivity_grid(method=[])
    with pytest.raises(EyeProcessValidationError):
        ep.process_sensitivity_grid(method=("a",), max_specifications=-1)
    with pytest.raises(EyeProcessValidationError, match="expands"):
        ep.process_sensitivity_grid(method=("a", "b"), threshold=(1, 2), max_specifications=2)

    assert gov._default_sensitivity_extract(1.5, None).effect.iloc[0] == 1.5
    frame = pd.DataFrame({"effect": [0.2]})
    assert gov._default_sensitivity_extract(frame, None).equals(frame)
    assert gov._default_sensitivity_extract({"effect": 0.3, "array": [1, 2]}, None).effect.iloc[0] == 0.3
    with pytest.raises(EyeProcessValidationError):
        gov._default_sensitivity_extract(object(), None)

    grid = ep.process_sensitivity_grid(mode=("ok", "analysis_fail", "extract_fail", "empty"))
    with pytest.raises(EyeProcessValidationError, match="analysis_fun"):
        ep.run_process_sensitivity({}, grid, 3)
    with pytest.raises(EyeProcessValidationError, match="extract_fun"):
        ep.run_process_sensitivity({}, grid, lambda d, s: 1, extract_fun=3)
    with pytest.raises(EyeProcessValidationError, match="specification_id"):
        ep.run_process_sensitivity({}, pd.DataFrame({"x": [1]}), lambda d, s: 1)

    def analysis(data, spec):
        mode = str(spec.iloc[0].mode)
        if mode == "analysis_fail":
            raise RuntimeError("analysis failed")
        warnings.warn("analysis warning", RuntimeWarning)
        return 0.25 if mode != "empty" else 0.0

    def extract(fit, spec):
        mode = str(spec.iloc[0].mode)
        if mode == "extract_fail":
            raise RuntimeError("extract failed")
        warnings.warn("extract warning", RuntimeWarning)
        if mode == "empty":
            return pd.DataFrame()
        return pd.DataFrame({"effect": [fit], "p_value": [0.04]})

    run = ep.run_process_sensitivity({}, grid, analysis, extract, progress=True)
    assert len(run.results) == 1
    assert len(run.failures) == 2
    assert not run.warnings.empty
    assert "sensitivity" in capsys.readouterr().out

    no_effect = _sensitivity_result(results=pd.DataFrame({"specification_id": ["S1"], "effect": [np.nan], "p_value": [np.nan]}))
    assert np.isnan(ep.sensitivity_sign_stability(no_effect))
    assert np.isnan(ep.sensitivity_significance_stability(no_effect))
    assert np.isnan(ep.sensitivity_threshold_stability(no_effect))
    assert np.isnan(ep.sensitivity_fragility_index(no_effect))

    with pytest.raises(EyeProcessValidationError, match="alpha"):
        ep.sensitivity_significance_stability(no_effect, alpha=-0.1)
    with pytest.raises(EyeProcessValidationError, match="direction"):
        ep.sensitivity_threshold_stability(no_effect, direction="sideways")
    with pytest.raises(EyeProcessValidationError, match="threshold"):
        ep.sensitivity_threshold_stability(no_effect, threshold=np.nan)

    empty = _sensitivity_result()
    stable = ep.decision_stability(empty)
    assert stable.stable_sign is None
    assert np.isnan(ep.specification_coverage(_sensitivity_result(grid=pd.DataFrame({"specification_id": []}))))
    with pytest.raises(EyeProcessValidationError):
        ep.specification_coverage({})

    intervals = _sensitivity_result(
        results=pd.DataFrame(
            {
                "specification_id": ["S00001", "S00002"],
                "method": ["a", "b"],
                "effect": [0.1, 0.2],
                "lo": [0.0, 0.1],
                "hi": [0.2, 0.3],
            }
        )
    )
    curve = ep.specification_curve_data(intervals, lower="lo", upper="hi")
    assert {".lower", ".upper"}.issubset(curve.columns)

    assert np.isnan(ep.sensitivity_rank_stability([[1, 2]]))
    with pytest.raises(EyeProcessValidationError, match="equal length"):
        ep.sensitivity_rank_stability([[1, 2], [1]])
    rank_df = pd.DataFrame(
        {
            "item": ["a", "b", "a", "b"],
            "rank": [1, 2, 2, 1],
            "spec": ["s1", "s1", "s2", "s2"],
        }
    )
    assert np.isfinite(ep.sensitivity_rank_stability(rank_df, id="item", rank="rank", specification="spec"))
    assert np.isnan(
        ep.sensitivity_rank_stability(
            pd.DataFrame({"item": ["a"], "rank": [1], "spec": ["s1"]}),
            id="item",
            rank="rank",
            specification="spec",
        )
    )
    with pytest.raises(EyeProcessValidationError, match="methods"):
        gov._compare_methods({}, ["a"], lambda *args: 1, gov._default_sensitivity_extract, "x")


def test_decision_manifest_validation_flatten_io_blinding_and_entropy_paths(tmp_path):
    with pytest.raises(EyeProcessValidationError):
        ep.validate_decision_manifest({})
    x = ep.eye_decision_manifest()
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        ep.validate_decision_manifest(x, required_domains=["absent"])
    with pytest.raises(EyeProcessValidationError, match="empty"):
        ep.validate_decision_manifest(x, required_domains=["sampling"], require_nonempty=True)

    complex_manifest = ep.eye_decision_manifest(
        sampling={"rate": 60, "nested": {}},
        model={"families": ["gaussian", "robust"], "none": None, "empty": []},
        provenance={"data_source": "demo", "software_version": "0.1.0", "analysis_commit": "abc"},
    )
    table = ep.decision_manifest_table(complex_manifest)
    assert table.path.str.contains(r"\[\[1\]\]", regex=True).any()
    assert table.value.isna().any()
    assert gov.decision_manifest_hash({"a": 1})

    with pytest.raises(EyeProcessValidationError):
        ep.verify_decision_manifest_lock({})
    lock = ep.lock_decision_manifest(complex_manifest)
    lock.manifest.domains["model"]["families"] = ["changed"]
    assert ep.verify_decision_manifest_lock(lock) is False

    a = ep.eye_decision_manifest(model={"x": 1})
    b = ep.eye_decision_manifest(model={"y": 2})
    diff = ep.compare_decision_manifests(a, b)
    assert diff.changed.any()

    with pytest.raises(EyeProcessValidationError, match="format"):
        ep.write_decision_manifest(a, tmp_path / "bad", format="yaml")
    rds = tmp_path / "manifest.rds"
    dput = tmp_path / "manifest.txt"
    ep.write_decision_manifest(a, rds, format="rds")
    ep.write_decision_manifest(a, dput, format="dput")
    assert ep.read_decision_manifest(rds).hash == a.hash
    assert ep.read_decision_manifest(dput).hash == a.hash

    tampered = tmp_path / "tampered.json"
    payload = gov._manifest_payload(a)
    payload["domains"]["model"]["x"] = 999
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EyeProcessValidationError, match="hash"):
        ep.read_decision_manifest(tampered, format="json")

    full = ep.eye_decision_manifest(
        sampling={"x": 1},
        validity={"x": 1},
        fixation={"x": 1},
        pupil={"x": 1},
        aoi={"x": 1},
        model={"x": 1},
        sensitivity={"x": 1},
        exclusions={"x": 1},
        provenance={"data_source": "demo", "software_version": "0.1.0", "analysis_commit": "abc"},
    )
    assert ep.audit_decision_provenance(full).complete is True

    data = pd.DataFrame({"id": [1, 2], "x": [3, 4], "y": [5, 6]})
    with pytest.raises(EyeProcessValidationError):
        ep.outcome_blind_snapshot(data, "missing")
    with pytest.raises(EyeProcessValidationError):
        ep.outcome_blind_snapshot(data, "y", id="missing")
    snap = ep.outcome_blind_snapshot(data, ["y"], id=["id"])
    assert ep.verify_outcome_blind_snapshot(snap) is True
    with pytest.raises(EyeProcessValidationError):
        ep.verify_outcome_blind_snapshot({})
    snap.data.loc[0, "x"] = 999
    assert ep.verify_outcome_blind_snapshot(snap) is False

    with pytest.raises(EyeProcessValidationError):
        ep.analysis_decision_entropy("bad", method=("a", "b"))
    with pytest.raises(EyeProcessValidationError):
        ep.analysis_decision_entropy()
    for base in [1, 0, np.nan]:
        with pytest.raises(EyeProcessValidationError, match="base"):
            ep.analysis_decision_entropy(base=base, method=("a", "b"))
    with pytest.raises(EyeProcessValidationError, match="empty"):
        ep.analysis_decision_entropy(method=[])

    grid = ep.process_sensitivity_grid(method=("a", "b"))
    sens = _sensitivity_result(
        grid=grid,
        results=pd.DataFrame({"specification_id": ["S00001"], "effect": [0.1]}),
    )
    assert ep.decision_space_coverage(grid, sens).coverage.iloc[0] == 0.5
    assert np.isnan(ep.decision_space_coverage(pd.DataFrame({"specification_id": []}), []).coverage.iloc[0])


def test_governance_plot_empty_invalid_and_alternate_paths():
    ax = plt.subplots()[1]
    assert gplots._axis(ax) is ax
    empty_ax = gplots._empty(ax, title="Empty")
    assert empty_ax.eyeprocess_plot_data.empty
    plt.close(ax.figure)

    with pytest.raises(EyeProcessValidationError, match="type"):
        ep.plot_eye_process_validation_result(_validation_result(), type="unknown")

    x = _validation_result()
    ax = ep.plot_eye_process_validation_result(x, type="recovery")
    assert ax.eyeprocess_plot_data.empty
    plt.close(ax.figure)

    fitted = ep.run_process_validation(_design())
    ax = ep.plot_eye_process_validation_result(fitted, type="failure")
    assert ax.eyeprocess_plot_data.empty
    plt.close(ax.figure)
    ax = ep.plot_eye_process_validation_result(fitted, type="recovery", parameter="missing")
    assert ax.eyeprocess_plot_data.empty
    plt.close(ax.figure)

    est = pd.DataFrame(
        {
            "parameter": ["b"],
            "estimate": [np.nan],
            "truth": [np.nan],
            "lower": [np.nan],
            "upper": [np.nan],
        }
    )
    nan_result = _validation_result(estimates=est)
    ax = ep.plot_eye_process_validation_result(nan_result, type="recovery")
    assert len(ax.eyeprocess_plot_data) == 1
    plt.close(ax.figure)
    ax = ep.plot_eye_process_validation_result(nan_result, type="coverage")
    assert ax.eyeprocess_plot_data.empty
    plt.close(ax.figure)

    ref = ep.freeze_validation_reference(fitted)
    comp = ep.validate_against_reference(fitted, ref)
    ax = ep.plot_eye_validation_reference_comparison(comp, metric="not_here")
    assert ax.eyeprocess_plot_data.empty
    plt.close(ax.figure)

    p = ep.eye_analysis_pipeline(ep.eye_pipeline_step("a", lambda: 1))
    audit = ep.audit_eye_pipeline(p)
    ax = ep.plot_eye_pipeline_audit(audit)
    assert hasattr(ax, "eyeprocess_plot_data")
    plt.close(ax.figure)

    empty_api_audit = gov._result("eye_api_audit", table=pd.DataFrame())
    ax = ep.plot_eye_api_audit(empty_api_audit)
    assert ax.eyeprocess_plot_data.empty
    plt.close(ax.figure)

    with pytest.raises(EyeProcessValidationError, match="type"):
        ep.plot_eye_process_sensitivity(_sensitivity_result(), type="wrong")
    no_decisions = _sensitivity_result(
        grid=pd.DataFrame({"specification_id": ["S1"]}),
        results=pd.DataFrame({"specification_id": ["S1"], "effect": [0.1]}),
    )
    ax = ep.plot_eye_process_sensitivity(no_decisions, type="decision_leverage")
    assert ax.eyeprocess_plot_data.empty
    plt.close(ax.figure)

    intervals = _sensitivity_result(
        results=pd.DataFrame(
            {
                "specification_id": ["S00001", "S00002"],
                "method": ["a", "b"],
                "effect": [0.1, 0.2],
                "lo": [0.0, 0.1],
                "hi": [0.2, 0.3],
            }
        )
    )
    ax = ep.plot_eye_process_sensitivity(intervals, lower="lo", upper="hi")
    assert {".lower", ".upper"}.issubset(ax.eyeprocess_plot_data.columns)
    plt.close(ax.figure)

    blank_stability = gov._result("eye_decision_stability", summary=pd.DataFrame({"other": [1]}))
    ax = ep.plot_eye_decision_stability(blank_stability)
    assert ax.eyeprocess_plot_data.empty
    plt.close(ax.figure)
