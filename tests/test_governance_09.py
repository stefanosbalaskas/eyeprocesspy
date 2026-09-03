from __future__ import annotations

import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.irt import EyeResult

SOURCE_FILES = {
    "R/069-validation-program-0-9.R",
    "R/070-governed-pipelines-0-9.R",
    "R/071-api-lifecycle-0-9.R",
    "R/072-sensitivity-multiverse-0-9.R",
    "R/073-decision-manifests-0-9.R",
}


def _frozen_exports():
    p = pd.read_csv(Path(__file__).resolve().parents[1] / "parity" / "PARITY_MATRIX.csv")
    return p.loc[p.source_file.isin(SOURCE_FILES), "r_name"].tolist()


def _param_names(fn):
    out = []
    for p in inspect.signature(fn).parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            out.append("...")
        else:
            out.append(p.name)
    # Python cannot place R's ... before later named arguments; compare sets/order
    # for ordinary arguments while requiring a variadic parameter where R has ... .
    return out


def test_all_80_frozen_governance_exports_resolve_and_signatures_cover_r_arguments():
    exports = _frozen_exports()
    assert len(exports) == 80
    sigs = json.loads((Path(__file__).resolve().parents[1] / "reference" / "R_SIGNATURES.json").read_text())
    for name in exports:
        fn = getattr(ep, name, None)
        assert callable(fn), name
        py = _param_names(fn)
        rr = ["..." if a["name"] == "..." else a["name"] for a in sigs[name]["args"]]
        if "..." not in rr:
            assert py == rr, (name, py, rr)
        else:
            assert "..." in py, name
            assert [z for z in py if z != "..."] == [z for z in rr if z != "..."], (name, py, rr)


def _small_design(misspec=False, reps=2, seed=9):
    return ep.process_validation_design(
        n_persons=20, n_trials=8, missingness=0,
        sampling_rate_hz=60, aoi_error="low", calibration_error=0,
        pupil_dropout=0, heterogeneity="low",
        model_misspecification=misspec, replications=reps, seed=seed,
    )


def test_validation_design_expands_runs_freezes_and_matches_reference():
    d = _small_design()
    assert ep.validate_process_validation_design(d) is True
    g = ep.expand_process_validation_design(d)
    assert d.eyeprocess_class == "eye_process_validation_design"
    assert len(g) == 1
    assert ep.validation_condition_id(g) == g.condition_id.astype(str).tolist()
    x = ep.run_process_validation(g)
    assert x.eyeprocess_class == "eye_process_validation_result"
    assert len(x.estimates) > 0
    assert {"bias", "rmse"}.issubset(ep.summarise_process_validation(x).columns)
    assert len(ep.validation_recovery_table(x)) == 1
    assert len(ep.validation_coverage_table(x)) == 1
    assert len(ep.validation_summary_mcse(x)) == 1
    assert isinstance(ep.validation_condition_ranking(x), pd.DataFrame)
    assert isinstance(ep.validation_failure_profile(x), pd.DataFrame)
    assert np.isfinite(ep.validation_robustness_score(x))
    ref = ep.freeze_validation_reference(x)
    cmp = ep.validate_against_reference(x, ref, tolerance=1e-8)
    assert cmp["pass"] is True
    mat = ep.validation_evidence_matrix(model=x)
    assert bool(mat.loc[0, "recovery"])


def test_validation_misspecification_changes_dgp_and_reference_requires_complete_keys():
    d = ep.process_validation_design(
        n_persons=20, n_trials=8, missingness=0,
        sampling_rate_hz=60, aoi_error="low", calibration_error=0,
        pupil_dropout=0, heterogeneity="low",
        model_misspecification=(False, True), replications=1, seed=19,
    )
    g = ep.expand_process_validation_design(d)
    ok = ep.simulate_process_validation_data(g[g.model_misspecification == False], 1, 919)  # noqa: E712
    miss = ep.simulate_process_validation_data(g[g.model_misspecification == True], 1, 919)  # noqa: E712
    np.testing.assert_allclose(ok.data.omitted_structure, miss.data.omitted_structure)
    assert not np.allclose(ok.data.process_value, miss.data.process_value)

    x = ep.run_process_validation(_small_design(reps=2, seed=29))
    ref = ep.freeze_validation_reference(x)
    extra = ref.summary.iloc[[0]].copy(); extra["condition_id"] = "C_MISSING_FROM_CURRENT"
    ref["summary"] = pd.concat([ref.summary, extra], ignore_index=True)
    cmp = ep.validate_against_reference(x, ref, tolerance=1e-8)
    assert cmp["pass"] is False
    assert cmp.table[".present_current"].isna().any()


def test_governed_pipeline_dependencies_outputs_resumption_and_exports(tmp_path):
    spec = ep.eye_analysis_spec(method="declared")
    p = ep.eye_analysis_pipeline(
        [
            ep.eye_pipeline_step("a", lambda context, spec: 2),
            ep.eye_pipeline_step("b", lambda a, context, spec: a + 3, requires="a"),
        ],
        spec=spec,
    )
    assert ep.validate_eye_pipeline(p) is True
    graph = ep.eye_pipeline_graph(p)
    assert graph.vertices.step.tolist() == ["a", "b"]
    assert len(graph.edges) == 1
    manifest = ep.eye_pipeline_manifest(p)
    assert manifest.execution_order.tolist() == [1, 2]
    r = ep.run_eye_pipeline(p)
    status = ep.pipeline_step_status(r)
    assert set(status.status) == {"success"}
    assert ep.pipeline_failures(r).empty
    assert ep.pipeline_result(r, "b") == 5
    assert r.completed is True
    assert ep.audit_eye_pipeline(r).valid is True
    assert "digraph" in ep.eye_pipeline_dot(p)
    assert "flowchart" in ep.eye_pipeline_mermaid(p)
    assert len(ep.eye_targets_manifest(p)) == 2
    assert Path(ep.write_eye_targets_template(p, tmp_path / "_targets.R")).exists()
    assert Path(ep.write_eye_pipeline_report(r, tmp_path / "report.md")).exists()
    assert Path(ep.export_eye_pipeline(r, tmp_path / "pipeline.csv")).exists()

    p2 = ep.eye_analysis_pipeline([ep.eye_pipeline_step("dot", lambda _context, _spec: len(_context) + len(_spec.decisions))], spec=spec)
    assert np.isfinite(ep.pipeline_result(ep.run_eye_pipeline(p2, context={"x": 1}), "dot"))
    with pytest.raises(Exception, match="syntactic"):
        ep.eye_pipeline_step("not valid", lambda x: x)

    a = ep.eye_pipeline_step("a", lambda context, spec: 1)
    b = ep.eye_pipeline_step("b", lambda a, context, spec: a + 1)
    with pytest.raises(Exception, match="not declared"):
        ep.eye_analysis_pipeline([a, b], strict=True)

    p3 = ep.eye_analysis_pipeline([ep.eye_pipeline_step("a", lambda context, spec: context["value"])])
    rr = ep.run_eye_pipeline(p3, context={"value": 2})
    assert ep.pipeline_result(ep.resume_eye_pipeline(p3, rr, context={"value": 2}), "a") == 2
    with pytest.raises(Exception, match="different context"):
        ep.resume_eye_pipeline(p3, rr, context={"value": 3})


def test_api_lifecycle_registry_is_frozen_complete_and_auditable(tmp_path):
    reg = ep.eye_api_lifecycle()
    assert len(reg) == 1182
    assert reg.name.nunique() == 1182
    assert not (reg.status == "unreviewed").any()
    reg2 = ep.register_eye_api_status(reg, "run_eye_pipeline", "workflow", canonical="run_eye_pipeline")
    assert ep.eye_api_status("run_eye_pipeline", reg2).status.iloc[0] == "workflow"
    assert ep.eye_api_status("unknown_symbol", reg2).status.iloc[0] == "unreviewed"
    assert "run_eye_pipeline" in set(ep.canonical_eye_api(reg2).name)
    lifecycle_diff = ep.api_lifecycle_diff(reg, reg2)
    assert len(lifecycle_diff) == 1182
    assert {"name", "old_status", "new_status", "changed"}.issubset(lifecycle_diff.columns)
    assert isinstance(ep.eye_api_superseded(reg2), pd.DataFrame)
    inv = ep.eye_api_inventory()
    assert len(inv) == 1182
    audit = ep.audit_eye_api(inv, reg)
    assert audit.reviewed_fraction == 1
    assert audit.valid is True
    assert not audit.unreviewed
    assert len(ep.api_surface_summary(inv)) > 0
    assert len(ep.api_family_map(inv)) == len(inv)
    path = tmp_path / "life.csv"
    ep.write_api_lifecycle_registry(reg, path)
    assert len(ep.read_api_lifecycle_registry(path)) == 1182

    small = pd.DataFrame({"name": ["run_eye_pipeline"], "status": ["unreviewed"], "canonical": [np.nan], "replacement": [np.nan]})
    badreg = ep.register_eye_api_status(reg, "run_eye_pipeline", "workflow", canonical="missing_canonical_api")
    bad = ep.audit_eye_api(small, badreg)
    assert bad.valid is False
    assert bad.invalid_canonical == ["run_eye_pipeline"]
    assert ep.eye_api_recommendation(bad).recommendation.iloc[0] == "repair_canonical"


def test_sensitivity_grid_run_summary_stability_and_comparison_helpers():
    g = ep.process_sensitivity_grid(method=("a", "b", "c"))
    x = EyeResult({
        "grid": g,
        "results": pd.DataFrame({"specification_id": g.specification_id, "effect": [.2, .3, .1], "p_value": [.04, .01, .2], "method": ["a", "b", "c"]}),
        "failures": pd.DataFrame(), "warnings": pd.DataFrame(), "grid_hash": "x",
    }, eyeprocess_class="eye_process_sensitivity")
    s = ep.summarise_process_sensitivity(x, p_value="p_value")
    assert s.specifications.iloc[0] == 3
    assert np.isfinite(ep.sensitivity_sign_stability(x))
    assert ep.sensitivity_significance_stability(x, alpha=.05) == pytest.approx(2 / 3)
    assert ep.sensitivity_threshold_stability(x, threshold=.15) == pytest.approx(2 / 3)
    assert ep.decision_stability(x, p_value="p_value").eyeprocess_class == "eye_decision_stability"
    assert ep.sensitivity_rank_stability([(1, 2, 3), (1, 3, 2)]) == pytest.approx(.5)
    assert len(ep.specification_curve_data(x)) == 3
    assert ep.specification_coverage(x) == 1
    assert len(ep.sensitivity_decision_leverage(x)) == 1
    assert 0 <= ep.sensitivity_fragility_index(x) <= 1
    assert len(ep.sensitivity_multiverse_manifest(x)) == 3
    assert isinstance(ep.sensitivity_branch_fingerprint(g.iloc[[0]]), str)

    data = pd.DataFrame({"v": [1, 2, 3]})
    run = ep.run_process_sensitivity(data, g, lambda d, spec: float({"a": .2, "b": .3, "c": .1}[spec.method.iloc[0]]))
    assert len(run.results) == 3
    methods = {"m1": 1, "m2": 2}
    analysis = lambda d, method, spec: float(method)
    assert len(ep.compare_aoi_methods(data, methods, analysis).results) == 2
    assert len(ep.compare_fixation_methods(data, methods, analysis).results) == 2
    assert len(ep.compare_pupil_preprocessing(data, methods, analysis).results) == 2
    assert len(ep.compare_process_models(data, methods, analysis).results) == 2


def test_decision_manifests_hash_lock_compare_io_blinding_entropy_and_coverage(tmp_path):
    x = ep.eye_decision_manifest(preprocessing={"blink": "linear"}, model={"family": "gaussian"})
    assert x.eyeprocess_class == "eye_decision_manifest"
    assert ep.validate_decision_manifest(x) is True
    assert ep.decision_manifest_hash(x)
    lock = ep.lock_decision_manifest(x)
    assert ep.verify_decision_manifest_lock(lock) is True
    y = ep.eye_decision_manifest(preprocessing={"blink": "none"}, model={"family": "gaussian"})
    assert len(ep.compare_decision_manifests(x, y)) >= 1
    assert len(ep.decision_manifest_diff(x, y)) >= 1
    assert len(ep.decision_manifest_table(x)) > 0
    p = tmp_path / "manifest.json"
    ep.write_decision_manifest(x, p, format="json")
    xr = ep.read_decision_manifest(p, format="json")
    assert xr.hash == x.hash
    snap = ep.outcome_blind_snapshot(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), outcome="y")
    assert ep.verify_outcome_blind_snapshot(snap) is True
    ent = ep.analysis_decision_entropy(method=("a", "b"), threshold=(0, 1), base=2)
    assert ent.attrs["joint_specifications"] == 4
    grid = ep.process_sensitivity_grid(method=("a", "b"))
    cov = ep.decision_space_coverage(grid, ["S00001"])
    assert cov.coverage.iloc[0] == .5
    audit = ep.audit_decision_provenance(x)
    assert audit.complete is False


def test_governance_plot_counterparts_return_axes_with_plot_data():
    d = _small_design(reps=1)
    ax = ep.plot_eye_process_validation_design(d); assert hasattr(ax, "eyeprocess_plot_data"); plt.close(ax.figure)
    x = ep.run_process_validation(d)
    for kind in ["recovery", "bias", "coverage", "failure"]:
        ax = ep.plot_eye_process_validation_result(x, type=kind); assert hasattr(ax, "eyeprocess_plot_data"); plt.close(ax.figure)
    ref = ep.freeze_validation_reference(x); cmp = ep.validate_against_reference(x, ref)
    ax = ep.plot_eye_validation_reference_comparison(cmp); assert hasattr(ax, "eyeprocess_plot_data"); plt.close(ax.figure)

    p = ep.eye_analysis_pipeline([ep.eye_pipeline_step("a", lambda context, spec: 1)])
    ax = ep.plot_eye_analysis_pipeline(p); assert hasattr(ax, "eyeprocess_plot_data"); plt.close(ax.figure)
    audit = ep.audit_eye_pipeline(ep.run_eye_pipeline(p))
    ax = ep.plot_eye_pipeline_audit(audit); assert hasattr(ax, "eyeprocess_plot_data"); plt.close(ax.figure)
    aa = ep.audit_eye_api(ep.eye_api_inventory().head(20), ep.eye_api_lifecycle())
    ax = ep.plot_eye_api_audit(aa); assert hasattr(ax, "eyeprocess_plot_data"); plt.close(ax.figure)

    sg = ep.process_sensitivity_grid(method=("a", "b"))
    sx = EyeResult({"grid": sg, "results": pd.DataFrame({"specification_id": sg.specification_id, "effect": [.1, .2], "method": ["a", "b"]}), "failures": pd.DataFrame(), "warnings": pd.DataFrame()}, eyeprocess_class="eye_process_sensitivity")
    ax = ep.plot_eye_process_sensitivity(sx, type="decision_leverage"); assert hasattr(ax, "eyeprocess_plot_data"); plt.close(ax.figure)
    ds = ep.decision_stability(sx)
    ax = ep.plot_eye_decision_stability(ds); assert hasattr(ax, "eyeprocess_plot_data"); plt.close(ax.figure)
    dm = ep.eye_decision_manifest(model={"family": "gaussian"})
    ax = ep.plot_eye_decision_manifest(dm); assert hasattr(ax, "eyeprocess_plot_data"); plt.close(ax.figure)
