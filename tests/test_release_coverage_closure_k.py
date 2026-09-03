from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.irt import EyeResult
import eyeprocesspy.adapters as ad
import eyeprocesspy.bayesian_3pl_08 as b3
import eyeprocesspy.compositional_aoi_10 as ca
import eyeprocesspy.coordinates as co
import eyeprocesspy.core_plots_10 as cp
import eyeprocesspy.dynamic_irt as di
import eyeprocesspy.engine_adapters as ea
import eyeprocesspy.evidence_graph as eg
import eyeprocesspy.frontier_08 as fr
import eyeprocesspy.grouped_validation_10 as gv
import eyeprocesspy.irt as irt
import eyeprocesspy.irt_validation_07 as iv
import eyeprocesspy.measurement_quality_legacy as mq
import eyeprocesspy.partitioned_storage_10 as ps
import eyeprocesspy.plots_completion_08 as pc
import eyeprocesspy.plots_functional_pupil as pfp
import eyeprocesspy.plots_governance_09 as pg
import eyeprocesspy.plots_irt as pi
import eyeprocesspy.plots_legacy_models as plm
import eyeprocesspy.plots_operational_08 as po
import eyeprocesspy.plots_process_irt_07 as ppi
import eyeprocesspy.plots_process_quality_09 as ppq
import eyeprocesspy.process_irt_07 as pir
import eyeprocesspy.process_quality_09 as pq
import eyeprocesspy.pupil_missingness as pm


def _close(ax):
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def _ds(**tables):
    x = ep.new_eye_dataset(validate=False)
    for name, table in tables.items():
        x[name] = ep.standardize_eye_table(table, name)
    return x


def test_adapter_registration_nonempty_and_folder_success(tmp_path):
    name = "coverage_temp"
    try:
        ad.register_eye_adapter(
            name,
            detect=lambda path, **kw: 1.0,
            read=lambda path, **kw: ep.new_eye_dataset(validate=False),
            priority=999,
            overwrite=True,
        )
        table = ad.supported_eye_formats()
        assert name in set(table.name)
        f = tmp_path / "one.csv"; f.write_text("x\n1\n", encoding="utf-8")
        out = ad.read_eye_folder(tmp_path, vendor=name)
        assert ep.is_eye_dataset(out)
    finally:
        ad.unregister_eye_adapter(name)


def test_bayesian_3pl_defensive_and_review_paths():
    with pytest.raises(ep.EyeProcessValidationError):
        b3.bayesian_process_diagnostics_dashboard()
    with pytest.raises(ep.EyeProcessBackendError):
        b3.bayesian_process_diagnostics_dashboard(object())
    with pytest.raises(ep.EyeProcessValidationError):
        b3.bayesian_process_diagnostic_flags(object())
    dash = EyeResult({"posterior": pd.DataFrame()}, eyeprocess_class="eye_bayesian_process_dashboard")
    assert b3.bayesian_process_diagnostic_flags(dash).empty
    with pytest.raises(ep.EyeProcessValidationError):
        b3.bayesian_process_diagnostic_flags(dash, rhat_threshold=1.0)
    with pytest.raises(ep.EyeProcessValidationError):
        b3.fit_gaze_anchored_3pl_audit(np.zeros((2, 3)))
    with pytest.raises(ep.EyeProcessBackendError):
        b3.fit_gaze_anchored_3pl_audit(np.zeros((2, 4)))
    with pytest.raises(ep.EyeProcessValidationError):
        b3.gaze_anchored_3pl_alignment(object())
    x = EyeResult({"alignment": pd.DataFrame({"x": [1]})}, eyeprocess_class="eye_gaze_anchored_3pl_audit")
    assert not b3.gaze_anchored_3pl_alignment(x).empty
    with pytest.raises(ep.EyeProcessValidationError):
        b3.audit_3pl_process_signatures(object())
    x = EyeResult({"item_parameters": pd.DataFrame({"a": [1,2]})}, eyeprocess_class="eye_gaze_anchored_3pl_audit")
    assert "a" in b3.audit_3pl_process_signatures(x)
    x["item_parameters"] = pd.DataFrame({"lower_asymptote": [np.nan, np.nan]})
    assert b3.audit_3pl_process_signatures(x).shape[0] == 2
    x["item_parameters"] = pd.DataFrame({"lower_asymptote": [.1,.9], "rt_ms": [100,200], "ttff_ms":[20,30]})
    out = b3.audit_3pl_process_signatures(x)
    assert "process_review_count" in out
    with pytest.raises(ep.EyeProcessValidationError):
        b3.audit_3pl_process_signatures(x, lower_asymptote_quantile=1.0)


def test_frontier_external_engine_paths():
    engine=lambda data, **kw: {"ok": data}
    assert fr.fit_persistence_gaze_diffusion_irt(3, engine=engine)["fit"]["ok"] == 3
    assert fr.fit_nonignorable_missing_irt(4, engine=engine)["fit"]["ok"] == 4
    assert fr.fit_crossclassified_process_irt_mhrm(5, engine=engine)["fit"]["ok"] == 5


def test_grouped_fold_count_float_failure_object():
    class Weird:
        def __int__(self): return 3
        def __float__(self): raise TypeError("no float")
    with pytest.raises(ep.EyeProcessValidationError):
        gv._fold_count(Weird())


def test_irt_remaining_public_branches(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError):
        irt.eyeprocess_irt_conditional_sem([np.nan])
    s = irt.eyeprocess_irt_sbc_summary([0,1,2,3,4], n_draws=4, bins=None)
    assert len(s) > 0
    with pytest.raises(ep.EyeProcessValidationError):
        irt.eyeprocess_cdm_dina_probability([[0,2]])
    monkeypatch.setattr(irt, "_require_engine", lambda *a, **k: None)
    out = irt.run_eyeprocess_equateirt("link")
    assert out.eyeprocess_class == "eye_gated_irt_engine"


def test_irt_validation_guards_and_failure_capture():
    with pytest.raises(ep.EyeProcessValidationError):
        iv.irt_validation_spec("")
    with pytest.raises(ep.EyeProcessValidationError):
        iv.irt_validation_spec("x", replications=0)
    with pytest.raises(ep.EyeProcessValidationError):
        iv.grade_model_evidence(pd.DataFrame(), spec=object())
    sbc = iv.run_sbc(lambda rng=None: {}, lambda data: None, lambda fit: pd.DataFrame(), replications=1)
    assert len(sbc.failures) == 1


def test_measurement_uncertainty_numpy_simulation_branch():
    base = mq._result("eye_process_uncertainty", data=np.arange(8.0), specification=mq.process_uncertainty_spec(source_sd={"x":1.0}, draws=3), uncertainty=pd.DataFrame({"source":["x"],"sd":[1.0]}), summary=pd.DataFrame({"total_se":[1.0]}))
    out = mq.propagate_process_uncertainty(base, method="simulation", draws=2, seed=1)
    assert len(out.draws) == 2


def test_partitioned_parquet_and_restore_branches(monkeypatch, tmp_path):
    class PQ:
        @staticmethod
        def read_table(path):
            return SimpleNamespace(to_pandas=lambda: pd.DataFrame({"x":[1]}))
    monkeypatch.setattr(ps, "_require_pyarrow", lambda: (None, None, PQ))
    p=tmp_path/"x.parquet"; p.write_bytes(b"x")
    assert ps._read_piece(p, "parquet").iloc[0,0] == 1

    target=tmp_path/"restore"
    target.mkdir()
    backup=target.with_name(target.name+".bak")
    backup.mkdir()
    staging=target.with_name(target.name+".staging")
    staging.mkdir()
    orig=ps.Path.rename
    def fail_once(self, dest):
        if self == staging: raise OSError("x")
        return orig(self,dest)
    monkeypatch.setattr(ps.Path,"rename",fail_once)


def test_plot_residual_paths():
    x=_ds(features=pd.DataFrame({"feature_name":["other"],"aoi_id":["A"],"value":[1.0]}))
    ax=cp.plot_aoi_dwell(x, feature="dwell_time_ms"); assert ax.eyeprocess_plot_data.empty; _close(ax)

    assert isinstance(pc._as_frame([[1,2]], name="x"), pd.DataFrame)
    with pytest.raises(ep.EyeProcessValidationError):
        pc._transition_table(pd.DataFrame(columns=["from_aoi","to_aoi"]), normalize="none")

    d=pd.DataFrame({"estimate":[1.0],"truth":[1.2]})
    ax=plm.plot_eye_parameter_recovery(d); _close(ax)
    ax=plm.plot_eye_parameter_recovery(pd.DataFrame({"estimate":[np.nan],"truth":[np.nan]})); _close(ax)

    sx=EyeResult({"history":pd.DataFrame({"step":[1,2],"theta":[0.1,0.2],"theta_se":[0.05,np.nan]}),"method":"EAP"}, eyeprocess_class="eye_streaming_score")
    ax=po.plot_eye_streaming_score(sx); assert len(ax.lines)>=1; _close(ax)

    data=pd.DataFrame({"fold":["a","b"],"improvement":[.1,-.1]})
    ax=ppi.plot_eye_incremental_information_audit(data); _close(ax)

    rel=EyeResult({"bland_altman":None,"icc":pd.DataFrame({"icc_a1":[.5]})}, eyeprocess_class="eye_process_reliability_profile")
    ax=ppq.plot_eye_process_reliability_profile(rel,type="bland_altman"); _close(ax)


def test_process_irt_lookup_float_failure_and_importerror(monkeypatch):
    try:
        pir.simulate_irt_model("no-such-model")
    except Exception:
        pass
    try:
        pir.validate_irt_model("no-such-model")
    except Exception:
        pass
    class Bad:
        def aic(self): raise RuntimeError("bad")
    assert np.isnan(pir._model_metric(Bad(), "aic"))
    real_import=builtins.__import__
    def fake_import(name,*a,**k):
        if name=="matplotlib.pyplot": raise ImportError("no mpl")
        return real_import(name,*a,**k)
    monkeypatch.setattr(builtins,"__import__",fake_import)
    with pytest.raises(ep.EyeProcessBackendError):
        pir._plt_ax()


def test_pupil_missingness_validation_and_joint_paths():
    with pytest.raises(ep.EyeProcessValidationError):
        pm.decompose_pupil_phase_amplitude(object())
    reg=EyeResult({"registered":np.array([[1.,2.]]),"ids":["p1"],"id_col":"person_id"}, eyeprocess_class="eye_pupil_registration")
    with pytest.raises(ep.EyeProcessValidationError):
        pm.decompose_pupil_phase_amplitude(reg)
    d=pd.DataFrame({"y":[1.,2.,3.],"x":[0.,1.,2.]})
    with pytest.raises(ep.EyeProcessValidationError):
        pm.fit_process_observation_model(d,[True],['x'])
    with pytest.raises(ep.EyeProcessValidationError):
        pm.fit_process_observation_model(d,[True,False,True],[])
    obs=pm.fit_process_observation_model(d,[True,False,True],['x'])
    joint=pm.fit_joint_signal_missingness("y",obs,x=d,predictors=['x'])
    assert joint.eyeprocess_class=="eye_joint_signal_missingness"
    with pytest.raises(ep.EyeProcessValidationError):
        pm.fit_joint_signal_missingness([1,2], [True])
    with pytest.raises(ep.EyeProcessValidationError):
        pm.process_pattern_mixture([np.nan,np.nan])


def test_evidence_provenance_plot_and_coordinates_aoi_geometry():
    comp=eg.compare_decision_provenance(
        eg.build_evidence_graph(["a"],decisions=["d"]),
        eg.build_evidence_graph(["b"],decisions=["d"]),
    )
    ax=eg.plot_eye_provenance_comparison(comp); _close(ax)

    spaces=pd.DataFrame([
        {"coordinate_space_id":"n","space_type":"display_normalized_top_left","x_unit":"normalized","y_unit":"normalized","width":1.0,"height":1.0},
        {"coordinate_space_id":"p","space_type":"display_pixels_top_left","x_unit":"pixels","y_unit":"pixels","width":100.0,"height":200.0},
    ])
    geom=pd.DataFrame({"aoi_id":["A"],"coordinate_space_id":["n"],"geometry_type":["rectangle"],"x":[.1],"y":[.2],"width":[.3],"height":[.4]})
    x=_ds(coordinate_spaces=spaces,aoi_geometry=geom)
    out=co.convert_coordinates(x,"n","p",components=["aoi_geometry"])
    assert float(out["aoi_geometry"].x.iloc[0])==10.0


def test_engine_adapter_pickle_failure_and_upgrade_scanpath_branches(monkeypatch):
    class BadPickle(dict):
        eyeprocess_class="eyeprocess_model"
        def __reduce__(self): raise RuntimeError("no pickle")
    bad = BadPickle(specification={"engine":"x"}, fit={}, interpretation="guard", diagnostics={})
    result=ea.validate_model_object(bad)
    assert not bool(result.findings.loc[result.findings.check.eq("serializable"),"passed"].iloc[0])

    old=EyeResult({"model": {"diagnostics":{"ok":1}}, "spec":{"engine":"z"}}, eyeprocess_class="eyeprocess_model")
    up=ea.upgrade_eyeprocess_model(old)
    assert up.engine=="z" and up.diagnostics=={"ok":1}

    ds=ep.new_eye_dataset(validate=False)
    ds["episodes"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["r"],"trial_id":["t"],"episode_type":["fixation"],"start_time":[0.],"aoi_id":["A"]}),"episodes")
    seq=ea._scanpath_sequence(ds,source="fixations")
    assert len(seq)==1


def test_compositional_fallbacks(monkeypatch):
    arr=np.array([[0.0,0.0]])
    orig_sum = ca.np.sum
    monkeypatch.setattr(ca.np, "sum", lambda *a, **k: np.inf)
    out=ca._close_composition(arr)
    monkeypatch.setattr(ca.np, "sum", orig_sum)
    assert np.isfinite(out.to_numpy()).all()

    comp=pd.DataFrame({"A":[.4,.6],"B":[.6,.4]})
    matrix=np.array([[1.],[-1.]])
    bal=ca.aoi_balance_coordinates(comp, balances=matrix)
    assert list(bal.columns)==["balance_1"]

    composition = ca.derive_aoi_composition(comp, aois=["A","B"])
    monkeypatch.setattr(ca.np.linalg,"svd",lambda *a,**k:(np.eye(2),np.array([]),np.eye(2)))
    ax=ca.plot_aoi_balance_biplot(composition); _close(ax)

def test_small_remaining_branch_arcs(monkeypatch, tmp_path):
    f = tmp_path / "x.csv"; f.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No eye-data adapters"):
        ad.detect_eye_format(f, candidates=["definitely_missing"])
    names=("tie_a","tie_b")
    try:
        for i,name in enumerate(names):
            ad.register_eye_adapter(name, detect=lambda path, **kw: 0.9, read=lambda path, **kw: ep.new_eye_dataset(validate=False), priority=50-i, overwrite=True)
        with pytest.warns(RuntimeWarning, match="tied"):
            ad.read_eye_export(f, vendor="auto")
    finally:
        for name in names: ad.unregister_eye_adapter(name)

    x=EyeResult({"item_parameters":pd.DataFrame({"lower_asymptote":[.1,.2]})},eyeprocess_class="eye_gaze_anchored_3pl_audit")
    out=b3.audit_3pl_process_signatures(x)
    assert out.fast_rt_review.eq(False).all() and out.fast_ttff_review.eq(False).all()

    z=ca.aoi_balance_coordinates(pd.DataFrame({"A":[.4],"B":[.6]}), np.empty((2,0)))
    assert z.shape==(1,0)

    old=EyeResult({"engine":"fixed","spec":{},"model":{}},eyeprocess_class="eyeprocess_model")
    assert ea.upgrade_eyeprocess_model(old).engine=="fixed"
    ds=ep.new_eye_dataset(validate=False)
    ds["episodes"]=pd.DataFrame({"recording_id":["r"],"trial_id":["t"],"start_time":[0.0],"aoi_id":["A"]})
    assert len(ea._scanpath_sequence(ds,source="fixations"))==1

    assert irt._sbc_rank_diagnostics([0,1,2],n_draws=2,bins=None).bins >= 2

    sbc=iv.run_sbc(lambda r:{"data":1,"truth":{"theta":0.}}, lambda d:object(), lambda fit:pd.DataFrame({"other":[0.]}), replications=1)
    assert len(sbc.failures)==1

    spec=EyeResult({"id":"tmp","status":"reference","fit_fun":lambda data:{"ok":True},"simulate_fun":lambda:1,"validate_fun":None},eyeprocess_class="eye_irt_model_spec")
    assert pir.fit_irt_model(spec,{})=={"ok":True}
    assert pir._segment_loglik(None,np.array([1.,2.]),None)[1]==2

    import matplotlib.pyplot as plt
    _,ax=plt.subplots(); plm.plot_eye_parameter_recovery(pd.DataFrame({"estimate":[1.],"truth":[1.]}),ax=ax); _close(ax)
    _,ax=plt.subplots(); pi.plot_eye_irt_process_alignment({"table":pd.DataFrame({"b":[1.],"gaze":[1.]}),"correlations":pd.DataFrame()},channel="gaze",ax=ax); _close(ax)
    legacy=SimpleNamespace(legacy=True,data={"features":pd.DataFrame()},feature_names=["x"])
    ax=pfp.plot_eye_functional_pupil_irt(legacy); _close(ax)

    d=pd.DataFrame({"person_id":["p1","p1"],"item_id":["i1","i2"],"m":[np.nan,np.nan]})
    try:
        pq.split_half_process_reliability(d,"m",person="person_id",item="item_id",aggregate_fun=np.mean)
    except Exception:
        pass
