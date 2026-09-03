from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.irt import EyeResult
import eyeprocesspy.adapters as ad
import eyeprocesspy.advanced_process_irt_07 as ap
import eyeprocesspy.coordinates as co
import eyeprocesspy.core_plots_10 as cp
import eyeprocesspy.dynamic_irt as di
import eyeprocesspy.evidence_graph as eg
import eyeprocesspy.foundation_09 as fd
import eyeprocesspy.frontier_08 as fr
import eyeprocesspy.importers as imp
import eyeprocesspy.plots_completion_08 as pc
import eyeprocesspy.plots_operational_08 as po
import eyeprocesspy.preprocess_features_09 as pf
import eyeprocesspy.process_governance_08 as pg
import eyeprocesspy.process_quality_09 as pq
import eyeprocesspy.pupil_missingness as pm
import eyeprocesspy.timebase as tb
import eyeprocesspy.validation_orchestration_completion_10 as voc


def _close(ax):
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_foundation_invalid_regex_own_handler_and_empty_visit_group(monkeypatch):
    from pandas.core.strings.accessor import StringMethods
    monkeypatch.setattr(StringMethods, "contains", lambda *a, **k: (_ for _ in ()).throw(re.error("bad")))
    with pytest.raises(ValueError, match="Invalid event pattern"):
        fd._event_matches(["["], pd.Series(["abc"]))

    x = ep.new_eye_dataset(validate=False)
    x["gaze_samples"] = ep.standardize_eye_table(pd.DataFrame({
        "recording_id":["R1"], "stream_id":["g"], "sample_id":["s"], "timestamp_seconds":[0.0],
        "gaze_x":[.1], "gaze_y":[.1], "valid":[True], "trial_id":["T1"], "stimulus_id":["S"],
        "coordinate_space_id":["coord"], "aoi_id":["A"],
    }), "gaze_samples")
    real_groupby = pd.DataFrame.groupby
    class EmptyGroups:
        def __iter__(self):
            yield ("R1","T1"), pd.DataFrame(columns=x["gaze_samples"].columns)
    def fake_groupby(self, by=None, *args, **kwargs):
        if by == ["recording_id", "trial_id"] and "aoi_id" in self.columns:
            return EmptyGroups()
        return real_groupby(self, by, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "groupby", fake_groupby)
    out = fd.build_aoi_visits(x)
    assert out["episodes"].empty


def test_plots_completion_defensive_crosstab_and_zero_total(monkeypatch):
    transitions = pd.DataFrame({"from":["A"], "to":["B"]})
    class EmptyTab:
        shape=(0,0)
        def reindex(self, **kwargs): return self
    monkeypatch.setattr(pc.pd, "crosstab", lambda *a, **k: EmptyTab())
    with pytest.raises(ep.EyeProcessValidationError, match="No complete AOI transitions"):
        pc._transition_table(transitions, normalize="none")

    monkeypatch.undo()
    real_crosstab = pc.pd.crosstab
    def zero_tab(*a, **k):
        z = real_crosstab(*a, **k)
        return z * 0
    monkeypatch.setattr(pc.pd, "crosstab", zero_tab)
    out = pc._transition_table(transitions, normalize="all")
    assert float(out.to_numpy().sum()) == 0.0


def test_preprocess_scalar_multikey_and_zero_transition_total(monkeypatch):
    d = pd.DataFrame({"recording_id":["R1","R1"],"stream_id":["g","g"],"sample_id":["s1","s2"],"timestamp_seconds":[0.,.1],"gaze_x":[.1,.2],"gaze_y":[.1,.2],"valid":[True,True],"trial_id":["T1","T1"],"aoi_id":["A","B"]})
    x = ep.new_eye_dataset(validate=False); x["gaze_samples"] = ep.standardize_eye_table(d, "gaze_samples")
    real_groupby = pd.DataFrame.groupby
    class ScalarGroup:
        def __init__(self, frame): self.frame=frame
        def __iter__(self): yield "R1", self.frame
    def scalar_groupby(self, by=None, *args, **kwargs):
        if by == ["recording_id", "trial_id"] and "aoi_id" in self.columns:
            return ScalarGroup(self)
        return real_groupby(self, by, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "groupby", scalar_groupby)
    out = pf.gaze_entropy(x, level="trial", source="samples")
    assert out.recording_id.iloc[0] == "R1"

    monkeypatch.undo()
    monkeypatch.setattr(pf, "scanpath_sequence", lambda *a, **k: pd.DataFrame({"sequence":["A > B"]}))
    real_to_numpy = pd.DataFrame.to_numpy
    def zero_matrix(self, *args, **kwargs):
        arr = real_to_numpy(self, *args, **kwargs)
        if list(self.index)==["A","B"] and list(self.columns)==["A","B"]:
            return np.zeros_like(arr, dtype=float)
        return arr
    monkeypatch.setattr(pd.DataFrame, "to_numpy", zero_matrix)
    m = pf.transition_matrix(object(), normalize="all")
    assert m.shape == (2,2)


def test_adapter_non_tie_auto_read(tmp_path):
    name = "coverage_unique"
    p = tmp_path / "x.csv"; p.write_text("x\n1\n", encoding="utf-8")
    try:
        ad.register_eye_adapter(name, detect=lambda path, **kw: .99, read=lambda path, **kw: ep.new_eye_dataset(validate=False), priority=9999, overwrite=True)
        out = ad.read_eye_export(p, vendor="auto", confidence_threshold=.5)
        assert ep.is_eye_dataset(out)
    finally:
        ad.unregister_eye_adapter(name)


def test_hmm_singleton_sequence_branch():
    d = pd.DataFrame({
        "trial_id":["a","b","b"], "timestamp":[0.,0.,1.], "x":[-1.,0.,1.], "y":[-1.,0.,1.],
        "response":[0,1,1], "participant_id":["p1","p2","p2"], "item_id":["i1","i1","i2"]
    })
    out = ap.fit_process_hmm_irt(d, n_states=2, max_iter=1, seed=4)
    assert out.eyeprocess_class == "eye_process_hmm_irt"


def test_coordinate_gaze_only_and_fixation_no_reverse():
    x = ep.new_eye_dataset(validate=False)
    x["coordinate_spaces"] = ep.standardize_eye_table(pd.DataFrame([
        {"coordinate_space_id":"n","space_type":"display_normalized_top_left","x_unit":"normalized","y_unit":"normalized","width":1.,"height":1.},
        {"coordinate_space_id":"p","space_type":"display_pixels_top_left","x_unit":"pixels","y_unit":"pixels","width":100.,"height":200.},
    ]), "coordinate_spaces")
    x["gaze_samples"] = ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"stream_id":["g"],"sample_id":["s"],"timestamp_seconds":[0.],"gaze_x":[.2],"gaze_y":[.3],"valid":[True],"coordinate_space_id":["n"]}), "gaze_samples")
    y = co.convert_coordinates(x, "n", "p", components=["gaze_samples"])
    assert len(y["gaze_samples"]) == 2

    x["episodes"] = ep.standardize_eye_table(pd.DataFrame({"episode_id":["f"],"recording_id":["R1"],"episode_type":["fixation"],"start_time":[0.],"end_time":[.1],"duration_ms":[100.],"centroid_x":[.2],"centroid_y":[.3],"derived_by":["vendor"]}), "episodes")
    ax = cp.plot_fixations(x, reverse_y=False)
    assert not ax.yaxis_inverted(); _close(ax)


def _transition_design():
    return EyeResult({
        "X":pd.DataFrame({"(Intercept)":[1.,1.,1.,1.]}), "y":np.array([1,2,1,2]),
        "allowed":np.ones((4,2),bool), "states":["A","B"],
        "data":pd.DataFrame({"state_probability":[1.,1.,1.,1.]})
    }, eyeprocess_class="eye_transition_design")


def test_dynamic_control_directions_and_unnamed_signature_guard():
    di.fit_multinomial_transition(_transition_design(), control={"maxit":3})
    di.fit_multinomial_transition(_transition_design(), control={"reltol":1e-6})
    with pytest.raises(ep.EyeProcessValidationError, match="Unnamed strategy signatures"):
        di.theory_strategy_spec({"a":[1.,2.], "b":[1.]}, feature_columns=["x","y"])


def test_evidence_graph_remaining_paths(monkeypatch):
    graph=eg.build_evidence_graph(["raw"], decisions=["yes"])
    with pytest.raises(ep.EyeProcessValidationError, match="No matching decision"):
        eg.trace_item_decision(graph, "absent")
    bad_edges=pd.DataFrame({"from":["unknown"],"to":["also_unknown"],"relation":["x"]})
    ax=eg._graph_plot(graph.nodes,bad_edges,"x"); _close(ax)
    monkeypatch.setattr(eg,"cross_recurrence",lambda *a,**k:{})
    monkeypatch.setattr(eg,"recurrence_features",lambda r: pd.DataFrame({"rr":[.1]}))
    with pytest.raises(ep.EyeProcessValidationError, match="covariates must align"):
        eg.crossmodal_recurrence_model([1],[1],outcome=[1.,2.],covariates=pd.DataFrame({"z":[1.,2.,3.]}))
    trace=eg.trace_item_decision(graph,"yes")
    ax=eg.plot_eye_decision_trace(trace); _close(ax)
    model=eg.crossmodal_recurrence_model([1],[1])
    ax=eg.plot_eye_crossmodal_recurrence_model(model,type="coefficients"); _close(ax)


def test_frontier_fold_none():
    out=fr.prepare_structured_unstructured_process_features(pd.DataFrame({"person_id":["p"],"item_id":["i"],"x":[1.]}), fold=None)
    assert out.status == "representation_contract_only"


def test_importer_residual_guards(tmp_path):
    p=tmp_path/"x.csv"; p.write_text("a,b\n1,2\n",encoding="utf-8")
    assert imp._read_delimited(p, delimiter=",").shape==(1,2)
    mapping={"timestamp":"t","x":"x","y":"y"}; assert imp.validate_eye_mapping(mapping, data=None) is mapping
    data=pd.DataFrame({"ev":[pd.NA],"r":[pd.NA],"score":[np.nan],"rt":[np.nan]})
    rec=pd.Series(["R1"]); t=pd.Series([0.])
    assert imp._make_events(data,{"event_name":"ev"},rec,t,t).empty
    assert imp._make_responses(data,{"response":"r","score":"score","response_time":"rt"},rec).empty
    with pytest.raises(ValueError, match="biometric_channels"):
        imp._make_biometrics(data,{"biometric_channels":["x"]},rec,t,t)
    assert imp._make_biometrics(data,{"biometric_channels":{"eda":"missing"}},rec,t,t).empty
    with pytest.raises(ValueError, match="no rows"):
        imp.read_eye_generic(pd.DataFrame(), mapping={})


def test_operational_plot_invalid_objects_and_empty_feature_branch():
    for func in (po.plot_eye_streaming_score, po.plot_eye_validation_bundle, po.plot_eye_preaction_process_features, po.plot_eye_decision_process_proxy):
        with pytest.raises(ep.EyeProcessValidationError): func(object())
    x=EyeResult({"data":pd.DataFrame()}, eyeprocess_class="eye_preaction_process_features")
    ax=po.plot_eye_preaction_process_features(x); _close(ax)
    with pytest.raises(ep.EyeProcessValidationError):
        po.plot_process_feature_stability(pd.DataFrame({"x":[1]}))


def test_process_governance_short_and_wrapper_paths():
    arr=pg._interp([np.nan,1.0]); assert np.isnan(arr).all()
    data=pd.DataFrame({"person_id":["p"],"trial_id":["t"],"time_ms":[0.]})
    spec=pg.process_window_spec(start_ms=0,end_ms=100,width_ms=100,step_ms=100,min_samples=2)
    out=pg.extract_process_windows(data,spec=spec); assert out.data.empty
    filt=pg.filter_pupil_signal([1,2,3,4,5],width=3,method="runmed"); assert filt.method=="runmed"


def test_process_quality_all_nan_aggregate():
    d=pd.DataFrame({"p":["a","a","b","b"],"t":[1,2,1,2],"m":[np.nan,np.nan,np.nan,np.nan]})
    out=pq.split_half_process_reliability(d,person="p",trial="t",measure="m",split="odd_even")
    assert isinstance(out,pd.DataFrame)


def test_pupil_missingness_nonmodel_and_plot_wrappers():
    d=pd.DataFrame({"y":[1.,2.,3.],"obs":[True,False,True],"x":[0.,1.,2.]})
    out=pm.fit_joint_signal_missingness("y","obs",x=d,predictors=["x"]); assert np.isnan(out.shared_correlation)
    sensitivity=pm.process_pattern_mixture([1.,np.nan,2.],delta=[-1,1])
    tip=pm.sensitivity_mnar_process(sensitivity)
    for func,obj in [(pm.plot_eye_mnar_sensitivity,sensitivity),(pm.plot_eye_mnar_tipping_point,tip)]:
        ax=func(obj); _close(ax)
    obs=pm.fit_process_observation_model(d,[True,False,True],["x"])
    ax=pm.plot_eye_process_observation_model(obs); _close(ax)


def test_timebase_residual_branches():
    assert np.isnan(tb.estimate_sampling_rate([1.,1.,np.nan]))
    x=ep.new_eye_dataset(validate=False)
    with pytest.raises(ValueError, match="origin"):
        tb.normalize_timebase(x, origin="bad")
    out=tb.normalize_timebase(x, origin="absolute"); assert ep.is_eye_dataset(out)
    audit=tb.audit_timebase(x); assert isinstance(audit,pd.DataFrame)


def test_validation_completion_artifact_fallbacks():
    class VendorFrame(pd.DataFrame):
        eyeprocess_class = "eye_vendor_validation"
    assert voc._evidence_pass(VendorFrame({"status":["pass"]}),"multi_vendor")
    vendor_obj = EyeResult({"status":1}, eyeprocess_class="eye_vendor_validation")
    assert not voc._evidence_pass(vendor_obj,"multi_vendor")
    assert voc._evidence_pass({"status":"success"},"other")
    assert not voc._evidence_pass({"status":1},"other")
