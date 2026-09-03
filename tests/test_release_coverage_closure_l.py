from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.benchmark_reproducibility_10 as br
import eyeprocesspy.dynamic_irt as di
import eyeprocesspy.foundation_09 as fd
import eyeprocesspy.governance_09 as gov
import eyeprocesspy.legacy_models as lm
import eyeprocesspy.preprocess_features_09 as pf
import eyeprocesspy.process_governance_08 as pg
import eyeprocesspy.sensitivity_08 as se
import eyeprocesspy.software_paper_evidence_09 as sp
import eyeprocesspy.validation_evidence_10 as ve
import eyeprocesspy.validation_extras_09 as vx
from eyeprocesspy.irt import EyeResult


def _foundation_dataset():
    recordings = pd.DataFrame([{"recording_id": "R1", "participant_id": "P1", "nominal_sampling_rate": 2.0}])
    spaces = ep.new_coordinate_space("coord_display_normalized_top_left")
    gaze = pd.DataFrame({"recording_id": ["R1"] * 5,"stream_id": ["G1"] * 5,"sample_id": [f"S{i}" for i in range(5)],"timestamp_seconds": [0.0, .5, 1.0, 1.5, 2.0],"gaze_x": [.1, .2, .8, .8, .2],"gaze_y": [.1, .2, .8, .7, .2],"valid": [True] * 5,"trial_id": [pd.NA] * 5,"stimulus_id": ["stim"] * 5,"coordinate_space_id": ["coord_display_normalized_top_left"] * 5})
    events = pd.DataFrame([{"event_id":"E1","recording_id":"R1","timestamp_seconds":0.0,"event_type":"trial","event_name":"TRIAL_START","event_value":"T1","trial_id":pd.NA,"stimulus_id":"stim"},{"event_id":"E2","recording_id":"R1","timestamp_seconds":2.0,"event_type":"trial","event_name":"TRIAL_END","event_value":"T1","trial_id":pd.NA,"stimulus_id":"stim"}])
    eye = pd.DataFrame({"recording_id":["R1","R1"],"sample_id":["P1","P2"],"timestamp_seconds":[0.0,1.0],"eye":["left","left"],"pupil_diameter":[3.0,3.2],"pupil_valid":[True,True],"trial_id":[pd.NA,pd.NA],"stimulus_id":["stim","stim"]})
    bio = pd.DataFrame({"recording_id":["R1","R1"],"stream_id":["B1","B1"],"timestamp_seconds":[.25,1.25],"channel":["eda","eda"],"value":[1.0,2.0],"valid":[True,True],"trial_id":[pd.NA,pd.NA],"stimulus_id":["stim","stim"]})
    return ep.new_eye_dataset(recordings=recordings, coordinate_spaces=spaces, gaze_samples=gaze, events=events, eye_samples=eye, biometrics=bio)


def _features_dataset(seed=501):
    x = ep.simulate_eye_dataset(n_person=16, n_item=6, sampling_rate=15, trial_duration=.35, samples_per_trial=8, seed=seed)
    r = x["responses"].reset_index(drop=True)
    rows=[]
    for i,row in r.iterrows():
        for j,(name,value) in enumerate([("gaze_feature", np.sin(i*.37)+(i%5)*.03),("pupil_feature", np.cos(i*.29)+.1*np.log(float(row["response_time"])))]):
            rows.append({"feature_id":f"f{i:04d}_{j}","recording_id":row["recording_id"],"participant_id":row["participant_id"],"trial_id":row["trial_id"],"item_id":row["item_id"],"stimulus_id":pd.NA,"aoi_id":pd.NA,"feature_name":name,"value":value,"unit":"arbitrary","level":"trial","method":"test","parameters":""})
    x["features"] = ep.standardize_eye_table(pd.DataFrame(rows), "features")
    return x


def _drift_data():
    rows=[]
    for item_i,item in enumerate(["i1","i2"]):
        for j,batch in enumerate([1,2,3]):
            rows.append({"deployment_batch":batch,"item_id":item,"dwell_ms":700.+80*j+10*item_i,"valid_gaze_prop":.95-.08*j})
    return pd.DataFrame(rows)


def _validation_result_varying():
    design = pd.DataFrame({"condition_id":["C1","C2"],"replications":[2,2]}); design.attrs["master_seed"] = 1
    est = pd.DataFrame({"parameter":["b"]*4,"condition_id":["C1","C1","C2","C2"],"estimate":[.0,.1,.9,1.2],"truth":[0.,0.,0.,0.],"lower":[-.2,-.1,.7,1.0],"upper":[.2,.3,1.1,1.4],"converged":[True]*4})
    failures=pd.DataFrame({"condition_id":["C2"],"replication":[1],"stage":["fit"],"error":["x"]})
    return EyeResult({"design":design,"estimates":est,"failures":failures,"warnings":pd.DataFrame(),"design_hash":"x"}, eyeprocess_class="eye_process_validation_result")


def test_foundation_remaining_executable_paths():
    x=_foundation_dataset()
    out=ep.synchronize_eye_biometrics(x, x, source_markers=[0.0,1.0], target_markers=[.1,1.1], method="offset")
    assert ep.is_eye_dataset(out)
    x2=x.copy(); extra=x2["gaze_samples"].copy(); extra["recording_id"]="R2"; extra["sample_id"]=[f"X{i}" for i in range(len(extra))]; extra["stimulus_id"]=pd.NA
    x2["gaze_samples"] = ep.standardize_eye_table(pd.concat([x2["gaze_samples"],extra],ignore_index=True),"gaze_samples")
    built=ep.build_stimulus_intervals(x2); assert built["intervals"].interval_type.eq("stimulus").any()
    aoi=ep.new_aoi("A",x=0,y=0,width=.5,height=.5); x3=ep.register_aois(x,aoi)
    episodes=pd.DataFrame({"episode_id":["f1"],"recording_id":["R1"],"episode_type":["fixation"],"start_time":[.1],"end_time":[.2],"duration_ms":[100.],"centroid_x":[.2],"centroid_y":[.2],"coordinate_space_id":["coord_display_normalized_top_left"],"aoi_id":["OLD"]})
    x3["episodes"] = ep.standardize_eye_table(episodes,"episodes")
    assigned=ep.assign_aois(x3,component="episodes",overwrite=True); assert assigned["episodes"].aoi_id.iloc[0] == "A"
    with pytest.raises(ValueError):
        fd._event_matches(["["], pd.Series(["abc"]))


def test_process_governance_success_paths():
    np.testing.assert_allclose(pg._z([1.,2.,3.]), [-1.,0.,1.])
    d=pd.DataFrame({"person_id":[f"P{i}" for i in range(12)],"m1":np.arange(12,dtype=float),"m2":np.arange(12,dtype=float)**2,"m3":np.linspace(0,1,12)})
    anomaly=ep.audit_process_anomalies(d,metrics=None,aggregate=False); assert len(anomaly.table)==12
    access=ep.audit_presentation_accessibility(d,rt="m1",dwell="m2",revisits="m3",review_quantile=.8); assert len(access.table)==12
    variants=ep.simulate_presentation_variants(access); assert not variants.empty
    cmp=ep.compare_deployment_batches(_drift_data(),batch_a=1,batch_b=2,metrics=["dwell_ms","valid_gaze_prop"]); assert {"dwell_ms_delta","valid_gaze_prop_delta"} <= set(cmp.columns)


def test_governance_finite_limit_ranking_and_invalid_reference():
    design=ep.process_validation_design(n_persons=4,n_trials=3,replications=1,seed=7)
    out=ep.run_process_validation(design,max_conditions=1); assert len(out.design)==1
    ranked=ep.validation_condition_ranking(_validation_result_varying()); assert len(ranked)==2 and ranked.robustness_score.nunique()>1
    with pytest.raises(ep.EyeProcessValidationError, match="eye_process_validation_result"):
        ep.freeze_validation_reference({})


def test_legacy_model_append_and_response_formula_paths():
    x=_features_dataset(); pupil=ep.functional_pupil_features(x,df=3,append=True)
    assert ep.is_eye_dataset(pupil) and pupil["features"].feature_name.astype(str).str.startswith("pupil_basis_").any()
    strategy=ep.fit_strategy_mixture(x,["gaze_feature","pupil_feature"],centers=2,response_formula="score ~ strategy_class",seed=2)
    assert strategy.model.fit.response_model is not None


def test_validation_small_residual_lines():
    with pytest.raises(ep.EyeProcessBackendError, match="stepwiseIt"):
        se.audit_item_reduction_sensitivity(None,alpha=.05,maxstep=1)
    with pytest.raises(ep.EyeProcessValidationError, match="non-empty"):
        vx._normalize_windows({})
    good_sbc=EyeResult({"results":pd.DataFrame()}, eyeprocess_class="eye_sbc")
    old=ve.sbc_summary
    try:
        ve.sbc_summary=lambda value: pd.DataFrame({"status":["pass"]}); assert ve._sbc_pass(good_sbc)
    finally:
        ve.sbc_summary=old
    dup=pd.Series([1.,2.],index=["a","a"]); assert ve._truth_map(dup) is None
    bundle=ep.software_paper_evidence_bundle(claims=object()); gaps=sp.software_paper_gap_analysis(bundle); assert gaps["claim_gaps"].empty


def test_dynamic_transition_nonfinite_scale_fallback():
    d=pd.DataFrame({"from_state":pd.Categorical(["A","B","A"],categories=["A","B"]),"to_state":pd.Categorical(["B","A","B"],categories=["A","B"]),"to_index":[2,1,2],"step":[1,2,3],"time_gap":[0.,0.,0.],"participant_id":["P1"]*3,"item_id":["I1"]*3,"trial_index":[1,1,1],"x":[1e308,-1e308,0.]})
    d.attrs["eyeprocess_class"]="eye_dynamic_transition_data"; d.attrs["states"]=["A","B"]
    spec=di.dynamic_irtree_spec(include_item=False,include_response=False,include_time_gap=False,transition_predictors=("x",))
    out=di.dynamic_transition_design(d,spec=spec); assert float(out.scaling.loc[out.scaling.term.eq("x"),"scale"].iloc[0]) == 1.0


def test_pandas_scalar_group_key_compatibility_paths(monkeypatch):
    class ScalarGroup:
        def __init__(self, key, frame): self.key=key; self.frame=frame
        def __iter__(self): yield self.key, self.frame
    x=_foundation_dataset(); x["gaze_samples"]["aoi_id"]="A"
    original_groupby=pd.DataFrame.groupby
    def scalar_groupby(self, by=None, *args, **kwargs):
        if by == ["recording_id"] and "aoi_id" in self.columns: return ScalarGroup("R1", self)
        return original_groupby(self,by,*args,**kwargs)
    monkeypatch.setattr(pd.DataFrame,"groupby",scalar_groupby)
    ent=pf.gaze_entropy(x,source="samples",level="recording"); assert ent.recording_id.iloc[0]=="R1"
    f=pd.DataFrame({"feature_id":["f1"],"recording_id":["R1"],"feature_name":["m"],"value":[1.],"unit":["u"],"level":["trial"],"method":["x"],"parameters":[""]})
    x2=ep.new_eye_dataset(validate=False); x2["features"]=ep.standardize_eye_table(f,"features")
    wide=pf.features_wide(x2,id_cols=("recording_id",)); assert wide.recording_id.iloc[0]=="R1"


def test_reproducibility_optional_import_failures(monkeypatch,tmp_path):
    p=tmp_path/"x.txt"; p.write_text("x",encoding="utf-8")
    real_import=builtins.__import__
    def fake_import(name,*args,**kwargs):
        if name == "scipy" or (name == "eyeprocesspy" and kwargs.get("fromlist") == ("__version__",)): raise ImportError(name)
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",fake_import)
    m=br.package_reproducibility_manifest([p],include_session=True); assert any(s.startswith("Python ") for s in m.session)
