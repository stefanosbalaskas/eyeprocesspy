from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.irt import EyeResult
import eyeprocesspy.io_validation_10 as io
import eyeprocesspy.gazepoint as gp
import eyeprocesspy.governance_09 as gov
import eyeprocesspy.plots_governance_09 as pgov
import eyeprocesspy.requested_api_07 as ra
import eyeprocesspy.timebase as tb
import eyeprocesspy.validation_orchestration_completion_10 as voc


def _close(ax):
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def _empty_dataset():
    return ep.new_eye_dataset(validate=False)


def test_io_small_residuals(monkeypatch,tmp_path):
    assert io._polygon_to_text([[0,1],[2,3]]) == "0,1;2,3"
    x=_empty_dataset()
    dest=tmp_path/"ds"; io.write_eye_dataset(x,dest,manifest=False)
    meta=dest/".eyeprocess-serialization.json"; d=json.loads(meta.read_text()); d.setdefault("column_families",{}).setdefault("recordings",{})["missing_col"]="string"; meta.write_text(json.dumps(d))
    y=io.read_eye_dataset(dest,validate=False); assert ep.is_eye_dataset(y)

    frame=pd.DataFrame({"t":[0.],"bio":[1.],".eye_x":[0.],".eye_y":[0.]})
    mapping={"timestamp":"t","x":".eye_x","y":".eye_y","biometric_channels":{"eda":"bio"}}
    b=io.as_eye_biometrics(frame,mapping=mapping); assert len(b)==1

    monkeypatch.setattr(io,"_read_delimited",lambda *a,**k: (_ for _ in ()).throw(ValueError("bad")))
    f=tmp_path/"bad.csv"; f.write_text("a,b\n1,2\n")
    ins=io.inspect_eye_source(f,include_hash=False); assert bool(ins.tabular.iloc[0])

    assert not io.compare_eye_datasets(x,x,tables=["recordings"],ignore_volatile=False).empty
    asc=tmp_path/"ok.asc"; asc.write_text("MSG 1 TRIAL_START\n")
    assert io.validate_eyelink_export(asc).empty


def test_format_matrix_fail_tally():
    summary=pd.DataFrame({"format_family":[pd.NA],"vendor":["generic"],"status":["fail"]})
    obj=io.EyeCorpusValidation(pd.DataFrame(),{},summary,"fail","now")
    out=io.format_compatibility_matrix(obj)
    assert int(out.empirical_failures.sum())==1


def test_gazepoint_timebase_and_folder_branches(monkeypatch,tmp_path):
    info=gp._gp_time_info(pd.DataFrame({"TIMETICK(F=1000)":[1],"TIME(start)":[0]})); assert info["tick_frequency"]==1000.0
    x=_empty_dataset(); x["recordings"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"participant_id":["P1"],"nominal_sampling_rate":[np.nan]}),"recordings")
    x["gaze_samples"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"stream_id":["g"],"sample_id":["s"],"timestamp_native":[1000.],"timestamp_seconds":[np.nan],"gaze_x":[.1],"gaze_y":[.2],"valid":[True]}),"gaze_samples")
    out=gp._gp_apply_timebase(x,pd.DataFrame({"TIMETICK(F=1000)":[1000.]}),info,origin_tick=1000.)
    assert float(out["gaze_samples"].timestamp_seconds.iloc[0])==0.0
    assert out["streams"].empty

    bio=_empty_dataset(); bio["biometrics"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"stream_id":["b"],"timestamp_native":[1.],"timestamp_seconds":[0.],"channel":["unknown"],"value":[1.],"unit":["u"],"valid":[True]}),"biometrics")
    z=gp._gp_apply_biometric_validity(bio,pd.DataFrame({"x":[1]})); assert len(z["biometrics"])==1

    monkeypatch.setattr(gp,"gp_pair_exports",lambda p: pd.DataFrame({"file":[str(tmp_path/'fix.csv')],"export_type":["fixations"],"group":["g"]}))
    (tmp_path/"fix.csv").write_text("x\n1\n")
    monkeypatch.setattr(gp,"read_gazepoint_fixations",lambda *a,**k: _empty_dataset())
    monkeypatch.setattr(gp,"combine_eye_datasets",lambda objs,resolve_ids=False: _empty_dataset())
    res=gp.read_gazepoint_folder(tmp_path,include=("fixations",)); assert ep.is_eye_dataset(res)


def test_governance_residual_guards():
    with pytest.raises(ep.EyeProcessValidationError): gov.validate_eye_pipeline(object())
    empty=EyeResult({"steps":{},"spec":gov.eye_analysis_spec(),"strict":False},eyeprocess_class="eye_analysis_pipeline")
    with pytest.raises(ep.EyeProcessValidationError,match="no steps"): gov.validate_eye_pipeline(empty)
    with pytest.raises(ep.EyeProcessValidationError): gov.pipeline_step_status(object())
    with pytest.raises(ep.EyeProcessValidationError,match="extract_fun"): gov._default_sensitivity_extract({"x":object()},None)
    with pytest.raises(ep.EyeProcessValidationError): gov.summarise_process_sensitivity(object())
    with pytest.raises(ep.EyeProcessValidationError): gov.sensitivity_multiverse_manifest(object())
    manifest=gov.eye_decision_manifest(sampling={"x":1},validity={"x":1},fixation={"x":1},pupil={"x":1},aoi={"x":1},model={"x":1},sensitivity={"x":1},exclusions={"x":1})
    assert gov.validate_decision_manifest(manifest,require_nonempty=True)
    rows=gov._flatten_manifest(np.array([1,2])); assert rows[0]["value"]=="1;2"


def test_requested_api_remaining_guards():
    assert isinstance(ra._extract_samples({"x":1}), pd.DataFrame)
    with pytest.raises(ep.EyeProcessValidationError): ra._logistic_fit(np.empty((0,1)),np.array([]))
    d=pd.DataFrame({"response":[np.nan,np.nan],"participant_id":["p1","p2"],"item_id":["i1","i1"],"gaze_exposure":[1.,2.],"theta":[0.,1.]})
    miss=ra.fit_gaze_informed_missingness_irt(d,theta="theta"); assert miss.response_model is None
    import matplotlib.pyplot as plt
    ax=ra.plot_distractor_information(pd.DataFrame({"gaze_contrast":[.1],"choice_contrast":[.2]})); plt.close(ax.figure)
    assert ra.latent_distribution_stress_test is not None
    sim={"simulate_fun":lambda n=1: {"n":n}}
    assert ra.simulate_from_model(sim,n=2)["n"]==2


def test_timebase_native_unit_and_empty_audit():
    x=_empty_dataset()
    x["gaze_samples"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"stream_id":["g"],"sample_id":["s"],"timestamp_native":[1.],"timestamp_seconds":[np.nan],"gaze_x":[.1],"gaze_y":[.1],"valid":[True]}),"gaze_samples")
    with pytest.raises(ep.EyeProcessTimebaseError): tb.normalize_timebase(x,native_unit="fortnights")
    empty=_empty_dataset(); out=tb.audit_timebase(empty); assert isinstance(out,pd.DataFrame)


def test_governance_plot_empty_branches(monkeypatch):
    monkeypatch.setattr(pgov,"eye_pipeline_graph",lambda x: EyeResult({"vertices":pd.DataFrame(),"edges":pd.DataFrame()},eyeprocess_class="eye_pipeline_graph"))
    ax=pgov.plot_eye_analysis_pipeline(object()); _close(ax)
    monkeypatch.setattr(pgov,"specification_curve_data",lambda *a,**k: pd.DataFrame())
    ax=pgov.plot_eye_process_sensitivity(object()); _close(ax)
    monkeypatch.setattr(pgov,"decision_manifest_table",lambda x: pd.DataFrame())
    ax=pgov.plot_eye_decision_manifest(object()); _close(ax)


def test_validation_evidence_dataframe_and_mapping_status():
    class VendorFrame(pd.DataFrame): eyeprocess_class="eye_vendor_validation"
    assert voc._evidence_pass(VendorFrame({"status":["pass"]}),"multi_vendor")
    assert not voc._evidence_pass({"status":1},"whatever")
