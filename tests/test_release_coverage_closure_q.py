from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.irt import EyeResult
import eyeprocesspy.coordinates as co
import eyeprocesspy.foundation_09 as fd
import eyeprocesspy.gazepoint as gp
import eyeprocesspy.gazepoint_real_10 as gr
import eyeprocesspy.legacy_models as lm
import eyeprocesspy.multimodal_staged as ms
import eyeprocesspy.preprocess_features_09 as pf
import eyeprocesspy.process_governance_08 as pg
import eyeprocesspy.pupil_missingness as pm
import eyeprocesspy.requested_api_07 as ra
import eyeprocesspy.semantic_validation_07 as sv
import eyeprocesspy.timebase as tb
import eyeprocesspy.vendor_corpus_10 as vc
import eyeprocesspy.vendor_importers_10 as vi


def _empty():
    return ep.new_eye_dataset(validate=False)


def _base():
    rec=pd.DataFrame([{"recording_id":"R1","participant_id":"P1"}])
    spaces=ep.new_coordinate_space("coord_display_normalized_top_left")
    gaze=pd.DataFrame({"recording_id":["R1"]*3,"stream_id":["g"]*3,"sample_id":["s1","s2","s3"],"timestamp_seconds":[0.,.1,.2],"gaze_x":[.1,.2,.3],"gaze_y":[.1,.2,.3],"valid":[True]*3,"trial_id":["T1"]*3,"stimulus_id":["M1"]*3,"coordinate_space_id":["coord_display_normalized_top_left"]*3})
    return ep.new_eye_dataset(recordings=rec, coordinate_spaces=spaces, gaze_samples=gaze)


def test_coordinates_fallthrough_component():
    x=_empty()
    x["coordinate_spaces"]=ep.standardize_eye_table(pd.DataFrame([
        {"coordinate_space_id":"from","space_type":"display_normalized_top_left","x_unit":"normalized","y_unit":"normalized","width":1.,"height":1.},
        {"coordinate_space_id":"to","space_type":"display_normalized_top_left","x_unit":"normalized","y_unit":"normalized","width":1.,"height":1.},
    ]),"coordinate_spaces")
    x["streams"]=ep.standardize_eye_table(pd.DataFrame({"stream_id":["st"],"recording_id":["R1"],"stream_type":["gaze_combined"],"coordinate_space_id":["from"]}),"streams")
    out=co.convert_coordinates(x,"from","to",components=["streams"])
    assert out["streams"].coordinate_space_id.iloc[0]=="from"


def test_foundation_empty_component_and_empty_gaze_paths():
    x=_empty()
    x["events"]=ep.standardize_eye_table(pd.DataFrame({"event_id":["e"],"recording_id":["R1"],"timestamp_seconds":[0.],"event_name":["TRIAL_START"],"event_value":["T1"]}),"events")
    x["gaze_samples"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"stream_id":["g"],"sample_id":["s"],"timestamp_seconds":[1.],"gaze_x":[.1],"gaze_y":[.2],"valid":[True]}),"gaze_samples")
    out=fd.build_trials(x,close_open="recording_end")
    assert len(out["intervals"])==1
    e=_empty(); q=fd.audit_signal_quality(e); assert isinstance(q,pd.DataFrame)


def test_gazepoint_residual_false_paths(monkeypatch,tmp_path):
    info=gp._gp_time_info(pd.DataFrame({"TIMETICK":[1000.]})); assert np.isnan(info["tick_frequency"])
    x=_empty(); info2={"tick_col":"TIMETICK(F=1000)","tick_frequency":1000.,"recording_start":pd.NA,"media_time_col":None}
    out=gp._gp_apply_timebase(x,pd.DataFrame({"TIMETICK(F=1000)":[1000.]}),info2,origin_tick=1000.); assert out["recordings"].empty
    p1=tmp_path/"a.csv"; p2=tmp_path/"b.csv"; p1.write_text("x\n1\n"); p2.write_text("x\n1\n")
    profile=pd.DataFrame({"file":[str(p1),str(p2)],"export_type":["gaze","gaze"],"group":["a","b"]})
    monkeypatch.setattr(gp,"gp_pair_exports",lambda p: profile)
    monkeypatch.setattr(gp,"read_gazepoint",lambda *a,**k: _empty())
    monkeypatch.setattr(gp,"combine_eye_datasets",lambda *a,**k: _empty())
    monkeypatch.setattr(gp,"validate_eye_dataset",lambda x: pd.DataFrame())
    r=gp.read_gazepoint_folder(tmp_path,include=("gaze",)); assert ep.is_eye_dataset(r)


def test_gazepoint_summary_metadata_false_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(gr, "_gp_is_summary_report", lambda p: True)

    one = tmp_path / "one.csv"
    one.write_text("Gazepoint Analysis,7.2.0\n", encoding="utf-8")
    first = gr.read_gazepoint_summary(one)
    assert pd.isna(first.processed_on)

    two = tmp_path / "two.csv"
    two.write_text(
        "Gazepoint Analysis,7.2.0\nProcessed On\n",
        encoding="utf-8",
    )
    second = gr.read_gazepoint_summary(two)
    assert pd.isna(second.processed_on)
def test_preprocess_bounds_existing_mark_and_empty_ivt_group(monkeypatch):
    def ds(unit,width=np.nan,height=np.nan):
        x=_empty(); x["coordinate_spaces"]=ep.standardize_eye_table(pd.DataFrame({"coordinate_space_id":["c"],"space_type":["display_pixels_top_left"],"x_unit":[unit],"y_unit":[unit],"width":[width],"height":[height]}),"coordinate_spaces")
        x["gaze_samples"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"stream_id":["g"],"sample_id":["s"],"timestamp_seconds":[0.],"gaze_x":[2.],"gaze_y":[2.],"valid":[True],"trial_id":["T"],"coordinate_space_id":["c"]}),"gaze_samples"); return x
    assert len(pf.flag_gaze_outliers(ds("degrees"),method="bounds")["gaze_samples"])==1
    assert len(pf.flag_gaze_outliers(ds("pixels"),method="bounds")["gaze_samples"])==1
    x=_empty(); x["eye_samples"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1","R1"],"sample_id":["a","b"],"timestamp_seconds":[0.,.1],"eye":["left","left"],"pupil_diameter":[3.,np.nan],"pupil_valid":[True,False],"interpolated":[False,False]}),"eye_samples")
    pf.interpolate_pupil(x,max_gap_ms=1000)
    z=ds("normalized")
    original=pd.DataFrame.groupby
    class EmptyGroup:
        def __iter__(self):
            yield ("R1","T"), self.iloc[0:0] if False else pd.DataFrame(columns=z["gaze_samples"].columns)
    def fake(self,by=None,*a,**k):
        if by==["recording_id","trial_id"] and "gaze_x" in self.columns: return EmptyGroup()
        return original(self,by,*a,**k)
    monkeypatch.setattr(pd.DataFrame,"groupby",fake)
    out=pf.detect_fixations_ivt(z,coordinate_units="normalized"); assert ep.is_eye_dataset(out)


def test_small_wrapper_and_structural_branches(monkeypatch):
    assert pg._entropy_numeric([0.,1.],bins=1)==0.0
    assert pg.filter_eye_signal([1.,2.,3.,4.,5.],width=3,method="runmed").width == 3
    assert tb.estimate_sampling_rate([0.,1.]) == 1.0
    monkeypatch.setattr(pm,"_pupil_plot",lambda *a,**k: "pupil")
    assert pm.plot_eye_pupil_phase_amplitude(object())=="pupil"
    monkeypatch.setattr(ra,"stress_test_latent_distribution",lambda *a,**k: "stress")
    assert ra.latent_distribution_stress_test()=="stress"
    monkeypatch.setattr(ra,"simulate_irt_model",lambda *a,**k: "sim")
    assert ra.simulate_from_model("rasch")=="sim"
    monkeypatch.setattr(sv,"_as_df",lambda x,name: pd.DataFrame())
    assert sv._df({"samples":1,"data":2,"gaze":3},"x").empty
    x=_empty(); assert ep.is_eye_dataset(tb.normalize_timebase(x,origin="absolute"))


def test_legacy_without_median_indicator(monkeypatch):
    d=pd.DataFrame({"f":[1.,2.],"score":[0,1]})
    monkeypatch.setattr(lm,"_require_dataset",lambda x:x)
    monkeypatch.setattr(lm,"model_data",lambda *a,**k:d.copy())
    monkeypatch.setattr(lm,"_fit_binomial",lambda *a,**k: object())
    monkeypatch.setattr(lm,"_result",lambda cls,**k: EyeResult(k,eyeprocess_class=cls))
    out=lm.sensitivity_missing_process(object(),"f","score ~ f",methods=("complete_case",)); assert out.eyeprocess_class=="eye_missing_sensitivity"


def test_multimodal_invalid_identifiability_and_empty_state_run(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError): ms.audit_multimodal_identifiability(object())
    d=pd.DataFrame({"person_id":["P"],"item_id":["I"],"sequence_id":["S"],"trial_index":[1],"rt":[1.],"gaze":[1.],"pupil":[1.]})
    x=EyeResult({"data":d,"truth":{"state":[1],"n_states":1}},eyeprocess_class="eye_multimodal_m4_simulation")
    original=pd.DataFrame.groupby
    class EmptySeq:
        def __iter__(self): yield "S", pd.DataFrame(columns=["MAP_state"])
        @property
        def groups(self): return {"P":np.array([0])}
    calls={"n":0}
    def fake(self,by=None,*a,**k):
        if by=="sequence_id" and "MAP_state" in self.columns:
            calls["n"]+=1
            if calls["n"]==1: return EmptySeq()
        return original(self,by,*a,**k)
    monkeypatch.setattr(pd.DataFrame,"groupby",fake)
    out=ms.multimodal_m4_state_diagnostics(x); assert out.eyeprocess_class=="eye_multimodal_m4_states"


def test_vendor_importer_false_paths(monkeypatch,tmp_path):
    base=_base()
    tob=pd.DataFrame({"Recording timestamp":[1.,2.],"Gaze point X":[.1,.2],"Gaze point Y":[.1,.2],"Validity left":[0,0]})
    monkeypatch.setattr(vi,"_read_delimited",lambda *a,**k:tob.copy())
    monkeypatch.setattr(vi,"read_eye_generic",lambda *a,**k: _empty())
    out=vi.read_tobii(tmp_path/"t.csv"); assert out["gaze_samples"].empty
    neon_path=tmp_path/"neon.csv"; neon_path.write_text("x\n")
    neon=pd.DataFrame({"timestamp [ns]":[1],"gaze x [px]":[1.],"gaze y [px]":[2.]})
    monkeypatch.setattr(vi,"_read_delimited",lambda *a,**k:neon.copy())
    vi.read_pupil_neon(neon_path)
    core_path=tmp_path/"core.csv"; core_path.write_text("x\n")
    core=pd.DataFrame({"gaze_timestamp":[1.],"norm_pos_x":[.1],"norm_pos_y":[.2]})
    monkeypatch.setattr(vi,"_read_delimited",lambda *a,**k:core.copy())
    vi.read_pupil_core(core_path)
    folder=tmp_path/"neon"; folder.mkdir(); pd.DataFrame({"timestamp [ns]":[1],"other":[1]}).to_csv(folder/"3d_eye_states.csv",index=False)
    monkeypatch.setattr(vi,"_read_delimited",lambda p,*a,**k: pd.read_csv(p))
    vi._read_neon_companions(base,folder,keep_raw=True); assert base["eye_samples"].empty
    asc=tmp_path/"empty.asc"; asc.write_text("MSG 1 hello\n",encoding="utf-8")
    z=vi.read_eyelink_asc(asc); assert z["gaze_samples"].empty


def test_vendor_corpus_false_paths(monkeypatch,tmp_path):
    source=tmp_path/"source.csv"; pd.DataFrame({"id":[1],"number":[2]}).to_csv(source,index=False)
    out=vc.redact_validation_case(source,tmp_path/"redacted",salt="salt",text_redactor=lambda s,c:s)
    assert len(out.manifest)==1
    sem=pd.DataFrame({"vendor":["tobii"],"native_field":["x"],"canonical_table":["gaze_samples"],"canonical_field":["gaze_x"],"loss_risk":["none"]})
    cmp=vc.compare_vendor_semantics(sem,vendors=["Tobii"]); assert len(cmp)==1
    corpus=tmp_path/"corpus"; corpus.mkdir()
    monkeypatch.setattr(vc,"read_vendor_registry",lambda p: pd.DataFrame({"case_id":["c"],"support_level":["fixture"]}))
    path=vc.write_vendor_case_report(corpus,"c",tmp_path/"report.md",validation=None); assert Path(path).exists()

import eyeprocesspy.gazepoint_workflow_10 as gw
import eyeprocesspy.io_validation_10 as io10


def _workflow_dataset(two_trials=False, matching=False):
    x=_base()
    trials=[{"interval_id":"i1","recording_id":"R1","interval_type":"trial","start_time":0.,"end_time":.1,"trial_id":"T1","participant_id":"P1","item_id":"I1","stimulus_id":"M1","valid_interval":True}]
    if two_trials:
        trials.append({"interval_id":"i2","recording_id":"R1","interval_type":"trial","start_time":.1,"end_time":.2,"trial_id":"T2","participant_id":"P1","item_id":"I2","stimulus_id":"M2","valid_interval":True})
    x["intervals"]=ep.standardize_eye_table(pd.DataFrame(trials),"intervals")
    if not matching:
        x["gaze_samples"]["trial_id"]="OTHER"
    return x


def test_gazepoint_workflow_feature_false_paths(monkeypatch):
    x=_base(); out=gw.build_gazepoint_media_trials(x,overwrite=False); assert len(out["intervals"])>=1
    y=_workflow_dataset(); y["biometrics"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"stream_id":["b"],"timestamp_seconds":[0.],"channel":["eda"],"value":[1.],"unit":["u"],"valid":[True],"trial_id":["UNKNOWN"],"stimulus_id":["M1"]}),"biometrics")
    f=gw._workflow_biometric_features(y); assert len(f)==3 and f.participant_id.isna().all()
    z=_workflow_dataset(matching=True)
    monkeypatch.setattr(gw,"derive_gaze_features",lambda x,**k:x)
    monkeypatch.setattr(gw,"_workflow_trial_features",lambda x: ep.empty_eye_table("features"))
    monkeypatch.setattr(gw,"_workflow_biometric_features",lambda x: ep.empty_eye_table("features"))
    r=gw.derive_gazepoint_workflow_features(z); assert ep.is_eye_dataset(r)


def test_gazepoint_workflow_plot_false_paths(tmp_path):
    empty_trials=_base(); empty_trials["gaze_samples"]=empty_trials["gaze_samples"].iloc[[0]].copy(); empty_trials["gaze_samples"]["trial_id"]=pd.NA
    a=gw.plot_gazepoint_workflow(empty_trials,tmp_path/"p1"); assert not a.empty
    two=_workflow_dataset(two_trials=True,matching=False); two["gaze_samples"]=two["gaze_samples"].iloc[[0]].copy()
    b=gw.plot_gazepoint_workflow(two,tmp_path/"p2"); assert not b.empty


def test_gazepoint_workflow_writer_loop_and_item_map_false(monkeypatch,tmp_path):
    monkeypatch.setattr(gw,"_write_csv",lambda data,path: str(path))
    irt={"readiness":pd.DataFrame(),"response_template":pd.DataFrame(),"process_covariates":pd.DataFrame(),"irt_long":pd.DataFrame(),"response_matrix":None,"response_time_matrix":None}
    paths=gw._write_tables({"skip":object(),"ok":pd.DataFrame()},irt,{"skip":object(),"ok":pd.DataFrame()},tmp_path); assert "table_ok" in paths and "qc_ok" in paths
    wf=SimpleNamespace(spec=gw.gazepoint_workflow_spec(),status="ok",source_path="src",output_dir=str(tmp_path),responses_supplied=False,irt={"status":"not_requested"},dataset=_empty(),tables={"trials":pd.DataFrame()},paths={},item_map=None)
    gw._reproducibility_files(wf); assert "workflow_spec" in wf.paths


def test_io_polygon_report_matrix_and_source_false_paths(monkeypatch,tmp_path):
    class Arrayish:
        def __array__(self,dtype=None): return np.asarray([[0.,1.],[2.,3.]],dtype=dtype)
    assert io10._polygon_to_text(Arrayish())=="0,1;2,3"
    monkeypatch.setattr(io10,"_save_plot",lambda *a,**k: None)
    e=_empty(); io10.report_eye_dataset(e,tmp_path/"empty.md",include_plots=True)
    y=_empty(); y["eye_samples"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"sample_id":["p"],"timestamp_seconds":[0.],"eye":["left"],"pupil_diameter":[3.],"pupil_valid":[True]}),"eye_samples")
    monkeypatch.setattr(ep,"plot_pupil_timeseries",None)
    io10.report_eye_dataset(y,tmp_path/"eye.md",include_plots=True)
    summary=pd.DataFrame({"format_family":[pd.NA,pd.NA],"vendor":["generic","generic"],"status":["fail","fail"]})
    obj=io10.EyeCorpusValidation(pd.DataFrame(),{},summary,"fail","now"); m=io10.format_compatibility_matrix(obj); assert int(m.empirical_failures.sum())==2
    binary=tmp_path/"x.bin"; binary.write_bytes(b"x"); ins=io10.inspect_eye_source(binary,include_hash=False); assert not bool(ins.tabular.iloc[0])
    txt=tmp_path/"x.txt"; txt.write_text("hello\n",encoding="utf-8"); assert io10.validate_eyelink_export(txt).empty


def test_io_validate_corpus_missing_override_and_anonymize_false_paths(monkeypatch):
    manifest=pd.DataFrame({"case_id":["c"],"path":["x"],"vendor":["generic"]})
    monkeypatch.setattr(io10,"_validate_manifest_rows",lambda x,n:x.copy())
    fake=SimpleNamespace(case_id="c")
    monkeypatch.setattr(io10,"validate_eye_source",lambda *a,**k:fake)
    monkeypatch.setattr(io10,"_format_validation_summary",lambda r:pd.DataFrame({"case_id":["c"],"status":["pass"],"imported":[True]}))
    out=io10.validate_eye_corpus(manifest); assert out.status=="pass"
    x=_empty(); anon=io10.anonymize_eye_dataset(x); assert ep.is_eye_dataset(anon)
    y=_empty(); y["recordings"]=y["recordings"].drop(columns=["source_file_set"])
    anon2=io10.anonymize_eye_dataset(y,redact_free_text=False); assert ep.is_eye_dataset(anon2)


def _format_validation(source):
    return io10.EyeFormatValidation(case_id="c",path="x",vendor="v",status="pass",started="s",completed="c",spec=io10.format_validation_spec(),source=source,detection=pd.DataFrame(),adapter_issues=pd.DataFrame(),checks=pd.DataFrame(),validation=pd.DataFrame(),coverage=pd.DataFrame(),preservation=pd.DataFrame(),audits={},roundtrip=None,import_error=None,dataset=None)


def test_io_redact_validation_source_false_loops():
    a=io10._redact_validation_result(_format_validation(pd.DataFrame())); assert a.path=="<redacted>"
    src=pd.DataFrame({"extension":["csv"],"other":[1]})
    b=io10._redact_validation_result(_format_validation(src)); assert len(b.source)==1

import eyeprocesspy.governance_09 as gov09
import eyeprocesspy.interoperability_storage_10 as ist
import eyeprocesspy.partitioned_storage_10 as ps


def test_governance_missing_required_after_optional_failure():
    pipe=ep.eye_analysis_pipeline(
        ep.eye_pipeline_step("up",lambda: (_ for _ in ()).throw(RuntimeError("x")),optional=True),
        ep.eye_pipeline_step("down",lambda up:up,requires="up"),
    )
    with pytest.raises(ep.EyeProcessValidationError,match="cannot run"):
        gov09.run_eye_pipeline(pipe,stop_on_error=True)


def test_interoperability_native_missing_loop(tmp_path):
    x=_base(); x["recordings"]["nominal_sampling_rate"]=60.
    x["gaze_samples"]["timestamp_native"]=[0.,1.,2.]
    x["eye_samples"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1","R1"],"sample_id":["e1","e2"],"timestamp_native":[pd.NA,1.],"timestamp_seconds":[0.,.1],"eye":["left","left"],"pupil_diameter":[3.,3.1],"pupil_valid":[True,True]}),"eye_samples")
    written=ist.export_eye_bids(x,tmp_path/"bids"); assert len(written)==1 and (tmp_path/"bids").exists()


def test_io_polygon_false_and_matrix_other_status(monkeypatch):
    real=io10.pd.isna
    class Arrayish:
        def __array__(self,dtype=None): return np.asarray([[0.,1.],[2.,3.]],dtype=dtype)
    monkeypatch.setattr(io10.pd,"isna",lambda value: False if isinstance(value,Arrayish) else real(value))
    assert io10._polygon_to_text(Arrayish())=="0,1;2,3"
    summary=pd.DataFrame({"format_family":[pd.NA,pd.NA],"vendor":["generic","generic"],"status":["other","pass"]})
    obj=io10.EyeCorpusValidation(pd.DataFrame(),{},summary,"pass","now"); out=io10.format_compatibility_matrix(obj); assert int(out.empirical_cases.sum())==2


def test_timebase_absolute_origin_with_rows():
    x=_empty(); x["gaze_samples"]=ep.standardize_eye_table(pd.DataFrame({"recording_id":["R1"],"stream_id":["g"],"sample_id":["s"],"timestamp_native":[10.],"timestamp_seconds":[pd.NA],"gaze_x":[.1],"gaze_y":[.2],"valid":[True]}),"gaze_samples")
    out=tb.normalize_timebase(x,component="gaze_samples",native_unit="seconds",origin="absolute",overwrite=True); assert float(out["gaze_samples"].timestamp_seconds.iloc[0])==10.


def test_partition_inner_commit_restore(monkeypatch,tmp_path):
    spec=ps.partition_eye_storage(by=["participant_id"],format="csv",max_rows=1)
    target=tmp_path/"store"; target.write_text("original",encoding="utf-8")
    original=ps.Path.rename
    def fail_staging(self,destination):
        if ".staging-" in self.name:
            raise OSError("staging commit")
        return original(self,destination)
    monkeypatch.setattr(ps.Path,"rename",fail_staging)
    with pytest.raises(OSError,match="staging commit"):
        ps.write_partitioned_eye_storage({"responses":pd.DataFrame({"participant_id":["P1"],"score":[1.]})},target,spec,overwrite=True)
    assert target.is_file() and target.read_text()=="original"

import math
import time
import eyeprocesspy.validation_completion_10 as vcomp
import eyeprocesspy.validation_orchestration_10 as vo


def test_partition_inner_commit_no_backup_false(monkeypatch,tmp_path):
    spec=ps.partition_eye_storage(by=["participant_id"],format="csv",max_rows=1)
    target=tmp_path/"new-store"
    original=ps.Path.rename
    def fail_staging(self,destination):
        if ".staging-" in self.name: raise OSError("new staging commit")
        return original(self,destination)
    monkeypatch.setattr(ps.Path,"rename",fail_staging)
    with pytest.raises(OSError,match="new staging commit"):
        ps.write_partitioned_eye_storage({"responses":pd.DataFrame({"participant_id":["P1"],"score":[1.]})},target,spec,overwrite=True)
    assert not target.exists()


def test_validation_completion_reporting_and_public_benchmark_false(monkeypatch,tmp_path):
    audit=vcomp.reporting_guideline_audit(_empty()); assert not bool(audit.loc[audit.section.eq("sampling"),"covered"].iloc[0])
    x=_empty(); x["recordings"]=x["recordings"].drop(columns=["participant_id"])
    monkeypatch.setattr(vcomp,"anonymize_eye_dataset",lambda x,**k:x.copy())
    def fake_export(y,path,**k): Path(path).mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(vcomp,"export_canonical",fake_export)
    out=vcomp.create_public_benchmark(x,tmp_path/"bench",include_samples=True); assert Path(out).exists()


def test_validation_completion_program_false_evidence_paths(monkeypatch,tmp_path):
    corpus={"summary":pd.DataFrame({"status":["pass"]})}
    vendor=pd.DataFrame({"vendor":["generic"],"pass_rate":[1.0],"cases":[1]})
    monkeypatch.setattr(vcomp,"audit_vendor_validation",lambda x:vendor)
    monkeypatch.setattr(vcomp,"write_vendor_validation_report",lambda *a,**k:None)
    monkeypatch.setattr(vcomp,"_save_bar",lambda *a,**k:None)
    monkeypatch.setattr(vcomp,"_write_snapshot",lambda *a,**k:None)
    monkeypatch.setattr(vcomp,"_merge_evidence",lambda *a,**k:None)
    sbc_results=iter([
        EyeResult({"ranks":pd.DataFrame()},eyeprocess_class="eye_sbc"),
        EyeResult({"ranks":pd.DataFrame({"parameter":["a","b"],"normalized_rank":[np.nan,.5]})},eyeprocess_class="eye_sbc"),
    ])
    monkeypatch.setattr(vcomp,"simulation_based_calibration",lambda **k:next(sbc_results))
    monkeypatch.setattr(vcomp,"sbc_summary",lambda r:pd.DataFrame({"status":["pass"]}))
    multi=EyeResult({"results":pd.DataFrame({"specification":["s"]})},eyeprocess_class="eye_preprocessing_multiverse")
    monkeypatch.setattr(vcomp,"preprocessing_multiverse",lambda **k:multi)
    monkeypatch.setattr(vcomp,"write_software_paper_scaffold",lambda path:str(path))
    evidence=pd.DataFrame({"model":["m"],"completed":[0],"required":[0]})
    monkeypatch.setattr(vcomp,"audit_advanced_model_evidence",lambda *a,**k:evidence)
    monkeypatch.setattr(vcomp,"write_advanced_model_evidence_report",lambda *a,**k:None)
    out=vcomp.run_eyeprocess_validation_program(corpus,tmp_path/"program",sbc_jobs={"empty":{},"groups":{}},multiverse_jobs={"plain":{}})
    assert out.eyeprocess_class=="eye_validation_program"


def _vo_plan():
    return vo.validation_job_plan({"n":[4]},replications=1,base_seed=17,model_family="q")

def _vo_job(): return _vo_plan()["jobs"].iloc[0].to_dict()
def _vo_sim(n=4,seed=1,**k): return {"values":np.arange(int(n),dtype=float),"truth":{"mu":1.5}}
def _vo_fit(sim,**k): return {"estimate":float(np.mean(sim["values"])),"se":.25,"converged":True,"iterations":3}
def _vo_extract(fit): return pd.DataFrame({"parameter":["mu"],"estimate":[fit["estimate"]],"std_error":[fit["se"]]})
def _vo_truth(sim): return sim["truth"]


def test_validation_orchestration_remaining_false_paths(monkeypatch,tmp_path):
    a=vo._default_diagnostics(object()); assert len(a)==1
    b=vo._default_diagnostics({"iterations":2}); assert int(b.iterations.iloc[0])==2
    monkeypatch.setattr(vo,"_deep_size",lambda value:1)
    r=vo._run_job_core(_vo_job(),_vo_sim,_vo_fit,_vo_extract,_vo_truth,memory_limit_mb=1); assert r["status"] in {"complete","nonconverged"}
    lock=tmp_path/"held.lock"; lock.mkdir(); assert vo._acquire_lock(lock,stale_after_seconds=3600) is False
    plan=_vo_plan(); output=tmp_path/"run"
    first=vo.run_validation_jobs(plan,_vo_sim,_vo_fit,_vo_extract,_vo_truth,output); assert first["results"][0]["status"]=="complete"
    second=vo.run_validation_jobs(plan,_vo_sim,_vo_fit,_vo_extract,_vo_truth,output); assert second["results"][0]["status"]=="complete"
