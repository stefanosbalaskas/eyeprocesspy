from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.gazepoint_real_10 as gr
import eyeprocesspy.gazepoint_workflow_10 as gw
import eyeprocesspy.io_validation_10 as io10
import eyeprocesspy.vendor_corpus_10 as vc
import eyeprocesspy.vendor_importers_10 as vi


def _base_dataset():
    rec=pd.DataFrame([{"recording_id":"R1","participant_id":"P1"}])
    spaces=ep.new_coordinate_space("coord_display_normalized_top_left")
    gaze=pd.DataFrame({"recording_id":["R1"]*4,"stream_id":["g"]*4,"sample_id":["s1","s2","s3","s4"],"timestamp_seconds":[0.,.1,.2,.3],"gaze_x":[.1,.2,.3,.4],"gaze_y":[.1,.2,.3,.4],"valid":[True]*4,"trial_id":["T1"]*4,"stimulus_id":["M1"]*4,"coordinate_space_id":["coord_display_normalized_top_left"]*4,"aoi_id":["A1"]*4})
    intervals=pd.DataFrame([{"interval_id":"i1","recording_id":"R1","interval_type":"trial","start_time":0.,"end_time":.3,"trial_id":"T1","participant_id":"P1","item_id":"I1","stimulus_id":"M1","condition_id":"C1","valid_interval":True}])
    episodes=pd.DataFrame([{"episode_id":"f1","recording_id":"R1","episode_type":"fixation","start_time":.05,"end_time":.15,"duration_ms":100.,"centroid_x":.2,"centroid_y":.2,"derived_by":"vendor","trial_id":"T1","aoi_id":"A1","coordinate_space_id":"coord_display_normalized_top_left"}])
    eye=pd.DataFrame({"recording_id":["R1"]*3,"sample_id":["p1","p2","p3"],"timestamp_seconds":[0.,.1,.2],"eye":["left"]*3,"pupil_diameter":[3.,3.1,3.2],"pupil_valid":[True]*3,"trial_id":["T1"]*3,"stimulus_id":["M1"]*3})
    responses=pd.DataFrame({"recording_id":["R1"],"participant_id":["P1"],"trial_id":["T1"],"item_id":["I1"],"response":["1"],"score":[1.],"response_time":[.8]})
    return ep.new_eye_dataset(recordings=rec,coordinate_spaces=spaces,gaze_samples=gaze,intervals=intervals,episodes=episodes,eye_samples=eye,responses=responses)


def _summary_text(rows: str = "M01,AOI 1,Headline,Alice,1,0,5,0.25,1.5,30,4,1,2,0.7,420,72,0.83") -> str:
    return f"""Gazepoint Analysis,7.2.0
Processed On,2026-08-29 12:34:56

AOI Summary
Media ID,AOI ID,AOI Name
M01,AOI 1,Headline

AOI Statistics (for each user)
Media ID,AOI ID,AOI Name,User Name,User ID,AOI Start,AOI Duration (sec - U=UserControlled),Time to 1st View (sec) -1.0 means not viewed,Time Viewed (sec),Time Viewed (%),Fixations (#),Revisits (#),Clicks (#),Ave Dial (0-1),Ave GSR (kOhm),Ave Heart Rate (BPM),Ave Interbeat Interval (s)
{rows}
"""


def test_gazepoint_real_remaining_paths(monkeypatch,tmp_path):
    real_read=gr.pd.read_csv; real_reader=gr.csv.reader
    monkeypatch.setattr(gr.pd,"read_csv",lambda *a,**k: (_ for _ in ()).throw(ValueError("x"))); monkeypatch.setattr(gr.csv,"reader",lambda *a,**k: [])
    assert gr._parse_csv_block(["TITLE","a,b","1,2"],"TITLE").empty
    monkeypatch.setattr(gr.pd,"read_csv",real_read); monkeypatch.setattr(gr.csv,"reader",real_reader)
    row="M01,AOI 1,Headline,Alice,1,0,5,0.25,1.5,30,4,1,2,0.7,420,72,0.83"
    p=tmp_path/"Data_Summary_duplicate.csv"; p.write_text(_summary_text(row+"\n"+row),encoding="utf-8")
    summary=gr.read_gazepoint_summary(p); features=gr._summary_features(summary); assert features.feature_id.astype(str).str.contains("_row0000").any()
    plain=tmp_path/"CurrentAOIStatistics.csv"; pd.DataFrame({"foo":[1]}).to_csv(plain,index=False)
    with pytest.raises(ValueError,match="AOI identifier"):
        gr.read_gazepoint_aoi_statistics(plain,quiet=True)
    empty=tmp_path/"Data_Summary_empty.csv"; empty.write_text("Gazepoint Analysis,7.2.0\nProcessed On,now\n\nAOI Summary\n\nAOI Statistics (for each user)\n",encoding="utf-8")
    out=gr.read_gazepoint_aoi_statistics(empty,participant_id="PX",recording_id="RX",quiet=True); assert out["aoi_definitions"].empty and out["recordings"].participant_id.iloc[0]=="PX"
    one=tmp_path/"Data_Summary_one.csv"; one.write_text(_summary_text(),encoding="utf-8")
    a=gr.read_gazepoint_aoi_statistics(one,participant_id="PX",recording_id="RX",quiet=True); assert a["features"].participant_id.eq("PX").all() and a["features"].recording_id.eq("RX").all()
    b=gr.read_gazepoint_aoi_statistics(one,recording_id="ONLYREC",quiet=True); assert b["features"].recording_id.eq("ONLYREC").all()


def test_gazepoint_workflow_feature_reset_aoi_and_scalar_group(monkeypatch):
    x=_base_dataset(); existing=pd.DataFrame({"feature_id":["old"],"recording_id":["R1"],"participant_id":["P1"],"trial_id":["T1"],"item_id":["I1"],"stimulus_id":["M1"],"aoi_id":[pd.NA],"feature_name":["old"],"value":[1.],"unit":["u"],"level":["trial"],"method":["derive_gaze_features old"],"parameters":[""]})
    x["features"]=ep.standardize_eye_table(existing,"features"); out=gw.derive_gazepoint_workflow_features(x,reset_workflow_features=True)
    assert "old" not in set(out["features"].feature_name.astype(str)); assert out["features"].aoi_id.notna().any()
    original=pd.DataFrame.groupby
    class ScalarGroup:
        def __init__(self,frame): self.frame=frame
        def __iter__(self): yield "R1",self.frame
    def scalar(self,by=None,*args,**kwargs):
        if by == ["recording_id"] and "episode_type" in self.columns: return ScalarGroup(self)
        return original(self,by,*args,**kwargs)
    monkeypatch.setattr(pd.DataFrame,"groupby",scalar); s=gw._summarize_fixations(x,by=("recording_id",),source="vendor"); assert s.recording_id.iloc[0]=="R1"


def test_vendor_companion_keep_raw_and_edf_finally(monkeypatch,tmp_path):
    x=_base_dataset(); neon=tmp_path/"neon"; neon.mkdir(); pd.DataFrame({"timestamp [ns]":[1_000_000_000],"name":["mark"]}).to_csv(neon/"events.csv",index=False)
    vi._read_neon_companions(x,neon,keep_raw=True); assert "neon_events" in x.raw
    core=tmp_path/"core"; core.mkdir(); pd.DataFrame({"start_timestamp":[0.],"duration":[.1],"norm_pos_x":[.2],"norm_pos_y":[.3]}).to_csv(core/"fixations.csv",index=False)
    vi._read_core_companions(x,core,keep_raw=True); assert "core_fixations" in x.raw
    edf=tmp_path/"x.edf"; edf.write_bytes(b"edf"); converter=tmp_path/"edf2asc"; converter.write_text("x",encoding="utf-8")
    real_mkstemp=vi.tempfile.mkstemp
    def missing_mkstemp(*a,**k):
        fd,path=real_mkstemp(*a,**k); os.unlink(path); return fd,path
    monkeypatch.setattr(vi.tempfile,"mkstemp",missing_mkstemp); monkeypatch.setattr(vi.subprocess,"run",lambda *a,**k: SimpleNamespace(returncode=1,stdout="",stderr="failed"))
    with pytest.raises(ep.EyeProcessBackendError,match="conversion failed"):
        vi.read_eyelink_edf(edf,edf2asc=str(converter),keep_asc=False)


def test_vendor_corpus_directory_missing_semantics_and_roundtrip_paths(monkeypatch,tmp_path):
    root=tmp_path/"case"; root.mkdir(); (root/"subdir").mkdir(); (root/"x.txt").write_text("x",encoding="utf-8"); files=vc._iter_case_files(root,include_hidden=True); assert files==[root/"x.txt"]
    corpus=tmp_path/"corpus"; corpus.mkdir(); sem=vc._semantics_path(corpus); sem.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame({"vendor":["V"]}).to_csv(sem,index=False)
    frame=vc._read_semantics(corpus); assert set(vc.SEMANTICS_COLUMNS) <= set(frame.columns)
    x=_base_dataset(); monkeypatch.setattr(vc,"_compare_columns",lambda *a,**k: pd.DataFrame()); audit=vc.audit_roundtrip_loss(x,x,tables=None); assert audit.eyeprocess_class == "eye_roundtrip_loss_audit"
    y=x.copy(); y["events"]=None; audit2=vc.audit_roundtrip_loss(y,y,tables=["events"]); assert audit2.summary.empty


def test_io_polygon_nan_and_owned_roundtrip_path(tmp_path):
    assert io10._polygon_to_text(np.nan) is pd.NA
    x=_base_dataset(); result=io10.roundtrip_eye_dataset(x,path=None,cleanup=False); assert Path(result.path).exists()
    import shutil; shutil.rmtree(result.path,ignore_errors=True)
