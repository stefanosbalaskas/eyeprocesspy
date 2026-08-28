from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import eyeprocesspy as ep


def test_generic_mapping_produces_canonical_tables_like_r():
    d=pd.DataFrame({"id":["P1"]*3,"rec":["R1"]*3,"t":[0,.01,.02],"gx":[.1,.2,.3],"gy":[.4,.5,.6],"pupil":[3.1,3.2,3.3]})
    m=ep.eye_mapping(participant="id",recording="rec",timestamp="t",x="gx",y="gy",pupil_left="pupil")
    x=ep.read_eye_generic(d,mapping=m,pupil_unit="millimetres",quiet=True)
    assert ep.is_eye_dataset(x)
    assert len(x["gaze_samples"])==3
    assert len(x["eye_samples"])==3
    assert x["recordings"].vendor.iloc[0]=="generic"
    assert len(x["provenance"])>=1
    assert x.validation.empty


def test_generic_inference_and_time_units():
    d=pd.DataFrame({"participant_id":["P1","P1"],"timestamp":[0,10],"gaze_x":[.1,.2],"gaze_y":[.2,.3]})
    m=ep.infer_eye_mapping(d)
    assert m["timestamp"]=="timestamp" and m["x"]=="gaze_x" and m["y"]=="gaze_y"
    x=ep.read_eye_generic(d,time_unit="milliseconds",quiet=True)
    np.testing.assert_allclose(x["gaze_samples"].timestamp_seconds,[0,.01])


def test_generic_events_responses_biometrics():
    d=pd.DataFrame({"p":["P1"]*2,"t":[0,1],"x":[.1,.2],"y":[.2,.3],"event":["TRIAL_START",None],"trial":["T1","T1"],"answer":[None,"A"],"score":[np.nan,1],"hr":[60,61]})
    m=ep.eye_mapping(participant="p",timestamp="t",x="x",y="y",event_name="event",trial="trial",response="answer",score="score",biometric_channels={"heart_rate":"hr"})
    x=ep.read_eye_generic(d,mapping=m,quiet=True)
    assert len(x["events"])==1
    assert len(x["responses"])==1
    assert set(x["biometrics"].channel)=={"heart_rate"}
    assert set(x["biometrics"].unit)=={"beats_per_minute"}


def test_mapping_validation_failure_contract():
    d=pd.DataFrame({"t":[0],"x":[.1],"y":[.2]})
    with pytest.raises(ValueError,match="Required mapping"):
        ep.validate_eye_mapping({"timestamp":"t","x":"x"},d)
    with pytest.raises(ValueError,match="Mapped source"):
        ep.validate_eye_mapping({"timestamp":"t","x":"x","y":"missing"},d)


def test_adapter_registry_detection_and_read(tmp_path: Path):
    p=tmp_path/'sample.csv'; p.write_text('t,x,y\n0,.1,.2\n',encoding='utf-8')
    detected=ep.detect_eye_format(p)
    assert detected.iloc[0]["format"]=="generic"
    assert detected.iloc[0]["confidence"]==pytest.approx(.1)
    # generic confidence intentionally cannot pass auto's 0.55 threshold
    with pytest.raises(ValueError,match="confidence threshold"):
        ep.read_eye_export(p)
    x=ep.read_eye_export(p,vendor="generic",mapping=ep.eye_mapping(timestamp="t",x="x",y="y"),quiet=True)
    assert ep.is_eye_dataset(x)


def test_custom_adapter_tie_priority_and_unregister(tmp_path: Path):
    p=tmp_path/'x.dat'; p.write_text('x',encoding='utf-8')
    d=pd.DataFrame({"t":[0],"x":[.1],"y":[.2]})
    reader=lambda path,**kw: ep.read_eye_generic(d,mapping=ep.eye_mapping(timestamp="t",x="x",y="y"),quiet=True)
    ep.register_eye_adapter("unit_a",lambda path,inspect_rows=20:.8,reader,priority=5,overwrite=True)
    ep.register_eye_adapter("unit_b",lambda path,inspect_rows=20:.8,reader,priority=7,overwrite=True)
    det=ep.detect_eye_format(p,candidates=["unit_a","unit_b"])
    assert list(det["format"])==["unit_b","unit_a"]
    ep.unregister_eye_adapter("unit_a"); ep.unregister_eye_adapter("unit_b")


def test_combine_resolves_duplicate_recording_ids():
    d=pd.DataFrame({"t":[0],"x":[.1],"y":[.2]})
    m=ep.eye_mapping(timestamp="t",x="x",y="y")
    a=ep.read_eye_generic(d,mapping=m,recording_id="R",quiet=True)
    b=ep.read_eye_generic(d,mapping=m,recording_id="R",quiet=True)
    out=ep.combine_eye_datasets(a,b)
    assert list(out["recordings"].recording_id)==["R","R_set2"]
    assert set(out["gaze_samples"].recording_id)=={"R","R_set2"}
