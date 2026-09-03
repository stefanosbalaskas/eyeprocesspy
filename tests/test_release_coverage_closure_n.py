from __future__ import annotations

import builtins
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.benchmark_reproducibility_10 as br
import eyeprocesspy.gazepoint as gp
import eyeprocesspy.interoperability_storage_10 as ios
import eyeprocesspy.requested_api_07 as ra
import eyeprocesspy.semantic_validation_07 as sv
import eyeprocesspy.validation_completion_10 as vcomp
import eyeprocesspy.validation_evidence_10 as ve
import eyeprocesspy.validation_orchestration_10 as vo
import eyeprocesspy.validation_orchestration_completion_10 as voc
import eyeprocesspy.validation_program_10 as vp
import eyeprocesspy.multimodal_staged as mm


def _simulator(n=4, seed=1, **kwargs):
    return {"values": np.arange(int(n), dtype=float), "truth": {"mu": 1.5}}

def _fitter(simulation, **kwargs):
    return {"estimate": float(np.mean(simulation["values"])), "se": .25, "converged": True}

def _extractor(fit, **kwargs):
    return pd.DataFrame({"parameter":["mu"],"estimate":[fit["estimate"]],"std_error":[fit["se"]]})

def _truth(simulation, **kwargs):
    return simulation["truth"]


def _write_bids_with_pupil(root: Path):
    path=root/"sub-P01"/"beh"/"sub-P01_task-read_run-01_recording-eye1_physio.tsv.gz"
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wt",encoding="utf-8") as fh:
        fh.write("0\t0.1\t0.2\t3.0\n")
    meta={"Columns":["timestamp","x_coordinate","y_coordinate","pupil_size"],"PhysioType":"eyetrack","StartTime":0.0,"RecordedEye":"left","SamplingFrequency":60.0,"timestamp":{"Units":"seconds"},"pupil_size":{"Units":"mm"}}
    Path(str(path)[:-7]+".json").write_text(json.dumps(meta),encoding="utf-8")
    return path


def test_validation_orchestration_fingerprint_path_and_missing_resume(monkeypatch,tmp_path):
    real_getsource=vo.inspect.getsource
    monkeypatch.setattr(vo.inspect,"getsource",lambda f: (_ for _ in ()).throw(OSError("no source")))
    fp=vo._function_fingerprint(lambda x: x); assert fp.startswith("fn-")
    monkeypatch.setattr(vo.inspect,"getsource",real_getsource)
    plan=vo.validation_job_plan({"n":[4]},replications=1,base_seed=17,model_family="n")
    manifest=vo.write_validation_job_manifest(plan,tmp_path/"manifest")
    run=vo.run_validation_jobs(manifest,_simulator,_fitter,_extractor,_truth,tmp_path/"run"); assert run.results[0]["status"] == "complete"
    captured={}
    def fake_run(plan_arg, **kwargs):
        captured["job_ids"]=kwargs["job_ids"]; return vo.EyeValidationRun(results=[])
    monkeypatch.setattr(vo,"run_validation_jobs",fake_run)
    resumed=vo.resume_validation_jobs(manifest,tmp_path/"fresh",simulator=_simulator,fitter=_fitter,extractor=_extractor,truth_extractor=_truth)
    assert captured["job_ids"] and isinstance(resumed,vo.EyeValidationRun)


def test_validation_program_overwrite_and_default_spec_lines(tmp_path):
    occupied=tmp_path/"validation"; occupied.mkdir(); (occupied/"stale.txt").write_text("x",encoding="utf-8")
    with pytest.raises(Exception):
        vcomp.run_eyeprocess_validation_program(object(),occupied,overwrite=True)
    assert not (occupied/"stale.txt").exists()
    with pytest.raises(Exception):
        vp.run_model_validation(_simulator,_fitter,_extractor,_truth,grid=object(),spec=None)


def test_requested_api_and_multimodal_invalid_paths():
    frame=ra._extract_samples({"samples":pd.DataFrame({"x":[1]})}); assert frame.x.iloc[0]==1
    with pytest.raises(ep.EyeProcessValidationError): ra._extract_samples(object())
    with pytest.raises(ep.EyeProcessValidationError,match="eye_multimodal_measurement"): mm.audit_multimodal_measurement(object())


def test_gazepoint_invalid_frequency_and_read_oserror(monkeypatch,tmp_path):
    info=gp._gp_time_info(pd.DataFrame({"TIMETICK(F=.)":[1.]})); assert np.isnan(info["tick_frequency"])
    p=tmp_path/"plain.csv"; p.write_text("x",encoding="utf-8")
    real=Path.read_text; monkeypatch.setattr(Path,"read_text",lambda self,*a,**k: (_ for _ in ()).throw(OSError("blocked")))
    assert gp._gp_is_summary_report(p) is False; monkeypatch.setattr(Path,"read_text",real)


def test_semantic_coercion_and_nonlinear_correlated_branch():
    assert sv._df([{"x":1}],"x").x.iloc[0]==1
    source=pd.DataFrame({"x":[1.,2.,3.,4.]}); roundtrip=pd.DataFrame({"x":[1.,2.2,3.7,5.6]})
    spec=sv.semantic_fidelity_spec(correlation_floor=.9,coordinate_tolerance=1e-12)
    report=sv.field_fidelity_report(source,roundtrip,fields=["x"],tolerance=1e-12,spec=spec)
    assert report.fields.status.iloc[0] == "SEMANTICALLY_EQUIVALENT"


def test_validation_release_report_auto_completion(monkeypatch,tmp_path):
    collection=vo.EyeValidationCollection(plan=None,paths=[],results=[],jobs=pd.DataFrame(),estimates=pd.DataFrame(),diagnostics=pd.DataFrame(),predictions=pd.DataFrame(),draws=pd.DataFrame(),corrupt=[],collected_utc="test")
    sentinel=object(); monkeypatch.setattr(voc,"audit_validation_completion",lambda x: sentinel)
    with pytest.raises(ep.EyeProcessValidationError,match="completion"):
        voc.write_validation_release_report(collection,tmp_path/"x.md",completion=None)


def test_bids_defensive_unmapped_pupil_guard(monkeypatch,tmp_path):
    root=tmp_path/"bids"; _write_bids_with_pupil(root)
    real_map=pd.Series.map; calls={"sample":0}
    def map_once(self,arg,*args,**kwargs):
        if self.name=="sample_id":
            calls["sample"]+=1
            if calls["sample"]==2: return pd.Series([np.nan]*len(self),index=self.index,name=self.name)
        return real_map(self,arg,*args,**kwargs)
    monkeypatch.setattr(pd.Series,"map",map_once)
    with pytest.raises(ValueError,match="Could not map BIDS pupil"):
        ios.import_eye_bids(root)


def test_reproducibility_relative_version_import_failure(monkeypatch,tmp_path):
    p=tmp_path/"x.txt"; p.write_text("x",encoding="utf-8")
    real=builtins.__import__
    def blocked(name,globals=None,locals=None,fromlist=(),level=0):
        fl=fromlist or ()
        if name=="" and level==1 and "__version__" in fl: raise ImportError("blocked version")
        return real(name,globals,locals,fromlist,level)
    monkeypatch.setattr(builtins,"__import__",blocked)
    m=br.package_reproducibility_manifest([p],include_session=True); assert not any(s.startswith("eyeprocesspy ") for s in m.session)


def test_truth_map_unique_series_line():
    assert ve._truth_map(pd.Series([1.],index=["a"])) == {"a":1.0}
