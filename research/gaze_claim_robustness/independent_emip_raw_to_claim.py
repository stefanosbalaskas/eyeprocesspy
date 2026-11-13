#!/usr/bin/env python3
"""Independent raw-to-claim reconstruction for the public 31-person EMIP mirror.

SOURCE: public raw participant TSV streams, original EMIP metadata, and four
original image-specific AOI CSV files. No raw recordings are committed.
INDEPENDENCE: detector implementations and AOI-label assignment below are
separate from the first derivation that produced frozen batch files.

This is an *acceptance test*, not a result generator: success requires exact
participant/task/sample validity support, exact fixation counts, assigned
fixation counts, AOI transition counts, and 24 model-level results.

If a step differs from the frozen outputs, the command raises and reports
discrepancies. Do not retrofit thresholds to obtain a pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PIDS = (1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,20,21,22,23,24,25,26,28,30,31,33,34,37,39,41)
DETECTORS = ("ivt30_100","ivt40_50","idt25_50")
EYES = ("left","right")
AOIS = ("line_bbox","element_union")
QUALITIES = ("source_valid","valid80")
RAW_ROOT = "https://raw.githubusercontent.com/egeiran/Data-Preprocessing-EMIP/main/emip_dataset/rawdata"
BASE_META = "https://raw.githubusercontent.com/hiqb269/EMIP-Toolkit/main/emtk/datasets/EMIP/EMIP-Toolkit-%20replication%20package/emip_dataset"
METADATA_URL = BASE_META + "/emip_metadata.csv"
AOI_ROOT = BASE_META + "/stimuli"
START = re.compile(r"\b(rectangle|vehicle)_java(?:2)?\.jpg\b",re.I)
STOP = re.compile(r"\b(?:mupliple|multiple)_choice_(rectangle|vehicle)\.jpg\b",re.I)

def retrieve(session, url):
    r=session.get(url,timeout=(30,180))
    r.raise_for_status()
    return r.content

def strnum(value, default=float("nan")):
    try:
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default

def metadata(session):
    text=retrieve(session,METADATA_URL).decode("utf-8-sig")
    table=pd.read_csv(io.StringIO(text))
    table["id"]=pd.to_numeric(table["id"],errors="raise").astype(int)
    assert not table["id"].duplicated().any()
    out=table.set_index("id")
    for pid in PIDS:
        assert pid in out.index
        assert str(out.loc[pid,"experiment_language"]).strip().lower()=="java", pid
    return out

def aoi_tables(session):
    result={}
    for task in ("rectangle","vehicle"):
        for suffix in ("","2"):
            key=f"{task}_java{suffix}.jpg"
            url=f"{AOI_ROOT}/{task}_java{suffix}.csv"
            text=retrieve(session,url).decode("utf-8-sig")
            table=pd.read_csv(io.StringIO(text))
            assert set(["level","x1","y1","x2","y2"]).issubset(table.columns)
            lines=table[table.level.eq("line")][["x1","y1","x2","y2"]].to_numpy(dtype=float)
            elems=table[table.level.eq("element")][["x1","y1","x2","y2"]].to_numpy(dtype=float)
            assert len(lines)>=2 and len(elems)>=len(lines)
            labels=[]
            for e in elems:
                overlaps=np.minimum(lines[:,3],e[3])-np.maximum(lines[:,1],e[1])
                overlaps=np.maximum(overlaps,0)
                maxima=np.flatnonzero(overlaps==overlaps.max())
                if overlaps.max()<=0 or len(maxima)!=1:
                    raise ValueError("Unresolved or ambiguous element-to-line mapping: "+key)
                labels.append(int(maxima[0]))
            result[key]={"line":lines,"elements":elems,"element_label":np.array(labels,dtype=int)}
    return result

def source_record(session,pid):
    raw=retrieve(session,f"{RAW_ROOT}/{pid}_rawdata.tsv")
    source_hash=hashlib.sha256(raw).hexdigest()
    file=io.StringIO(raw.decode("utf-8-sig",errors="replace"))
    header=None
    windows={}
    current=None
    for line in file:
        tokens=line.rstrip("\n\r").split("\t")
        if len(tokens)<4:
            continue
        if tokens[0]=="Time" and tokens[1]=="Type":
            header={name:i for i,name in enumerate(tokens)}
            needed=["Time","Type","L POR X [px]","L POR Y [px]","R POR X [px]","R POR Y [px]","L Validity","R Validity"]
            if not all(k in header for k in needed):
                raise ValueError(f"{pid}: required raw header columns missing")
            continue
        if header is None:
            continue
        typ=tokens[1]
        if typ=="MSG":
            msg=" ".join(tokens[3:]).lower()
            b=START.search(msg); e=STOP.search(msg)
            if b:
                if current is not None:
                    raise ValueError(f"{pid}: unclosed stimulus interval before next code stimulus")
                current={"task":b.group(1),"stimulus":b.group(0),"start":int(strnum(tokens[0])),
                         "times":[],"left_x":[],"left_y":[],"right_x":[],"right_y":[],
                         "left_valid":[],"right_valid":[]}
            elif e and current is not None and e.group(1)==current["task"]:
                current["stop"]=int(strnum(tokens[0]))
                if current["task"] in windows:
                    raise ValueError(f"{pid}: duplicate code stimulus for task {current['task']}")
                windows[current["task"]]=current
                current=None
            continue
        if typ!="SMP" or current is None:
            continue
        t=int(strnum(tokens[0]))
        current["times"].append(t)
        for eye,letter in (("left","L"),("right","R")):
            x=strnum(tokens[header[f"{letter} POR X [px]"]])
            y=strnum(tokens[header[f"{letter} POR Y [px]"]])
            v=tokens[header[f"{letter} Validity"]].strip()=="1"
            v=v and math.isfinite(x) and math.isfinite(y) and 0<x<=1920 and 0<y<=1080
            current[f"{eye}_x"].append(x)
            current[f"{eye}_y"].append(y)
            current[f"{eye}_valid"].append(v)
    if current is not None:
        raise ValueError(f"{pid}: missing final code-to-question marker")
    if set(windows)!={"rectangle","vehicle"}:
        raise ValueError(f"{pid}: incomplete code windows {sorted(windows)}")
    for task,w in windows.items():
        if w["stop"]<=w["start"]:raise ValueError("Nonpositive exposure")
        for name in ("times","left_x","left_y","right_x","right_y","left_valid","right_valid"):
            w[name]=np.array(w[name])
        assert w["times"].size>0
    return windows,source_hash

def valid_runs(t,x,y,valid):
    """Drop invalid samples, but never bridge over invalid data, time gaps or reversals."""
    open_at=None
    for i in range(len(t)):
        good=bool(valid[i])
        connected=(good and i>0 and bool(valid[i-1]) and t[i]>t[i-1]
                   and t[i]-t[i-1]<=100_000)
        if open_at is not None and (not good or not connected):
            if i-open_at>=2:
                yield t[open_at:i],x[open_at:i],y[open_at:i]
            open_at=None
        if good and open_at is None:
            open_at=i
    if open_at is not None and len(t)-open_at>=2:
        yield t[open_at:],x[open_at:],y[open_at:]

def ivt(t,x,y,threshold,min_ms):
    if len(t)<2:return []
    # Exact physical geometry from the released EMIP header, not real head position.
    px_mm_x=344.0/1920.0
    px_mm_y=194.0/1080.0
    # Per-pair small-angle pixel displacement: degrees ≈ dx_mm/700 * (180/pi).
    # The frozen protocol specifies instantaneous velocity of pixel displacement
    # under the 344×194-mm / 700-mm nominal source geometry, not absolute
    # eccentricity under a nonlinear coordinate projection.
    degx=(x-960.0)*px_mm_x/700.0*(180.0/np.pi)
    degy=(y-540.0)*px_mm_y/700.0*(180.0/np.pi)
    dt=np.diff(t).astype(float)/1e6
    spd=np.hypot(np.diff(degx),np.diff(degy))/dt
    edges=spd<=threshold
    out=[]
    i=0
    while i<len(edges):
        if not edges[i]:
            i+=1;continue
        start=i
        while i<len(edges) and edges[i]:
            i+=1
        stop=i # includes sample at index stop
        duration=(t[stop]-t[start])/1000.0
        if duration>=min_ms:
            out.append((float(x[start:stop+1].mean()),
                        float(y[start:stop+1].mean()),int(t[start]),int(t[stop])))
    return out

def idt(t,x,y,dispersion=25.0,min_ms=50):
    out=[];n=len(t);i=0
    minimum_us=min_ms*1000
    while i<n-1:
        j=int(np.searchsorted(t,t[i]+minimum_us,side="left"))
        if j>=n:
            break
        slice_x=x[i:j+1];slice_y=y[i:j+1]
        d=float(np.ptp(slice_x)+np.ptp(slice_y))
        if d>dispersion:
            i+=1
            continue
        e=j+1
        while e<n:
            newd=float(np.ptp(x[i:e+1])+np.ptp(y[i:e+1]))
            if newd>dispersion:
                break
            e+=1
        last=e-1
        out.append((float(np.mean(x[i:last+1])),
                    float(np.mean(y[i:last+1])),
                    int(t[i]),int(t[last])))
        i=e
    return out

def aoi_label(x,y,definition,mode):
    if mode=="line_bbox":
        rects=definition["line"]
        labels=np.arange(len(rects))
    else:
        rects=definition["elements"]
        labels=definition["element_label"]
    matches=np.flatnonzero((rects[:,0]<=x)&(x<=rects[:,2])&
                           (rects[:,1]<=y)&(y<=rects[:,3]))
    if not len(matches):
        return None
    selected=set(int(labels[j]) for j in matches)
    if len(selected)!=1:
        return None
    return selected.pop()

def gaze_outcome(w,eye,det,mode,definition):
    t=w["times"]
    valid=w[f"{eye}_valid"].astype(bool)
    frac=float(np.count_nonzero(valid)/len(valid))
    fixations=[]
    for st,sx,sy in valid_runs(t,w[f"{eye}_x"],w[f"{eye}_y"],valid):
        if det=="ivt30_100":out=ivt(st,sx,sy,30,100)
        elif det=="ivt40_50":out=ivt(st,sx,sy,40,50)
        else:out=idt(st,sx,sy,25,50)
        fixations.extend((xx,yy,first,last) for xx,yy,first,last in out)
        fixations.append((None,None,None,None)) # explicit gap sentinel
    total_fix=sum(1 for x,_,_,_ in fixations if x is not None)
    labs=[None if x is None else aoi_label(x,y,definition,mode)
          for x,y,_,_ in fixations]
    assigned=sum(a is not None for a in labs)
    transitions=sum(a is not None and b is not None and a!=b
                    for a,b in zip(labs,labs[1:]))
    return dict(Y=transitions,fix=total_fix,assigned=assigned,
                n_samples=int(len(t)),n_valid=int(np.count_nonzero(valid)),valid_ratio=frac)

def expected_rows():
    frames=[pd.read_csv(RESULTS/f"emip_mirror_derived_batch_{i}.csv")
            for i in range(1,5)]
    old=pd.concat(frames,ignore_index=True)
    keys=["pid","task","eye","det","mask"]
    assert len(old)==744 and not old.duplicated(keys).any()
    return old.set_index(keys)

def reproduce(out_dir):
    sess=requests.Session()
    sess.headers["User-Agent"]="THMS-EMIP-independent-raw-to-claim-audit/1.0"
    md=metadata(sess); aois=aoi_tables(sess)
    original=expected_rows()
    differences=[]
    generated=[]
    source_hashes={}
    for pid in PIDS:
        wins,source_hash=source_record(sess,pid)
        source_hashes[str(pid)]=source_hash
        for task,w in wins.items():
            expected_stim=str(md.loc[pid,f"stimulus_{task}"]).strip().lower()
            if w["stimulus"]!=expected_stim:
                raise AssertionError(f"Stimulus/AOI source mismatch for pid {pid}, task {task}: {w['stimulus']} vs {expected_stim}")
            assert w["stimulus"] in aois
            correct=int(md.loc[pid,f"correct_{task}"])
            order=int(md.loc[pid,f"order_{task}"])
            duration=(w["stop"]-w["start"])/1e6
            for eye in EYES:
                for det in DETECTORS:
                    for mask in AOIS:
                        observed=gaze_outcome(w,eye,det,mask,aois[w["stimulus"]])
                        key=(pid,task,eye,det,mask)
                        ref=original.loc[key]
                        row={"pid":pid,"task":task,"correct":correct,"order":order,
                             "eye":eye,"det":det,"mask":mask,"duration":duration,**observed}
                        generated.append(row)
                        columns=["correct","order","n_samples","n_valid","Y","fix","assigned"]
                        bad=[c for c in columns if int(row[c])!=int(ref[c])]
                        for c in ["duration","valid_ratio"]:
                            if abs(row[c]-float(ref[c]))>1e-5:bad.append(c)
                        if bad:
                            differences.append({"pid":pid,"task":task,"eye":eye,"det":det,
                                                "mask":mask,"fields":bad,
                                                "expected_Y":int(ref["Y"]),"obtained_Y":int(row["Y"]),
                                                "expected_fix":int(ref["fix"]),"obtained_fix":row["fix"]})
        print(f"Source audit: {pid}, generated {len(generated)}/744 records",flush=True)
    if len(generated)!=744:
        raise AssertionError("Not all 744 rows reconstructed")
    df=pd.DataFrame(generated)
    overview={
        "audited_participants":len(PIDS),"audited_code_trials":62,
        "audited_derived_rows":len(df),"source_sha256_by_pid":source_hashes,
        "mismatching_rows":len(differences),
        "mismatch_fields":dict(Counter(k for r in differences for k in r["fields"])),
        "illustrative_differences":differences[:20],
        "raw_to_gaze_outcome_verified":len(differences)==0,
        "model_refit_verified":False,
        "fully_independently_certified":False,
    }
    if differences:
        return overview
    summary=pd.read_csv(RESULTS/"emip_mirror_model_summary.csv")
    n_models=0
    for det in DETECTORS:
        for eye in EYES:
            for mask in AOIS:
                for quality in QUALITIES:
                    subset=df[(df.det==det)&(df.eye==eye)&(df["mask"]==mask)]
                    if quality=="valid80":
                        subset=subset[subset.valid_ratio>=0.8]
                    X=pd.DataFrame({"intercept":np.ones(len(subset)),
                                    "correct":subset.correct.to_numpy().astype(float),
                                    "vehicle":subset.task.eq("vehicle").astype(float).to_numpy(),
                                    "second":subset.order.eq(2).astype(float).to_numpy()})
                    fit=sm.GLM(subset.Y.to_numpy(),X,family=sm.families.Poisson(),
                               offset=np.log(subset.duration.to_numpy())).fit(
                        cov_type="cluster",cov_kwds={"groups":subset.pid.to_numpy(),
                                                    "use_correction":True,"df_correction":False})
                    beta=float(fit.params["correct"]);se=float(fit.bse["correct"])
                    found={"logRR":beta,"SE":se,"RR":math.exp(beta),
                           "lower":math.exp(beta-1.96*se),"upper":math.exp(beta+1.96*se)}
                    ref=summary[(summary.det==det)&(summary.eye==eye)&
                                (summary["mask"]==mask)&(summary.quality==quality)]
                    if len(ref)!=1:raise AssertionError("Missing comparison model")
                    if len(subset)!=int(ref.iloc[0].N) or subset.pid.nunique()!=int(ref.iloc[0].G):
                        raise AssertionError("Model support mismatch")
                    for col,val in found.items():
                        if not math.isclose(val,float(ref.iloc[0][col]),abs_tol=1e-7,rel_tol=1e-7):
                            raise AssertionError(f"Model {det}/{eye}/{mask}/{quality} {col} mismatch")
                    n_models+=1
    overview["model_refit_verified"]=n_models==24
    overview["fully_independently_certified"]=n_models==24 and len(differences)==0
    return overview

def selftest():
    # Exactly one 100-ms, low-velocity I-VT fixation; 2 distant AOIs.
    t=np.array([0,25000,50000,75000,100000],dtype=np.int64)
    x=np.array([600.,601.,600.,600.,601.])
    y=np.array([180.,180.,180.,180.,180.])
    assert len(ivt(t,x,y,30,100))==1
    assert len(idt(t,x,y,25,50))==1
    t2=np.array([0,25000,50000,75000,100000,350000,375000,400000,425000,450000],dtype=np.int64)
    v=np.ones(10,dtype=bool)
    assert len(list(valid_runs(t2,np.r_[x,x+100],np.r_[y,y],v)))==2
    rect={"line":np.array([[500,150,700,200],[700,250,900,300]],dtype=float),
          "elements":np.array([[500,150,700,200],[700,250,900,300]],dtype=float),
          "element_label":np.array([0,1])}
    assert aoi_label(600,180,rect,"line_bbox")==0
    assert aoi_label(600,180,rect,"element_union")==0
    assert aoi_label(600,220,rect,"line_bbox") is None
    print("PASS synthetic unit tests (not independent empirical validation)")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("emip_independent_audit"))
    parser.add_argument("--selftest",action="store_true")
    args=parser.parse_args()
    if args.selftest:
        selftest();return
    args.out.mkdir(parents=True,exist_ok=True)
    outcome={"fully_independently_certified":False,"source":"public EMIP mirror"}
    try:
        outcome=reproduce(args.out)
    except Exception as e:
        outcome["fatal"]=type(e).__name__+": "+str(e)
    (args.out/"audit.json").write_text(json.dumps(outcome,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in outcome.items() if k!="source_sha256_by_pid"},indent=2))
    if not outcome.get("fully_independently_certified"):
        raise SystemExit(2)

if __name__=="__main__":
    main()
