#!/usr/bin/env python3
"""Read original EMIP mirror TSV streams and independently audit 31x2 code windows.

This audit validates source windowing/exposure and per-eye validity/sample
counts against frozen derived trial records. It does NOT reconstruct detector
fixations, AOI-hit sequences, or model estimates. Raw gaze files are streamed,
kept ephemeral, and never added to Git or uploaded as artifacts.
"""
from __future__ import annotations
import csv
import io
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
S_DIR=ROOT/"results"
MIRROR="https://raw.githubusercontent.com/egeiran/Data-Preprocessing-EMIP/main/emip_dataset/rawdata"
PIDS=(1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,20,21,22,23,24,25,26,28,30,31,33,34,37,39,41)
BEGIN=re.compile(r"(rectangle|vehicle)_java(?:2)?\.jpg",re.I)
END=re.compile(r"(?:mupliple|multiple)_choice_(rectangle|vehicle)\.jpg",re.I)

def source_stream(session,pid):
    url=f"{MIRROR}/{pid}_rawdata.tsv"
    r=session.get(url,timeout=(20,140))
    r.raise_for_status()
    return io.StringIO(r.content.decode("utf-8-sig",errors="replace"))

def read_intervals(file):
    head=None; active=None; saved={}
    for line in file:
        tok=line.rstrip("\r\n").split("\t")
        if len(tok)<4:continue
        if tok[0].strip()=="Time" and tok[1].strip()=="Type":
            head={v:i for i,v in enumerate(tok)}
            continue
        if head is None:continue
        typ=tok[head["Type"]].strip()
        if typ=="MSG":
            msg=" ".join(tok[3:])
            b=BEGIN.search(msg)
            e=END.search(msg)
            if b:
                if active is not None:
                    raise ValueError("Code interval lacks question-screen end before next onset")
                active={"task":b.group(1).lower(),"onset":int(float(tok[0])),"samples":0,
                        "n_valid_left":0,"n_valid_right":0}
            if e and active is not None and e.group(1).lower()==active["task"]:
                active["offset"]=int(float(tok[0]))
                if active["task"] in saved:raise ValueError("Repeated code stimulus task")
                saved[active["task"]]=active
                active=None
        elif typ=="SMP" and active is not None:
            active["samples"]+=1
            for eye in ("L","R"):
                key="n_valid_left" if eye=="L" else "n_valid_right"
                v=tok[head[f"{eye} Validity"]]
                try:
                    x=float(tok[head[f"{eye} POR X [px]"]]);y=float(tok[head[f"{eye} POR Y [px]"]])
                    okay=v.strip()=="1" and math.isfinite(x) and math.isfinite(y) and (0<x<=1920) and (0<y<=1080)
                except (ValueError,IndexError,KeyError):
                    okay=False
                if okay:active[key]+=1
    if active is not None:raise ValueError("Unclosed code interval "+active["task"])
    if set(saved)!={"rectangle","vehicle"}:raise ValueError("Incomplete 2-task intervals: "+str(sorted(saved)))
    return saved

def load_targets():
    rows=[]
    for i in range(1,5):
        with (S_DIR/f"emip_mirror_derived_batch_{i}.csv").open(newline="",encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    assert len(rows)==744
    targets={}
    for r in rows:
        key=(int(r["pid"]),r["task"],r["eye"])
        v=(float(r["duration"]),int(r["n_samples"]),int(r["n_valid"]))
        if key in targets and targets[key]!=v:
            raise AssertionError("Derived source counts vary with detector/AOI "+str(key))
        targets[key]=v
    if len(targets)!=31*2*2:raise AssertionError("Missing participant-task-eye targets")
    return targets

def main():
    targets=load_targets()
    session=requests.Session(); session.headers["User-Agent"]="EMIP-independent-source-window-audit/1.0"
    mismatches=[]; checked=0
    for pid in PIDS:
        with source_stream(session,pid) as text:
            windows=read_intervals(text)
        for task,window in windows.items():
            duration=(window["offset"]-window["onset"])/1e6
            for eye in ("left","right"):
                expected=targets[(pid,task,eye)]
                measured=(duration,window["samples"],window["n_valid_"+eye])
                if abs(measured[0]-expected[0])>0.002 or measured[1:]!=expected[1:]:
                    mismatches.append((pid,task,eye,expected,measured))
                checked+=1
        print("audited source record",pid,flush=True)
    if mismatches:
        for row in mismatches[:20]:
            print("MISMATCH:",row,file=sys.stderr)
        raise SystemExit(f"FAILED {len(mismatches)}/{checked} source-derived support rows")
    print(f"PASS: {len(PIDS)} distinct raw TSVs, 62 source code intervals, {checked} eye-specific support rows")
    print("SCOPE: source time windows, sample counts, valid counts; NO fixation/AOI independent audit")
if __name__=="__main__":main()
