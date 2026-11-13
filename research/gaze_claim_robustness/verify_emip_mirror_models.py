#!/usr/bin/env python3
"""Independent THMS Route B statistical audit of frozen EMIP mirror summaries.

Checks 31 persons x 2 tasks x 3 detectors x 2 eye-validity masks x 2 AOI
representations = 744 derived rows and refits the 24 planned Poisson models.

IMPORTANT: a model-layer refit is NOT independent validation of raw-to-fixation
or AOI extraction. Raw participant recordings are NOT committed to this repo.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE=Path(__file__).resolve().parent
DETECTORS=("ivt30_100","ivt40_50","idt25_50")
EYES=("left","right")
MASKS=("line_bbox","element_union")
QUALITY=("source_valid","valid80")

def run(base:Path):
    rows=pd.concat(
        [pd.read_csv(base/f"emip_mirror_derived_batch_{i}.csv") for i in range(1,5)],
        ignore_index=True,
    )
    expected=pd.read_csv(base/"emip_mirror_model_summary.csv")
    unique=["pid","task","eye","det","mask"]
    assert rows.shape[0]==744, f"Unexpected derived-row count {len(rows)}"
    assert not rows.duplicated(unique).any(), "Duplicate participant-task-decision rows"
    assert rows.pid.nunique()==31 and rows.groupby("pid").size().eq(24).all()
    assert set(rows.det)==set(DETECTORS) and set(rows.eye)==set(EYES)
    assert set(rows["mask"])==set(MASKS) and set(rows.task)=={"rectangle","vehicle"}
    assert rows.valid_ratio.between(0,1).all()
    assert rows.duration.gt(0).all()
    assert rows[["Y","fix","assigned"]].ge(0).all().all()
    assert len(expected)==24 and not expected.duplicated(["det","eye","mask","quality"]).any()
    report=[]
    for det in DETECTORS:
      for eye in EYES:
       for mask in MASKS:
        for q in QUALITY:
          x=rows[(rows.det==det)&(rows.eye==eye)&(rows["mask"]==mask)].copy()
          if q=="valid80": x=x[x.valid_ratio>=0.8].copy()
          X=pd.DataFrame({
            "intercept":np.ones(len(x),dtype=float),
            "correct":x.correct.astype(float).to_numpy(),
            "vehicle":(x.task=="vehicle").astype(float).to_numpy(),
            "second":(x.order==2).astype(float).to_numpy(),
          })
          fit=sm.GLM(x.Y.to_numpy(),X,family=sm.families.Poisson(),
                     offset=np.log(x.duration.to_numpy())).fit(
                     cov_type="cluster",
                     cov_kwds={"groups":x.pid.to_numpy(),
                               "use_correction":True,"df_correction":False})
          beta=float(fit.params["correct"])
          se=float(fit.bse["correct"])
          obs={"N":len(x),"G":x.pid.nunique(),"y_total":int(x.Y.sum()),
               "logRR":beta,"SE":se,"RR":float(np.exp(beta)),
               "lower":float(np.exp(beta-1.96*se)),
               "upper":float(np.exp(beta+1.96*se))}
          ref=expected[(expected.det==det)&(expected.eye==eye)&
                       (expected["mask"]==mask)&(expected.quality==q)]
          assert len(ref)==1
          for col in ("N","G","y_total"):
              assert obs[col]==int(ref.iloc[0][col]), (det,eye,mask,q,col)
          for col in ("logRR","SE","RR","lower","upper"):
              assert np.isclose(obs[col],float(ref.iloc[0][col]),
                       atol=1e-7,rtol=1e-7), (det,eye,mask,q,col,obs[col],ref.iloc[0][col])
          report.append({"det":det,"eye":eye,"mask":mask,"quality":q,**obs})
    assert len(report)==24
    out=pd.DataFrame(report)
    assert out.lower.le(1).all() and out.upper.ge(1).all()
    print("PASS: 31 participants, 744 derived rows, 24/24 independently refitted Poisson models")
    print("Positive log-rate ratios:",int(out.logRR.gt(0).sum()))
    print("Rate ratio range:",round(out.RR.min(),6),round(out.RR.max(),6))
    print("All Wald intervals include RR=1")
    print("SCOPE: model-layer audit only; source-to-events reconstruction not independently certified.")
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results-dir",type=Path,default=HERE/"results")
    args=p.parse_args()
    run(args.results_dir)
