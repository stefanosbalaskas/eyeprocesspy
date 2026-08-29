"""Bayesian process diagnostics and gaze-anchored 3PL audit contracts."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
import pandas as pd
from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .irt import EyeResult, _result


def bayesian_process_diagnostics_dashboard(*models:Any,model_names:Sequence[str]|None=None,compute_loo:bool=True,compute_bayes_factor:bool=False,posterior_summary:bool=True)->EyeResult:
    del model_names,compute_loo,compute_bayes_factor,posterior_summary
    if not models: raise EyeProcessValidationError("Supply at least one fitted Bayesian model.")
    raise EyeProcessBackendError("bayesian_process_diagnostics_dashboard() requires R `brms`/`posterior`/`loo` model objects for exact parity.")


def bayesian_process_diagnostic_flags(x:Any,rhat_threshold:float=1.01,ess_threshold:float=400)->pd.DataFrame:
    if getattr(x,"eyeprocess_class",None)!="eye_bayesian_process_dashboard": raise EyeProcessValidationError("x must be an eye_bayesian_process_dashboard.")
    if not np.isfinite(rhat_threshold) or rhat_threshold<=1 or not np.isfinite(ess_threshold) or ess_threshold<=0: raise EyeProcessValidationError("Require rhat_threshold > 1 and ess_threshold > 0.")
    p=x.get("posterior")
    if not isinstance(p,pd.DataFrame) or p.empty: return pd.DataFrame()
    out=p.copy()
    for col in ["rhat","ess_bulk","ess_tail"]:
        if col not in out: out[col]=np.nan
        out[col]=pd.to_numeric(out[col],errors="coerce")
    out["rhat_review"]=out["rhat"].notna() & (out["rhat"]>rhat_threshold)
    out["ess_bulk_review"]=out["ess_bulk"].notna() & (out["ess_bulk"]<ess_threshold)
    out["ess_tail_review"]=out["ess_tail"].notna() & (out["ess_tail"]<ess_threshold)
    out["review_required"]=out[["rhat_review","ess_bulk_review","ess_tail_review"]].any(axis=1)
    return out


def fit_gaze_anchored_3pl_audit(response_matrix:Any,process_data:Any=None,item:str="item_id",process_features:Sequence[str]=("ttff_ms","dwell_ms","pupil_bc","pupil_peak","rt_ms","accuracy"),model:Any=1,SE:bool=False)->EyeResult:
    X=pd.DataFrame(response_matrix)
    if X.shape[1]<4: raise EyeProcessValidationError("At least four items are required for this 3PL audit.")
    del process_data,item,process_features,model,SE
    raise EyeProcessBackendError("fit_gaze_anchored_3pl_audit() requires the exact R `mirt` 3PL engine; no approximate lower-asymptote fit is substituted.")


def gaze_anchored_3pl_alignment(x:Any)->pd.DataFrame:
    if getattr(x,"eyeprocess_class",None)!="eye_gaze_anchored_3pl_audit": raise EyeProcessValidationError("x must be an eye_gaze_anchored_3pl_audit.")
    a=x.get("alignment")
    return a.copy() if isinstance(a,pd.DataFrame) else pd.DataFrame()


def audit_3pl_process_signatures(x:Any,lower_asymptote_quantile:float=.80,fast_rt_quantile:float=.20,fast_ttff_quantile:float=.20)->pd.DataFrame:
    if getattr(x,"eyeprocess_class",None)!="eye_gaze_anchored_3pl_audit": raise EyeProcessValidationError("x must be an eye_gaze_anchored_3pl_audit.")
    qs=np.asarray([lower_asymptote_quantile,fast_rt_quantile,fast_ttff_quantile],dtype=float)
    if not np.all(np.isfinite(qs)) or np.any((qs<=0)|(qs>=1)): raise EyeProcessValidationError("All review quantiles must lie in (0,1).")
    d=x["item_parameters"].copy()
    if "lower_asymptote" not in d: return d
    g=pd.to_numeric(d["lower_asymptote"],errors="coerce")
    if not g.notna().any(): return d
    qg=float(g.quantile(lower_asymptote_quantile)); d["high_lower_asymptote_review"]=g.notna() & (g>=qg)
    d["fast_rt_review"]=False
    if "rt_ms" in d:
        r=pd.to_numeric(d["rt_ms"],errors="coerce")
        if r.notna().any(): d["fast_rt_review"]=r.notna() & (r<=float(r.quantile(fast_rt_quantile)))
    d["fast_ttff_review"]=False
    if "ttff_ms" in d:
        t=pd.to_numeric(d["ttff_ms"],errors="coerce")
        if t.notna().any(): d["fast_ttff_review"]=t.notna() & (t<=float(t.quantile(fast_ttff_quantile)))
    d["process_review_count"]=d[["high_lower_asymptote_review","fast_rt_review","fast_ttff_review"]].sum(axis=1).astype(int)
    d["review_label"]=np.where(d["process_review_count"]>=2,"review_item_response_process_alignment","no_combined_review_flag")
    return d

__all__=["bayesian_process_diagnostics_dashboard","bayesian_process_diagnostic_flags","fit_gaze_anchored_3pl_audit","gaze_anchored_3pl_alignment","audit_3pl_process_signatures"]
