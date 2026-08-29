"""Advanced IRT/process sensitivity diagnostics from frozen eyeprocess 0.11.1."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import pandas as pd
from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .irt import EyeResult, _result


def _df(x: Any,name: str="data") -> pd.DataFrame:
    if isinstance(x,pd.DataFrame): return x.copy()
    try: return pd.DataFrame(x)
    except Exception as exc: raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc

def _req(d:pd.DataFrame,cols:Sequence[str],name:str="data") -> None:
    miss=[c for c in cols if c not in d.columns]
    if miss: raise EyeProcessValidationError(f"{name} is missing required columns: {', '.join(miss)}")

def _mean(x:Any)->float:
    a=pd.to_numeric(pd.Series(x),errors="coerce").to_numpy(float); a=a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def fit_mixture_irt_process_classes(response_matrix: Any,n_classes:int=2,model:Any=1,itemtype:str="2PL",SE:bool=False)->EyeResult:
    del response_matrix,model,itemtype,SE
    if int(n_classes)!=2: raise EyeProcessValidationError("The internally verified route currently supports n_classes = 2 only.")
    raise EyeProcessBackendError("fit_mixture_irt_process_classes() requires the exact R `mirt` mixture-2 engine; no surrogate is substituted.")


def map_latent_classes_to_process_profiles(class_membership:Any,process_data:Any,person:str="person_id",class_col:str="class",process_features:Sequence[str]=())->EyeResult:
    cm=_df(class_membership,"class_membership"); pd0=_df(process_data,"process_data")
    _req(cm,[person,class_col],"class_membership"); process_features=list(process_features)
    if not process_features: raise EyeProcessValidationError("Supply at least one process feature.")
    _req(pd0,[person,*process_features],"process_data")
    if cm[person].astype(str).duplicated().any(): raise EyeProcessValidationError("class_membership must contain one assignment row per person for class-profile mapping.")
    p=pd0[[person,*process_features]].copy(); p[person]=p[person].astype(str)
    for v in process_features: p[v]=pd.to_numeric(p[v],errors="coerce")
    p=p.groupby(person,sort=False,dropna=False)[process_features].mean().reset_index()
    c=cm[[person,class_col]].copy(); c[person]=c[person].astype(str); d=c.merge(p,on=person,how="left")
    summary=d.groupby(class_col,sort=True,dropna=False)[process_features].mean().reset_index().rename(columns={class_col:"class"})
    return _result("eye_latent_process_alignment",data=d,summary=summary,process_features=process_features,class_col=class_col,
                   caveat="Process summaries can aid interpretation of latent response classes but cannot prove substantive strategy labels.")


def audit_nonparametric_rasch(response_matrix:Any,methods:Sequence[str]=("T1","T10"),n:int=100,splitcr:str="median",seed:int=321)->EyeResult:
    del response_matrix,splitcr,seed
    methods=[str(m) for m in methods if str(m)]
    if not methods: raise EyeProcessValidationError("Supply at least one NPtest method.")
    if int(n)<1: raise EyeProcessValidationError("n must be positive.")
    raise EyeProcessBackendError("audit_nonparametric_rasch() requires the exact R `eRm::NPtest` engine; no Python surrogate is substituted.")


def audit_item_reduction_sensitivity(erm_model:Any,criterion:Any=None,alpha:float=.05,maxstep:int=5)->EyeResult:
    del erm_model,criterion
    if not np.isfinite(alpha) or alpha<=0 or alpha>=1: raise EyeProcessValidationError("alpha must be in (0,1).")
    if int(maxstep)<1: raise EyeProcessValidationError("maxstep must be positive.")
    raise EyeProcessBackendError("audit_item_reduction_sensitivity() requires the exact R `eRm::stepwiseIt` engine.")


def biometric_imputation_sensitivity(data:Any,variables:Sequence[str],methods:Sequence[str]=("mice","missForest"),m:int=3,maxit:int=3,seed:int=521)->EyeResult:
    del seed
    d=_df(data); variables=list(variables)
    if not variables: raise EyeProcessValidationError("Supply at least one variable to impute.")
    _req(d,variables)
    methods=list(dict.fromkeys(str(x) for x in methods if str(x) in {"mice","missForest"}))
    if int(m)<1 or int(maxit)<1: raise EyeProcessValidationError("m and maxit must be positive.")
    missing=pd.DataFrame({"variable":variables,"missing_prop":[float(d[v].isna().mean()) for v in variables]})
    # Exact R backends are not silently replaced.  Availability is recorded as the R contract does.
    status=pd.DataFrame({"method":methods,"status":["package_unavailable" for _ in methods]})
    return _result("eye_biometric_imputation_sensitivity",missingness=missing,results={},status=status,variables=variables,
                   caveat="Imputed biometric datasets are sensitivity analyses by default. Do not silently replace complete-case or explicitly modeled missingness analyses.")


def audit_biometric_imputation(*args:Any,**kwargs:Any)->EyeResult:
    return biometric_imputation_sensitivity(*args,**kwargs)


def fit_process_rasch_tree(response_matrix:Any,covariates:Any,formula:Any=None,maxit:int=60)->EyeResult:
    X=np.asarray(response_matrix); c=_df(covariates,"covariates")
    if X.shape[0]!=len(c): raise EyeProcessValidationError("response_matrix and covariates must have the same number of persons.")
    if formula is None and not len(c.columns): raise EyeProcessValidationError("At least one splitting covariate is required.")
    if int(maxit)<1: raise EyeProcessValidationError("maxit must be positive.")
    raise EyeProcessBackendError("fit_process_rasch_tree() requires the exact R `psychotree::raschtree` engine.")


def compare_bayesian_process_models(*models:Any,method:str="loo")->Any:
    if method not in {"loo","bayes_factor"}: raise EyeProcessValidationError("method must be loo or bayes_factor.")
    if len(models)<2: raise EyeProcessValidationError("Supply at least two fitted Bayesian models.")
    raise EyeProcessBackendError("compare_bayesian_process_models() requires the exact R `brms` LOO/Bayes-factor backend.")

__all__=["fit_mixture_irt_process_classes","map_latent_classes_to_process_profiles","audit_nonparametric_rasch","audit_item_reduction_sensitivity","biometric_imputation_sensitivity","audit_biometric_imputation","fit_process_rasch_tree","compare_bayesian_process_models"]
