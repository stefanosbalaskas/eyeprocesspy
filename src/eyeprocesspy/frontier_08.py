"""Evidence-gated frontier estimator contracts from eyeprocess 0.11.1."""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from typing import Any
import pandas as pd
from .exceptions import EyeProcessValidationError
from .irt import EyeResult, _result


def _df(x: Any, name: str="data") -> pd.DataFrame:
    if isinstance(x,pd.DataFrame): return x.copy()
    try: return pd.DataFrame(x)
    except Exception as exc: raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _gated(model_id: str, purpose: str, required: Sequence[str], engine: Any=None, fit: Any=None) -> EyeResult:
    return _result("eye_gated_process_model", id=model_id, purpose=purpose,
                   required_evidence=list(required), engine=engine, fit=fit,
                   status="gated" if fit is None else "experimental_external_engine",
                   caveat="This frontier estimator is evidence-gated; no internal surrogate is substituted for the requested model.")


def fit_kde_latent_distribution_irt(response_matrix: Any, engine: Callable[...,Any]|None=None, **kwargs: Any) -> EyeResult:
    req=["nonparametric latent-density integration inside the IRT marginal likelihood","bandwidth-selection specification","parameter-recovery simulation","comparison against misspecified Gaussian latent distributions"]
    if engine is None: return _gated("kde_latent_distribution_irt","IRT calibration with a nonparametric KDE latent distribution rather than post-hoc density estimation.",req)
    if not callable(engine): raise EyeProcessValidationError("engine must be None or a function.")
    return _gated("kde_latent_distribution_irt","External exact/nonparametric latent-density IRT engine.",req,{"name":getattr(engine,"__name__","engine")},engine(response_matrix,**kwargs))


def fit_persistence_gaze_diffusion_irt(data: Any, engine: Callable[...,Any]|None=None, **kwargs: Any) -> EyeResult:
    req=["separate capability, caution, and persistence identification","competing censoring/persistence likelihood","parameter recovery","behavior-label guardrails"]
    if engine is None: return _gated("persistence_gaze_diffusion_irt","Extend gaze-diffusion measurement with a competing maximum-time/persistence process.",req)
    if not callable(engine): raise EyeProcessValidationError("engine must be None or a function.")
    return _gated("persistence_gaze_diffusion_irt","External persistence-augmented gaze-diffusion estimator.",req,{"name":getattr(engine,"__name__","engine")},engine(data,**kwargs))


def fit_nonignorable_missing_irt(data: Any, engine: Callable[...,Any]|None=None, **kwargs: Any) -> EyeResult:
    req=["explicit nonignorable missingness mechanism","prior specification","identifiability checks","missingness-mechanism recovery simulations","sensitivity to prior/missingness misspecification"]
    if engine is None: return _gated("nonignorable_missing_irt","Bayesian IRT with a jointly modeled nonignorable missingness mechanism.",req)
    if not callable(engine): raise EyeProcessValidationError("engine must be None or a function.")
    return _gated("nonignorable_missing_irt","External Bayesian nonignorable-missing IRT estimator.",req,{"name":getattr(engine,"__name__","engine")},engine(data,**kwargs))


def prepare_structured_unstructured_process_features(structured: Any, unstructured: Any=None, fold: str|None=None,
                                                     builder: Callable[...,Any]|None=None,
                                                     id: Sequence[str]=("person_id","item_id"), **kwargs: Any) -> EyeResult:
    d=_df(structured,"structured"); ids=[c for c in id if c in d.columns]
    contract={"structured_columns":list(d.columns),"id_columns":ids,"has_unstructured":unstructured is not None,
              "fold_column":fold,"leakage_rule":"Any learned representation, scaling, vocabulary, embedding, or feature selection must be fitted inside the training fold only."}
    if fold is None: return _result("eye_structured_unstructured_process_features",contract=contract,structured=d,unstructured=unstructured,status="representation_contract_only")
    if fold not in d.columns: raise EyeProcessValidationError(f"structured is missing required columns: {fold}")
    vals=pd.Series(d[fold]).dropna().drop_duplicates().tolist()
    if builder is None: return _result("eye_structured_unstructured_process_features",contract=contract,structured=d,unstructured=unstructured,fold_registry=vals,status="fold_registry_requires_builder")
    if not callable(builder): raise EyeProcessValidationError("builder must be None or a function.")
    if len(vals)<2: raise EyeProcessValidationError("At least two non-missing folds are required for fold-local representation building.")
    results={}
    for v in vals:
        test=d[fold].notna() & d[fold].eq(v); train_s=d.loc[~test].copy(); test_s=d.loc[test].copy(); train_u=unstructured; test_u=unstructured
        if isinstance(unstructured,pd.DataFrame) and fold in unstructured.columns:
            train_u=unstructured.loc[unstructured[fold].notna() & unstructured[fold].ne(v)].copy(); test_u=unstructured.loc[unstructured[fold].notna() & unstructured[fold].eq(v)].copy()
        results[str(v)]=builder(train_s,train_u,test_s,test_u,v,**kwargs)
    return _result("eye_structured_unstructured_process_features",contract=contract,folds=results,status="fold_local_representation_built")


def fit_crossclassified_process_irt_mhrm(data: Any, engine: Callable[...,Any]|None=None, **kwargs: Any) -> EyeResult:
    req=["cross-classified outcome/process likelihood","study-design nesting","scalable MH-RM estimation","parameter recovery across cluster sizes","runtime/memory scaling benchmark"]
    if engine is None: return _gated("crossclassified_process_irt_mhrm","Scalable MH-RM estimation for cross-classified outcome/process IRT.",req)
    if not callable(engine): raise EyeProcessValidationError("engine must be None or a function.")
    return _gated("crossclassified_process_irt_mhrm","External scalable cross-classified process-IRT estimator.",req,{"name":getattr(engine,"__name__","engine")},engine(data,**kwargs))


def audit_frontier_model_contract(x: Any, evidence: Mapping[str,Any]|None=None) -> pd.DataFrame:
    if getattr(x,"eyeprocess_class",None)!="eye_gated_process_model": raise EyeProcessValidationError("x must be eye_gated_process_model.")
    if evidence is None: evidence={}
    if not isinstance(evidence,Mapping): raise EyeProcessValidationError("evidence must be a list/mapping.")
    return pd.DataFrame([{"model":x["id"],"engine_supplied":x.get("fit") is not None,
                          "n_required_evidence_elements":len(x["required_evidence"]),"n_named_evidence_objects":len(evidence),
                          "status":"candidate_for_validation_review" if x.get("fit") is not None and len(evidence)>=3 else "remains_gated"}])

__all__=["fit_kde_latent_distribution_irt","fit_persistence_gaze_diffusion_irt","fit_nonignorable_missing_irt","prepare_structured_unstructured_process_features","fit_crossclassified_process_irt_mhrm","audit_frontier_model_contract"]
