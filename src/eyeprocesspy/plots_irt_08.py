"""Matplotlib counterparts for frozen 0.8 IRT/process S3 plots."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .exceptions import EyeProcessValidationError


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires the 'plots' extra.") from exc
    return plt

def _ax(ax=None):
    plt=_plt()
    if ax is None: _,ax=plt.subplots()
    return ax

def _empty(title:str,message:str="No validated internal plot is available",ax=None):
    ax=_ax(ax); ax.set_title(title); ax.text(.5,.5,message,ha="center",va="center",transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]); return ax


def plot_eye_gated_process_model(x:Any,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_gated_process_model": raise EyeProcessValidationError("x must be eye_gated_process_model.")
    ax=_empty(f"Gated frontier model: {x['id']}",f"status: {x['status']}\nNo validated internal plot is claimed.",ax); ax.gp3_data=x; return ax

def plot_eye_mixture_irt_process(x:Any,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_mixture_irt_process": raise EyeProcessValidationError("x must be eye_mixture_irt_process.")
    ax=_empty("Mixture IRT",f"latent response classes: {x.get('n_classes')}\nInspect class-specific item parameters before process interpretation.",ax); ax.gp3_data=x; return ax

def plot_eye_latent_process_alignment(x:Any,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_latent_process_alignment": raise EyeProcessValidationError("x must be eye_latent_process_alignment.")
    s=x["summary"]; vars=[v for v in x["process_features"] if v in s.columns]; ax=_ax(ax)
    if s.empty or not vars: return _empty("Latent-class/process alignment","No process summaries available",ax)
    xx=np.arange(len(vars));
    for _,row in s.iterrows(): ax.plot(xx,[row[v] for v in vars],marker="o",label=str(row["class"]))
    ax.set_xticks(xx,vars,rotation=45,ha="right"); ax.set_ylabel("Class mean"); ax.set_title("Latent response classes vs process summaries"); ax.legend(); ax.gp3_data=s; return ax

def plot_eye_nonparametric_rasch_audit(x:Any,method:str|None=None,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_nonparametric_rasch_audit": raise EyeProcessValidationError("x must be eye_nonparametric_rasch_audit.")
    return _empty("Nonparametric Rasch diagnostic",f"External eRm plot required ({method or 'default'}).",ax)

def plot_eye_item_reduction_sensitivity(x:Any,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_item_reduction_sensitivity": raise EyeProcessValidationError("x must be eye_item_reduction_sensitivity.")
    items=list(x.get("eliminated_items",[])); ax=_ax(ax)
    if not items: return _empty("Item-reduction sensitivity","No items were eliminated",ax)
    ax.scatter(np.arange(1,len(items)+1),np.ones(len(items))); [ax.text(i,1,s,rotation=45) for i,s in enumerate(items,1)]; ax.set_xlabel("Elimination step"); ax.set_yticks([]); ax.set_title("Stepwise item-reduction sensitivity"); ax.gp3_data=items; return ax

def plot_eye_biometric_imputation_sensitivity(x:Any,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_biometric_imputation_sensitivity": raise EyeProcessValidationError("x must be eye_biometric_imputation_sensitivity.")
    d=x["missingness"]; ax=_ax(ax); ax.bar(d["variable"],d["missing_prop"]); ax.set_ylim(0,1); ax.set_ylabel("Missing proportion"); ax.set_title("Biometric-feature missingness before imputation"); ax.tick_params(axis="x",rotation=45); ax.gp3_data=d; return ax

def plot_eye_process_rasch_tree(x:Any,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_process_rasch_tree": raise EyeProcessValidationError("x must be eye_process_rasch_tree.")
    return _empty("Process Rasch tree","External psychotree plot required",ax)

def plot_eye_bayesian_process_dashboard(x:Any,type:str="loo",ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_bayesian_process_dashboard": raise EyeProcessValidationError("x must be eye_bayesian_process_dashboard.")
    ax=_ax(ax)
    if type=="loo":
        d=x.get("loo_table",pd.DataFrame())
        if not isinstance(d,pd.DataFrame) or d.empty or "elpd_loo" not in d or not pd.to_numeric(d["elpd_loo"],errors="coerce").notna().any(): return _empty("Bayesian process diagnostics","LOO summary unavailable",ax)
        ax.bar(d["model"],pd.to_numeric(d["elpd_loo"],errors="coerce")); ax.set_ylabel("ELPD-LOO"); ax.set_title("Bayesian process-model LOO comparison"); ax.gp3_data=d; return ax
    d=x.get("posterior",pd.DataFrame())
    if not isinstance(d,pd.DataFrame) or d.empty: return _empty("Bayesian process diagnostics","Posterior summary unavailable",ax)
    col="rhat" if type=="rhat" else "ess_bulk"; z=d[["model",col]].copy() if col in d else pd.DataFrame()
    if z.empty: return _empty(f"{type} diagnostics",f"{col} unavailable",ax)
    groups=[pd.to_numeric(g[col],errors="coerce").dropna().to_numpy() for _,g in z.groupby("model",sort=False)]; labels=list(z["model"].drop_duplicates())
    ax.boxplot(groups,tick_labels=labels); ax.axhline(1.01 if type=="rhat" else 400,linestyle="--"); ax.set_title("Posterior R-hat by model" if type=="rhat" else "Posterior bulk ESS by model"); ax.gp3_data=z; return ax

def plot_eye_gaze_anchored_3pl_audit(x:Any,type:str="lower_asymptote",feature:str|None=None,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_gaze_anchored_3pl_audit": raise EyeProcessValidationError("x must be eye_gaze_anchored_3pl_audit.")
    d=x["item_parameters"]; ax=_ax(ax)
    if type=="lower_asymptote":
        if "lower_asymptote" not in d: return _empty("3PL process audit","Lower-asymptote parameter unavailable",ax)
        ax.bar(d["item_id"],d["lower_asymptote"]); ax.tick_params(axis="x",rotation=45); ax.set_ylabel("3PL lower asymptote"); ax.set_title("Item lower-asymptote parameters")
    elif type=="difficulty_discrimination":
        if not {"a","b"}.issubset(d.columns): return _empty("3PL item parameters","Difficulty/discrimination unavailable",ax)
        ax.scatter(d["b"],d["a"]); ax.set_xlabel("Difficulty b"); ax.set_ylabel("Discrimination a"); ax.set_title("3PL item parameter map")
    else:
        available=[v for v in x.get("process_features",[]) if v in d.columns];
        if not available or "lower_asymptote" not in d: return _empty("3PL/process alignment","No process features available",ax)
        feature=feature or available[0]
        if feature not in available: raise EyeProcessValidationError("Unknown process feature.")
        ax.scatter(d[feature],d["lower_asymptote"]); ax.set_xlabel(feature); ax.set_ylabel("3PL lower asymptote"); ax.set_title("Descriptive 3PL/process alignment")
    ax.gp3_data=d; return ax


def plot_eye_multiblock_process_map(x:Any,type:str="individuals",ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_multiblock_process_map": raise EyeProcessValidationError("x must be eye_multiblock_process_map.")
    ax=_ax(ax)
    if type=="individuals":
        d=x["person_coordinates"]; dims=[c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if len(dims)<2: return _empty("Multiblock map","Fewer than two dimensions",ax)
        ax.scatter(d[dims[0]],d[dims[1]]); ax.set_xlabel(dims[0]); ax.set_ylabel(dims[1]); ax.set_title(f"Multiblock individuals -- {x['engine']}")
    elif type=="variables":
        d=x["variable_coordinates"]; dims=[c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if len(dims)<2: return _empty("Multiblock variables","Fewer than two dimensions",ax)
        ax.scatter(d[dims[0]],d[dims[1]])
        for _,r in d.iterrows(): ax.text(r[dims[0]],r[dims[1]],str(r["variable"]))
        ax.set_xlabel(dims[0]); ax.set_ylabel(dims[1]); ax.set_title("Multiblock variable coordinates")
    else:
        d=x["block_coordinates"]; num=[c for c in d.columns if c!="block" and pd.api.types.is_numeric_dtype(d[c])]
        if d.empty or not num: return _empty("Multiblock blocks","No numeric block coordinates",ax)
        vals=d[num].abs().mean(axis=1); ax.bar(d["block"],vals); ax.tick_params(axis="x",rotation=45); ax.set_ylabel("Mean absolute component coordinate"); ax.set_title("Multiblock contribution summary")
    ax.gp3_data=d; return ax

def plot_eye_process_profile_mixture(x:Any,type:str="profiles",ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_process_profile_mixture": raise EyeProcessValidationError("x must be eye_process_profile_mixture.")
    ax=_ax(ax)
    if type=="posterior":
        d=x["assignment"]; cols=[c for c in d.columns if c.startswith("profile_probability_")]
        if not cols: return _empty("Process profiles","No posterior/proximity probabilities",ax)
        M=d[cols].to_numpy(float)
        for row in M: ax.plot(np.arange(1,len(cols)+1),row)
        ax.set_xticks(np.arange(1,len(cols)+1),cols,rotation=45,ha="right"); ax.set_ylabel("Probability/proximity"); ax.set_title("Process-profile membership uncertainty"); ax.gp3_data=d; return ax
    if type in {"profiles","parallel"}:
        d=x["summary"]; vars=[v for v in x["variables"] if v in d.columns]; xx=np.arange(len(vars))
        for _,r in d.iterrows(): ax.plot(xx,[r[v] for v in vars],marker="o",label=str(r["profile"]))
        ax.set_xticks(xx,vars,rotation=45,ha="right"); ax.set_ylabel("Profile mean"); ax.set_title(f"Process profiles -- {x['status']}"); ax.legend(); ax.gp3_data=d; return ax
    Z=np.asarray(x["scaled_data"],dtype=float); d=x["assignment"]
    if Z.shape[1]<2: return _empty("Process-profile scatter","Fewer than two features",ax)
    for prof,g in d.groupby("profile",sort=False):
        idx=g.index.to_numpy(); ax.scatter(Z[idx,0],Z[idx,1],label=str(prof))
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2"); ax.set_title("Process-profile scatter"); ax.legend(); ax.gp3_data=d; return ax

def plot_eye_process_external_validity(x:Any,type:str="associations",ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_process_external_validity": raise EyeProcessValidationError("x must be eye_process_external_validity.")
    ax=_ax(ax)
    if type=="associations":
        d=x["associations"]; yy=np.arange(len(d)); ax.scatter(d["correlation"],yy); ax.set_yticks(yy,d["predictor"]); ax.axvline(0,linestyle="--"); ax.set_xlim(-1,1); ax.set_xlabel("Correlation with external criterion"); ax.set_title("Process external-validity associations")
    elif type=="incremental":
        d=pd.DataFrame([{"baseline_r2":x["baseline_model"]["r_squared"],"full_r2":x["full_model"]["r_squared"],"incremental_r2":x["incremental_r2"]}]); ax.bar(["baseline","full"],[d.baseline_r2.iloc[0],d.full_r2.iloc[0]]); ax.set_ylabel("R²"); ax.set_title(f"Incremental process validity; delta R2 = {d.incremental_r2.iloc[0]:.3f}")
    else:
        model=x["full_model"]; idx=model["training_index"]; y=x["data"].loc[idx,x["criterion"]].to_numpy(float); f=np.asarray(model["fitted"]); r=np.asarray(model["residuals"]); d=x["data"]
        if type=="observed_fitted": ax.scatter(f,y); ax.plot([np.nanmin(f),np.nanmax(f)],[np.nanmin(f),np.nanmax(f)],linestyle="--"); ax.set_xlabel("Fitted external criterion"); ax.set_ylabel("Observed external criterion")
        else: ax.scatter(f,r); ax.axhline(0,linestyle="--"); ax.set_xlabel("Fitted external criterion"); ax.set_ylabel("Residual")
    ax.gp3_data=d; return ax

def plot_eye_item_parameter_seed(x:Any,candidate_data:Any=None,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_item_parameter_seed": raise EyeProcessValidationError("x must be eye_item_parameter_seed.")
    ax=_ax(ax)
    if candidate_data is None:
        d=x["training_data"]; ax.scatter(d[x["difficulty"]],d[x["discrimination"]]); ax.set_xlabel("Calibrated difficulty"); ax.set_ylabel("Calibrated discrimination"); ax.set_title("Item-parameter seed training space")
    else:
        from .context_structure_08 import predict_item_parameter_priors
        d=predict_item_parameter_priors(x,candidate_data); ax.scatter(d["predicted_pre_pilot_difficulty"],d["predicted_pre_pilot_discrimination"]); ax.set_xlabel("Predicted pre-pilot difficulty"); ax.set_ylabel("Predicted pre-pilot discrimination"); ax.set_title("Candidate item screening map")
    ax.gp3_data=d; return ax

def plot_eye_candidate_item_bank_audit(x:Any,ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_candidate_item_bank_audit": raise EyeProcessValidationError("x must be eye_candidate_item_bank_audit.")
    d=x["table"]; ax=_ax(ax); normal=d.loc[~d["review_required"]]; review=d.loc[d["review_required"]]
    if len(normal): ax.scatter(normal["predicted_pre_pilot_difficulty"],normal["predicted_pre_pilot_discrimination"],marker="o",label="no review flag")
    if len(review): ax.scatter(review["predicted_pre_pilot_difficulty"],review["predicted_pre_pilot_discrimination"],marker="x",label="review")
    ax.set_xlabel("Predicted pre-pilot difficulty"); ax.set_ylabel("Predicted pre-pilot discrimination"); ax.set_title("Candidate item-bank screening audit"); ax.legend(); ax.gp3_data=d; return ax

def plot_eye_visual_context_irt(x:Any,type:str="context_registry",ax=None):
    if getattr(x,"eyeprocess_class",None)!="eye_visual_context_irt": raise EyeProcessValidationError("x must be eye_visual_context_irt.")
    if type!="context_registry": return _empty("Visual-context IRT","Exact mirt coefficients required for this plot",ax)
    d=x["registry"]["mapping"]; counts=d["visual_context_id"].value_counts(); ax=_ax(ax); ax.bar(counts.index.astype(str),counts.to_numpy()); ax.tick_params(axis="x",rotation=45); ax.set_ylabel("Items"); ax.set_title("Visual-context item registry"); ax.gp3_data=d; return ax

__all__=[n for n in globals() if n.startswith("plot_eye_")]
