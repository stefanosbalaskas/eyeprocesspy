"""Plot counterparts for frozen 0.8 operational validation and decision features."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .exceptions import EyeProcessValidationError
from .operational_validation_08 import validation_bundle_manifest

__all__ = ["plot_eye_streaming_score", "plot_eye_validation_bundle", "plot_eye_preaction_process_features",
           "plot_eye_decision_process_proxy", "plot_process_feature_stability"]

def _ax(ax=None):
    import matplotlib.pyplot as plt
    return plt.subplots()[1] if ax is None else ax

def plot_eye_streaming_score(x: Any, **kwargs: Any):
    if getattr(x,"eyeprocess_class",None)!="eye_streaming_score": raise EyeProcessValidationError("x must be eye_streaming_score.")
    ax=_ax(kwargs.pop("ax",None));d=x.history.copy();good=np.isfinite(pd.to_numeric(d.theta,errors="coerce")) if len(d) else np.array([],bool)
    if good.any():
        ax.plot(d.loc[good,"step"],d.loc[good,"theta"],marker="o")
        se=pd.to_numeric(d.loc[good,"theta_se"],errors="coerce").to_numpy(float);th=pd.to_numeric(d.loc[good,"theta"],errors="coerce").to_numpy(float);st=d.loc[good,"step"].to_numpy(float);g=np.isfinite(se)
        if g.any():ax.vlines(st[g],th[g]-se[g],th[g]+se[g])
    else: ax.text(.5,.5,"No finite score estimates",ha="center",va="center",transform=ax.transAxes)
    ax.set(xlabel="Observed-response step",ylabel="Theta estimate",title=f"Streaming person score -- {x.method}");ax.eyeprocess_plot_data=d;return ax

def plot_eye_validation_bundle(x: Any, **kwargs: Any):
    if getattr(x,"eyeprocess_class",None)!="eye_validation_bundle": raise EyeProcessValidationError("x must be eye_validation_bundle.")
    ax=_ax(kwargs.pop("ax",None));m=validation_bundle_manifest(x);score=m.status.map({"missing":0,"empty":.25,"error":0,"available":1}).fillna(0);ax.bar(m.slot.astype(str),score);ax.tick_params(axis="x",rotation=90);ax.set(ylim=(0,1),ylabel="Evidence availability",title=f"Validation bundle: {x.model_name}");ax.eyeprocess_plot_data=m;return ax

def plot_eye_preaction_process_features(x: Any, feature: str="pupil_mean", **kwargs: Any):
    if getattr(x,"eyeprocess_class",None)!="eye_preaction_process_features": raise EyeProcessValidationError("x must be eye_preaction_process_features.")
    ax=_ax(kwargs.pop("ax",None));d=x.data.copy()
    if d.empty or feature not in d: a=pd.DataFrame(columns=["window_ms","value"]);ax.text(.5,.5,"No feature rows",ha="center",transform=ax.transAxes)
    else:a=d.groupby("pre_window_ms",as_index=False)[feature].mean().rename(columns={"pre_window_ms":"window_ms",feature:"value"});ax.plot(a.window_ms,a.value,marker="o")
    ax.set(xlabel="Look-back window (ms)",ylabel=feature,title="Pre-action process feature by horizon");ax.eyeprocess_plot_data=a;return ax

def plot_eye_decision_process_proxy(x: Any, **kwargs: Any):
    if getattr(x,"eyeprocess_class",None)!="eye_decision_process_proxy": raise EyeProcessValidationError("x must be eye_decision_process_proxy.")
    ax=_ax(kwargs.pop("ax",None));d=x.features.copy();vars=[c for c in d.select_dtypes(include=np.number) if c not in x.by];vals=pd.DataFrame({"feature":vars,"value":[pd.to_numeric(d[v],errors="coerce").mean() for v in vars]});ax.bar(vals.feature,vals.value);ax.tick_params(axis="x",rotation=60);ax.set(ylabel="Mean proxy value",title="aDDM/GLAM-inspired process proxies");ax.eyeprocess_plot_data=d;return ax

def plot_process_feature_stability(data: Any, feature: str="feature", stability: str="selection_rate", top_n: int=20, **kwargs: Any):
    d=pd.DataFrame(data).copy()
    if feature not in d or stability not in d: raise EyeProcessValidationError("data is missing feature/stability columns.")
    q=d.assign(_v=pd.to_numeric(d[stability],errors="coerce")).sort_values("_v",ascending=False).head(int(top_n));ax=_ax(kwargs.pop("ax",None));ax.barh(q[feature].astype(str).iloc[::-1],q._v.iloc[::-1]);ax.set(xlabel="Stability / selection rate",title="Process-feature stability");ax.eyeprocess_plot_data=q.drop(columns="_v");return ax
