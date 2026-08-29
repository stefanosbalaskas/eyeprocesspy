"""Matplotlib counterparts for functional-pupil S3 plots."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def plot_eye_functional_pupil_irt(x: Any, type: str = "trajectories", ax=None):
    import matplotlib.pyplot as plt
    if ax is None: _,ax=plt.subplots()
    if getattr(x,"legacy",False):
        data=x.data["features"] if isinstance(x.data,dict) else pd.DataFrame(); d=data.loc[data.feature_name.astype(str).isin(x.feature_names)] if not data.empty else data
        if not d.empty:
            pivot=d.pivot_table(index=[c for c in ["participant_id","trial_id","item_id"] if c in d],columns="feature_name",values="value",aggfunc="mean")
            for row in pivot.to_numpy(float): ax.plot(np.arange(1,len(row)+1),row,alpha=.25)
        ax.set_title("Functional pupil coefficients"); ax.set_xlabel("Basis coefficient"); ax.set_ylabel("Value"); ax.gp3_data=d; return ax
    if type=="trajectories":
        d=x.data.data.copy()
        for _,z in list(d.groupby("trial_index",sort=False))[:30]: ax.plot(z[".time"],z.pupil_adjusted,alpha=.15)
        ax.set_xlabel("Aligned time (ms)"); ax.set_ylabel("Adjusted pupil"); ax.set_title("Functional pupil trajectories"); pdata=d
    elif type=="coefficients":
        pdata=x.trial_coefficients[x.feature_names].copy()
        for row in pdata.to_numpy(float): ax.plot(np.arange(1,len(row)+1),row,alpha=.25)
        ax.set_xlabel("Basis coefficient"); ax.set_ylabel("Coefficient"); ax.set_title("Functional pupil coefficients")
    else:
        if getattr(x.model,"eyeprocess_class",None)!="eye_functional_pupil_stan": raise ValueError("Posterior recovery plots require the Stan engine.")
        s=x.model.summary.copy(); names=s.index.astype(str) if "variable" not in s else s.variable.astype(str); mask=names.str.contains("theta_loading|response_loading",regex=True); pdata=s.loc[mask].copy(); means=pdata["Mean"] if "Mean" in pdata else pdata.get("mean"); ax.plot(means,np.arange(len(pdata)),"o"); ax.set_xlabel("Posterior mean"); ax.set_title("Shared pupil effects")
    ax.gp3_data=pdata; return ax


def plot_eye_functional_pupil_diagnostics(x: Any, ax=None):
    import matplotlib.pyplot as plt
    if ax is None: _,ax=plt.subplots()
    d=x.residual_acf.copy(); vals=pd.to_numeric(d.get("lag1"),errors="coerce").dropna()
    if len(vals): ax.hist(vals)
    ax.set_xlabel("Within-trial lag-1 correlation"); ax.set_title("Pupil residual autocorrelation"); ax.gp3_data=d; return ax


def plot_eye_functional_pupil_sensitivity(x: Any, parameter: Any = None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None: _,ax=plt.subplots()
    d=x.results.copy()
    if parameter is not None:
        p=[parameter] if isinstance(parameter,str) else list(parameter); d=d.loc[d.parameter.isin(p)]
    d=d.loc[np.isfinite(pd.to_numeric(d.estimate,errors="coerce"))].copy()
    if len(d): ax.plot(d.specification,d.estimate,"o-")
    ax.set_xlabel("Specification"); ax.set_ylabel("Estimate"); ax.set_title("Pupil preprocessing sensitivity"); ax.gp3_data=d; return ax

__all__=["plot_eye_functional_pupil_irt","plot_eye_functional_pupil_diagnostics","plot_eye_functional_pupil_sensitivity"]
