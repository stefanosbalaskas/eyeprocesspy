"""Matplotlib counterparts for frozen process-IRT 0.7 S3 plot methods."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .advanced_process_irt_07 import (
    process_state_occupancy, process_state_transition_summary, process_residual_map,
    compare_parametric_nonparametric_irf,
)
from .process_irt_07 import distractor_process_map, facet_effects

__all__ = [
    "plot_eye_joint_gaze_rt_irt", "plot_eye_joint_graded_rt_process_irt",
    "plot_eye_nominal_gaze_irt", "plot_eye_omission_survival_irt",
    "plot_eye_manyfacet_process_irt", "plot_eye_irt_changepoints",
    "plot_eye_process_hmm_irt", "plot_eye_latent_space_irt", "plot_eye_process_person_fit",
    "plot_eye_irt_equating", "plot_eye_gpirt", "plot_eye_process_cat_simulation",
    "plot_eye_irt_recovery_summary", "plot_eye_irt_sbc", "plot_eye_sbc_audit",
    "plot_eye_irt_ppc", "plot_eye_incremental_information_audit",
    "plot_eye_process_negative_control",
]


def _ax(ax=None):
    import matplotlib.pyplot as plt
    return ax if ax is not None else plt.subplots()[1]


def _attach(ax, data):
    ax.gp3_data = data
    return ax


def plot_eye_joint_gaze_rt_irt(x, type="latent", ax=None):
    ax=_ax(ax); z=getattr(x,"person_scores",None)
    if type=="latent" and isinstance(z,pd.DataFrame):
        nums=z.select_dtypes(include="number").columns
        if len(nums)>=2:
            ax.scatter(z[nums[0]],z[nums[1]]); ax.set(xlabel=nums[0],ylabel=nums[1],title="Joint process-IRT person dimensions"); return _attach(ax,z.copy())
    ax.set_title("Joint gaze-RT-response IRT fit"); return _attach(ax,pd.DataFrame())


def plot_eye_joint_graded_rt_process_irt(x, ax=None):
    ax=_ax(ax); z=getattr(x,"person_scores",None)
    if isinstance(z,pd.DataFrame):
        nums=z.select_dtypes(include="number").columns
        if len(nums)>=2: ax.scatter(z[nums[0]],z[nums[1]]); ax.set(xlabel=nums[0],ylabel=nums[1],title="Graded-response process dimensions"); return _attach(ax,z.copy())
    ax.set_title("Graded-response + RT/process IRT"); return _attach(ax,pd.DataFrame())


def plot_eye_nominal_gaze_irt(x, type="distractor_map", ax=None):
    ax=_ax(ax)
    z=distractor_process_map(x)
    if not z.empty:
        ax.bar(np.arange(len(z)),z["coefficient"].to_numpy(float)); ax.axhline(0,ls="--"); ax.set_title("Nominal gaze-IRT coefficients")
    return _attach(ax,z)


def plot_eye_omission_survival_irt(x, type="missingness", ax=None):
    ax=_ax(ax); z=getattr(x,"missingness",getattr(x,"classified_missingness",None))
    if isinstance(z,pd.DataFrame) and "missingness_class" in z:
        tab=z.missingness_class.astype(str).value_counts(); ax.bar(tab.index,tab.values); ax.tick_params(axis='x',rotation=45); d=tab.rename_axis("missingness_class").reset_index(name="count")
    else: d=pd.DataFrame()
    ax.set_title("Process-informed missingness"); return _attach(ax,d)


def plot_eye_manyfacet_process_irt(x, facet=None, ax=None):
    ax=_ax(ax); facets=list(getattr(x,"facets",{})); facet=facet or (facets[0] if facets else None)
    try:
        fx=facet_effects(x,"process"); d=fx.effects if hasattr(fx,"effects") else pd.DataFrame()
    except Exception: d=pd.DataFrame()
    if isinstance(d,pd.DataFrame) and not d.empty:
        num=d.select_dtypes(include="number").columns[0]; ax.barh(np.arange(len(d)),d[num]); ax.set_title("Many-facet process effects")
    else: ax.set_title("Many-facet process IRT")
    return _attach(ax,d)


def plot_eye_irt_changepoints(x, person=None, ax=None):
    ax=_ax(ax); d=getattr(x,"results",getattr(x,"data",x)); d=pd.DataFrame(d).copy()
    if person is not None and "person" in d: d=d[d.person==person]
    pos=next((c for c in ("changepoint","position","index") if c in d),None); score=next((c for c in ("score","sic","evidence","delta_sic") if c in d),None)
    if pos and score and len(d): ax.plot(d[pos],d[score],marker='o')
    elif pos and len(d): ax.scatter(d[pos],np.arange(len(d)))
    ax.set_title("Process changepoint evidence"); return _attach(ax,d)


def plot_eye_process_hmm_irt(x, type="occupancy", ax=None):
    ax=_ax(ax)
    if type=="transition":
        d=process_state_transition_summary(x); mat=d.pivot(index="from_state",columns="to_state",values="probability"); ax.imshow(mat.to_numpy(),aspect="auto"); ax.set_title("Process-state transition matrix"); return _attach(ax,d)
    d=process_state_occupancy(x); cols=[c for c in d if c.endswith("_occupancy")]; vals=d[cols].mean() if cols else pd.Series(dtype=float); ax.bar(range(len(vals)),vals.to_numpy()); ax.set_title("Process-state occupancy"); return _attach(ax,d)


def plot_eye_latent_space_irt(x, ax=None):
    ax=_ax(ax); d=process_residual_map(x); nums=d.select_dtypes(include="number").columns
    if len(nums)>=2: ax.scatter(d[nums[0]],d[nums[1]])
    ax.set_title("Person-item latent space"); return _attach(ax,d)


def plot_eye_process_person_fit(x, top=25, ax=None):
    ax=_ax(ax); d=pd.DataFrame(x).copy(); score=next((c for c in ("joint_discrepancy","process_discrepancy","combined_rms","score","rms") if c in d),None)
    if score:
        z=d.nlargest(min(int(top),len(d)),score); ax.barh(np.arange(len(z)),z[score]); d=z
    ax.set_title("Largest model-process discrepancies"); return _attach(ax,d)


def plot_eye_irt_equating(x, theta=None, ax=None):
    ax=_ax(ax); th=np.linspace(-4,4,201) if theta is None else np.asarray(theta,float); linked=float(getattr(x,"A",1))*th+float(getattr(x,"B",0)); d=pd.DataFrame({"theta":th,"linked":linked}); ax.plot(th,linked); ax.plot(th,th,ls="--"); ax.set(xlabel="New-form scale",ylabel="Reference scale",title="IRT scale linking"); return _attach(ax,d)


def plot_eye_gpirt(x, item=None, ax=None):
    ax=_ax(ax); d=compare_parametric_nonparametric_irf(x.response_matrix,gpirt_object=x); items=list(pd.unique(d.item)); chosen=items[0] if item is None else (items[int(item)-1] if isinstance(item,(int,np.integer)) else str(item)); z=d[d.item==chosen]; ax.plot(z.theta,z.parametric,label="parametric"); ax.plot(z.theta,z.flexible,label="flexible"); ax.legend(); ax.set_title("Item-response shape audit"); return _attach(ax,z)


def plot_eye_process_cat_simulation(x, ax=None):
    ax=_ax(ax); d=pd.DataFrame(x).copy(); y=next((c for c in ("information","utility","selection_utility","cumulative_information") if c in d),None)
    if y: ax.plot(d["step"],d[y],marker='o')
    ax.set_title("Process-aware adaptive information"); return _attach(ax,d)


def plot_eye_irt_recovery_summary(x, metric="rmse", ax=None):
    ax=_ax(ax); d=pd.DataFrame(x).copy(); ax.barh(np.arange(len(d)),d[metric]); ax.set_title("IRT validation recovery"); return _attach(ax,d)


def plot_eye_irt_sbc(x, parameter=None, breaks=10, ax=None):
    ax=_ax(ax); d=x.ranks.copy(); parameter=parameter or d.parameter.iloc[0]; z=d[d.parameter==parameter]; ax.hist(z.normalized_rank,bins=int(breaks),range=(0,1)); ax.set_title(f"SBC: {parameter}"); return _attach(ax,z)


def plot_eye_sbc_audit(x, ax=None):
    ax=_ax(ax); d=pd.DataFrame(x).copy(); ax.scatter(d.mean_rank,d.rank_variance); ax.axvline(.5,ls="--"); ax.axhline(1/12,ls="--"); ax.set_title("SBC uniformity screen"); return _attach(ax,d)


def plot_eye_irt_ppc(x, ax=None):
    ax=_ax(ax); d=pd.DataFrame(x).copy(); ax.barh(np.arange(len(d)),d.p_two_sided); ax.axvline(.01,ls="--"); ax.axvline(.05,ls=":"); ax.set_title("Posterior predictive discrepancies"); return _attach(ax,d)


def plot_eye_incremental_information_audit(x, ax=None):
    ax=_ax(ax); d=pd.DataFrame(x).copy(); ax.bar(d.fold.astype(str),d.improvement); ax.axhline(0,ls="--"); ax.set_title("Incremental information from process channel"); return _attach(ax,d)


def plot_eye_process_negative_control(x, ax=None):
    ax=_ax(ax); arr=np.asarray(x.null,float); ax.hist(arr,bins=min(20,max(5,len(arr)//5))); ax.axvline(float(x.observed),ls="--"); ax.set_title("Process-channel negative control"); return _attach(ax,pd.DataFrame({"null":arr}))
