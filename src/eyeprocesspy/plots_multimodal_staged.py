"""Matplotlib counterparts for the frozen eyeprocess 0.10/0.11 multimodal S3 plots."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .exceptions import EyeProcessValidationError
from .multimodal_staged import (
    audit_multimodal_measurement, multimodal_m4_state_diagnostics,
    audit_multimodal_m4_identifiability,
)


def _ax(ax=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires the 'plots' extra.") from exc
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _attach(ax, data):
    ax.gp3_data = data
    return ax


def _df(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame): return x.copy()
    if isinstance(x, dict) and isinstance(x.get("data"), pd.DataFrame): return x["data"].copy()
    if hasattr(x, "data") and isinstance(x.data, pd.DataFrame): return x.data.copy()
    return pd.DataFrame()


def _empty(title: str, message: str, ax=None):
    ax=_ax(ax); ax.set_title(title); ax.text(.5,.5,message,ha="center",va="center",transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]); return _attach(ax,pd.DataFrame())


def plot_eye_multimodal_measurement(x, type="availability", ax=None):
    a=audit_multimodal_measurement(x); d=a.channel_table.copy(); ax=_ax(ax)
    if type=="availability":
        ax.bar(d.channel, d.observed / d.n); ax.set_ylim(0,1); ax.set_ylabel("Observed fraction")
    elif type=="missingness":
        ax.bar(d.channel, d.missing_fraction); ax.set_ylim(0,1); ax.set_ylabel("Missing fraction")
    else: raise EyeProcessValidationError("type must be 'availability' or 'missingness'.")
    ax.set_title("Multimodal measurement channels"); return _attach(ax,d)


def plot_eye_multimodal_simulation(x, type="latent_correlation", ax=None):
    ax=_ax(ax)
    if type=="latent_correlation":
        persons=x.truth.get("persons") if isinstance(x.truth,dict) else None
        if isinstance(persons,pd.DataFrame):
            nums=persons.select_dtypes(include="number"); mat=nums.corr(); im=ax.imshow(mat,aspect="auto",vmin=-1,vmax=1); ax.set_xticks(range(len(mat)),mat.columns,rotation=45,ha="right"); ax.set_yticks(range(len(mat)),mat.index); ax.set_title("Simulated latent correlation"); return _attach(ax,mat.reset_index(names="latent"))
    d=_df(x)
    if type=="channels":
        cols=[c for c in ["response","rt","gaze_fixation_count","gaze","pupil_response","pupil"] if c in d]; means=[pd.to_numeric(d[c],errors="coerce").mean() for c in cols]; out=pd.DataFrame({"channel":cols,"mean":means}); ax.bar(out.channel,out["mean"]); ax.set_title("Simulated multimodal channels"); return _attach(ax,out)
    raise EyeProcessValidationError("Unknown multimodal simulation plot type.")


def plot_eye_process_information(x, ax=None):
    d=pd.DataFrame(x).copy(); ax=_ax(ax); ax.barh(np.arange(len(d)),d.value); ax.set_yticks(np.arange(len(d)),d.target); ax.axvline(0,linestyle="--"); ax.set_xlabel(str(d.metric.iloc[0]) if len(d) else "information"); ax.set_title("Incremental process information"); return _attach(ax,d)


def plot_eye_multimodal_validation(x, ax=None):
    d=x.checks.copy(); ax=_ax(ax); vals=d["pass"].astype(int); ax.barh(np.arange(len(d)),vals); ax.set_yticks(np.arange(len(d)),d["check"]); ax.set_xlim(0,1); ax.set_title("Multimodal validation checks"); return _attach(ax,d)


def plot_eye_multimodal_m2_simulation(x, type="person_latent", ax=None):
    ax=_ax(ax)
    if type=="person_latent":
        t=x.truth; d=pd.DataFrame({"theta":t["theta"],"tau":t["tau"],"omega":t["omega"]}); ax.scatter(d.theta,d.tau); ax.set(xlabel="theta",ylabel="tau",title="M2 person latent dimensions"); return _attach(ax,d)
    d=_df(x)
    if type=="channel_distributions":
        out=pd.DataFrame({"channel":["response","rt","gaze"],"mean":[d.response.mean(),d.rt.mean(),d.gaze.mean()]}); ax.bar(out.channel,out["mean"]); ax.set_title("M2 channel summaries"); return _attach(ax,out)
    if type=="missingness":
        out=d[["response","rt","gaze"]].isna().mean().rename_axis("channel").reset_index(name="missing_fraction"); ax.bar(out.channel,out.missing_fraction); ax.set_ylim(0,1); ax.set_title("M2 channel missingness"); return _attach(ax,out)
    raise EyeProcessValidationError("Unknown M2 simulation plot type.")


def plot_eye_multimodal_m2_fit(x, type="latent", ax=None):
    d=getattr(x,"summary",pd.DataFrame());
    if not isinstance(d,pd.DataFrame) or d.empty: return _empty("M2 fitted model","Posterior summary unavailable",ax)
    ax=_ax(ax); nums=d.select_dtypes(include="number").columns
    if len(nums)>=2: ax.scatter(d[nums[0]],d[nums[1]])
    ax.set_title(f"M2 fit: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m2_ppc(x, ax=None):
    d=x.summary.copy(); ax=_ax(ax); ax.bar(d.channel,d.observed_mean); ax.set_title("M2 posterior-predictive/observed summaries"); return _attach(ax,d)


def plot_eye_multimodal_m2_information(x, ax=None):
    d=x.summary.copy(); ax=_ax(ax); y="observed_fraction" if "observed_fraction" in d else d.select_dtypes(include="number").columns[-1]; ax.bar(d.channel,d[y]); ax.set_title("M2 process-channel information"); return _attach(ax,d)


def plot_eye_multimodal_m2_validation(x, ax=None):
    d=x.checks.copy(); ax=_ax(ax); ax.barh(np.arange(len(d)),d["pass"].astype(int)); ax.set_yticks(np.arange(len(d)),d.criterion); ax.set_xlim(0,1); ax.set_title("M2 validation"); return _attach(ax,d)


def plot_eye_multimodal_m2_recovery(x, ax=None):
    d=x.design.copy(); ax=_ax(ax); ax.scatter(d.replicate,d.seed); ax.set(xlabel="Replicate",ylabel="Seed",title="M2 recovery design"); return _attach(ax,d)


def plot_eye_multimodal_m2_negative_controls(x, ax=None):
    d=x.diagnostics.copy(); ax=_ax(ax); pivot=d.pivot(index="dataset",columns="channel",values="mean"); pivot.plot(kind="bar",ax=ax); ax.set_title("M2 negative-control channel means"); ax.set_ylabel("Mean"); return _attach(ax,d)


def plot_eye_multimodal_m3_simulation(x, type="pupil_nuisance", ax=None):
    d=_df(x); ax=_ax(ax)
    if type=="pupil_nuisance":
        out=d[["pupil","pupil_nuisance_effect"]].dropna(); ax.scatter(out.pupil_nuisance_effect,out.pupil,alpha=.5); ax.set(xlabel="Pupil nuisance effect",ylabel="Pupil",title="M3 pupil nuisance structure"); return _attach(ax,out)
    if type=="missingness":
        cols=["response","rt","gaze","pupil"]; out=d[cols].isna().mean().rename_axis("channel").reset_index(name="missing_fraction"); ax.bar(out.channel,out.missing_fraction); ax.set_ylim(0,1); ax.set_title("M3 channel missingness"); return _attach(ax,out)
    raise EyeProcessValidationError("Unknown M3 simulation plot type.")


def plot_eye_multimodal_m3_fit(x, type="latent", ax=None):
    d=getattr(x,"summary",pd.DataFrame());
    if not isinstance(d,pd.DataFrame) or d.empty: return _empty("M3 fitted model","Posterior summary unavailable",ax)
    ax=_ax(ax); nums=d.select_dtypes(include="number").columns
    if len(nums)>=2: ax.scatter(d[nums[0]],d[nums[1]])
    ax.set_title(f"M3 fit: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m3_ppc(x, type="item_channel", ax=None):
    d=x.summary.copy(); ax=_ax(ax); ax.bar(d.channel,d.observed_mean); ax.set_title(f"M3 PPC: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m3_information(x, type="incremental_pupil", ax=None):
    d=pd.DataFrame([{"pupil_observed_fraction":x.pupil_observed_fraction,"pupil_cost":x.pupil_cost,"decisive_z":x.decisive_z}]); ax=_ax(ax); ax.bar(["observed fraction","cost"],[d.pupil_observed_fraction.iloc[0],d.pupil_cost.iloc[0]]); ax.set_title(f"M3 process information: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m3_validation(x, type="checks", ax=None):
    d=x.checks.copy(); ax=_ax(ax); ax.barh(np.arange(len(d)),d["pass"].astype(int)); ax.set_yticks(np.arange(len(d)),d.criterion); ax.set_xlim(0,1); ax.set_title(f"M3 validation: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m3_recovery(x, type="stress", ax=None):
    d=x.design.copy(); ax=_ax(ax); tab=d.groupby(["scenario","missingness"]).size().reset_index(name="replicates"); xx=np.arange(len(tab)); ax.bar(xx,tab.replicates); ax.set_xticks(xx,[f"{a}/{b}" for a,b in zip(tab.scenario,tab.missingness)],rotation=45,ha="right"); ax.set_title(f"M3 recovery: {type}"); return _attach(ax,tab)


def plot_eye_multimodal_m3_negative_controls(x, type="pupil_alignment", ax=None):
    rows=[]
    for name,d in x.datasets.items(): rows.append({"dataset":name,"pupil_mean":pd.to_numeric(d.pupil,errors="coerce").mean(),"pupil_sd":pd.to_numeric(d.pupil,errors="coerce").std()})
    out=pd.DataFrame(rows); ax=_ax(ax); ax.scatter(out.pupil_mean,out.pupil_sd); [ax.text(r.pupil_mean,r.pupil_sd,r.dataset) for r in out.itertuples()]; ax.set(xlabel="Pupil mean",ylabel="Pupil SD",title=f"M3 negative controls: {type}"); return _attach(ax,out)


def plot_eye_multimodal_m3_identifiability(x, type="checks", ax=None):
    ax=_ax(ax)
    if type=="missingness":
        d=x.missing_fraction.rename_axis("channel").reset_index(name="missing_fraction"); ax.bar(d.channel,d.missing_fraction); ax.set_ylim(0,1)
    else:
        d=x.checks.copy(); ax.barh(np.arange(len(d)),d["pass"].astype(int)); ax.set_yticks(np.arange(len(d)),d.criterion); ax.set_xlim(0,1)
    ax.set_title(f"M3 identifiability: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m4_simulation(x, type="state_sequence", person=None, ax=None):
    d=_df(x).copy(); st=np.asarray(x.truth["state"]); d["state"]=st; ax=_ax(ax)
    if person is not None: d=d[d.person_id.astype(str)==str(person)]
    if type=="state_sequence":
        z=d.iloc[:min(len(d),250)]; ax.step(np.arange(len(z)),z.state,where="mid"); ax.set_ylabel("State")
    elif type=="channel_profile":
        z=d.groupby("state")[["rt","gaze","pupil"]].mean().reset_index(); z.set_index("state").plot(kind="bar",ax=ax); d=z
    elif type=="missingness":
        z=d[["rt","gaze","pupil"]].isna().mean().rename_axis("channel").reset_index(name="missing_fraction"); ax.bar(z.channel,z.missing_fraction); d=z
    else: raise EyeProcessValidationError("Unknown M4 simulation plot type.")
    ax.set_title(f"M4 simulation: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m4_fit(x, type="state_probability", person=None, ax=None):
    try: states=multimodal_m4_state_diagnostics(x)
    except Exception: return _empty("M4 fit","Posterior state probabilities unavailable",ax)
    return plot_eye_multimodal_m4_states(states,type="probability" if type=="state_probability" else type,ax=ax)


def plot_eye_multimodal_m4_states(x, type="probability", ax=None):
    ax=_ax(ax)
    if type=="probability":
        d=x.probability.copy(); p=[c for c in d if c.startswith("state_") and c.endswith("_probability")]; z=d.iloc[:min(len(d),250)]; [ax.plot(np.arange(len(z)),z[c],label=c) for c in p]; ax.legend(); ax.set_ylabel("Probability")
    elif type=="occupancy":
        d=x.occupancy.copy(); y="mean_probability" if "mean_probability" in d else "occupancy"; ax.bar(d.state,d[y]); ax.set_ylabel(y)
    elif type=="entropy":
        d=x.probability[["posterior_entropy"]].copy(); ax.hist(d.posterior_entropy,bins=20)
    else:
        d=x.probability.copy(); p=[c for c in d if c.startswith("state_") and c.endswith("_probability")]; ax.plot(np.arange(len(d)),d[p].max(axis=1)); ax.set_ylabel("Maximum state probability")
    ax.set_title(f"M4 state diagnostics: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m4_identifiability(x, type="checks", ax=None):
    d=x.checks.copy(); ax=_ax(ax); score=d.status.map({"PASS":1,"PASS_WITH_CAUTION":.75,"REVIEW":.5,"NOT_EVALUATED":.25,"FAIL":0}).fillna(.25); ax.barh(np.arange(len(d)),score); ax.set_yticks(np.arange(len(d)),d.criterion); ax.set_xlim(0,1); ax.set_title(f"M4 identifiability: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m4_ppc(x, type="measurement", ax=None):
    d=x.measurement.copy(); ax=_ax(ax); ax.bar(d.channel,d["mean"]); ax.set_title(f"M4 PPC: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m4_information(x, type="occupancy", ax=None):
    d=x.occupancy.copy(); ax=_ax(ax)
    if not d.empty:
        y="mean_probability" if "mean_probability" in d else "occupancy"; ax.bar(d.state,d[y])
    ax.set_title(f"M4 process information: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m4_negative_controls(x, type="design", ax=None):
    d=pd.DataFrame({"control":x.controls,"index":np.arange(1,len(x.controls)+1)}); ax=_ax(ax); ax.barh(np.arange(len(d)),d["index"]); ax.set_yticks(np.arange(len(d)),d.control); ax.set_title(f"M4 negative controls: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m4_sensitivity(x, type="state_count", ax=None):
    d=x.design.copy(); ax=_ax(ax); ax.scatter(d.n_states,np.zeros(len(d))); ax.set_xticks(d.n_states); ax.set_title(f"M4 sensitivity: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m4_recovery(x, type="parameter_rmse", ax=None):
    d=x.design.copy(); ax=_ax(ax); ax.barh(np.arange(len(d)),np.ones(len(d))); ax.set_yticks(np.arange(len(d)),d.scenario); ax.set_title(f"M4 recovery design: {type}"); return _attach(ax,d)


def plot_eye_multimodal_m4_validation(x, type="domains", ax=None):
    d=x.checks.copy(); ax=_ax(ax); counts=d.status.value_counts().rename_axis("status").reset_index(name="count"); ax.bar(counts.status,counts["count"]); ax.set_title(f"M4 validation: {type}"); return _attach(ax,d)


__all__=[name for name in globals() if name.startswith("plot_eye_")]