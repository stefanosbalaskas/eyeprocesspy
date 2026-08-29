"""Matplotlib counterparts for 0.9 process reliability/calibration quality plots."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import pandas as pd
from .exceptions import EyeProcessValidationError

__all__=["plot_eye_process_reliability_profile","plot_eye_calibration_error_model","plot_eye_calibration_drift_profile","plot_eye_data_quality_profile","plot_eye_probabilistic_aoi_assignment","plot_eye_sampling_irregularity_audit"]

def _ax(ax=None):
    import matplotlib.pyplot as plt
    return plt.subplots()[1] if ax is None else ax

def plot_eye_process_reliability_profile(x: Any, y: Any=None, type: str="bland_altman", ax: Any=None):
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_process_reliability_profile": raise EyeProcessValidationError("x must be an eye_process_reliability_profile.")
    ax=_ax(ax)
    if type=="bland_altman":
        ba=x.get("bland_altman")
        if ba is None: ax.text(.5,.5,"At least two sessions are required",ha="center"); ax.eyeprocess_plot_data=pd.DataFrame(); return ax
        d=ba["pairs"].copy(); sm=ba["summary"].iloc[0]; ax.scatter(d.pair_mean,d.difference); [ax.axhline(float(sm[k]),linestyle="--" if k!='bias' else "-") for k in ["bias","loa_lower","loa_upper"]]; ax.set(xlabel="Pair mean",ylabel="Difference",title="Bland-Altman process reliability"); ax.eyeprocess_plot_data=d; return ax
    if type!="summary": raise EyeProcessValidationError("type must be 'bland_altman' or 'summary'.")
    vals=pd.DataFrame({"metric":["ICC_A1","BA_bias"],"value":[float(x["icc"].icc_a1.iloc[0]),float(x["bland_altman"]["summary"].bias.iloc[0]) if x.get("bland_altman") is not None else np.nan]}); ax.bar(vals.metric,vals.value); ax.set_title("Process reliability summary"); ax.eyeprocess_plot_data=vals; return ax

def plot_eye_calibration_error_model(x: Any, y: Any=None, ax: Any=None):
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_calibration_error_model": raise EyeProcessValidationError("x must be an eye_calibration_error_model.")
    ax=_ax(ax); E=np.asarray(x["errors"],float); d=pd.DataFrame(E,columns=["horizontal_error","vertical_error"]); ax.scatter(d.horizontal_error,d.vertical_error); ax.axhline(0,linestyle="--"); ax.axvline(0,linestyle="--"); ax.scatter([x["mean_error"][0]],[x["mean_error"][1]],marker="x"); ax.set(xlabel="Horizontal error",ylabel="Vertical error",title="Empirical calibration-error cloud"); ax.eyeprocess_plot_data=d; return ax

def plot_eye_calibration_drift_profile(x: Any, y: Any=None, ax: Any=None):
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_calibration_drift_profile": raise EyeProcessValidationError("x must be an eye_calibration_drift_profile.")
    ax=_ax(ax); d=x["table"].copy(); metric=next((m for m in ["delta_from_first","mean_radial_error","bias_x"] if m in d),None); ax.plot(np.arange(1,len(d)+1),d[metric],marker="o"); ax.set(xlabel="Session/group order",ylabel=metric,title="Calibration drift profile"); ax.eyeprocess_plot_data=d; return ax

def plot_eye_data_quality_profile(x: Any, y: Any=None, metric: str|None=None, ax: Any=None):
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_data_quality_profile": raise EyeProcessValidationError("x must be an eye_data_quality_profile.")
    ax=_ax(ax); d=x["table"].copy(); metric=metric or next((m for m in ["valid_fraction","effective_hz","rms_s2s","missing_fraction"] if m in d),None)
    if metric is None or metric not in d: ax.text(.5,.5,"No requested quality metric",ha="center"); ax.eyeprocess_plot_data=d; return ax
    ax.bar(np.arange(1,len(d)+1),pd.to_numeric(d[metric],errors="coerce")); ax.set(ylabel=metric,title="Eye-tracking data quality"); ax.eyeprocess_plot_data=d; return ax

def plot_eye_probabilistic_aoi_assignment(x: Any, y: Any=None, ax: Any=None):
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_probabilistic_aoi_assignment": raise EyeProcessValidationError("x must be an eye_probabilistic_aoi_assignment.")
    ax=_ax(ax); d=x["probabilities"].copy()
    if d.empty:
        ax.text(.5,.5,"No probabilistic AOI assignments",ha="center"); ax.eyeprocess_plot_data=d; ax.eyeprocess_plot_matrix=np.empty((0,0)); return ax
    aois=list(pd.unique(d["aoi"])); samples=sorted(pd.unique(d["sample_id"]))
    M=np.zeros((len(aois),len(samples)),dtype=float)
    aidx={v:i for i,v in enumerate(aois)}; sidx={v:i for i,v in enumerate(samples)}
    for row in d.itertuples(index=False):
        M[aidx[row.aoi],sidx[row.sample_id]]=float(row.probability) if pd.notna(row.probability) else np.nan
    ax.imshow(M,aspect="auto",origin="lower")
    ax.set(xlabel="Sample",ylabel="AOI",title="Probabilistic AOI membership")
    ax.set_xticks(np.arange(len(samples)),labels=[str(v) for v in samples])
    ax.set_yticks(np.arange(len(aois)),labels=[str(v) for v in aois])
    ax.eyeprocess_plot_data=d; ax.eyeprocess_plot_matrix=M; return ax

def plot_eye_sampling_irregularity_audit(x: Any, y: Any=None, ax: Any=None):
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_sampling_irregularity_audit": raise EyeProcessValidationError("x must be an eye_sampling_irregularity_audit.")
    ax=_ax(ax); d=x["table"].copy(); ax.bar(np.arange(1,len(d)+1),d.interval_cv); ax.axhline(float(x["cv_threshold"]),linestyle="--"); ax.set(ylabel="Interval CV",title="Sampling irregularity"); ax.eyeprocess_plot_data=d; return ax
