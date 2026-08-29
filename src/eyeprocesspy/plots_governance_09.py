"""Matplotlib counterparts for the eyeprocess 0.9 validation/governance plots."""
from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .governance_09 import (
    expand_process_validation_design,
    validation_failure_profile,
    validation_coverage_table,
    eye_pipeline_graph,
    sensitivity_decision_leverage,
    specification_curve_data,
    decision_manifest_table,
)


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib>=3.9 is required for plotting; install eyeprocesspy[plots].") from exc
    return plt


def _axis(ax: Any = None):
    plt = _plt()
    return plt.subplots()[1] if ax is None else ax


def _attach(ax: Any, data: Any):
    ax.eyeprocess_plot_data = data
    return ax


def _empty(ax: Any, message: str = "No plottable data", title: str | None = None):
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes)
    return _attach(ax, pd.DataFrame())


def plot_eye_process_validation_design(x: Any, y: Any = None, ax: Any = None, **kwargs: Any):
    ax = _axis(ax)
    g = expand_process_validation_design(x)
    names = ["n_persons", "n_trials", "missingness", "sampling_rate_hz", "aoi_error", "calibration_error", "pupil_dropout", "heterogeneity", "model_misspecification"]
    counts = pd.DataFrame({"dimension": names, "levels": [g[c].nunique(dropna=False) for c in names]})
    ax.bar(np.arange(len(counts)), counts.levels.to_numpy(float), **kwargs)
    ax.set_xticks(np.arange(len(counts)), counts.dimension, rotation=90)
    ax.set_ylabel("Levels"); ax.set_title(f"Validation design: {len(g)} conditions")
    return _attach(ax, counts)


def plot_eye_process_validation_result(x: Any, y: Any = None, type: str = "recovery", parameter: Any = None, ax: Any = None, **kwargs: Any):
    ax = _axis(ax)
    if type not in {"recovery", "bias", "coverage", "failure"}:
        raise EyeProcessValidationError("type must be recovery, bias, coverage, or failure.")
    if type == "failure":
        d = validation_failure_profile(x)
        if not len(d): return _empty(ax, "No recorded failures", "Validation failure profile")
        ax.bar(d.stage.astype(str), pd.to_numeric(d.failure_rate, errors="coerce"), **kwargs); ax.set_ylabel("Failure rate"); ax.set_title("Validation failures")
        return _attach(ax, d)
    d = x.estimates.copy()
    if not len(d): return _empty(ax)
    if parameter is not None:
        vals = [parameter] if isinstance(parameter, str) else list(parameter); d = d[d.parameter.isin(vals)]
    if not len(d): return _empty(ax, "Requested parameter not found")
    truth = pd.to_numeric(d.truth, errors="coerce").to_numpy(float); est = pd.to_numeric(d.estimate, errors="coerce").to_numpy(float)
    if type == "recovery":
        ax.scatter(truth, est, **kwargs); finite = np.r_[truth[np.isfinite(truth)], est[np.isfinite(est)]]
        if finite.size:
            lo, hi = finite.min(), finite.max(); ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Truth"); ax.set_ylabel("Estimate"); ax.set_title("Parameter recovery")
    elif type == "bias":
        b = est - truth; groups = []
        for nm, z in d.assign(_bias=b).groupby("parameter", sort=False): groups.append((nm, pd.to_numeric(z._bias, errors="coerce").dropna().to_numpy(float)))
        ax.boxplot([z for _, z in groups], tick_labels=[n for n, _ in groups]); ax.axhline(0, linestyle="--"); ax.set_xlabel("Parameter"); ax.set_ylabel("Estimate - truth"); ax.set_title("Validation bias")
    else:
        s = validation_coverage_table(x); cov = pd.to_numeric(s.coverage, errors="coerce").to_numpy(float)
        if not np.isfinite(cov).any(): return _empty(ax, "No finite interval coverage available")
        ax.plot(np.arange(1, len(s)+1), cov, marker="o", **kwargs); nom = float(pd.to_numeric(s.nominal, errors="coerce").dropna().iloc[0]); ax.axhline(nom, linestyle="--"); ax.set_xlabel("Summary row"); ax.set_ylabel("Coverage"); ax.set_title("Interval coverage")
        d = s
    return _attach(ax, d)


def plot_eye_validation_reference_comparison(x: Any, y: Any = None, metric: str = "rmse_delta", ax: Any = None, **kwargs: Any):
    ax = _axis(ax); d = x.table.copy()
    if metric not in d: return _empty(ax, f"Missing {metric}")
    v = pd.to_numeric(d[metric], errors="coerce").to_numpy(float); xx = np.arange(1, len(v)+1)
    ax.vlines(xx, 0, v, **kwargs); ax.axhline(0, linestyle="--"); ax.set_xlabel("Matched summary row"); ax.set_ylabel(metric); ax.set_title(f"Frozen-reference delta: {metric}")
    return _attach(ax, d)


def plot_eye_analysis_pipeline(x: Any, y: Any = None, ax: Any = None, **kwargs: Any):
    ax = _axis(ax); g = eye_pipeline_graph(x); v = g.vertices.copy()
    if not len(v): return _empty(ax)
    xpos = pd.to_numeric(v.order, errors="coerce").to_numpy(float); ypos = np.zeros(len(v))
    if len(g.edges):
        pos = {r.step:(float(r.order),0.0) for _,r in v.iterrows()}
        for _,e in g.edges.iterrows():
            a,b=pos[e["from"]],pos[e["to"]]; ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->"})
    ax.scatter(xpos,ypos,s=500,**kwargs)
    for _,r in v.iterrows(): ax.text(float(r.order),0.05,str(r.step),ha="center")
    ax.set_yticks([]); ax.set_xlabel("Execution order"); ax.set_title("Governed eyeprocess pipeline")
    return _attach(ax, g)


def plot_eye_pipeline_audit(x: Any, y: Any = None, ax: Any = None, **kwargs: Any):
    ax = _axis(ax); d = x.table.copy()
    if "status" in d:
        mapping={"not_run":1,"success":2,"optional_error":3,"error":4}; vals=d.status.map(mapping).fillna(0).to_numpy(float)
    else: vals=d.decision_declared.astype(int).to_numpy(float)
    ax.bar(np.arange(len(d)),vals,**kwargs); ax.set_xticks(np.arange(len(d)),d.step.astype(str),rotation=90); ax.set_ylabel("Audit/status code"); ax.set_title("Pipeline audit")
    return _attach(ax,d)


def plot_eye_api_audit(x: Any, y: Any = None, ax: Any = None, **kwargs: Any):
    ax=_axis(ax); d=x.table
    if not isinstance(d,pd.DataFrame) or not len(d) or "status" not in d: return _empty(ax)
    s=d.status.value_counts(); ax.bar(s.index.astype(str),s.to_numpy(float),**kwargs); ax.tick_params(axis="x",rotation=90); ax.set_ylabel("Exports"); ax.set_title("API lifecycle audit")
    tab = s.rename_axis("status").reset_index(name="count")
    return _attach(ax, tab)


def plot_eye_process_sensitivity(x: Any, y: Any = None, type: str = "specification_curve", effect: str = "effect", lower: str | None = None, upper: str | None = None, ax: Any = None, **kwargs: Any):
    ax=_axis(ax)
    if type not in {"specification_curve","decision_leverage"}: raise EyeProcessValidationError("type must be specification_curve or decision_leverage.")
    if type=="decision_leverage":
        d=sensitivity_decision_leverage(x,effect)
        if not len(d): return _empty(ax)
        ax.bar(np.arange(len(d)),d.effect_range.to_numpy(float),**kwargs); ax.set_xticks(np.arange(len(d)),d.decision.astype(str),rotation=90); ax.set_ylabel("Within-decision effect range"); ax.set_title("Decision leverage")
        return _attach(ax,d)
    d=specification_curve_data(x,effect,lower,upper)
    if not len(d): return _empty(ax)
    ax.scatter(d.curve_order,d[".effect"],**kwargs); ax.axhline(0,linestyle="--")
    if {".lower",".upper"}.issubset(d): ax.vlines(d.curve_order,d[".lower"],d[".upper"])
    ax.set_xlabel("Specification (ordered)"); ax.set_ylabel(effect); ax.set_title("Specification curve")
    return _attach(ax,d)


def plot_eye_decision_stability(x: Any, y: Any = None, ax: Any = None, **kwargs: Any):
    ax=_axis(ax); s=x.summary
    cols=[c for c in ["sign_stability","threshold_stability","significance_stability"] if c in s]
    if not cols: return _empty(ax)
    vals=pd.to_numeric(s.iloc[0][cols],errors="coerce"); ax.bar(np.arange(len(cols)),vals.to_numpy(float),**kwargs); ax.set_xticks(np.arange(len(cols)),cols,rotation=30); ax.set_ylim(0,1); ax.axhline(.9,linestyle="--"); ax.set_ylabel("Stability"); ax.set_title("Decision stability")
    return _attach(ax,pd.DataFrame({"metric":cols,"stability":vals.to_numpy(float)}))


def plot_eye_decision_manifest(x: Any, y: Any = None, ax: Any = None, **kwargs: Any):
    ax=_axis(ax); d=decision_manifest_table(x)
    if not len(d): return _empty(ax,"No declared decisions","Analysis decision manifest")
    section=d.path.astype(str).str.replace(r"\..*$","",regex=True); tab=section.value_counts()
    ax.bar(np.arange(len(tab)),tab.to_numpy(float),**kwargs); ax.set_xticks(np.arange(len(tab)),tab.index.astype(str),rotation=90); ax.set_ylabel("Declared decisions"); ax.set_title("Analysis decision manifest")
    pdata = tab.rename_axis("section").reset_index(name="count")
    return _attach(ax, pdata)


__all__=[n for n in globals() if n.startswith("plot_eye_")]
