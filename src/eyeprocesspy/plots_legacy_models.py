"""Plots for legacy/core model parity."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def plot_eye_parameter_recovery(x: Any, ax=None):
    import matplotlib.pyplot as plt
    d = pd.DataFrame(x).copy()
    if ax is None:
        _, ax = plt.subplots()
    est = pd.to_numeric(d.get("estimate"), errors="coerce")
    truth = pd.to_numeric(d.get("truth"), errors="coerce")
    ok = np.isfinite(est) & np.isfinite(truth)
    pdata = d.loc[ok].copy()
    if len(pdata):
        ax.scatter(truth[ok], est[ok])
        lo = float(min(truth[ok].min(), est[ok].min()))
        hi = float(max(truth[ok].max(), est[ok].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True value")
    ax.set_ylabel("Estimate")
    ax.set_title("Parameter recovery")
    ax.gp3_data = pdata
    return ax


__all__ = ["plot_eye_parameter_recovery"]
