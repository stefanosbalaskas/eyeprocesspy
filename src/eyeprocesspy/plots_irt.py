"""Matplotlib counterparts for frozen eyeprocess 0.9 IRT S3 plot methods.

The R package dispatches these through ``plot.<class>``.  Python exposes named
functions so the scientific plot surface is explicit and discoverable.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised in minimal-env CI
        raise ImportError("IRT plotting requires the 'plots' extra: pip install eyeprocesspy[plots]") from exc
    return plt


def _ax(ax=None):
    if ax is not None:
        return ax
    _, ax = _plt().subplots()
    return ax


def _df(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x)


def _mapping(x: Any) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise EyeProcessValidationError("plot input must be an eyeprocess result mapping.")
    return x


def _empty(ax, title="No finite data"):
    ax.set_title(title)
    return ax


def plot_eye_irt_information_profile(x: Any, *, show_sem: bool = False, ax=None):
    ax = _ax(ax); d = _df(x)
    if d.empty: return _empty(ax)
    y = "conditional_sem" if show_sem else "information"
    ax.plot(d["theta"], d[y]); ax.set_xlabel("theta"); ax.set_ylabel("Conditional SEM" if show_sem else "Test information")
    return ax


def plot_eye_irt_test_characteristic_curve(x: Any, *, ax=None):
    ax = _ax(ax); d = _df(x)
    if d.empty: return _empty(ax)
    ax.plot(d["theta"], d["expected_score"]); ax.set_xlabel("theta"); ax.set_ylabel("Expected test score"); return ax


def plot_eye_irt_identification_audit(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x)
    vals = [bool(x.get("location_identified")), bool(x.get("scale_identified"))]
    ax.bar(["location", "scale"], np.asarray(vals, dtype=int)); ax.set_ylim(0, 1); ax.set_ylabel("Identified (0/1)"); return ax


def plot_eye_irt_sparse_design_audit(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x)
    pc = _df(x.get("person_counts", []))
    if pc.empty or "n_items" not in pc: return _empty(ax)
    ax.hist(pd.to_numeric(pc["n_items"], errors="coerce").dropna()); ax.set_xlabel("Observed items per person"); ax.set_title("Sparse IRT design"); return ax


def plot_eye_irt_q3_matrix(x: Any, *, ax=None):
    ax = _ax(ax); z = np.asarray(x, dtype=float)
    if z.size == 0: return _empty(ax)
    ax.imshow(z, aspect="auto", origin="upper"); ax.set_xlabel("Item"); ax.set_ylabel("Item"); return ax


def plot_eye_irt_item_fit(x: Any, *, statistic: str = "infit", ax=None):
    if statistic not in {"infit", "outfit"}: raise EyeProcessValidationError("statistic must be 'infit' or 'outfit'.")
    ax = _ax(ax); d = _df(x)
    if d.empty: return _empty(ax)
    labels = d["item_id"].astype(str) if "item_id" in d else pd.Series(np.arange(1, len(d)+1).astype(str))
    ax.scatter(np.arange(1, len(d)+1), d[statistic]); ax.axhline(1, linestyle="--"); ax.set_xticks(np.arange(1, len(d)+1), labels, rotation=90); ax.set_xlabel("Item"); ax.set_ylabel(statistic); return ax


def plot_eye_irt_person_fit(x: Any, *, statistic: str = "infit", ax=None):
    if statistic not in {"infit", "outfit"}: raise EyeProcessValidationError("statistic must be 'infit' or 'outfit'.")
    ax = _ax(ax); d = _df(x); v = pd.to_numeric(d.get(statistic, pd.Series(dtype=float)), errors="coerce"); v = v[np.isfinite(v)]
    if len(v) == 0: return _empty(ax)
    ax.hist(v); ax.axvline(1, linestyle="--"); ax.set_xlabel(statistic); ax.set_title(f"Person {statistic}"); return ax


def plot_eye_irt_fit_dashboard(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x); c = x.get("components", {}) or {}
    names = ["item_fit", "person_fit", "q3", "parameter_audit", "identification"]
    vals = [c.get(k) is not None for k in names]
    ax.bar(["item_fit", "person_fit", "q3", "parameters", "identification"], np.asarray(vals, dtype=int)); ax.set_ylim(0, 1); ax.set_ylabel("Component present"); ax.tick_params(axis="x", rotation=45); return ax


def plot_eye_irt_score_uncertainty(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x); vals = np.asarray([x.get("mean_se", np.nan), x.get("median_se", np.nan), x.get("p95_se", np.nan)], dtype=float)
    if not np.isfinite(vals).any(): return _empty(ax)
    ax.bar(["mean_SE", "median_SE", "p95_SE"], vals); ax.set_ylabel("Conditional SE"); return ax


def plot_eye_irt_adaptive_trace(x: Any, *, ax=None):
    ax = _ax(ax); d = _df(x)
    if d.empty: return _empty(ax)
    ax.plot(d["step"], d["theta_after"], marker="o"); ax.set_xlabel("Administered item"); ax.set_ylabel("theta estimate"); return ax


def plot_eye_irt_link_stability(x: Any, *, parameter: str = "A", ax=None):
    if parameter not in {"A", "B"}: raise EyeProcessValidationError("parameter must be 'A' or 'B'.")
    ax = _ax(ax); x = _mapping(x); d = _df(x.get("table", []))
    if d.empty: return _empty(ax)
    ax.scatter(np.arange(1, len(d)+1), d[parameter]); ax.set_xticks(np.arange(1, len(d)+1), d["set"].astype(str), rotation=90); ax.set_xlabel("Anchor set"); ax.set_ylabel(parameter); return ax


def plot_eye_irt_dif_curve(x: Any, *, ax=None):
    ax = _ax(ax); d = _df(x)
    if d.empty: return _empty(ax)
    ax.plot(d["theta"], d["signed_difference"]); ax.axhline(0, linestyle="--"); ax.set_xlabel("theta"); ax.set_ylabel("Focal - reference probability"); return ax


def plot_eye_irt_dtf_curve(x: Any, *, ax=None):
    ax = _ax(ax); d = _df(x)
    if d.empty: return _empty(ax)
    ax.plot(d["theta"], d["signed_difference"]); ax.axhline(0, linestyle="--"); ax.set_xlabel("theta"); ax.set_ylabel("Focal - reference expected score"); return ax


def plot_eye_irt_process_alignment(x: Any, *, channel: str | None = None, parameter: str = "b", ax=None):
    if parameter not in {"a", "b"}: raise EyeProcessValidationError("parameter must be 'a' or 'b'.")
    ax = _ax(ax); x = _mapping(x); d = _df(x.get("table", [])); cor = _df(x.get("correlations", []))
    if channel is None and not cor.empty: channel = str(cor.iloc[0]["channel"])
    if channel is None or channel not in d: return _empty(ax, "No process channel")
    a = pd.to_numeric(d[parameter], errors="coerce"); b = pd.to_numeric(d[channel], errors="coerce"); good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 2: return _empty(ax)
    ax.scatter(a[good], b[good]); ax.set_xlabel(f"IRT {parameter}"); ax.set_ylabel(channel); return ax


def plot_eye_irt_recovery_result(x: Any, *, parameter: str = "b", ax=None):
    if parameter not in {"a", "b"}: raise EyeProcessValidationError("parameter must be 'a' or 'b'.")
    ax = _ax(ax); x = _mapping(x); d = _df(x.get("estimates", []))
    if d.empty: return _empty(ax, "No successful recovery fits")
    tx = pd.to_numeric(d[f"{parameter}_truth"], errors="coerce"); ex = pd.to_numeric(d[f"{parameter}_estimate"], errors="coerce"); good = np.isfinite(tx) & np.isfinite(ex)
    ax.scatter(tx[good], ex[good]); ifin = np.r_[tx[good].to_numpy(), ex[good].to_numpy()];
    if ifin.size: ax.plot([ifin.min(), ifin.max()], [ifin.min(), ifin.max()], linestyle="--")
    ax.set_xlabel(f"{parameter} truth"); ax.set_ylabel(f"{parameter} estimate"); return ax


def plot_eye_irt_sbc_evidence(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x); d = _mapping(x.get("diagnostics", {})); counts = np.asarray(d.get("counts", []), dtype=float)
    if counts.size == 0: return _empty(ax)
    ax.bar(np.arange(1, len(counts)+1), counts); exp = np.asarray(d.get("expected_count", []), dtype=float)
    if exp.size: ax.plot(np.arange(1, len(exp)+1), exp, linestyle="--")
    ax.set_xlabel("SBC rank bin"); ax.set_ylabel("Count"); return ax


def plot_eye_cdm_qmatrix_audit(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x); d = _df(x.get("attribute", []))
    if d.empty: return _empty(ax)
    ax.bar(d["attribute"].astype(str), d["n_items"]); ax.set_ylabel("Items measuring attribute"); return ax


def plot_eye_irt_bank_coverage(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x); d = _df(x.get("curve", []))
    if d.empty: return _empty(ax)
    ax.plot(d["theta"], d["information"]); ax.axhline(float(x.get("target_information", np.nan)), linestyle="--")
    target = np.asarray(x.get("target", []), dtype=float).ravel()
    for t in target[np.isfinite(target)]: ax.axvline(t, linestyle=":")
    ax.set_xlabel("theta"); ax.set_ylabel("test information"); return ax


def plot_eye_irt_targeting_gap(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x); d = _df(x.get("table", []))
    if d.empty: return _empty(ax)
    ax.plot(d["theta"], d["person_mass"]); ax.plot(d["theta"], d["information_mass"], linestyle="--"); ax.set_xlabel("theta"); ax.set_ylabel("normalized mass"); return ax


def plot_eye_irt_missing_design_audit(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x); denom = max(1, int(x.get("n_persons", 0)) * int(x.get("n_items", 0)))
    vals = [float(x.get("observed_fraction", np.nan)), float(x.get("structural_missing", 0))/denom, float(x.get("unexpected_missing", 0))/denom]
    ax.bar(["observed", "structural_missing", "unexpected_missing"], vals); ax.set_ylabel("fraction"); ax.tick_params(axis="x", rotation=25); return ax


def plot_eye_irt_prior_sensitivity(x: Any, *, ax=None):
    ax = _ax(ax); x = _mapping(x); d = _df(x.get("table", []))
    if "estimate" not in d: return _empty(ax, "Prior sensitivity table has no `estimate` column.")
    y = pd.to_numeric(d["estimate"], errors="coerce"); ax.plot(np.arange(1, len(y)+1), y, marker="o"); ax.set_xlabel("prior specification"); ax.set_ylabel("estimate"); return ax


__all__ = [name for name, value in globals().items() if name.startswith("plot_eye_") and callable(value)]
