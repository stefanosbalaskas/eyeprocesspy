"""Operational validation, streaming-score contracts, and decision-process proxies.

Ports frozen eyeprocess 0.11.1 public APIs from R/063 and R/066. Exact mirt
streaming scoring remains an explicit backend boundary; validation bundles and
process-feature representations are implemented natively and conservatively.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .irt import EyeResult, _result
from .process_governance_08 import _df, _req, _groups, _group_values, _num, _mean, _slope, _entropy_factor, _switch_count, _as_bool

__all__ = [
    "score_partial_response_pattern", "score_response_stream", "update_person_score",
    "streaming_score_history", "collect_validation_evidence", "validation_bundle_manifest",
    "validation_report", "write_validation_report", "export_validation_bundle",
    "preaction_process_features", "addm_glam_proxy_features", "process_feature_family_registry",
    "assign_process_feature_family", "process_feature_stability",
]

_VALIDATION_SLOTS = [
    "model_spec", "evidence_grade", "recovery", "bias", "rmse", "coverage",
    "interval_width", "convergence", "identifiability", "mcse", "sbc", "ppc",
    "stress_tests", "external_validation", "transportability", "process_ablation",
    "incremental_information", "negative_controls", "semantic_validation",
    "preflight", "drift", "software_provenance", "session_provenance",
]


def score_partial_response_pattern(model: Any, response_pattern: Any,
                                   method: str = "MAP", **kwargs: Any) -> pd.DataFrame:
    """Score a partial response pattern using the exact frozen mirt backend.

    The R implementation delegates to ``mirt::fscores``. No Python estimator is
    labelled algorithmically identical, so this route is explicitly gated.
    """
    method = str(method).upper()
    if method not in {"MAP", "EAP"}:
        raise EyeProcessValidationError("method must be MAP or EAP.")
    rp = np.asarray(response_pattern, dtype=float).reshape(-1)
    if rp.size == 0:
        raise EyeProcessValidationError("response_pattern must contain at least one item position.")
    raise EyeProcessBackendError(
        "The frozen score_partial_response_pattern() contract requires R package `mirt` "
        "and mirt::fscores(). No algorithmically identical Python backend is substituted."
    )


def score_response_stream(model: Any, response_pattern: Any, observed_order: Any = None,
                          method: str = "MAP", **kwargs: Any) -> EyeResult:
    method = str(method).upper()
    if method not in {"MAP", "EAP"}:
        raise EyeProcessValidationError("method must be MAP or EAP.")
    rp = np.asarray(response_pattern, dtype=float).reshape(-1)
    n = rp.size
    if n == 0:
        raise EyeProcessValidationError("response_pattern must contain at least one item position.")
    if observed_order is None:
        order = np.flatnonzero(~np.isnan(rp)) + 1
    else:
        order = np.asarray(observed_order, dtype=int).reshape(-1)
    if order.size == 0:
        raise EyeProcessValidationError("No observed responses were supplied for streaming scoring.")
    if np.any((order < 1) | (order > n)) or len(np.unique(order)) != len(order):
        raise EyeProcessValidationError("observed_order must contain unique valid item positions.")
    rows = []
    current = np.full(n, np.nan)
    for k, pos in enumerate(order, start=1):
        current[pos - 1] = rp[pos - 1]
        theta = theta_se = math.nan
        try:
            fs = score_partial_response_pattern(model, current, method=method, **kwargs)
            if len(fs):
                theta_col = "F1" if "F1" in fs else fs.columns[0]
                se_cols = [c for c in fs if str(c).startswith("SE")]
                theta = float(pd.to_numeric(fs[theta_col], errors="coerce").iloc[0])
                theta_se = float(pd.to_numeric(fs[se_cols[0]], errors="coerce").iloc[0]) if se_cols else math.nan
        except (EyeProcessBackendError, EyeProcessValidationError):
            pass
        rows.append({"step": k, "item_position": int(pos), "latest_response": rp[pos - 1],
                     "theta": theta, "theta_se": theta_se})
    return _result(
        "eye_streaming_score", history=pd.DataFrame(rows), response_pattern=rp,
        observed_order=order, method=method, model=model,
        status="streaming_scoring_simulation",
        caveat=("Streaming scores are demonstrations/operational building blocks. High-stakes use requires "
                "calibrated banks, latency testing, privacy review, stopping rules, and score-governance validation."),
    )


def update_person_score(model: Any, current_pattern: Any, item_position: int, response: Any,
                        method: str = "MAP", **kwargs: Any) -> dict[str, Any]:
    rp = np.asarray(current_pattern, dtype=float).reshape(-1).copy()
    pos = int(item_position)
    if pos < 1 or pos > len(rp):
        raise EyeProcessValidationError("item_position is out of range.")
    value = float(np.asarray(response).reshape(-1)[0])
    if np.isnan(value):
        raise EyeProcessValidationError("response must be non-missing when updating a person score.")
    rp[pos - 1] = value
    return {"pattern": rp, "score": score_partial_response_pattern(model, rp, method=method, **kwargs)}


def streaming_score_history(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_streaming_score":
        raise EyeProcessValidationError("x must be eye_streaming_score.")
    return x.history.copy()


def collect_validation_evidence(*args: Any, model_name: str | None = None,
                                notes: Any = None, **evidence: Any) -> EyeResult:
    if args:
        raise EyeProcessValidationError("Validation evidence supplied through ... must be named.")
    unknown = [k for k in evidence if k not in _VALIDATION_SLOTS]
    if unknown:
        warnings.warn("Non-standard validation evidence slot(s): " + ", ".join(unknown), UserWarning, stacklevel=2)
    session = {"python_version": platform.python_version(), "platform": platform.platform()}
    return _result(
        "eye_validation_bundle", model_name="unnamed_model" if model_name is None else str(model_name),
        evidence=dict(evidence), notes=notes,
        created_at=datetime.now(timezone.utc).isoformat(), session=session,
        status="validation_evidence_bundle",
        caveat="A validation bundle organizes evidence; the presence of an object does not itself establish adequacy.",
    )


def _evidence_status(x: Any) -> str:
    if x is None:
        return "missing"
    if isinstance(x, pd.DataFrame) and x.empty:
        return "empty"
    if isinstance(x, BaseException):
        return "error"
    return "available"


def validation_bundle_manifest(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_validation_bundle":
        raise EyeProcessValidationError("x must be eye_validation_bundle.")
    slots = list(dict.fromkeys(_VALIDATION_SLOTS + list(x.evidence.keys())))
    rows = []
    for slot in slots:
        obj = x.evidence.get(slot)
        cls = None if obj is None else type(obj).__name__
        rows.append({"slot": slot, "status": _evidence_status(obj), "class": cls})
    return pd.DataFrame(rows)


def validation_report(x: Any, include_session: bool = True) -> list[str]:
    if getattr(x, "eyeprocess_class", None) != "eye_validation_bundle":
        raise EyeProcessValidationError("x must be eye_validation_bundle.")
    man = validation_bundle_manifest(x)
    avail = man.loc[man.status.eq("available"), "slot"].tolist()
    missing = man.loc[man.status.eq("missing"), "slot"].tolist()
    lines = [
        f"eyeprocess validation report: {x.model_name}", "=" * (30 + len(x.model_name)), "",
        f"Created: {x.created_at}", "", "Evidence inventory", "------------------",
        "Available: " + (", ".join(avail) if avail else "none"),
        "Missing/not supplied: " + (", ".join(missing) if missing else "none"), "",
    ]
    labels = {
        "recovery": "Parameter recovery evidence was supplied.",
        "coverage": "Interval-coverage evidence was supplied.",
        "convergence": "Convergence evidence was supplied.",
        "process_ablation": "Process-channel ablation evidence was supplied.",
        "negative_controls": "Negative-control evidence was supplied.",
        "preflight": "Biometric pre-flight evidence was supplied.",
        "drift": "Deployment-drift evidence was supplied.",
    }
    for key, text in labels.items():
        if x.evidence.get(key) is not None:
            lines.append(text)
    if x.evidence.get("external_validation") is not None or x.evidence.get("transportability") is not None:
        lines.append("External/transportability evidence was supplied.")
    lines += ["", "Interpretation guardrails", "-------------------------",
              "- Convergence is not validation.",
              "- Predictive improvement is not causal evidence.",
              "- Gaze/pupil/process channels require sensitivity to preprocessing, missingness, and data quality.",
              "- External validity and transportability should be evaluated before generalization.",
              "- Screening, anomaly, accessibility, and profile outputs are review tools, not clinical or misconduct labels."]
    if x.notes is not None:
        lines += ["", "Notes", "-----", str(x.notes)]
    if include_session:
        lines += ["", "Software provenance", "-------------------",
                  f"Python version: {x.session['python_version']}", f"Platform: {x.session['platform']}"]
    return lines


def write_validation_report(x: Any, path: str | Path, **kwargs: Any) -> str:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(validation_report(x, **kwargs)) + "\n", encoding="utf-8")
    return str(p)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return {"class": type(obj).__name__, "repr": repr(obj)}


def export_validation_bundle(x: Any, directory: str | Path, overwrite: bool = False,
                             include_rds: bool = True) -> EyeResult:
    if getattr(x, "eyeprocess_class", None) != "eye_validation_bundle":
        raise EyeProcessValidationError("x must be eye_validation_bundle.")
    directory = Path(directory).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise EyeProcessValidationError("Target directory is not empty; set overwrite=True to continue.")
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    manifest = validation_bundle_manifest(x)
    mp = directory / "validation_bundle_manifest.csv"; manifest.to_csv(mp, index=False); files["manifest"] = str(mp)
    files["report"] = write_validation_report(x, directory / "validation_report.txt")
    for name, obj in x.evidence.items():
        if isinstance(obj, pd.DataFrame):
            p = directory / f"{name}.csv"; obj.to_csv(p, index=False); files[name] = str(p)
        elif obj is not None and include_rds:
            p = directory / f"{name}.json"; p.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8"); files[name] = str(p)
    payload = {"model_name": x.model_name, "created_at": x.created_at,
               "manifest": manifest.to_dict(orient="records"), "files": files,
               "serialization_note": "Python uses safe JSON/CSV; RDS is not emitted."}
    jp = directory / "validation_bundle_manifest.json"; jp.write_text(json.dumps(payload, indent=2), encoding="utf-8"); files["json_manifest"] = str(jp)
    if include_rds:
        bp = directory / "validation_bundle.json"
        bp.write_text(json.dumps({"model_name": x.model_name, "notes": _jsonable(x.notes), "evidence": _jsonable(x.evidence)}, indent=2), encoding="utf-8")
        files["bundle"] = str(bp)
    return _result("eye_validation_export", directory=str(directory), files=files, manifest=manifest,
                   serialization_note="RDS is R-specific; the Python port emits safe JSON/CSV instead.")


def preaction_process_features(data: Any, by: Sequence[str] = ("person_id", "trial_id"),
                               time: str = "time_ms", response_time: str = "response_time_ms",
                               windows_ms: Sequence[float] = (500, 1000, 2000), aoi: str = "aoi",
                               pupil: str = "pupil_bc", blink: str = "blink") -> EyeResult:
    d = _df(data); by = list(by); _req(d, [*by, time, response_time])
    windows = np.unique(_num(windows_ms)); windows = windows[np.isfinite(windows) & (windows > 0)]
    if windows.size == 0:
        raise EyeProcessValidationError("windows_ms must contain at least one positive finite window.")
    global_aois = []
    if aoi in d:
        global_aois = sorted(pd.Series(d[aoi]).dropna().astype(str).loc[lambda s: s.str.len() > 0].unique())
    rows = []
    for idx in _groups(d, by):
        z = d.iloc[idx]; tt = _num(z[time]); rtvals = _num(z[response_time]); fin = rtvals[np.isfinite(rtvals)]
        if not fin.size: continue
        rt = float(fin[0]); rel = tt - rt
        for w in windows:
            keep = np.isfinite(rel) & (rel >= -w) & (rel <= 0); zz = z.loc[keep]; rr = rel[keep]
            if len(zz) < 3: continue
            av = zz[aoi].astype(object).to_numpy() if aoi in zz else np.repeat(None, len(zz))
            pv = _num(zz[pupil]) if pupil in zz else np.repeat(np.nan, len(zz))
            bv = _as_bool(zz[blink]).astype(float) if blink in zz else np.repeat(np.nan, len(zz))
            row = {**_group_values(d, idx, by), "pre_window_ms": float(w), "n_samples": len(zz),
                   "aoi_entropy": _entropy_factor(av), "aoi_switch_count": _switch_count(av),
                   "pupil_mean": _mean(pv), "pupil_slope": _slope(pv, rr), "blink_prop": _mean(bv)}
            for lv in global_aois:
                good = pd.notna(av); row[f"aoi_prop__{str(lv).replace(' ', '.')}"] = float(np.mean(np.asarray(av[good], str) == lv)) if good.any() else math.nan
            rows.append(row)
    return _result("eye_preaction_process_features", data=pd.DataFrame(rows), windows_ms=windows, by=by,
                   status="preaction_process_representation",
                   caveat="Pre-action features describe observed process dynamics; they are not evidence of latent intention by themselves.")


def addm_glam_proxy_features(data: Any, by: Sequence[str] = ("person_id", "trial_id"),
                             time: str = "time_ms", aoi: str = "aoi", target_aoi: str = "target",
                             distractor_aoi: str = "distractor", action_aoi: str = "button") -> EyeResult:
    d = _df(data); by = list(by); _req(d, [*by, time, aoi]); rows = []
    for idx in _groups(d, by):
        z = d.iloc[idx]; av = z[aoi].astype(str).to_numpy(); tt = _num(z[time])
        evidence = np.where(av == target_aoi, 1.0, np.where(av == distractor_aoi, -1.0, 0.0))
        action = np.where(av == action_aoi, 1.0, np.where(np.isin(av, [target_aoi, distractor_aoi]), -1.0, 0.0))
        tp, dp, ap = _mean(av == target_aoi), _mean(av == distractor_aoi), _mean(av == action_aoi)
        eps = 1e-6; split = float(np.nanmedian(tt)); early = evidence[tt < split]; late = evidence[tt >= split]
        rows.append({**_group_values(d, idx, by),
                     "target_minus_distractor_prop": tp - dp,
                     "evidence_slope": _slope(evidence, tt), "action_evidence_slope": _slope(action, tt),
                     "late_minus_early_evidence": _mean(late) - _mean(early),
                     "relative_target_attention": tp / (tp + dp + eps),
                     "relative_action_attention": ap / (ap + tp + dp + eps),
                     "gaze_discount_proxy": dp / (tp + dp + eps),
                     "choice_caution_proxy": _switch_count(av) / (abs(_mean(evidence)) + .01)})
    return _result("eye_decision_process_proxy", features=pd.DataFrame(rows), by=by,
                   status="decision_process_proxy_features",
                   caveat=("These are aDDM/GLAM-inspired descriptive proxies, not fitted drift rate, gaze discount, "
                           "decision threshold, or causal attention parameters."))


def process_feature_family_registry() -> pd.DataFrame:
    return pd.DataFrame({
        "pattern": ["transition|switch|entropy|scanpath|sequence", "aoi|target|distractor|button|text",
                    "pupil|phasic|tonic|ripa|frequency", "gaze|fixation|saccade|dwell|ttff",
                    "valid|trackloss|missing|confidence|quality", "rt|response_time|trial_order",
                    "condition|task|stimulus|layout|luminance"],
        "family": ["Scanpath organization", "AOI attention", "Pupil dynamics", "Gaze dynamics", "Data quality", "Timing", "Design/context"],
        "warning": ["Correlated with dwell and task structure.", "May reflect task design or option relevance.",
                    "May reflect luminance, arousal, effort, fatigue, or motor preparation.",
                    "May reflect viewing constraints and layout as well as processing.", "Must not be interpreted psychologically.",
                    "May reflect task design, motor timing, or speededness.",
                    "Design/context variables can dominate prediction and must be audited separately."],
    })


def assign_process_feature_family(feature_names: Any, registry: pd.DataFrame | None = None) -> np.ndarray:
    reg = process_feature_family_registry() if registry is None else _df(registry, "registry")
    _req(reg, ["pattern", "family"])
    out = []
    for feature in map(str, np.asarray(feature_names, dtype=object).reshape(-1)):
        hit = [i for i, pat in enumerate(reg.pattern.astype(str)) if pd.Series([feature]).str.contains(pat, case=False, regex=True).iloc[0]]
        out.append(str(reg.family.iloc[hit[0]]) if hit else "Other")
    return np.asarray(out, dtype=object)


def process_feature_stability(data: Any, feature: str = "feature", split: str = "split",
                              importance: str = "importance", top_n: int = 20) -> pd.DataFrame:
    d = _df(data); _req(d, [feature, split, importance]); top_n = int(top_n)
    if top_n < 1: raise EyeProcessValidationError("top_n must be at least 1.")
    q = d[[feature, split, importance]].copy(); q.columns = ["feature", "split", "importance"]
    q["importance"] = pd.to_numeric(q.importance, errors="coerce"); q = q.dropna(subset=["feature", "split", "importance"])
    if q.empty: raise EyeProcessValidationError("No complete split/importance rows are available.")
    groups = [z for _, z in q.groupby("split", sort=True)]
    rows = []
    for f in pd.unique(q.feature):
        selected = []
        for z in groups:
            selected.append(f in z.sort_values("importance", ascending=False).head(top_n).feature.to_numpy())
        rows.append({"feature": f, "top_n_selection_rate": float(np.mean(selected)),
                     "mean_importance": _mean(q.loc[q.feature.eq(f), "importance"])})
    out = pd.DataFrame(rows); out["feature_family"] = assign_process_feature_family(out.feature)
    return out.sort_values(["top_n_selection_rate", "mean_importance"], ascending=False, kind="stable").reset_index(drop=True)
