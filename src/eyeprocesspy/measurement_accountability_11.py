"""Measurement-accountability diagnostics for eyeprocesspy.

These independent helpers complement, rather than replace, existing pupil,
timebase, multimodal, and grouped-validation APIs.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence
import math
import random


def _finite(values: Iterable[float]) -> list[float]:
    out = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _median(values: Sequence[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    if not n:
        return math.nan
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _mad(values: Sequence[float], center: float | None = None) -> float:
    if not values:
        return math.nan
    c = _median(values) if center is None else center
    return _median([abs(x - c) for x in values])


def _first_sustained(mask: Sequence[bool], run: int) -> int | None:
    run = max(1, int(run))
    streak = 0
    for i, flag in enumerate(mask):
        streak = streak + 1 if flag else 0
        if streak >= run:
            return i - run + 1
    return None


def pupil_latency_sensitivity(
    time: Sequence[float],
    pupil: Sequence[float],
    *,
    event_time: float = 0.0,
    baseline_window: tuple[float, float] = (-0.5, 0.0),
    search_window: tuple[float, float] = (0.0, 2.0),
    direction: str = "constriction",
    threshold_sigma: float = 3.0,
    sustain_ms: float = 40.0,
    simulations: int = 200,
    seed: int = 1,
) -> dict:
    """Estimate pupil-response latency with estimator-disagreement diagnostics.

    Returns several defensible onset estimates rather than presenting one latency
    as hardware- and algorithm-independent. A parametric resampling audit then
    characterizes resolvability under the observed sampling/noise regime.
    """
    if len(time) != len(pupil) or len(time) < 8:
        raise ValueError("time and pupil must have equal length and at least 8 samples")
    pairs = []
    for t, y in zip(time, pupil):
        try:
            t, y = float(t), float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and math.isfinite(y):
            pairs.append((t, y))
    pairs.sort()
    if len(pairs) < 8:
        raise ValueError("fewer than 8 finite samples")
    ts = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    dts = [b - a for a, b in zip(ts[:-1], ts[1:]) if b > a]
    dt = _median(dts)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("time must contain increasing samples")
    sampling_hz = 1.0 / dt

    b0, b1 = event_time + baseline_window[0], event_time + baseline_window[1]
    s0, s1 = event_time + search_window[0], event_time + search_window[1]
    base_idx = [i for i, t in enumerate(ts) if b0 <= t < b1]
    search_idx = [i for i, t in enumerate(ts) if s0 <= t <= s1]
    if len(base_idx) < 3 or len(search_idx) < 4:
        raise ValueError("baseline/search windows contain too few samples")
    baseline = _median([ys[i] for i in base_idx])
    noise = 1.4826 * _mad([ys[i] for i in base_idx], baseline)
    if not math.isfinite(noise) or noise == 0:
        centered = [ys[i] - baseline for i in base_idx]
        noise = math.sqrt(sum(x * x for x in centered) / max(1, len(centered) - 1)) or 1e-12

    sign = -1.0 if direction.lower() in {"constriction", "decrease", "negative"} else 1.0
    response = [sign * (ys[i] - baseline) for i in search_idx]
    st = [ts[i] for i in search_idx]
    sustain_n = max(1, round((sustain_ms / 1000.0) / dt))
    threshold = threshold_sigma * noise

    ix = _first_sustained([r >= threshold for r in response], sustain_n)
    threshold_latency = math.nan if ix is None else st[ix] - event_time

    slopes = [(response[i] - response[i - 1]) / (st[i] - st[i - 1]) for i in range(1, len(st))]
    peak_i = max(range(len(slopes)), key=lambda i: slopes[i]) + 1
    slope = slopes[peak_i - 1]
    tangent_latency = (st[peak_i] - response[peak_i] / slope) - event_time if slope > 0 else math.nan

    best = None
    for k in range(2, len(st) - 2):
        x0 = st[k]
        xs = [x - x0 for x in st[k:]]
        denom = sum(x * x for x in xs)
        if denom <= 0:
            continue
        beta = sum(x * r for x, r in zip(xs, response[k:])) / denom
        pred = [0.0] * k + [beta * x for x in xs]
        sse = sum((r - p) ** 2 for r, p in zip(response, pred))
        if best is None or sse < best[0]:
            best = (sse, k)
    breakpoint_latency = math.nan if best is None else st[best[1]] - event_time

    estimates = {
        "sustained_threshold": threshold_latency,
        "max_slope_tangent": tangent_latency,
        "piecewise_breakpoint": breakpoint_latency,
    }
    finite_est = _finite(estimates.values())
    spread_ms = (max(finite_est) - min(finite_est)) * 1000.0 if len(finite_est) >= 2 else math.nan
    amplitude = max(response) if response else 0.0
    snr = amplitude / noise if noise > 0 else math.inf

    rng = random.Random(seed)
    sim_latencies = []
    reference_latency = _median(finite_est) if finite_est else math.nan
    if simulations > 0 and math.isfinite(reference_latency) and amplitude > 0:
        tau = max(0.05, 4 * dt)
        for _ in range(int(simulations)):
            onset_abs = event_time + reference_latency
            sim = [
                (amplitude * (1.0 - math.exp(-(t - onset_abs) / tau)) if t >= onset_abs else 0.0)
                + rng.gauss(0.0, noise)
                for t in st
            ]
            sim_ix = _first_sustained([r >= threshold for r in sim], sustain_n)
            if sim_ix is not None:
                sim_latencies.append(st[sim_ix] - event_time)
    sim_error_ms = [abs(x - reference_latency) * 1000.0 for x in sim_latencies]
    p95_error_ms = (
        sorted(sim_error_ms)[max(0, math.ceil(0.95 * len(sim_error_ms)) - 1)]
        if sim_error_ms else math.nan
    )
    if len(finite_est) >= 2 and spread_ms <= 50 and (not sim_error_ms or p95_error_ms <= 100):
        resolvability = "high"
    elif finite_est and (not sim_error_ms or p95_error_ms <= 200):
        resolvability = "moderate"
    else:
        resolvability = "low"
    return {
        "estimates_s": estimates,
        "estimator_spread_ms": spread_ms,
        "sampling_hz": sampling_hz,
        "baseline": baseline,
        "noise_mad_sigma": noise,
        "response_amplitude": amplitude,
        "signal_to_noise": snr,
        "simulation": {"n": int(simulations), "successful": len(sim_latencies), "p95_abs_error_ms": p95_error_ms},
        "latency_resolvability": resolvability,
        "provenance": {
            "direction": direction, "threshold_sigma": threshold_sigma, "sustain_ms": sustain_ms,
            "baseline_window": baseline_window, "search_window": search_window, "seed": seed,
        },
    }


def event_marker_qc(
    offsets: Sequence[float],
    *,
    tolerance: float,
    expected_direction: str | None = None,
    effects: Sequence[float] | None = None,
    min_corroborating: int = 2,
) -> dict:
    """Assess event-marker plausibility from independent channel offsets.

    This is annotation/event plausibility QC, not hardware-clock synchronization.
    It reports consensus offset and uncertainty without modifying timestamps.
    """
    xs = _finite(offsets)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if not xs:
        return {
            "status": "implausible", "reason": "no corroborating offsets", "n": 0,
            "estimated_offset_s": math.nan, "uncertainty_s": math.nan,
        }
    center = _median(xs)
    uncertainty = 1.4826 * _mad(xs, center) if len(xs) > 1 else 0.0
    in_tol = sum(abs(x) <= tolerance for x in xs)
    consensus = sum(abs(x - center) <= tolerance for x in xs)
    direction_ok = True
    if effects is not None and expected_direction:
        es = _finite(effects)
        if es:
            m = _median(es)
            direction_ok = m <= 0 if expected_direction.lower() in {"negative", "decrease", "constriction"} else m >= 0
    if len(xs) >= min_corroborating and consensus >= min_corroborating and abs(center) <= tolerance and direction_ok:
        status = "confirmed"
    elif consensus >= 1 and abs(center) <= 2 * tolerance and direction_ok:
        status = "plausible"
    elif consensus >= 1:
        status = "ambiguous"
    else:
        status = "implausible"
    return {
        "status": status, "n": len(xs), "within_tolerance": in_tol,
        "estimated_offset_s": center, "uncertainty_s": uncertainty,
        "tolerance_s": tolerance, "direction_consistent": direction_ok,
        "note": "Event plausibility only; no clock-drift correction was applied.",
    }


def validation_ladder(stages: Mapping[str, str | bool | None], *, claim: str = "descriptive") -> dict:
    """Summarize evidence from acquisition QC through held-out-person validation."""
    required = ["acquisition_qc", "analytical_qc", "construct_check", "within_person", "held_out_person"]

    def norm(value):
        if value is True:
            return "pass"
        if value is False:
            return "fail"
        if value is None:
            return "not_assessed"
        value = str(value).lower().replace("-", "_")
        return {
            "ok": "pass", "passed": "pass", "warning": "warning", "warn": "warning",
            "failed": "fail", "na": "not_assessed", "missing": "not_assessed",
        }.get(value, value)

    values = {stage: norm(stages.get(stage)) for stage in required}
    invalid = {k: v for k, v in values.items() if v not in {"pass", "warning", "fail", "not_assessed"}}
    if invalid:
        raise ValueError(f"invalid stage status: {invalid}")
    first_blocker = next((s for s in required if values[s] in {"fail", "not_assessed"}), None)
    held_out = values["held_out_person"] == "pass"
    general_claim = claim.lower().replace("-", "_") in {"generalizable", "generalization", "out_of_person", "population"}
    if any(v == "fail" for v in values.values()) or (general_claim and not held_out):
        claim_status = "not_supported"
    elif any(v in {"warning", "not_assessed"} for v in values.values()):
        claim_status = "qualified"
    else:
        claim_status = "supported"
    return {
        "stages": values, "claim": claim, "claim_status": claim_status,
        "first_blocker": first_blocker, "held_out_person_generalization": held_out,
        "interpretation": "Within-person evidence is calibration/personalization evidence unless held-out-person validation passes.",
    }


__all__ = ["pupil_latency_sensitivity", "event_marker_qc", "validation_ladder"]
