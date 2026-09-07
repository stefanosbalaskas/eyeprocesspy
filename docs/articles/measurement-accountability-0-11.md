# Measurement accountability: latency, event plausibility, and validation ladders

`eyeprocesspy.measurement_accountability_11` adds three diagnostics that make uncertainty and validation claims explicit without replacing the package's existing pupil, timebase, multimodal, or grouped-validation engines.

## Pupil latency sensitivity

`pupil_latency_sensitivity()` returns a sustained-threshold onset, maximum-slope tangent onset, and piecewise breakpoint together with estimator spread, robust baseline noise, signal-to-noise ratio, and a small parametric resampling audit under the observed sampling regime. The result includes a `latency_resolvability` label rather than presenting one latency as algorithm- or hardware-independent.

```python
from eyeprocesspy.measurement_accountability_11 import pupil_latency_sensitivity

result = pupil_latency_sensitivity(
    time_s,
    pupil,
    event_time=0.0,
    baseline_window=(-0.5, 0.0),
    search_window=(0.0, 2.0),
    simulations=500,
)
print(result["estimates_s"])
print(result["estimator_spread_ms"])
print(result["latency_resolvability"])
```

This is an independent, transparent sensitivity harness. It is not a verbatim reproduction of any published implementation.

## Event-marker plausibility is not clock synchronization

`event_marker_qc()` evaluates whether independently estimated channel offsets corroborate a nominal event. It returns `confirmed`, `plausible`, `ambiguous`, or `implausible`, a consensus offset, and robust uncertainty. It never changes timestamps and explicitly records that no clock-drift correction was applied.

```python
from eyeprocesspy.measurement_accountability_11 import event_marker_qc

qc = event_marker_qc([0.012, 0.018, 0.016], tolerance=0.050)
```

Use the package's timebase/alignment tools for synchronization and drift correction; use this diagnostic to audit event/annotation plausibility or conduct timing-uncertainty sensitivity analyses.

## Validation ladder

`validation_ladder()` separates five evidence stages:

1. acquisition QC;
2. analytical QC;
3. construct check;
4. within-person evidence;
5. held-out-person generalization.

A generalization claim is `not_supported` unless the held-out-person stage passes. This prevents personalized/within-participant calibration performance from being reported as out-of-person generalization.

```python
from eyeprocesspy.measurement_accountability_11 import validation_ladder

ladder = validation_ladder(
    {
        "acquisition_qc": "pass",
        "analytical_qc": "pass",
        "construct_check": "pass",
        "within_person": "pass",
        "held_out_person": "not_assessed",
    },
    claim="generalizable",
)
```

## Methodological provenance

These additions were motivated by the September 2026 measurement-methods surveillance tranche, especially work on pupil-latency benchmarking, multimodal construct-validation ladders, and physiological validation of uncertain timestamps. Relevant primary references include DOI `10.1038/s41598-026-68921-9`, DOI `10.3389/fnrgo.2026.1911259`, and DOI `10.1007/s12028-026-02637-6`.
