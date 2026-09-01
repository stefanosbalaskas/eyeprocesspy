# Measurement-quality stress tests and negative controls

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/measurement-stress-negative-controls.Rmd`.

The measurement-validation layer turns declared measurement-quality perturbations into reproducible software-validation scenarios.

```python
import eyeprocesspy as ep

stress_plan = ep.eyeprocess_stress_evidence_plan()
stress_grid = ep.expand_eyeprocess_stress_evidence_plan(stress_plan)
reliability_plan = ep.eyeprocess_reliability_evidence_plan()
negative_control_plan = ep.eyeprocess_negative_control_evidence_plan()
```

Stress dimensions include gaze missingness, pupil dropout, calibration offsets, sampling jitter, AOI-label noise, device shifts, and trial imbalance. Reliability evidence includes split-half behavior, ICC-style stability, temporal stability, and agreement diagnostics. Negative controls include permutations, temporal shifts, placebo windows, and intentionally known leakage.

These scenarios answer different questions. Reliability describes repeatability, not validity. Robustness under one corruption family does not prove robustness to every empirical failure mechanism. A negative-control or leakage flag indicates that a declared falsification or information-boundary check was triggered; it is not a misconduct label.

Thresholds and perturbations are study-specific reporting conventions. They should be declared before interpreting the corresponding validation results and retained in provenance so that a release gate can be reconstructed exactly.
