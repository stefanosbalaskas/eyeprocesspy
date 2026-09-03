# Computational benchmarking and synthetic stress testing

Scientific validity and computational feasibility are separate questions. `eye_benchmark_design()` measures runtime and scaling under declared dataset sizes, while synthetic corruption plans probe robustness to missingness, pupil dropout, calibration offsets, timestamp jitter, AOI label noise, device shifts, and trial imbalance.

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/benchmarking-and-stress-testing.Rmd`.

```python
import eyeprocesspy as ep

plans = [
    ep.synthetic_corruption_plan(missingness=0.05),
    ep.synthetic_corruption_plan(
        missingness=0.20,
        sampling_jitter_sd=2,
    ),
    ep.synthetic_corruption_plan(
        pupil_dropout=0.30,
        gaze_offset_x=0.02,
    ),
]

stress = ep.stress_test_process_pipeline(data, plans, analysis_fun)
summary = ep.stress_test_summary(stress)
```

A benchmark design should declare the quantities being scaled—participants, items, samples, channels, or model complexity—and separate warm-up or compilation costs from steady-state computation when those distinctions matter.

Synthetic stress tests likewise describe sensitivity only to the perturbations that were actually supplied. A result that is stable under simulated missingness does not establish robustness to every real missing-data mechanism, and synthetic device shifts do not substitute for empirical cross-device transportability evidence.

Use computational benchmarks to answer *can this workflow run at the intended scale?* Use validation and stress tests to answer *how does the result change under declared perturbations?* Neither question alone establishes scientific validity.
