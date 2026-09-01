# Negative controls, placebo windows, and temporal leakage

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/negative-controls-and-temporal-leakage.Rmd`.

Predictive and process-feature workflows can accidentally use information that is unavailable at the intended decision boundary. The temporal provenance layer makes availability explicit.

```python
import eyeprocesspy as ep

provenance = ep.process_feature_time_provenance(
    ("dwell_pre", "rt_final"),
    (400, 1200),
    outcome_at=(1000, 1000),
)
ep.audit_temporal_leakage(provenance)
```

Negative controls deliberately break a declared process–outcome relation and rerun the same analysis.

```python
controls = ep.run_process_negative_controls(
    data,
    outcome="y",
    analysis_fun=analysis_fun,
    replications=200,
)
summary = ep.summarise_process_negative_controls(controls)
benchmark = ep.process_null_benchmark(observed_effect, controls)
```

A leakage flag denotes temporal or information contamination, not misconduct. Null-like negative controls are useful diagnostics but do not prove model validity. Conversely, a failed negative control should remain visible as evidence rather than being omitted because it makes a pipeline look less stable.
