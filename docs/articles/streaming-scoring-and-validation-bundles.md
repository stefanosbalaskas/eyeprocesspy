# Streaming scoring and validation evidence bundles

This article is the Python counterpart of the frozen `streaming-scoring-and-validation-bundles.Rmd` vignette from `eyeprocess 0.11.1`.

## Partial and streaming scoring

The frozen R implementation delegates partial MAP/EAP scoring to `mirt::fscores()`. `eyeprocesspy` preserves that backend identity: `score_partial_response_pattern()` raises an explicit backend error rather than silently substituting a different estimator.

`score_response_stream()` still implements the frozen cumulative-response contract. When the exact `mirt` backend is unavailable, each step is retained with `theta`/`theta_se` missing, making backend unavailability visible rather than fabricating a score.

```python
stream = ep.score_response_stream(
    calibrated_model,
    response_pattern=[1, 0, 1, 1, 0, 1],
    method="MAP",
)

history = ep.streaming_score_history(stream)
ax = ep.plot_eye_streaming_score(stream)
```

Streaming scores are operational building blocks. High-stakes deployment still requires calibrated item banks, latency/stopping validation, privacy governance, and explicit score-use rules.

## Unified validation evidence bundles

```python
bundle = ep.collect_validation_evidence(
    recovery=recovery_summary,
    convergence=convergence_audit,
    negative_controls=negative_controls,
    preflight=preflight,
    drift=drift,
    model_name="joint_process_model",
)

manifest = ep.validation_bundle_manifest(bundle)
report = ep.validation_report(bundle)
ax = ep.plot_eye_validation_bundle(bundle)
exported = ep.export_validation_bundle(bundle, "validation-export")
```

The bundle is organizational evidence, not proof that validation is adequate. The report preserves the frozen guardrails: convergence is not validation; predictive improvement is not causal evidence; gaze/pupil/process channels require preprocessing, missingness, and data-quality sensitivity; and review/screening outputs are not clinical or misconduct labels.

## Python serialization boundary

RDS is an R-specific binary format. The Python port does **not** imitate RDS with unsafe pickling. `export_validation_bundle()` writes CSV for tabular evidence and JSON/text metadata for the safe Python artifact. This difference is explicitly recorded as a backend/serialization parity exception rather than hidden.
