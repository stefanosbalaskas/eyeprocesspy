# Worked example: detector failure clinic

This synthetic example is intentionally **not** a successful-analysis showcase. It demonstrates how the workflow behaves when scientifically important failures occur and verifies that they remain visible instead of being repaired silently.

The runnable source is `examples/detector_multiverse_failure_clinic.py`.

## 1. Deliberate detector failure

The multiverse contains an I-VT branch and an external callback that deliberately raises an error. With `continue_on_error=True`, the I-VT branch runs and the external branch is retained as `status = "failed"` with its error message.

```python
detected = ep.run_detector_multiverse(
    data,
    [ivt30, external_fail],
    continue_on_error=True,
)

print(detected.status)
print(detected.failures)
```

The failed detector is not replaced by I-VT, REMoDNaV, or another surrogate.

## 2. Deliberate model-input attrition

The clinic then creates exactly one low-quality target-AOI row and one non-finite dwell outcome before model propagation.

```python
inference = ep.run_detector_inference_multiverse(
    features,
    {
        "engine": "callback",
        "outcome": "dwell_time_ms",
        "aoi_id": "disclosure",
    },
    model_callback=tidy_callback,
    minimum_valid_fraction=0.5,
)

print(inference.input_audit)
print(inference.warnings)
```

For the I-VT branch the audit records one `quality_excluded_rows`, one `outcome_missing_rows`, and the exact `model_rows_used`. The failed external branch has zero available propagated rows and remains `no_model_data` at the modelling stage.

This distinction matters: **AOI selection, quality exclusion, outcome missingness, and backend failure are different scientific states.**

## 3. Invalid callback output

Finally, a deliberately invalid callback returns the same coefficient term twice. The branch fails with an `EyeProcessValidationError` rather than allowing one detector to contribute multiple rows to the stability summary.

```python
invalid = ep.run_detector_inference_multiverse(
    features,
    model_spec,
    model_callback=duplicate_callback,
)

print(invalid.failures)
```

## Interpretation

A robust workflow is not one in which every branch is forced to succeed. It is one in which branch success, failure, exclusion, convergence, and term availability are **explicitly attributable and reproducible**.

Use this clinic when validating a new external detector adapter, specialist model callback, quality rule, or reporting pipeline.

## Reporting example

> Two detector specifications were planned. The internal I-VT branch completed, whereas the deliberately failing external branch was retained as a detector-stage failure and was not replaced. During model propagation, the input audit separately recorded target-AOI selection, one prespecified quality exclusion, one non-finite outcome, and the resulting model N. An invalid duplicate-term callback was rejected and recorded as a model-stage failure.

## Next steps

[Decision guide](../guides/detector-multiverse-decision-guide.md) · [Troubleshooting](../guides/detector-multiverse-troubleshooting.md) · [Reporting template](../guides/detector-multiverse-reporting-template.md) · [API reference](../reference/detector-multiverse.md)
