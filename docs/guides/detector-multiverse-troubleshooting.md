# Detector multiverse troubleshooting

Do not repair a failed branch by silently changing thresholds, units, observations, or estimators. Diagnose the contract failure, keep the branch visible, and either correct the declared input or report that the specification could not be evaluated.

## Failure clinic

The executable script `examples/detector_multiverse_failure_clinic.py` deliberately creates four auditable problems:

1. an external detector callback raises an error;
2. one target-AOI row fails a prespecified quality threshold;
3. one model outcome is non-finite;
4. a model callback returns the same coefficient term twice.

See the [worked failure clinic](../examples/detector-multiverse-failure-clinic.md) for the complete output interpretation.

## Common symptoms

| Symptom | What it usually means | Correct response |
| --- | --- | --- |
| `status = "failed"` at detection | Detector/backend could not produce a valid branch | Inspect `failures`; fix the declared backend/input or report the failed branch |
| Sampling-rate warning | Declared rate does not match realized timing closely enough | Verify timestamps and units; do not silently change the declared rate |
| Pixel data rejected by REMoDNaV bridge | Degree conversion is not defined | Supply an explicit `px2deg` conversion or transform coordinates first |
| AOI propagation fails | Geometry/overlap is ambiguous under the selected rule | Correct geometry or choose an explicit overlap rule; never auto-resolve ambiguity |
| `quality_excluded_rows > 0` | The declared quality rule removed model rows | Report the threshold and counts from `input_audit` |
| `outcome_missing_rows > 0` | Model outcome is non-finite after feature propagation | Investigate why it is missing; do not recode to zero |
| `status = "no_model_data"` | No finite rows remained after declared selection/quality/outcome rules | Keep the failure in the planned denominator and report the audit |
| Formula engine fails on missing predictors | `missing="raise"` protected the analysis from silent complete-case deletion | Construct an explicit analysis dataset only if scientifically justified |
| Callback duplicate-term error | One branch returned more than one row for the same coefficient term | Fix the callback so one branch contributes at most one row per term |
| Mixed model does not converge | The declared model failed for that branch | Retain diagnostic estimates/warnings; exclude it from converged stability summaries |
| Requested coefficient missing | Model parameterization did not emit the same term in that branch | Treat the term as unavailable; do not rename a different coefficient to force parity |

## Read the audit in order

For each detector, inspect:

```python
inference.input_audit[
    [
        "detector_id",
        "input_rows",
        "aoi_selected_rows",
        "quality_excluded_rows",
        "outcome_missing_rows",
        "model_rows_used",
        "status",
    ]
]
```

Then inspect `inference.failures` and `inference.warnings`. Only after the input and failure audit is understood should coefficient stability be summarized.

## What not to do

Do not:

- lower a quality threshold only for a detector with an inconvenient result;
- convert missing dwell to zero;
- delete a failed detector from the denominator;
- substitute a different external backend under the same detector label;
- choose the model engine separately for each detector;
- resolve AOI overlap differently across branches without declaring that as a separate sensitivity dimension;
- describe agreement as accuracy when no ground-truth event catalogue exists.

## Related pages

[Decision guide](detector-multiverse-decision-guide.md) · [Statistical inference](detector-statistical-inference.md) · [Reporting template](detector-multiverse-reporting-template.md) · [API reference](../reference/detector-multiverse.md)
