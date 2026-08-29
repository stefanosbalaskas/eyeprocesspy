# Empirical validation programmes in eyeprocesspy

`eyeprocesspy` mirrors the eyeprocess 0.9 validation layer by treating validation as an explicit research object rather than an implicit property of an estimator. `process_validation_design()` declares the measurement regimes to study; `run_process_validation()` records recovery, interval coverage, convergence, warnings, and failures; and `freeze_validation_reference()` supports regression-style evidence freezing.

```python
import eyeprocesspy as ep

design = ep.process_validation_design(
    n_persons=(50, 150),
    n_trials=(10, 30),
    missingness=(0, .15),
    sampling_rate_hz=(60, 120),
    replications=100,
)
result = ep.run_process_validation(design)
recovery = ep.validation_recovery_table(result)
coverage = ep.validation_coverage_table(result)
failures = ep.validation_failure_profile(result)
ax = ep.plot_eye_process_validation_result(result, type="recovery")
```

The built-in simulator is a neutral software-validation fixture with known truth. It is **not** a substantive psychological theory. Recovery under a supplied data-generating process does not establish external validity, and numerical parity with the R package remains a separate cross-language validation question.
