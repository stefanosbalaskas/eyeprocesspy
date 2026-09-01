# Research-scale validation execution

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/validation-execution-engine.Rmd`.

`eyeprocesspy` separates an executable model from evidence that the model is scientifically dependable. The validation execution engine converts a declared Monte Carlo design into deterministic jobs, atomic checkpoints, resumable runs, auditable failures, recovery summaries, calibration diagnostics, and promotion decisions.

## Deterministic plans

```python
import eyeprocesspy as ep

plan = ep.validation_job_plan(
    grid={
        "n_person": (50, 150, 500),
        "n_item": (10, 30),
        "process_effect": (0.0, 0.25, 0.50),
        "feature_reliability": (0.50, 0.80),
        "missingness": (0.0, 0.15),
    },
    replications=500,
    base_seed=20260805,
    model_family="dynamic_irtree",
    chunk_size=25,
)

ep.write_validation_job_manifest(
    plan,
    "validation/dynamic-irtree",
)
```

A job seed is determined by the complete design cell, replication, and base seed. Reordering the plan therefore should not alter the simulated study.

## Atomic execution and resumption

```python
result = ep.run_validation_jobs(
    plan,
    simulator=simulate_one_study,
    fitter=fit_one_model,
    extractor=extract_estimates,
    truth_extractor=extract_truth,
    diagnostics_extractor=extract_diagnostics,
    draws_extractor=extract_draws,
    output_dir="validation/dynamic-irtree",
    workers=8,
    backend="future",
    isolation="callr",
    timeout_seconds=3600,
    memory_limit_mb=8192,
)

resumed = ep.resume_validation_jobs(
    plan,
    "validation/dynamic-irtree",
    retry=("missing", "failed", "nonconverged"),
)
```

Backend names that originate in the R implementation remain explicit interoperability contracts in Python; an unavailable exact backend must be gated rather than silently replaced.

Every checkpoint should preserve the job specification, seed, warnings, messages, errors, runtime, estimates, diagnostics, optional posterior draws, predictions, and environment metadata. Failed jobs are evidence and are never silently removed.

## Collection and evidence

```python
collected = ep.collect_validation_jobs(
    "validation/dynamic-irtree",
    plan,
)
recovery = ep.validation_recovery_summary(collected)
failures = ep.validation_failure_summary(collected)
runtime = ep.validation_runtime_summary(collected)
calibration = ep.validation_calibration_summary(collected)
sbc = ep.validation_sbc_summary(collected)
audit = ep.audit_validation_completion(collected)
```

## Promotion remains gated

Model promotion requires the declared recovery, calibration/SBC, misspecification, grouped-validation, engine-equivalence, empirical-reproduction, and sensitivity evidence. Code execution alone is not a promotion criterion, and missing gates remain visible as missing evidence.
