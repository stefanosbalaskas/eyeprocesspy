"""Empirical process-validation design, recovery, and frozen-reference smoke."""
import eyeprocesspy as ep

design = ep.process_validation_design(
    n_persons=20, n_trials=8, missingness=0,
    sampling_rate_hz=60, aoi_error="low", calibration_error=0,
    pupil_dropout=0, heterogeneity="low",
    model_misspecification=False, replications=2, seed=9,
)
result = ep.run_process_validation(design)
summary = ep.summarise_process_validation(result)
reference = ep.freeze_validation_reference(result)
comparison = ep.validate_against_reference(result, reference, tolerance=1e-8)
assert len(summary) == 1
assert comparison["pass"] is True
