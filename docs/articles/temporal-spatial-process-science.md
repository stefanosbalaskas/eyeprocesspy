# Temporal and spatial process science

This article covers recurrence and fixation point-process workflows ported from `eyeprocess` 0.11.1. `gaze_recurrence()` and `cross_recurrence()` create recurrence matrices for gaze or synchronized channels, while `recurrence_features()` reports recurrence rate, determinism, laminarity, trapping time, and diagonal entropy. `windowed_recurrence()` tracks those quantities across local windows.

`fit_fixation_point_process()` provides a dependency-light gridded intensity model for fixation locations, optional history effects, and declared spatial/temporal covariates. `predict_fixation_intensity()` and `diagnose_gaze_point_process()` expose the fitted intensity and observed-versus-expected diagnostics.

These methods characterize spatial-temporal organization. Recurrence, intensity, and residual patterns are process summaries and are not automatic evidence for attention, strategy, difficulty, or diagnosis.
