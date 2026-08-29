# Uncertainty and calibration

This Python counterpart follows the frozen `eyeprocess` 0.11.1 measurement-intelligence workflow for separating process-measure uncertainty sources, propagating uncertainty to downstream estimands, detecting spatial calibration drift, and auditing offline recalibration.

The uncertainty budget treats calibration, AOI assignment, preprocessing, sampling, and model uncertainty as explicit components. The resulting variance shares are measurement-accounting quantities rather than psychological constructs. `propagate_process_uncertainty()` supports bootstrap and simulation propagation while retaining the declared uncertainty specification.

Offline recalibration is explicit and auditable. `detect_calibration_drift()` compares observed and reference coordinates over windows; `fit_offline_recalibration()` estimates a declared translation, affine, or polynomial transform; `apply_offline_recalibration()` writes corrected coordinates to new columns; and `audit_recalibration()` compares before/after spatial RMSE. Recalibration never silently overwrites the source coordinates.
