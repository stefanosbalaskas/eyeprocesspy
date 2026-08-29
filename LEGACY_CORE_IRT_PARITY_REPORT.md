# Legacy/core IRT parity checkpoint

Frozen R reference: `eyeprocess 0.11.1`.

## Scope completed in this tranche

- `R/013-models.R`: 15/15 frozen exports.
- `R/014-simulation.R`: 4/4 frozen exports.
- `R/016-advanced-experimental.R`: 12/12 frozen exports.
- `plot.eye_parameter_recovery` S3 counterpart implemented as `plot_eye_parameter_recovery()`.
- `vignettes/psychometric-process-models.Rmd` has a Python counterpart at `docs/articles/psychometric-process-models.md`.
- Executable example: `examples/irt_legacy_core.py`.

Total: **31/31 frozen exports** in this tranche.

## Implemented contracts

The tranche includes response and response-time matrix construction/alignment; generic model-data widening; Rasch-GLM, explanatory IRT and two-stage accuracy/RT reference paths; logistic DIF; shared-process PCA factors; item/person extraction; fit statistics; joint-process reference fitting; dynamic AOI transition modelling; eye/process simulation; parameter recovery and power simulation; process-IRT specification and informed wrappers; process diagnostics; functional pupil basis features; descriptive strategy mixtures; EZ diffusion; gaze-weighted choice; process-missingness modelling and sensitivity analysis.

## Algorithmic boundary

Sixteen functions are recorded as source-ported algorithms/contracts. Fifteen functions are conservatively recorded as `python_reference_differs` where the frozen R implementation uses R-specific engines (`mirt`, `TAM`, `LNIRT`, `lme4`, `brms`) or where the Python reference uses a different statistical/numerical backend (statsmodels, scikit-learn or Patsy natural splines).

Exact R-only engines remain explicit typed backend gates; they are never silently replaced while claiming R-engine identity.

No P4 numerical-parity credit is granted from Python-only tests.

## Validation

- Legacy/core focused tests: **7/7 passed**.
- Frozen export smoke: **31/31**.
- Frozen argument-name signature smoke: **31/31**.
- Full package suite after tranche: **138/138 passed**.
- Installed validation-wheel smoke: PASS.
- Canonical Stan resources in installed wheel: **13/13**.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `4c256e6c3e2e44c1a37c4447b53be538de67f4f64c0cb90cd2b63b00dff3c74a`.
