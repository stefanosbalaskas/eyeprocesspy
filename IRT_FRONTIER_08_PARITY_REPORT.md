# IRT / process 0.8 parity checkpoint

Frozen R reference: `eyeprocess 0.11.1`.

## Scope completed in this tranche

- `R/062-context-process-structure-0-8.R`: 21/21 frozen exports.
- `R/065-frontier-gates-0-8.R`: 6/6 frozen exports.
- `R/067-advanced-sensitivity-diagnostics-0-8.R`: 8/8 frozen exports.
- `R/068-bayesian-3pl-process-diagnostics-0-8.R`: 5/5 frozen exports.
- Six corresponding frozen 0.8 vignette/article counterparts.
- Fifteen relevant S3 plotting counterparts implemented with Matplotlib.
- Six new executable examples.

## Algorithmic boundary

Dependency-light frozen algorithms and evidence contracts are source-ported: visual-context registry, block-scaled PCA mapping, descriptive k-means profiles, external-validity models, linear item-parameter seeding, frontier model gates, fold-local representation contracts, latent-class/process summaries, missingness reporting, Bayesian diagnostic flags and 3PL process-signature review rules.

The following exact frozen R optional-engine routes remain explicit Python backend boundaries rather than silent substitutes: `mirt`, `eRm`, `psychotree`, `brms`, `ranger`, and FactoMineR-specific MFA. These functions are present in the API, validate their source contracts where possible, and fail with typed backend errors when the exact R-specific estimator would otherwise be required.

No P4 numerical-parity credit is granted by Python-only tests.

## Validation

- Full source test suite after tranche: 114 passed.
- Focused 0.8 tests: 12 passed.
- Installed-wheel smoke: PASS.
- Frozen 0.8 tranche export smoke: 40/40.
- Canonical Stan resources in wheel: 13/13.
