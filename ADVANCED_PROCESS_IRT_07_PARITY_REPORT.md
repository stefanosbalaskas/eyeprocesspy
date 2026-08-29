# Advanced process-IRT 0.7 parity report

Reference: frozen R `eyeprocess` 0.11.1.

## Scope

This tranche accounts for **63 frozen exports**:

- `R/051-advanced-process-irt-0-7.R`: **29 exports**
- `R/052-irt-validation-0-7.R`: **34 exports**
- `R/053-process-irt-plots-0-7.R`: **18 S3 plot counterparts** (not additional frozen exports)

## Implemented capabilities

- diagonal-Gaussian process-state HMM with forward/backward EM and occupancy summaries;
- process latent classes and explicit cognitive-diagnosis/latent-space engine gates;
- response-process sequence n-grams and TF-IDF/SVD embeddings;
- mean-sigma, mean-mean, Stocking-Lord and Haebara linking;
- process person-fit and process-adjusted DIF diagnostics;
- flexible IRF/GPIRT shape criticism with explicit non-GP status;
- dynamic-GPIRT, continuous-time IRT, Flow-MIRT and variational-IRT external-engine gates;
- process-aware item information and CAT simulation;
- recovery canonicalization, bias/RMSE/coverage/convergence/identifiability audits;
- Monte Carlo standard errors and replication planning;
- prior SBC, posterior-SBC contracts and rank-uniformity audits;
- posterior predictive discrepancy checks;
- latent-distribution/local-dependence/speededness/missingness/preprocessing stress tests;
- external and grouped holdout validation;
- calibration transfer, incremental-information and negative-control audits;
- conservative evidence grading.

## Validation

- 63/63 frozen exports resolve to callables.
- 11 focused advanced/validation tests pass.
- Full package suite at tranche freeze: **85/85 tests pass**.
- Executable IRT example smoke suite: **13/13**.
- Installed-wheel smoke: PASS.
- 13/13 canonical Stan resources are present in the installed wheel.

## Algorithmic boundary

Direct estimator-agnostic validation algorithms and external-engine gates are source-ported. Procedures that rely on R-specific mixed models, `smooth.spline`, or the exact R spline-reference implementation are labelled `python_reference_differs`. No P4 numerical parity is granted without R-oracle execution.
