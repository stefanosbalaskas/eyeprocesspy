# Process-IRT 0.7 parity report

Reference: frozen R `eyeprocess` 0.11.1.

## Scope

This tranche accounts for **45 frozen exported functions** from:

- `R/049-multimodal-irt-registry.R` — 17 exports
- `R/050-process-irt-models-0-7.R` — 17 exports
- `R/054-additional-process-measurement-0-7.R` — 7 exports
- `R/057-emerging-process-irt-0-7.R` — 4 exports

It also adds explicit Python plot counterparts for four S3 plot methods and four Python article/example workflows corresponding to the frozen 0.7 process-IRT articles.

## Validation

- 45/45 frozen exports resolve to callable Python functions.
- 11/11 focused process-IRT tests pass.
- 11/11 executable IRT examples pass after adding the four 0.7 examples.
- Full package suite: 72/72 tests pass.

## Algorithmic parity

Dependency-light algorithms, registries, missingness classification, process information, distractor audits, changepoints, censored-normal calibration, channel ablation, cross-device equating, response-combination encoding and local-dependence audits are direct source translations.

Where the frozen R implementation delegates to optional `lme4`, `brms`, `nnet`, `MASS`, or `survival` engines, the current Python implementation is explicitly labelled a Python reference estimator. These functions preserve contracts and scientific semantics but are not counted as cross-engine algorithmic parity until an exact or validated backend is established.

## Numerical parity boundary

No new P4 numerical-parity credit is granted in this tranche. The sandbox does not contain `Rscript`, so deterministic R-oracle comparisons must be executed in the configured GitHub Actions R lane or another R-enabled environment.
