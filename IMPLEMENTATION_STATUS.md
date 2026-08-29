# Implementation status

## Frozen scope

| Metric | Count |
|---|---:|
| R exports | 1182 |
| S3 registrations | 435 |
| R source files | 114 |
| Rd files | 969 |
| R testthat files | 113 |
| Articles/vignettes | 88 |
| Stan programs | 13 |
| Plot candidates (initial heuristic) | 341 |

## Python parity checkpoint

| Stage | Count |
|---|---:|
| P0 discovered | 1182 |
| P1 API implemented | 286 |
| P2 structural initial | 286 |
| P3 semantic initial | 286 |
| P4 cross-language numerical tested | 56 initial only; extended R oracle pending |
| P5 source-ported algorithms | 273 |
| P5 Python reference algorithms differing from R optional engines | 13 |
| Explicit Python IRT/process plot counterparts currently exported | 44 |
| Python article counterparts complete | 11 / 88 |
| Executable `irt_*.py` examples | 11 |

No generated placeholders are counted as implementations. Cross-language numerical parity is never inferred merely from a passing Python test.

## Completed implementation families

- **56** foundational/core/Gazepoint exports: schemas, provenance, timebase, coordinates, generic import/adapters and first-class Gazepoint ingestion.
- **115** frozen 0.9 IRT exports: dichotomous/polytomous models, information/scoring/diagnostics, CAT, linking/DIF, multidimensional/testlet/CDM, recovery/SBC, engine governance and joint-process contracts.
- **35** measurement-intelligence exports: device linking/equivalence, multi-objective item-bank selection, process DIF/fairness drift and conditional process norms.
- **35** dynamic/strategy/diffusion exports: dynamic IRTree, theory-constrained strategy mixtures and gaze-diffusion infrastructure.
- **45** frozen 0.7 process-IRT exports: multimodal registry/channels, joint/graded/nominal/omission/many-facet process models, changepoints, continuous-process calibration, ablation/equating, multiple-response and revisiting workflows.

Total frozen R exports with Python callables: **286 / 1182**.

## Current validation

- Full local pytest suite: **72 passed**.
- Process-IRT 0.7 focused suite: **11 passed**.
- Executable IRT example smoke suite: **11 passed**.
- All 13 canonical Stan sources remain packaged.
- Canonical M4 Stan source MD5 remains `c5af3e5d25ff63db42c58573eb42124b`.

## Important parity boundary

Functions that are direct dependency-light translations are marked `source_ported`. Thirteen functions whose R implementations rely on optional engines such as `lme4`, `brms`, `nnet`, `MASS`, or `survival` currently expose transparent Python reference estimators and are marked `python_reference_differs`; they are not claimed to be algorithmically identical to those R engines. Extended R-oracle numerical validation remains pending because `Rscript` is unavailable in this sandbox.
