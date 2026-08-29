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
| P1 API implemented | 386 |
| P2 structural initial | 386 |
| P3 semantic initial | 386 |
| P4 cross-language numerical tested | 56 initial only; extended R oracle pending |
| P5 source-ported algorithms/gates | 360 |
| P5 Python reference algorithms differing from R optional engines | 26 |
| Explicit exported `plot_*` counterparts | 68 |
| Python article counterparts complete | 13 / 88 |
| Executable `irt_*.py` examples | 15 |

No generated placeholders are counted as implementations. P4 numerical parity is never inferred from Python-only tests.

## Implemented families

- **56** foundational/core/Gazepoint exports.
- **115** frozen 0.9 IRT exports.
- **35** measurement-intelligence exports: device linking/equivalence, item-bank Pareto selection, process DIF/fairness drift and conditional process norms.
- **35** dynamic/strategy/diffusion exports.
- **45** frozen 0.7 process-IRT registry/model/continuous/multiple-response exports.
- **63** advanced process-IRT + validation exports, plus 18 corresponding S3 plot counterparts.
- **37** requested-API completion + semantic-validation exports, plus six corresponding plot counterparts.

Total frozen R exports with Python callables: **386 / 1182**.

## Current validation

- Full local pytest suite: **96 passed**.
- Executable IRT example smoke suite: **15 passed**.
- Installed-wheel import and frozen-export smoke: PASS.
- Installed wheel contains **13/13** canonical Stan programs.
- Current wheel SHA-256: `6da3fc2a831fe2c4d5b404937e2a29b1d59b4b4bb9a485db51b88c0a998219ce`.
- Canonical M4 Stan MD5 remains `c5af3e5d25ff63db42c58573eb42124b`.

## Important parity boundary

26 functions are deliberately marked `python_reference_differs` where the frozen R implementation depends on optional R engines/mixed models or an R-specific smoother/reference implementation. Exact specialist-engine methods remain explicit gates where appropriate. Extended cross-language R-oracle numerical validation remains pending because `Rscript` is unavailable in this sandbox.
