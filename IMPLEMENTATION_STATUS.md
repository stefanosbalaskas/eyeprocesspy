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
| P1 API implemented | 426 |
| P2 structural initial | 426 |
| P3 semantic initial | 426 |
| P4 cross-language numerical tested | 56 initial only; extended R oracle pending |
| P5 source-ported algorithms/gates | 389 |
| P5 Python reference/backend-different algorithms | 37 |
| Public Python `plot_*` callables currently present | 83 |
| Plot-ledger rows explicitly verified `implemented` | 21 / 341 |
| Python article counterparts complete | 19 / 88 |
| Executable `irt_*.py` examples | 21 |

No generated placeholders are counted as implementations. P4 numerical parity is never inferred from Python-only tests.

## Implemented families

- **56** foundational/core/Gazepoint exports.
- **115** frozen 0.9 IRT exports.
- **35** measurement-intelligence exports: device linking/equivalence, item-bank Pareto selection, process DIF/fairness drift and conditional process norms.
- **35** dynamic/strategy/diffusion exports.
- **45** frozen 0.7 process-IRT registry/model/continuous/multiple-response exports.
- **63** advanced process-IRT + validation exports.
- **37** requested-API completion + semantic-validation exports.
- **40** frozen 0.8 context/frontier/sensitivity/Bayesian-3PL exports.

Total frozen R exports with Python callables: **426 / 1182**.

## Current validation

- Full local pytest suite: **114 passed**.
- Executable IRT example smoke suite: **21 passed**.
- Installed-wheel import and 0.8 frozen-export smoke: PASS.
- Installed wheel contains **13/13** canonical Stan programs.
- Current wheel SHA-256: `ebf9d6938bd432f3d0a5c89086b58a3d5de85c2e25cf1eadeb78e3a3f695098d`.
- Canonical M4 Stan MD5 remains `c5af3e5d25ff63db42c58573eb42124b`.

## Important parity boundary

37 functions are deliberately marked `python_reference_differs` where the frozen R implementation depends on an R-specific optional engine or where the Python reference contract cannot yet claim algorithmic identity. Exact specialist-engine methods remain explicit gates. Extended cross-language R-oracle numerical validation remains pending because `Rscript` is unavailable in this sandbox.
