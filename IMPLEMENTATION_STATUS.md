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
| P1 API implemented | 500 |
| P2 structural initial | 500 |
| P3 semantic initial | 500 |
| P4 cross-language numerical tested | 56 initial only; extended R oracle pending |
| P5 source-ported algorithms/gates | 448 |
| P5 Python reference/backend-different algorithms | 52 |
| Public Python `plot_*` callables currently present | 113 |
| Plot-ledger rows explicitly verified `implemented` | 51 / 341 |
| Python article counterparts complete | 40 / 88 |
| Executable `irt_*.py` examples | 27 |

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
- **43** staged multimodal M0/M2/M3/M4 measurement, simulation, evidence, recovery and gated canonical-fit exports.
- **31** legacy/core IRT, response-time, process-IRT, simulation, recovery and experimental-reference exports.

Total frozen R exports with Python callables: **500 / 1182**.

## Current validation

- Full local pytest suite: **138 passed**.
- Focused staged multimodal contract/article/signature suite: **11 passed**.
- Executable IRT example smoke suite: **27 passed**.
- Installed-wheel import and staged M4 smoke: PASS.
- Installed wheel contains **13/13** canonical Stan programs.
- Current local validation-wheel SHA-256: `4c256e6c3e2e44c1a37c4447b53be538de67f4f64c0cb90cd2b63b00dff3c74a`.
- Canonical M4 Stan MD5 remains `c5af3e5d25ff63db42c58573eb42124b`.

## Staged M0-M4 scientific boundaries

- M2 retains the frozen three-channel response/RT/gaze model contract.
- M3 treats pupil as a measurement channel with explicit nuisance/confound handling; a scalar pupil summary is not automatically a cognitive-load measure.
- M4 retains REVIEW/evidence gating for trait-conditioned Markov latent process states.
- M4 state probabilities are primary uncertainty-bearing summaries; MAP labels are secondary summaries and do not automatically denote attention, strategy, effort, guessing, misconduct, comprehension or other psychological constructs.
- Canonical M2/M3/M4 fitters preserve CmdStan-backed model identity and do not silently fall back to unrelated Python estimators.

## Important parity boundary

52 functions are deliberately marked `python_reference_differs` where the frozen R implementation depends on an R-specific optional engine or where the Python reference contract cannot yet claim algorithmic identity. Exact specialist-engine methods remain explicit gates. Extended cross-language R-oracle numerical validation remains pending because `Rscript` is unavailable in this sandbox.

The sandbox also lacks the normal `build`/`wheel` build dependencies and cannot resolve PyPI. The current installed-artifact check therefore used an offline PEP 427 validation wheel assembled directly from the pure-Python source tree. GitHub CI remains responsible for the standard PEP 517 wheel/sdist build lane.
