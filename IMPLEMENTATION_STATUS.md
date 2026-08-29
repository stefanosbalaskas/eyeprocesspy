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
| P1 API implemented | 810 |
| P2 structural initial | 810 |
| P3 semantic initial | 810 |
| P4 cross-language numerical tested | 56 initial only; extended R oracle pending |
| P5 source-ported algorithms/gates | 724 |
| P5 Python reference/backend-different algorithms | 86 |
| Public Python `plot_*` callables currently present | 235 |
| Plot-ledger rows explicitly verified `implemented` | 172 / 341 |
| Python article counterparts complete | 61 / 88 |
| Executable `irt_*.py` examples | 49 |

No generated placeholders are counted as implementations. P4 numerical parity is never inferred from Python-only tests.

## Implemented families

The previously validated 730-export surface remains intact: canonical schema/import/Gazepoint infrastructure; IRT, process-IRT and multimodal M0-M4 modelling; measurement intelligence; reliability/calibration/uncertainty; dynamic strategy/diffusion; process dynamics; evidence provenance; process governance/window representations; advanced pupil methods; and operational validation/decision-process features.

This checkpoint adds **80 frozen 0.9 governance exports** from the validation-programme, governed-pipeline, API-lifecycle, multiverse/sensitivity and decision-manifest families. The packaged lifecycle resources preserve the frozen 1,182-row API registry and 108-row module policy.

Total frozen R exports with Python callables: **810 / 1182**.

## Current validation

- Full source pytest surface: **219/219 passed** in deterministic split batches because a single monolithic invocation exceeds the sandbox process-time ceiling.
- Governance 0.9 focused contract/export/signature/plot suite: **8/8 passed**.
- Executable IRT example surface: **49 examples**, including five new governance examples.
- CI-portability regression for dynamic IRTree, functional pupil and legacy-model paths: **21/21 passed**; the seven previously failing/example paths also pass locally.
- Canonical M4 Stan MD5 remains `c5af3e5d25ff63db42c58573eb42124b`.
- Installed validation-wheel smoke: **PASS**, including the packaged 1,182-row lifecycle registry and both lifecycle CSV resources.
- Installed wheel contains **13/13** canonical Stan programs.
- Validation-wheel SHA-256: `621c74a77e7a6137701e8d0c2ca7fe27b982ced4d000f6667331b120ba80429b`.

## Scientific boundaries

- M2 retains the frozen response/RT/gaze contract; M3 treats pupil as a measurement channel with explicit nuisance/confound handling.
- M4 remains REVIEW/evidence-gated; latent state probabilities are uncertainty-bearing statistical summaries and are not automatic labels for attention, strategy, effort, guessing, misconduct or comprehension.
- Exact R-specific specialist engines remain explicit backend boundaries rather than silent Python substitutions.
- Four governance I/O/inventory functions are marked `python_reference_differs` where Python serialization or namespace semantics necessarily differ from R.
- Extended R-oracle P4 numerical validation remains pending; Python-only tests do not upgrade P4 status.

## CI portability

The fresh GitHub environment identified dependency declarations that were implicit in the earlier development sandbox. The development extra now includes Matplotlib, patsy and statsmodels, and dynamic-IRTree NumPy conversion explicitly requests a writable copy for pandas 3 compatibility. These repairs do not alter frozen API counts.
