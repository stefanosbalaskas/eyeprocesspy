# Changelog

All notable changes to `eyeprocesspy` are documented here.

## Unreleased

### Event-detector multiverse and inference robustness

- Added an explicit vendor-neutral detector-sensitivity workflow spanning detector specifications, deterministic branch execution, temporal event matching/agreement, AOI and feature propagation, identical-model inference, stability summaries, visualization, and reproducible Markdown reporting.
- Added a planned-specification convergence denominator: failed fits, missing focal terms, unavailable external backends, and non-converged branches remain visible rather than disappearing from robustness accounting. Custom model callbacks reject duplicate coefficient terms within a detector branch.
- Added per-detector model-input accountability through `DetectorInferenceResult.input_audit`, separating propagated rows, target-AOI selection, declared quality exclusions, non-finite outcomes, model rows used, and final branch status. Missing outcomes are never converted to zero.
- Added a deterministic disclosure worked example, executable failure clinic, decision guide, troubleshooting guide, reporting template, interpretation/limitations guidance, API reference, figures, report asset, and expanded MkDocs discovery.
- Expanded the website with a detector-multiverse visual diagnostic atlas, global gallery coverage, a planned-specification denominator/accountability figure, a model-input attrition/failure figure, publication-oriented figure guidance, interpretation boundaries, and regression-protected navigation/assets.

**Validation record — 2026-09-19:** the detector implementation and regression suite passed **24/24 locally reconstructed detector tests**; the failure clinic and seeded worked example executed successfully; `py_compile` passed for the detector module/example. The merged feature commit is **not GitHub CI-certified** because exact-head Actions did not execute. Re-check Ruff delta, Python 3.11–3.14 across Linux/macOS/Windows, wheel/build, R-oracle parity smoke, documentation build/deployment, and any required status checks when Actions capacity is available.

**Website-visualization validation — 2026-09-19:** the locally reconstructed detector smoke suite passed **19/19** after restoring the exact repository contract fixture omitted from the historical sdist. Exact feature-branch structural checks confirmed every visual-atlas Markdown link and SVG target resolves, all five detector visuals are exposed, the new navigation and gallery entries are present, and regression tests protect the site surface. MkDocs/Material and Ruff are not installed in the local runtime, so a full local site build/lint was not claimed. Exact-head GitHub Actions documentation/build/deployment checks remain pending.

### Standardized spatial data quality

- Added a dedicated Data Quality plot gallery with exact seeded synthetic previews for the four-panel dashboard and irregular sampling intervals, runnable focused plot calls, interpretation boundaries, reporting wording, and API links.
- Added site-contract coverage for Data Quality navigation/assets and removed escaped-newline artifacts from the worked documentation.

- Added vendor-neutral target-referenced accuracy, RMS-S2S and population-SD precision, explicit-probability BCEA, empirical sampling interval/jitter/effective-Hz diagnostics, validity fractions, reason-aware data loss, and a canonical review-oriented quality report.
- Added metric-specific units, explicit coordinate conversion, missing-gap-safe RMS-S2S, localized timestamp diagnostics, NA-group preservation, threshold validation, provenance fingerprints, insufficiency flags, and distinct long-interval versus estimated dropped-sample counts.
- Added deterministic six-profile 9-point validation data, frozen R/Python parity fixtures, focused tests, dashboard/reporting helpers, a worked validation example, sensitivity example, methodological/reporting guides, conceptual figure, dedicated Data Quality API/site navigation, and homepage discovery.
- Quality thresholds remain study-specific review rules; no function performs automatic exclusion.

### Censored gaze-latency survival analysis

- Added a vendor-neutral survival-ready trial contract that retains valid never-inspected trials as right-censored observations while keeping incomplete/unusable gaze in explicit review states.
- Added Kaplan–Meier, Cox PH, participant-clustered Cox, Weibull/log-normal AFT, prediction, latency quantiles, PH diagnostics, sensitivity comparisons, plotting, reporting, provenance, and deterministic synthetic examples. Cox estimation delegates to `statsmodels`; parametric AFT estimation delegates to `lifelines>=0.30.3,<0.31` rather than duplicating the survival likelihood in eyeprocesspy.
- Added a first-class documentation pathway with method overview, when-to-use/not-use guidance, worked examples, visual output, reporting/limitations guidance, and dedicated API reference.
- Added cross-language contract fixture coverage with the R `eyeprocess` implementation. Python explicitly rejects latent frailty requests rather than substituting clustered standard errors.
- Added a standalone evidence-verification workflow for time to first source/evidence AOI entry plus a troubleshooting clinic covering incomplete windows, time-zero events, sparse events, PH flags, low-quality gaze, competing-event boundaries, and explicit failure reporting.
- Added a survival reproducibility/preregistration checklist and a documentation-integrity regression test that fails if required survival pages, examples, API links, or navigation entries disappear from the repository.
- Added a plot-rich website consolidation with evidence-verification Kaplan-Meier, single-event 1-KM, and censoring-audit visuals, while promoting detector-multiverse and standardized data-quality plots alongside the existing AOI robustness gallery.

## 0.1.0 — 2026-09-03

First public Python release aligned to the frozen R `eyeprocess` 0.11.1 reference.

### API parity

- Implemented **1,182 / 1,182** frozen public R exports.
- Preserved the frozen R 0.11.1 source commit and function-level parity ledger.
- Closed the P4 numerical evidence ledger with zero `not_started` rows.
- Added explicit gates for R-specific serialization, estimator backends, RNG streams, and other cases where byte-identical cross-language output is not scientifically defensible.

### Data and preprocessing

- Canonical vendor-neutral data model, mappings, adapters, schema validation, provenance, timebase and coordinate-space handling.
- Gazepoint import, pairing, downstream workflows, real-export validation, and benchmark corpus support.
- Pupil, gaze, AOI, trial, preprocessing, quality-control, and governance workflows.
- Branch-focused conformance tests cover outlier handling, interpolation, blink/fixation/saccade detection, AOI sequences, transition/entropy summaries, and feature derivation without excluding scientific code from coverage.

### Psychometrics and modelling

- IRT, process IRT, advanced IRT, dynamic IRT, functional pupil/IRT, model validation, and controlled optional-engine adapters.
- Multimodal, pupil, reliability, process-dynamics, validation, sensitivity, benchmark, negative-control, and evidence-governance families.
- Dynamic transition and functional-pupil adapters hardened for current pandas/NumPy semantics.

### Validation and reproducibility

- **1,458 tests passed** at the controlling deep-parity release gate.
- Exact package coverage: **23,085 / 23,085 statements** and **9,680 / 9,680 branches**.
- Frozen-R oracle validation.
- Full CI across Python 3.11–3.14 on Ubuntu, Windows, and macOS.
- Validation programmes, evidence manifests, benchmark/stress testing, reproducibility/provenance, software-paper evidence, validation stress/freeze, and validation atlas tooling.

### Documentation and release infrastructure

- Frozen workflow article counterparts: **88 / 88**.
- GitHub Pages documentation and article corpus.
- Published-release front page with direct PyPI, GitHub Release, and Zenodo DOI access; redesigned responsive documentation landing page with install, workflow, validation, and capability entry points.
- PEP 639 SPDX-style license metadata.
- `CITATION.cff` and `.zenodo.json` release metadata.
- PyPI Trusted Publishing, provenance attestation, and GitHub Release automation.
- Package-wide exact coverage and deep-parity release gates.

### Scientific guardrails

The release distinguishes API coverage from numerical identity. Exact R-specific engines, RDS serialization, language-specific object hashes, random-number streams, and platform timings are not fabricated. Legitimate Python-reference differences are documented in `parity/PARITY_MATRIX.csv` and require explicit conformance tests or blockers.
