# Changelog

All notable changes to `eyeprocesspy` are documented here.

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
