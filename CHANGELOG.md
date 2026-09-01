# Changelog

All notable changes to `eyeprocesspy` are documented here.

## 0.1.0 — 2026-09-01

First public Python release aligned to the frozen R `eyeprocess` 0.11.1 reference.

### API parity

- Implemented **1,182 / 1,182** frozen public R exports.
- Preserved the frozen R 0.11.1 source commit and function-level parity ledger.
- Added explicit gates for R-specific serialization, estimator backends, RNG streams, and other cases where byte-identical cross-language output is not scientifically defensible.

### Data and preprocessing

- Canonical vendor-neutral data model, mappings, adapters, schema validation, provenance, timebase and coordinate-space handling.
- Gazepoint import, pairing, downstream workflows, real-export validation, and benchmark corpus support.
- Pupil, gaze, AOI, trial, preprocessing, quality-control, and governance workflows.

### Psychometrics and modelling

- IRT, process IRT, advanced IRT, dynamic IRT, functional pupil/IRT, model validation, and controlled optional-engine adapters.
- Multimodal, pupil, reliability, process-dynamics, validation, sensitivity, benchmark, negative-control, and evidence-governance families.
- Hardened dynamic transition masks for pandas 3 / NumPy read-only array semantics while preserving structural-zero and self-transition behavior.

### Validation and reproducibility

- Frozen-R oracle smoke testing.
- Validation programmes, evidence manifests, benchmark/stress testing, reproducibility/provenance, software-paper evidence, validation stress/freeze, and validation atlas tooling.
- Platform CI across Python 3.11–3.14 on Ubuntu, Windows, and macOS.

### Documentation and release infrastructure

- GitHub Pages documentation source and article corpus.
- PEP 639 SPDX-style license metadata.
- `CITATION.cff` and `.zenodo.json` release metadata.
- PyPI Trusted Publishing workflow and GitHub release validation.
- Package-wide coverage and deep-parity release gates.

### Scientific guardrails

The release distinguishes API coverage from numerical identity. Exact R-specific engines, RDS serialization, language-specific object hashes, random-number streams, and platform timings are not fabricated. Legitimate Python-reference differences are documented in `parity/PARITY_MATRIX.csv` and require explicit conformance tests or blockers.
