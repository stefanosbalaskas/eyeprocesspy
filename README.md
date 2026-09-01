<p align="center">
  <img src="https://raw.githubusercontent.com/stefanosbalaskas/eyeprocesspy/main/docs/assets/python-suite-logo.png" width="260" alt="Python Suite research packages logo">
</p>

# eyeprocesspy

**Python parity implementation of R `eyeprocess` 0.11.1.**

`eyeprocesspy` is being built from the frozen `eyeprocess` 0.11.1 release as vendor-neutral infrastructure for eye-tracking, pupillometry, biometrics, behavioural and psychometric process data.

## Frozen reference

- R package: `eyeprocess` 0.11.1
- R tarball SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- R exports: 1,182
- R S3 registrations: 435
- R articles/vignettes: 88
- R testthat files: 113
- R Stan programs: 13

## Current status

This repository is in **Milestone 0/1 development**. The full frozen API is inventoried before implementation. No placeholder functions are counted as parity. See `IMPLEMENTATION_STATUS.md` and `parity/PARITY_MATRIX.csv`.

The current validated development checkpoint implements **810/1,182 frozen R exports**. It includes the canonical schema/import/Gazepoint foundation, extensive IRT and multimodal M0–M4 measurement APIs, reliability/calibration/uncertainty infrastructure, temporal/spatial process dynamics, evidence provenance, process preflight and drift governance, temporal process-window/AOI trajectory representations, advanced pupillometry, empirical validation programmes, governed pipelines, API lifecycle governance, multiverse/sensitivity analysis, and decision manifests. No generated placeholders are counted as parity.

## Scientific commitments

The Python port preserves the R package's core commitments: semantic harmonization rather than column renaming, native time/source retention, explicit coordinate/time transformations, provenance, no silent interpolation/resampling/exclusion, and responsible interpretation of gaze, pupil and physiology. M4 remains evidence-gated.
