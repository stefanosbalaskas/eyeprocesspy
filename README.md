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

The implemented Milestone 0/1 tranches establish the canonical schema, dataset validation, provenance, mappings, timebase and coordinate-space primitives, generic import/adapter infrastructure, and first-class Gazepoint 7.x gaze, pupil, fixation and biometric ingestion against the frozen package demo exports.

## Scientific commitments

The Python port preserves the R package's core commitments: semantic harmonization rather than column renaming, native time/source retention, explicit coordinate/time transformations, provenance, no silent interpolation/resampling/exclusion, and responsible interpretation of gaze, pupil and physiology. M4 remains evidence-gated.
