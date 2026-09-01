# eyeprocesspy

**Vendor-neutral Python infrastructure for eye-tracking, pupillometry, biometrics, psychometrics, and multimodal process data.**

`eyeprocesspy` is the Python port of the frozen R `eyeprocess` 0.11.1 reference. The complete 1,182-function public R API is represented in Python, with cross-language differences tracked explicitly in the parity ledger rather than hidden behind substitute estimators or serialization formats.

## Release target

- Python release: `eyeprocesspy` 0.1.0
- Frozen R reference: `eyeprocess` 0.11.1
- Frozen R commit: `d867555eecae46f262843501c07074cebe1f7aa9`
- Public API: **1,182 / 1,182**
- Python versions: 3.11–3.14
- Platforms: Linux, macOS, Windows

## What is included

The package provides:

- canonical schemas, mappings, adapters, provenance, coordinate and timebase handling;
- Gazepoint and vendor-neutral import/harmonization workflows;
- gaze, AOI, trial, fixation, scanpath, pupil and process-data tooling;
- quality control, governance, uncertainty, sensitivity and negative controls;
- psychometrics, IRT, process IRT, dynamic and functional pupil/IRT workflows;
- multimodal measurement and physiology-oriented interoperability;
- validation programmes, recovery/SBC/stress evidence, reproducibility and evidence freezes;
- plotting and paper-ready evidence tables;
- storage/interoperability and controlled optional scientific backends.

## Start here

1. Read [Getting started](getting-started.md).
2. Browse the complete [API reference](reference/api.md).
3. Explore the source-ported tutorials under [Articles](articles/).
4. Review [Parity and validation](parity-and-validation.md) before making claims that depend on cross-language numerical identity.
5. Consult [Release and reproducibility](release-and-reproducibility.md) when archiving or publishing an analysis.

## Scientific boundary

A green software-validation result is not a construct-validity certificate. `eyeprocesspy` preserves the R package's explicit distinction between software behavior, measurement quality, statistical model assumptions, and substantive scientific interpretation.
