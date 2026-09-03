# eyeprocesspy 0.1.0 release validation

Authoritative release checkpoint for `eyeprocesspy 0.1.0`, validated against frozen R `eyeprocess 0.11.1`.

## Frozen scientific reference

- R reference: `eyeprocess 0.11.1`
- Frozen R commit: `d867555eecae46f262843501c07074cebe1f7aa9`
- R tarball SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- Frozen public exports: **1,182**
- Frozen articles/vignettes: **88**
- Frozen Stan programs: **13**

## Final deep-parity evidence

Controlling release head before conflict-resolution metadata merge: `7271ce1baf14c5dec3f59e6c2207e727d9eda7b0`.

GitHub Actions Deep parity audit **#280** (`33747210010`) completed successfully on 2026-09-03:

- pytest: **1,458 passed**
- statements: **23,085 / 23,085 (100%)**
- branches: **9,680 / 9,680 (100%)**
- missing statements: **0**
- missing branches: **0**
- exact gate: `TEST_AND_COVERAGE_GATE=PASS`
- P4 numerical `not_started`: **0**
- P4 `python_reference_differs` without blocker: **0**
- P6 plot `not_started`: **0**
- frozen article manifest: **88 / 88 present**
- linked frozen articles: **88 / 88 present**
- public API symbols resolved: **1,182 / 1,182**

Coverage artifact: `deep-parity-coverage`, artifact ID `9890342699`.

## Cross-platform CI

GitHub Actions CI **#566** (`33747210046`) completed successfully on the same scientific release head.

The declared matrix covered:

- Ubuntu × Python 3.11, 3.12, 3.13, 3.14
- macOS × Python 3.11, 3.12, 3.13, 3.14
- Windows × Python 3.11, 3.12, 3.13, 3.14
- Ruff
- wheel build
- clean wheel installation/import
- frozen R 0.11.1 oracle

The clean wheel and frozen-R oracle jobs passed. The Windows-specific EDF tempfile regression that previously failed was repaired without altering scientific production behavior.

## Release artifact and documentation gates

The release workflow re-runs the scientific gate from the tag, then performs:

1. Ruff;
2. full tests with 100% branch coverage;
3. deep-parity release-gate audit;
4. public API documentation audit;
5. strict MkDocs build;
6. PEP 517 distribution build;
7. `twine check`;
8. clean wheel install/import smoke;
9. PyPI Trusted Publishing;
10. provenance attestation;
11. GitHub Release creation.

## Scientific boundary

`eyeprocesspy` does not infer byte-identical equivalence where R and Python necessarily differ. Governed cases include R-native serialization, package-specific estimators, random-number streams, object hashes, renderer-specific graphics, platform timing, and optional external backends. Such cases require an explicit parity blocker and independently tested shared scientific contract.

Pupil, gaze, biometric, and process measures remain measurements rather than automatic psychological constructs or causal explanations.
