# eyeprocesspy implementation and release status

## Frozen reference

- R package: `eyeprocess` 0.11.1
- Frozen R commit: `d867555eecae46f262843501c07074cebe1f7aa9`
- Frozen public exports: **1,182**
- Frozen articles/vignettes: **88**
- Frozen testthat files: **113**
- Frozen Stan programs: **13**

## Deep-parity status

The first public Python release surface is complete:

- `p1_api`: **1,182 / 1,182 implemented**
- public API remaining: **0**
- P4 numerical `not_started`: **0**
- P4 reference-difference rows without blocker: **0**
- P6 plot `not_started`: **0**
- frozen workflow articles: **88 / 88 present and linked**
- final deep-parity suite: **1,458 tests passed**
- statement coverage: **23,085 / 23,085 (100%)**
- branch coverage: **9,680 / 9,680 (100%)**

The authoritative pre-release scientific checkpoint is Deep parity audit #280 on `7271ce1baf14c5dec3f59e6c2207e727d9eda7b0`. CI #566 passed on the same head across Ubuntu, Windows, and macOS with Python 3.11–3.14, together with the clean wheel and frozen-R oracle gates.

## Cross-language policy

`eyeprocesspy` does not fabricate equality where R and Python cannot be byte-identical. Explicitly governed cases include:

- native RDS serialization;
- R package/namespace-specific estimators;
- R and NumPy random-number streams;
- language-specific object serialization/hashes;
- wall-clock timings and memory estimates;
- renderer-specific graphics pixels;
- optional external backends that do not have the exact same implementation contract.

Those rows may be marked `python_reference_differs` only when the parity matrix contains a concrete blocker and the Python behavior is independently tested against the shared scientific contract.

## Release target

Public release: **eyeprocesspy 0.1.0**.

The Python version identifies the first public release; the frozen scientific source reference remains R `eyeprocess` 0.11.1 and is reported separately as `eyeprocesspy.__r_reference_version__`.

The tag-triggered release workflow validates the source again before PyPI publication, provenance attestation, and GitHub Release creation.
