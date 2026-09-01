# eyeprocesspy implementation and release status

## Frozen reference

- R package: `eyeprocess` 0.11.1
- Frozen R commit: `d867555eecae46f262843501c07074cebe1f7aa9`
- Frozen public exports: **1,182**
- Frozen articles/vignettes: 88
- Frozen testthat files: 113
- Frozen Stan programs: 13

## API milestone

The public API surface is complete:

- `p1_api`: **1,182 / 1,182 implemented**
- public API remaining: **0**
- final API parity commit: `d1d38d6db8cb49ca6ec47b610b528422946a55be`
- hosted CI run: `33478160191` — green on the frozen API checkpoint

The hosted API-freeze matrix passed the frozen-R oracle smoke test, wheel build/install/import, Ruff, and pytest on Ubuntu, Windows, and macOS with Python 3.11–3.14.

## Current release phase

Branch: `release/0.1.0-deep-parity`

The release phase audits and closes the remaining evidence behind the API surface rather than treating callable-name parity as sufficient. Release is gated on all of the following:

1. no `p4_numerical == not_started` rows;
2. every `python_reference_differs` row has an explicit reason and conformance test;
3. semantic/algorithmic edge-case regression suites are green;
4. plot functions have stable data-contract tests and backend-safe render smoke tests;
5. the documentation/article corpus builds successfully;
6. package-wide statement and branch coverage reach the declared 100% release gate;
7. source distribution and wheel build, install, import, and `twine check` cleanly;
8. the frozen-R oracle and Python 3.11–3.14 cross-platform CI matrix remain green;
9. GitHub Pages documentation deploys successfully;
10. GitHub release, PyPI publication, and Zenodo archival are verified after publication.

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

Target public release: **eyeprocesspy 0.1.0**.

The version identifies the first public Python release; the frozen scientific source reference remains R `eyeprocess` 0.11.1 and is reported separately as `eyeprocesspy.__r_reference_version__`.
