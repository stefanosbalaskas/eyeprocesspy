# Reproducibility and release evidence

`eyeprocesspy` treats reproducibility as a scientific property of the workflow, not only a software-engineering convenience. The package includes provenance, deterministic benchmarks, validation evidence, software-paper evidence, parity ledgers, frozen-reference checks, and release gates.

## Record the computational identity

At minimum, record:

```python
import eyeprocesspy as ep

print(ep.__version__)
print(ep.__r_reference_version__)
```

For archival analysis, also record the exact Git commit, Python version, operating system, and relevant optional-backend versions.

## Preserve source provenance

Canonical datasets support provenance records and manifests:

```python
manifest = ep.provenance_manifest(eye)
```

Where possible, preserve:

- source filenames;
- file hashes;
- software/export versions;
- import mapping/vendor selection;
- time and coordinate transformations;
- preprocessing actions;
- exclusions;
- derived-feature definitions;
- warnings and non-reversible transformations.

## Use deterministic benchmark studies

```python
study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
data = ep.import_benchmark_study(study)
```

Bundled benchmarks provide a stable route for checking installation, transformation, numerical, and reporting workflows without requiring private research data.

For publication or package validation, deterministic fixtures make it easier to distinguish code drift from data changes.

## Distinguish four types of evidence

### 1. Unit/contract evidence

Does each function respect its documented input/output and validation contract?

### 2. Numerical/parity evidence

Does the Python implementation reproduce the frozen R reference where exact parity is scientifically appropriate?

### 3. Statistical validation evidence

Do estimators or derived measures recover known parameters, remain calibrated, and behave sensibly under stress/missingness/model violation?

### 4. Workflow/release evidence

Can a clean environment build, install, import, test, document, and audit the package reproducibly?

Passing one layer does not substitute for the others.

## Frozen-R parity

The Python release candidate is tied to frozen **eyeprocess 0.11.1**. The release workflow downloads the frozen R release, verifies its checksum, installs it, and runs an oracle smoke test.

Where Python intentionally differs because an R-specific engine or runtime cannot be faithfully reproduced, the parity ledger records the blocker/difference rather than silently using a different estimator.

## Validation programs

The package includes families for:

- simulation and parameter recovery;
- SBC-style evidence;
- stress testing;
- negative controls;
- grouped/leakage-aware validation;
- validation orchestration;
- evidence atlases;
- robustness/freeze evidence.

Use them proportionally to the scientific claim. A descriptive feature extractor does not need the same validation program as a new latent-variable estimator, but neither should rely only on a few happy-path unit tests.

## Release-candidate distribution verification

The CI build creates wheel and source distributions. The wheel is then installed in a clean virtual environment and imported before the distribution artifact is uploaded.

That artifact is the preferred manual release-candidate installer because it is tied to the tested commit.

See [Manual installation](../manual-install.md).

## Current deep-parity gate

The final release gate requires:

1. complete pytest success;
2. **100% statement coverage**;
3. **100% branch coverage**;
4. Ubuntu/macOS/Windows × Python 3.11–3.14 success;
5. Ruff success;
6. clean wheel build/install/import;
7. frozen R 0.11.1 oracle success;
8. strict documentation build.

The deep-parity workflow is intentionally red while statement or branch coverage remains below 100%. That red state must not be confused with a scientific test failure when all tests themselves pass.

## Reproducible manuscript checklist

For a methods/software paper or empirical analysis, archive or report:

- package version and commit;
- Python and backend versions;
- operating system/container details where relevant;
- data provenance and hashes if shareable;
- deterministic analysis configuration;
- preprocessing and feature definitions;
- random seeds;
- validation/evidence outputs;
- plots and tables generated from code;
- model formulas/priors/settings;
- exclusion rules;
- known parity differences or backend constraints.

## Release discipline

Do not infer readiness from a version number alone. A candidate can be installable and scientifically useful while still being withheld from PyPI because a stricter archival gate has not yet been satisfied.

That separation is deliberate: **manual-install availability is not the same as final archival release**.

[Parity and validation](../parity-and-validation.md){ .md-button }
[Release and reproducibility](../release-and-reproducibility.md){ .md-button .md-button--primary }
