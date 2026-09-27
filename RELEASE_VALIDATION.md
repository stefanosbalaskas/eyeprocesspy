# eyeprocesspy 0.2.2 release validation

Release candidate: **0.2.2**  
Frozen R scientific reference: **eyeprocess 0.11.1**

## Controlling qualification

The exact `main` scientific head immediately preceding the 0.2.2 version-only release preparation passed:

- pytest: **1,892 passed**
- statements: **36,556 / 36,556 (100%)**
- branches: **12,630 / 12,630 (100%)**
- missing statements: **0**
- missing branches: **0**
- Ruff: **pass**
- mypy: **pass**
- wheel and sdist clean-install smoke: **pass**
- frozen-R oracle: **pass**
- strict MkDocs build and asset validation: **pass**
- runtime matrix: **Ubuntu / macOS / Windows × Python 3.11–3.14: pass**

The 0.2.2 release workflow performs an independent release-candidate rerun after the version bump. A tag and publication are created only if those gates pass.

## Release artifact and documentation gates

The automated release workflow requires:

1. package/runtime/tag version consistency;
2. Ruff and mypy;
3. full tests with exact 100% statement and branch coverage;
4. zero missing coverage counters;
5. deep-parity release audit;
6. public-API documentation audit;
7. strict documentation build;
8. published-dependency audit;
9. wheel and source-distribution build plus strict metadata validation;
10. clean wheel and sdist install smoke tests;
11. SHA-256 release evidence;
12. PyPI Trusted Publishing;
13. build-provenance attestation;
14. GitHub Release creation.

## Scientific boundary

The 0.2.2 additions extend the vendor-neutral scientific core with measurement-validity and dynamic-gaze analyses and complete the static Bayesian-network robustness layer. Corrections, AOI choices, detector parameters, coordinate assumptions, missingness, and robustness branches remain explicit; failed or unsupported branches are not silently discarded.

`eyeprocesspy` does not infer byte-identical equivalence where R and Python necessarily differ. Governed cases include R-native serialization, package-specific estimators, random-number streams, object hashes, renderer-specific graphics, platform timing, and optional external backends. Such cases require an explicit parity blocker and independently tested shared scientific contract.
