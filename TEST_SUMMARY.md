# eyeprocesspy 0.1.0 test summary

## Authoritative final scientific gate

Deep parity audit #280 on release head `7271ce1baf14c5dec3f59e6c2207e727d9eda7b0`:

- **1,458 / 1,458 tests passed**.
- Statement coverage: **23,085 / 23,085 (100%)**.
- Branch coverage: **9,680 / 9,680 (100%)**.
- Combined exact coverage: **100%**.
- Coverage partial branches: only narrowly documented structurally unreachable directions; no scientific statements are excluded from the gate.
- `TEST_AND_COVERAGE_GATE=PASS`.

## Parity and corpus

- Frozen R public exports: **1,182**.
- Python public API symbols resolved: **1,182 / 1,182**.
- P4 numerical `not_started`: **0**.
- P4 `python_reference_differs` without explicit blocker: **0**.
- P6 plot `not_started`: **0**.
- Frozen article manifest rows: **88**.
- Frozen article counterparts present: **88 / 88**.
- Linked frozen articles present: **88 / 88**.
- Frozen R reference: `eyeprocess 0.11.1`.
- Frozen R tarball SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`.

## CI and artifact validation

CI #566 passed on the same scientific release head across Ubuntu, macOS, and Windows with Python 3.11–3.14. The workflow also passed:

- Ruff;
- standard wheel build;
- clean wheel installation/import;
- frozen R 0.11.1 oracle smoke.

The tag-triggered release workflow performs a final independent re-run of the tests, exact coverage gate, deep-parity release gate, documentation build, distribution build, `twine check`, clean-wheel smoke, PyPI publication, provenance attestation, and GitHub Release creation.
