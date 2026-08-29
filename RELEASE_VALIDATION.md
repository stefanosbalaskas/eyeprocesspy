# Release validation checkpoint

Development validation checkpoint for `eyeprocesspy 0.1.0.dev0`; this is not yet a full-parity release.

## Frozen source

- R reference: `eyeprocess 0.11.1`
- R tarball SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- Frozen exports: 1,182
- Frozen Stan programs: 13

## Python checkpoint

- Frozen exports with Python callables: **810/1,182**
- Source-ported algorithms/contracts: **724**
- Python-reference/backend-different algorithms: **86**
- Full source pytest surface: **219/219 passed in deterministic split batches**
- Article counterparts complete: **61/88**
- Verified plot-ledger rows: **172/341**
- Public Python `plot_*` callables: **235**
- Executable IRT examples: **49**

## CI portability repairs

- Development test extra includes `matplotlib>=3.9`, `patsy>=1.0`, and `statsmodels>=0.14`.
- Dynamic-IRTree `time_gap` conversion requests a writable NumPy copy for pandas 3 compatibility.
- The previously failing dynamic-IRTree, functional-pupil and legacy-model paths pass locally after these repairs.

## Installed artifact

The sandbox cannot rely on network installation of build requirements. The local artifact gate uses the repository's offline PEP 427 validation-wheel builder; GitHub CI remains authoritative for normal PEP 517 wheel/sdist construction.

- Installed-wheel import: PASS
- Packaged lifecycle registry/policy resources: **PASS (2/2 CSV resources; lifecycle registry 1,182 rows)**
- Canonical Stan resources: **13/13**
- Canonical M4 Stan MD5 expected: `c5af3e5d25ff63db42c58573eb42124b`
- Validation-wheel SHA-256: `621c74a77e7a6137701e8d0c2ca7fe27b982ced4d000f6667331b120ba80429b`

## Scientific boundary

No P4 cross-language numerical parity is inferred from Python-only tests. Exact R-specific model-engine identities remain explicit backend boundaries. Pupil trajectories remain physiological measurements rather than automatic psychological constructs, and M4 remains REVIEW/evidence-gated.
