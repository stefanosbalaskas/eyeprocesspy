# Release validation checkpoint

Development validation checkpoint for `eyeprocesspy 0.1.0.dev0`; this is not yet a full-parity release.

## Frozen source

- R reference: `eyeprocess 0.11.1`
- R tarball SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- Frozen exports: 1,182
- Frozen Stan programs: 13

## Python checkpoint

- Frozen exports with Python callables: **539/1,182**
- Source-ported algorithms/contracts: **470**
- Python-reference/backend-different algorithms: **69**
- Full pytest suite: **156 passed**
- Article counterparts complete: **43/88**
- Verified plot-ledger rows: **55/341**
- Executable IRT examples: **30**

## Installed artifact

The sandbox cannot download standard PEP 517 build requirements. The local installed-artifact gate therefore uses an offline PEP 427 pure-Python validation wheel; GitHub CI remains authoritative for normal wheel/sdist construction.

- Installed wheel import: PASS
- Functional-pupil specification/advanced-grid installed smoke: PASS
- External-engine adapter and engine-equivalence installed smoke: PASS
- Canonical Stan resources: **13/13**
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`
- Validation-wheel SHA-256: `0ae7c56fa006c798e13f8adcf0f000aed57d5193da1fd29377ace8894d5a5365`

## Scientific boundary

No P4 cross-language numerical parity is inferred from Python-only tests. Pupil trajectories remain physiological measurements rather than automatic psychological constructs. Exact R-specific `mirt`/`TAM`/`brms`/`LNIRT`/`GDINA`/`OpenMx`/`diffIRT` and related adapter paths are explicit backend boundaries; the canonical Stan route retains the bundled frozen Stan source. M4 remains REVIEW/evidence-gated.
