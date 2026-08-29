# Release validation checkpoint

Development validation checkpoint for `eyeprocesspy 0.1.0.dev0`; this is not yet a full-parity release.

## Frozen source

- R reference: `eyeprocess 0.11.1`
- R tarball SHA-256: `08058e820011f5877603c93886a31295b50e29a461f364ab01604f8c38c09c63`
- Frozen exports: 1,182
- Frozen Stan programs: 13

## Python checkpoint

- Frozen exports with Python callables: **730/1,182**
- Source-ported algorithms/contracts: **648**
- Python-reference/backend-different algorithms: **82**
- Full pytest suite: **206/206 passed in deterministic split batches**
- Article counterparts complete: **56/88**
- Verified plot-ledger rows: **163/341**
- Executable IRT examples: **44**

## Installed artifact

The sandbox cannot download standard PEP 517 build requirements. The local installed-artifact gate therefore uses an offline PEP 427 pure-Python validation wheel; GitHub CI remains authoritative for normal wheel/sdist construction.

- Installed wheel import: PASS
- Functional-pupil specification/advanced-grid installed smoke: PASS
- Process preflight/drift/window/advanced-pupil installed smoke: PASS
- Streaming validation-bundle/process-decision installed smoke: PASS
- External-engine adapter and engine-equivalence installed smoke: PASS
- Process registry/reliability and calibration-quality installed smoke: PASS
- Canonical Stan resources: **13/13**
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`
- Validation-wheel SHA-256: `08058e820011f5877603c93886a31295b50e29a461f364ab01604f8c38c09c63`

## Scientific boundary

No P4 cross-language numerical parity is inferred from Python-only tests. Pupil trajectories remain physiological measurements rather than automatic psychological constructs. Exact R-specific `mirt`/`TAM`/`brms`/`LNIRT`/`GDINA`/`OpenMx`/`diffIRT` and related adapter paths are explicit backend boundaries; the canonical Stan route retains the bundled frozen Stan source. M4 remains REVIEW/evidence-gated.
