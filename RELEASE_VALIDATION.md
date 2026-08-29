# Release validation checkpoint

This is a development validation checkpoint for `eyeprocesspy 0.1.0.dev0`, not a parity release.

## Frozen source

- R reference: `eyeprocess 0.11.1`
- R tarball SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- Frozen R exports: 1,182
- Frozen Stan programs: 13

## Current Python checkpoint

- Frozen exports with implemented Python callables: **500/1,182**
- P5 source-ported algorithms/contracts: **448**
- P5 Python-reference/backend-different algorithms: **52**
- Full pytest suite: **138 passed**
- Python article counterparts complete: **40/88**
- Verified plot-ledger rows: **51/341**
- Executable IRT examples: **27**

## Installed artifact validation

The current sandbox cannot download normal PEP 517 build requirements from PyPI, so the local installed-artifact gate uses an offline PEP 427 pure-Python validation wheel assembled from the package source tree. GitHub CI remains authoritative for the standard wheel/sdist build lane.

- Installed wheel import: PASS
- Legacy/core installed-package simulation/matrix smoke: PASS
- Canonical Stan resources in wheel: **13/13**
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`
- Validation-wheel SHA-256: `4c256e6c3e2e44c1a37c4447b53be538de67f4f64c0cb90cd2b63b00dff3c74a`

## Scientific boundary

P4 cross-language numerical parity is not inferred from Python-only tests. Exact R-specific optional engines remain explicit gates where appropriate. M4 remains REVIEW/evidence-gated and latent states are not automatically interpreted as psychological constructs.
