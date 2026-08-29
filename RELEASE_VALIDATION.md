# Release validation checkpoint

This is a development checkpoint, not a parity release.

- Python version: `0.1.0.dev0`
- Frozen R reference: `eyeprocess 0.11.1`
- Frozen tarball SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- Frozen exports: `1182`
- Python API implemented: `469`
- Full pytest: `130 passed`
- Installed validation-wheel smoke: PASS
- Stan resources: `13/13`
- Validation-wheel SHA-256: `e3b15ab79c2e93b846dbc7a528e8eeb8f190ddae12d9fd26a367f38729307e3f`
- M4 canonical Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`
- Article parity: `39/88`
- Plot-ledger parity: `50/341`
- Public Python plot callables: `112`
- Executable IRT examples: `26`

The newly validated staged programme adds 43 frozen exports and 29 explicit plot counterparts covering generic multimodal measurement plus M2, M3 and REVIEW-gated M4. M4 state labels remain statistical latent-state summaries and are not automatic psychological constructs.

P4 numerical parity is not inferred from Python-only tests. Standard PEP 517 wheel/sdist validation remains a CI task in this sandbox because local build dependencies cannot be downloaded; the installed-artifact check used an offline pure-Python PEP 427 validation wheel.
