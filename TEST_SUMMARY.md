# Test summary

- Full source pytest surface: **219/219 passed in deterministic split batches**.
- Governance 0.9 focused tests: **8/8 passed**.
- CI portability regression (dynamic IRTree + functional pupil + legacy models): **21/21 passed**.
- Seven previously failing CI/example paths: **7/7 passed locally**.
- Executable `irt_*.py` examples present: **49**.
- Frozen exports with implemented Python callables: **810/1,182**.
- Source-ported algorithms/contracts: **724**.
- Python-reference/backend-different algorithms: **86**.
- Public Python `plot_*` callables: **235**.
- Plot-ledger rows explicitly verified as implemented: **172/341**.
- Python article counterparts marked complete: **61/88**.
- Installed validation-wheel governance/resource smoke: **PASS**.
- Packaged canonical Stan resources: **13/13**.
- Validation-wheel SHA-256: `621c74a77e7a6137701e8d0c2ca7fe27b982ced4d000f6667331b120ba80429b`.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Frozen R reference: `eyeprocess 0.11.1`, SHA-256 `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`.
- Cross-language R-oracle numerical validation remains pending for the expanded surface.
- Standard PEP 517 build remains the GitHub CI lane; the sandbox uses the repository's offline PEP 427 validation-wheel builder for installed-artifact smoke.
