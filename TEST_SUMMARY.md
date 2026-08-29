# Test summary

- Full local pytest suite: **167 passed**.
- Functional-pupil focused tests: **7 passed**.
- External-engine adapter focused tests: **8 passed**.
- Process reliability/calibration-quality focused tests: **9 passed**.
- Legacy/core IRT focused tests: **7 passed**.
- Executable `irt_*.py` example smoke suite: **32 passed**.
- Frozen exports with implemented Python callables: **569/1,182**.
- Plot-ledger rows explicitly verified as implemented: **61/341**.
- Python article counterparts marked complete: **45/88**.
- Installed validation-wheel smoke: PASS.
- Packaged canonical Stan resources: **13/13**.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `3689803ed8460c1256bcd4048994e5014556828a423b94fd953a19a6d543dc89`.
- Frozen R reference: `eyeprocess 0.11.1`.
- Cross-language R-oracle numerical validation remains pending because `Rscript` is unavailable in this sandbox.
- Standard PEP 517 build dependencies are unavailable locally with network access disabled; installed-artifact validation uses an offline PEP 427 pure-Python wheel. GitHub CI remains authoritative for standard wheel/sdist construction.
- Ruff is not available in the current sandbox; CI remains responsible for the configured Ruff lane.
