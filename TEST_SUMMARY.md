# Test summary

- Full local pytest suite: **138 passed**.
- Legacy/core IRT focused tests: **7 passed**.
- Legacy/core frozen-export smoke: **31/31**.
- Legacy/core frozen-signature argument-name smoke: **31/31**.
- Executable `irt_*.py` example smoke suite: **27 passed**.
- Plot-ledger rows explicitly verified as implemented: **51/341**.
- Python article counterparts marked complete: **40/88**.
- Installed validation-wheel smoke: PASS.
- Packaged canonical Stan resources: **13/13**.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `4c256e6c3e2e44c1a37c4447b53be538de67f4f64c0cb90cd2b63b00dff3c74a`.
- Frozen R reference: `eyeprocess 0.11.1`.
- Cross-language R-oracle numerical validation remains pending in this sandbox because `Rscript` is unavailable.
- Standard PEP 517 build dependencies are unavailable locally with network access disabled; the installed-artifact gate therefore uses an offline PEP 427 pure-Python validation wheel. GitHub CI remains authoritative for normal wheel/sdist construction.
- Ruff is not available in the current sandbox; CI remains responsible for the configured Ruff lane.
