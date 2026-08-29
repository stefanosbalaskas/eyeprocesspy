# Test summary

- Full local pytest suite: **130 passed**.
- Focused staged M0/M2/M3/M4 tests, including article and signature smoke: **11 passed**.
- Executable `irt_*.py` example smoke suite: **26 passed**.
- Staged frozen-export smoke: **43/43**.
- Staged frozen-signature argument-name smoke: **43/43**.
- Staged multimodal plot counterparts tested with attached plot-data: **29**.
- Staged Python article counterparts smoke tested: **20**.
- Installed validation-wheel smoke: PASS.
- Packaged canonical Stan resources: **13/13**.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `e3b15ab79c2e93b846dbc7a528e8eeb8f190ddae12d9fd26a367f38729307e3f`.
- Frozen R reference: `eyeprocess 0.11.1`.
- Cross-language R-oracle numerical validation remains pending in this sandbox because `Rscript` is unavailable.
- Standard PEP 517 build could not be executed locally because build dependencies are absent and network access is disabled; the installed-artifact smoke used an offline pure-Python PEP 427 validation wheel. CI remains authoritative for normal wheel/sdist construction.
- Ruff is not available in the current sandbox; CI remains responsible for the configured Ruff lane.
