# Test summary — 0.1.0.dev0 initial tranche

- Local runtime: Python 3.13.5
- pytest: **13 passed**
- Python compileall: PASS
- Wheel build: PASS using installed setuptools/wheel without network isolation
- Wheel artifact: `eyeprocesspy-0.1.0.dev0-py3-none-any.whl`
- Installed-wheel import from isolated target directory: PASS
- Packaged Stan resources: **13/13**
- Full clean dependency installation: **not executable in this sandbox** because outbound PyPI DNS is unavailable. This is not marked PASS; GitHub CI must perform it.
- Ruff/mypy: configuration added, but local executables are unavailable in this sandbox; CI must perform them.
- R oracle execution: harness + GitHub Actions frozen-reference smoke lane added; local execution remains unavailable because `Rscript` is not installed in this sandbox. Cross-language numerical parity remains pending.
