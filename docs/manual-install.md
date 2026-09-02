# Manual installation

The `0.1.0` release candidate is intentionally installable **before PyPI publication**. CI builds a wheel and source distribution, installs the wheel in a clean environment, verifies the package import, and only then publishes the distributions as a workflow artifact.

## Recommended Windows route

Download and extract the manual-install bundle, open PowerShell inside the extracted directory, and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_eyeprocesspy.ps1
```

The installer automatically finds a supported Python **3.11–3.14**, upgrades `pip`, installs the canonical wheel, and verifies both package versions.

Successful verification reports:

```text
eyeprocesspy 0.1.0 | R reference 0.11.1
eyeprocesspy installation completed successfully.
```

The repository copy of the installer is [`scripts/install_eyeprocesspy.ps1`](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/scripts/install_eyeprocesspy.ps1), and a second smoke check is available as [`scripts/verify_manual_install.py`](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/scripts/verify_manual_install.py).

## Install recommended scientific extras

To install the wheel plus plotting, psychometrics, machine-learning and Arrow support in one command:

```powershell
.\install_eyeprocesspy.ps1 -WithAllRecommended
```

Or request only the extras you need:

```powershell
.\install_eyeprocesspy.ps1 -WithPlots
.\install_eyeprocesspy.ps1 -WithPsychometrics
.\install_eyeprocesspy.ps1 -WithML
.\install_eyeprocesspy.ps1 -WithArrow
```

Flags can be combined:

```powershell
.\install_eyeprocesspy.ps1 -WithPlots -WithPsychometrics
```

To force a specific supported Python interpreter through the Windows launcher:

```powershell
.\install_eyeprocesspy.ps1 -PythonVersion 3.12 -WithAllRecommended
```

The optional installer groups correspond to commonly used backends:

| Flag | Packages installed |
| --- | --- |
| `-WithPlots` | `matplotlib>=3.9` |
| `-WithPsychometrics` | `patsy`, `statsmodels`, `girth`, `catsim` |
| `-WithML` | `scikit-learn>=1.5` |
| `-WithArrow` | `pyarrow>=20` |
| `-WithAllRecommended` | all of the above |

Stan, Bayesian, streaming and specialist interoperability backends remain opt-in and should be installed only for workflows that require them.

## Install the wheel directly

The canonical wheel name is:

```text
eyeprocesspy-0.1.0-py3-none-any.whl
```

Install it with:

```powershell
python -m pip install --upgrade .\eyeprocesspy-0.1.0-py3-none-any.whl
```

Then verify:

```powershell
python -c "import eyeprocesspy as ep; print('eyeprocesspy:', ep.__version__); print('R reference:', ep.__r_reference_version__)"
```

For a deeper smoke test:

```powershell
python .\verify_eyeprocesspy.py
```

or, from a repository checkout:

```powershell
python .\scripts\verify_manual_install.py
```

The verifier imports the package and runs the bundled deterministic benchmark audit.

## Important Windows/browser filename issue

Browsers sometimes rename a duplicate download to a filename such as:

```text
eyeprocesspy-0.1.0-py3-none-any (1).whl
```

That is **not a valid wheel filename**. `pip` can report that it is unsupported even when the wheel contents are fine. Rename it back before installation:

```powershell
Rename-Item ".\eyeprocesspy-0.1.0-py3-none-any (1).whl" "eyeprocesspy-0.1.0-py3-none-any.whl"
python -m pip install --upgrade .\eyeprocesspy-0.1.0-py3-none-any.whl
```

The packaged manual-install ZIP preserves the canonical filename and avoids this problem.

## CI-built distributions

The release-branch CI publishes a fresh artifact named:

```text
eyeprocesspy-manual-install-<commit>
```

It contains the wheel and source distribution **after** clean wheel installation/import verification. This makes the downloadable distribution part of the release evidence rather than an ad hoc binary.

## After installation: run the examples

With plotting support installed:

```powershell
python .\examples\complete_workflow.py
python .\examples\calibration_probabilistic_aoi.py
python .\examples\process_reliability.py
python .\examples\irt_diagnostics.py
```

See [Runnable examples](examples/index.md), the [Cookbook](cookbook.md), and the [visual gallery](gallery.md) for the corresponding workflows and package-generated figures.

## Why the manual bundle exists

The manual path separates **distribution testing** from **public release publication**. It lets Windows, macOS and Linux users install the exact built package while the deep-parity branch still enforces the final scientific release gates.

!!! success "Verified installation path"
    The canonical wheel has passed CI clean-install/import verification and the manual workflow has been tested with the package reporting `eyeprocesspy 0.1.0` and frozen R reference `0.11.1`.
