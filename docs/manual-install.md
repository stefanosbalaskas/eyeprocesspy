# Manual installation

The release candidate is intentionally installable **before PyPI publication**. CI creates a wheel and source distribution, installs the wheel in a clean environment, verifies the package import, and then publishes the distributions as a workflow artifact.

## Recommended Windows route

Download and extract the Windows manual-install bundle, open PowerShell inside the extracted directory, and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_eyeprocesspy.ps1
```

The installer upgrades `pip`, installs the canonical wheel, and verifies both package versions:

```text
eyeprocesspy: 0.1.0
R reference: 0.11.1
```

This route has been verified successfully on **Windows with Python 3.11.9**.

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

## Important Windows/browser filename issue

Browsers sometimes rename a duplicate download to a filename such as:

```text
eyeprocesspy-0.1.0-py3-none-any (1).whl
```

That is **not a valid wheel filename**. `pip` can report:

```text
ERROR: ... is not a supported wheel on this platform.
```

The wheel itself may be valid; the renamed filename is the problem. Either use the extracted manual-install bundle, which preserves the canonical filename, or rename it back:

```powershell
Rename-Item ".\eyeprocesspy-0.1.0-py3-none-any (1).whl" "eyeprocesspy-0.1.0-py3-none-any.whl"
python -m pip install --upgrade .\eyeprocesspy-0.1.0-py3-none-any.whl
```

## Optional plotting and analysis dependencies

For plotting examples:

```powershell
python -m pip install "matplotlib>=3.9"
```

For common formula/statistical workflows:

```powershell
python -m pip install "patsy>=1.0" "statsmodels>=0.14"
```

Optional Stan, interoperability, and specialist backends should be installed only when the corresponding workflow requires them.

## CI-built distributions

The release-branch CI publishes a fresh artifact named:

```text
eyeprocesspy-manual-install-<commit>
```

It contains the wheel and source distribution **after** the wheel has passed clean installation/import verification. This makes the downloadable file part of the release evidence rather than a manually assembled binary.

## Why the manual bundle exists

The manual path separates **distribution testing** from **public release publication**. It lets Windows, macOS, and Linux users install the exact built package while the deep-parity branch is still enforcing the final scientific release gates.

!!! success "Verified installation"
    The canonical manual bundle has been installed successfully on Windows, importing `eyeprocesspy 0.1.0` and reporting frozen R reference `0.11.1`.
