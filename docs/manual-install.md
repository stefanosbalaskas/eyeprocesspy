# Manual installation

The `0.1.0` release candidate can be installed directly from the project-built wheel before any public PyPI release. The manual-install route makes the exact built package easy to validate on a real workstation while the deep-parity branch remains under release gating.

!!! success "Windows verification completed"
    The hardened installer has now been exercised successfully on a real Windows installation with **Python 3.11.9**. `eyeprocesspy 0.1.0` imported successfully against frozen R reference **0.11.1**; the bundled verifier reported **PASS**; the recommended `plots`, `psychometrics`, `ml`, `arrow`, `physio`, and `docs` extras installed; and `python -m pip check` reported **No broken requirements found**.

## Windows: recommended route

Download and extract the Windows manual-install bundle. The extracted directory should contain the canonical wheel, the PowerShell installer, the verifier, and the one-click launcher.

The easiest route is to double-click:

```text
RUN_INSTALL_RECOMMENDED.cmd
```

It launches PowerShell with a process-local execution-policy bypass and runs:

```powershell
.\install_eyeprocesspy.ps1 -WithAllRecommended
```

The installer installs the bundled wheel, installs the recommended scientific extras, and verifies the package import.

## Windows: PowerShell route

Open PowerShell in the extracted directory and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_eyeprocesspy.ps1 -WithAllRecommended
```

The installer **does not require the Windows `py` launcher**. It searches for a supported Python 3.11–3.14 using, in order, the Windows launcher when available, `python`, `python3`, the active virtual or Conda environment, common Miniconda/Anaconda locations, and standard per-user python.org installation directories.

Successful verification reports the installed package version and the frozen R reference used by this build.

## If Python is installed but is not detected

Pass the interpreter explicitly:

```powershell
.\install_eyeprocesspy.ps1 `
  -PythonCommand "C:\Path\To\python.exe" `
  -WithAllRecommended
```

You may also request a specific validated Python minor version:

```powershell
.\install_eyeprocesspy.ps1 -PythonVersion 3.13 -WithAllRecommended
```

The currently validated manual-install range is Python 3.11–3.14.

## If Python is not installed

Install a supported Python version, reopen PowerShell, and rerun the installer. On Windows systems with `winget`, for example:

```powershell
winget install -e --id Python.Python.3.13
```

The corrected installer also searches standard python.org install locations, so the Windows `py` launcher is optional. Adding `python.exe` to `PATH` is convenient but is no longer required for the common per-user python.org layout.

## Installer options

`-WithAllRecommended` installs the common scientific stack needed for most package examples and documentation workflows:

| Flag | Package extra |
| --- | --- |
| `-WithPlots` | `plots` |
| `-WithPsychometrics` | `psychometrics` |
| `-WithML` | `ml` |
| `-WithArrow` | `arrow` |
| `-WithPhysio` | `physio` |
| `-WithDocs` | `docs` |
| `-WithAllRecommended` | all six groups above |

Specialized backends remain explicit rather than being silently installed:

```powershell
.\install_eyeprocesspy.ps1 -WithStan
.\install_eyeprocesspy.ps1 -WithBayes
.\install_eyeprocesspy.ps1 -WithGaze
.\install_eyeprocesspy.ps1 -WithStreaming
```

Development tooling can be installed with:

```powershell
.\install_eyeprocesspy.ps1 -WithDev
```

If the selected Python installation does not permit system-wide package writes, add:

```powershell
-UserInstall
```

Other useful switches include `-ForceReinstall` and `-SkipPipUpgrade`.

## Direct wheel fallback

Once a supported `python` command works, the core package can always be installed without the helper script:

```powershell
python -m pip install --upgrade .\eyeprocesspy-0.1.0-py3-none-any.whl
python .\verify_eyeprocesspy.py
```

If `python` is not the command name for the interpreter you intend to use, substitute its full path.

## Verification

A basic verification is:

```powershell
python -c "import eyeprocesspy as ep; print('eyeprocesspy', ep.__version__); print('R reference', ep.__r_reference_version__)"
python -m pip check
```

A successful installation should report `eyeprocesspy 0.1.0`; `pip check` should report no broken requirements.

The bundled `verify_eyeprocesspy.py` performs the deeper manual-install smoke check. From a repository checkout, the equivalent repository verifier is:

```powershell
python .\scripts\verify_manual_install.py
```

## Troubleshooting

### PowerShell parser error around an exit code

An early Windows bundle contained a PowerShell interpolation form equivalent to `$LASTEXITCODE:`. PowerShell parses the colon as part of the variable reference and fails before installation begins. The canonical installer now stores the exit code separately and interpolates it as `$($exitCode)`, eliminating that parser failure.

Use the current `scripts/install_eyeprocesspy.ps1` or a manual-install bundle produced after this fix rather than reusing an older extracted installer.

### `py` is not recognized

That is no longer a blocker. The current installer automatically tries other supported interpreter locations. Do not install the Windows launcher solely for eyeprocesspy if a supported Python interpreter is already present.

### `tqdm.exe` or another helper script is not on PATH

Some optional dependencies install command-line helpers into the per-user Python `Scripts` directory. A pip warning about that directory not being on `PATH` does **not** mean the corresponding Python library failed to install. For eyeprocesspy itself, the package verifier and `python -m pip check` are the relevant checks.

### An optional backend fails to install

The installer handles recommended extras independently. A failure in one optional group is reported without hiding the status of the core package. Retry only the failed group after resolving its platform-specific dependency.

## Repository sources

The canonical Windows installer is [`scripts/install_eyeprocesspy.ps1`](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/scripts/install_eyeprocesspy.ps1). The one-click launcher is [`scripts/RUN_INSTALL_RECOMMENDED.cmd`](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/scripts/RUN_INSTALL_RECOMMENDED.cmd), and the repository smoke verifier is [`scripts/verify_manual_install.py`](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/scripts/verify_manual_install.py).

## After installation

With the recommended extras installed, start with the runnable workflows:

```powershell
python .\examples\complete_workflow.py
python .\examples\calibration_probabilistic_aoi.py
python .\examples\process_reliability.py
python .\examples\irt_diagnostics.py
```

The documentation site also provides the runnable-example index, cookbook, visual gallery, eye-tracking and pupillometry guides, psychometrics/IRT guide, and the full article library.

## Why the manual bundle exists

The manual path separates distribution testing from public release publication. It allows the exact built package to be installed and verified on Windows while release governance, deep parity, coverage, and publication gates remain independent.
