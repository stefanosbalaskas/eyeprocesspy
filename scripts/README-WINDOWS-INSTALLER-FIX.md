# Windows installer parser fix

The manual Windows bundle generated before 2026-09-03 contained an invalid PowerShell interpolation inside an error message:

```powershell
$LASTEXITCODE:
```

PowerShell treats the colon as part of a scoped variable reference. The corrected installer uses a parenthesized expansion:

```powershell
$($LASTEXITCODE):
```

Use `scripts/install_eyeprocesspy_fixed.ps1` or a manual-install artifact generated after this fix. Windows CI should parse the installer before the package tests run so the syntax regression cannot recur.
