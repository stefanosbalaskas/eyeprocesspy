# Manual GitHub upload — validated 730-export checkpoint

This checkpoint is intended for manual upload to:

`https://github.com/stefanosbalaskas/eyeprocesspy`

## Validated checkpoint

- Frozen R reference: `eyeprocess 0.11.1`
- Frozen R tarball SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- Frozen R exports implemented: **730 / 1,182**
- Local test surface: **206 / 206 passed** in deterministic batches
- Articles complete: **56 / 88**
- Plot-ledger rows implemented: **163 / 341**
- Executable IRT examples: **44**
- Canonical Stan resources packaged: **13 / 13**
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`

## Recommended upload method

Use the Git bundle and the supplied PowerShell script. This preserves the complete local commit history.

1. Put these two files in the same directory, normally `Downloads`:
   - `eyeprocesspy-0.1.0.dev0-through-730.bundle`
   - `upload_eyeprocesspy_730.ps1`
2. Open PowerShell.
3. Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& "$HOME\Downloads\upload_eyeprocesspy_730.ps1"
```

The script is guarded. It checks that GitHub `main` is still at the known README-only initialization commit before replacing it with the validated local history. It uses `--force-with-lease`, not an unconditional force push.

## Source-only alternative

The ZIP and TAR.GZ archives contain the complete source tree at the same checkpoint. They are useful for inspection or for creating a fresh clone, but the Git bundle is preferred because it preserves the development history.

## Validation wheel

`eyeprocesspy-0.1.0.dev0-py3-none-any.whl` is an **offline validation wheel** assembled directly from the pure-Python source tree because this sandbox cannot download the standard PEP 517 build dependencies. GitHub CI remains authoritative for the normal wheel/sdist build.

Verify all downloaded artifacts against `SHA256SUMS.txt` before uploading.
