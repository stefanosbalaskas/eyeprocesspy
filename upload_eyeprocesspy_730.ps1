param(
    [string]$BundlePath = (Join-Path $PSScriptRoot "eyeprocesspy-0.1.0.dev0-through-730.bundle"),
    [string]$Repository = "https://github.com/stefanosbalaskas/eyeprocesspy.git",
    [string]$ExpectedRemoteMain = "cb3a957d013b2edb75f97c610618e64a98f02d6c"
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required but was not found on PATH."
}

if (-not (Test-Path -LiteralPath $BundlePath)) {
    throw "Git bundle not found: $BundlePath"
}

Write-Host "==> Verifying bundle"
Invoke-Git bundle verify $BundlePath

if (Get-Command gh -ErrorAction SilentlyContinue) {
    Write-Host "==> Configuring Git to use the authenticated GitHub CLI credential helper"
    & gh auth setup-git
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "gh auth setup-git did not succeed; Git may still use another configured credential helper."
    }
}

$Work = Join-Path $env:TEMP ("eyeprocesspy-manual-upload-" + [guid]::NewGuid().ToString("N"))
Write-Host "==> Restoring validated history into $Work"
Invoke-Git clone $BundlePath $Work

Push-Location $Work
try {
    Invoke-Git remote remove origin
    Invoke-Git remote add origin $Repository

    Write-Host "==> Fetching current GitHub main"
    Invoke-Git fetch origin main
    $RemoteHead = (& git rev-parse refs/remotes/origin/main).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve origin/main." }

    $LocalHead = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve local HEAD." }

    Write-Host "Remote main : $RemoteHead"
    Write-Host "Local HEAD  : $LocalHead"

    if ($RemoteHead -ne $ExpectedRemoteMain) {
        throw @"
Safety stop: GitHub main is no longer at the expected initialization commit.
Expected: $ExpectedRemoteMain
Actual:   $RemoteHead
Nothing was pushed. Inspect the remote before retrying.
"@
    }

    Write-Host "==> Pushing validated history with force-with-lease"
    Invoke-Git push "--force-with-lease=refs/heads/main:$ExpectedRemoteMain" origin HEAD:main

    $Published = (& git ls-remote origin refs/heads/main).Split("`t")[0].Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not verify published main." }
    if ($Published -ne $LocalHead) {
        throw "Post-push verification failed. Expected $LocalHead but GitHub reports $Published."
    }

    Write-Host ""
    Write-Host "SUCCESS: GitHub main now points to $Published" -ForegroundColor Green
    Write-Host "Repository: https://github.com/stefanosbalaskas/eyeprocesspy"
    Write-Host "Next: open the Actions tab and confirm the CI workflows are green."
}
finally {
    Pop-Location
}
