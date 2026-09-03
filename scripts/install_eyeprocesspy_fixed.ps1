[CmdletBinding()]
param(
    [ValidateSet("3.11", "3.12", "3.13", "3.14")]
    [string]$PythonVersion = "",
    [switch]$WithPlots,
    [switch]$WithPsychometrics,
    [switch]$WithML,
    [switch]$WithArrow,
    [switch]$WithAllRecommended,
    [switch]$ForceReinstall,
    [switch]$UserInstall,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

function Test-PythonCandidate {
    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [string[]]$PrefixArgs = @()
    )
    try {
        $versionText = & $Exe @PrefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -ne 0) { return $false }
        $parts = $versionText.Trim().Split(".")
        if ($parts.Count -lt 2) { return $false }
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        return ($major -eq 3 -and $minor -ge 11 -and $minor -le 14)
    }
    catch {
        return $false
    }
}

function Resolve-PythonCommand {
    param([string]$RequestedVersion)

    if ($RequestedVersion) {
        if ((Get-Command py -ErrorAction SilentlyContinue) -and
            (Test-PythonCandidate -Exe "py" -PrefixArgs @("-$RequestedVersion"))) {
            return [pscustomobject]@{ Exe = "py"; PrefixArgs = @("-$RequestedVersion") }
        }
        throw "Python $RequestedVersion was requested but was not found through the Windows py launcher."
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @("3.14", "3.13", "3.12", "3.11")) {
            if (Test-PythonCandidate -Exe "py" -PrefixArgs @("-$version")) {
                return [pscustomobject]@{ Exe = "py"; PrefixArgs = @("-$version") }
            }
        }
    }

    if ((Get-Command python -ErrorAction SilentlyContinue) -and
        (Test-PythonCandidate -Exe "python")) {
        return [pscustomobject]@{ Exe = "python"; PrefixArgs = @() }
    }

    throw "Python 3.11-3.14 was not found. Install a supported Python version and rerun this script."
}

$python = Resolve-PythonCommand -RequestedVersion $PythonVersion
$script:PythonExe = $python.Exe
$script:PythonPrefixArgs = @($python.PrefixArgs)

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)

    $prefix = @($script:PythonPrefixArgs)
    Write-Host "> $script:PythonExe $($prefix -join ' ') $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $script:PythonExe @prefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        # The parenthesized expansion is intentional: `$LASTEXITCODE:` is invalid PowerShell syntax.
        throw "Python command failed with exit code $($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

$wheel = Get-ChildItem -Path $PSScriptRoot -Filter "eyeprocesspy-0.1.0-py3-none-any.whl" | Select-Object -First 1
if (-not $wheel) {
    throw "Could not find eyeprocesspy-0.1.0-py3-none-any.whl next to this installer."
}

$pythonVersionText = & $script:PythonExe @script:PythonPrefixArgs --version
Write-Host "Using $pythonVersionText" -ForegroundColor Cyan

if (-not $SkipPipUpgrade) {
    Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip")
}

$installArgs = @("-m", "pip", "install", "--upgrade")
if ($ForceReinstall) { $installArgs += "--force-reinstall" }
if ($UserInstall) { $installArgs += "--user" }
$installArgs += $wheel.FullName
Invoke-Python -Arguments $installArgs

$extras = New-Object System.Collections.Generic.List[string]
if ($WithAllRecommended -or $WithPlots) {
    $extras.Add("matplotlib>=3.9")
}
if ($WithAllRecommended -or $WithPsychometrics) {
    $extras.Add("patsy>=1.0")
    $extras.Add("statsmodels>=0.14")
    $extras.Add("girth")
    $extras.Add("catsim")
}
if ($WithAllRecommended -or $WithML) {
    $extras.Add("scikit-learn>=1.5")
}
if ($WithAllRecommended -or $WithArrow) {
    $extras.Add("pyarrow>=20")
}

if ($extras.Count -gt 0) {
    Write-Host "Installing requested optional scientific backends..." -ForegroundColor Cyan
    $extraArgs = @("-m", "pip", "install", "--upgrade")
    if ($UserInstall) { $extraArgs += "--user" }
    $extraArgs += @($extras)
    Invoke-Python -Arguments $extraArgs
}

Write-Host "Verifying eyeprocesspy..." -ForegroundColor Cyan
$verifyCode = @'
import eyeprocesspy as ep
print("eyeprocesspy", ep.__version__, "| R reference", ep.__r_reference_version__)
assert ep.__version__ == "0.1.0"
assert ep.__r_reference_version__ == "0.11.1"
print("Import verification: PASS")
'@
Invoke-Python -Arguments @("-c", $verifyCode)

$verifyScript = Join-Path $PSScriptRoot "verify_eyeprocesspy.py"
if (Test-Path -LiteralPath $verifyScript) {
    Invoke-Python -Arguments @($verifyScript)
}

Write-Host "eyeprocesspy installation completed successfully." -ForegroundColor Green
