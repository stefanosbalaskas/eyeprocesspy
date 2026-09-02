param(
    [ValidateSet("3.11", "3.12", "3.13", "3.14")]
    [string]$PythonVersion = "",
    [switch]$WithPlots,
    [switch]$WithPsychometrics,
    [switch]$WithML,
    [switch]$WithArrow,
    [switch]$WithAllRecommended
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Resolve-PythonCommand {
    param([string]$RequestedVersion)

    if ($RequestedVersion) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py "-$RequestedVersion" -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,14)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return [pscustomobject]@{ Exe = "py"; PrefixArgs = @("-$RequestedVersion") }
            }
        }
        throw "Python $RequestedVersion was requested but was not found through the Windows py launcher."
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @("3.14", "3.13", "3.12", "3.11")) {
            & py "-$version" -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,14)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return [pscustomobject]@{ Exe = "py"; PrefixArgs = @("-$version") }
            }
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,14)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return [pscustomobject]@{ Exe = "python"; PrefixArgs = @() }
        }
    }

    throw "Python 3.11-3.14 was not found. Install a supported Python version and rerun this script."
}

$python = Resolve-PythonCommand -RequestedVersion $PythonVersion
$pythonArgs = @($python.PrefixArgs)
$wheel = Get-ChildItem -Path $PSScriptRoot -Filter "eyeprocesspy-0.1.0-py3-none-any.whl" | Select-Object -First 1
if (-not $wheel) { throw "Could not find eyeprocesspy-0.1.0-py3-none-any.whl next to this installer." }

Write-Host "Using: $($python.Exe) $($pythonArgs -join ' ')" -ForegroundColor Cyan
& $python.Exe @pythonArgs -m pip install --upgrade pip
& $python.Exe @pythonArgs -m pip install --upgrade $wheel.FullName

$extras = New-Object System.Collections.Generic.List[string]
if ($WithAllRecommended -or $WithPlots) { $extras.Add("matplotlib>=3.9") }
if ($WithAllRecommended -or $WithPsychometrics) {
    $extras.Add("patsy>=1.0")
    $extras.Add("statsmodels>=0.14")
    $extras.Add("girth")
    $extras.Add("catsim")
}
if ($WithAllRecommended -or $WithML) { $extras.Add("scikit-learn>=1.5") }
if ($WithAllRecommended -or $WithArrow) { $extras.Add("pyarrow>=20") }

if ($extras.Count -gt 0) {
    Write-Host "Installing requested optional scientific backends..." -ForegroundColor Cyan
    & $python.Exe @pythonArgs -m pip install --upgrade @extras
}

Write-Host "Verifying eyeprocesspy..." -ForegroundColor Cyan
& $python.Exe @pythonArgs -c "import eyeprocesspy as ep; print('eyeprocesspy', ep.__version__, '| R reference', ep.__r_reference_version__)"
if ($LASTEXITCODE -ne 0) { throw "Installation verification failed." }
Write-Host "eyeprocesspy installation completed successfully." -ForegroundColor Green
