[CmdletBinding()]
param(
    [ValidateSet("3.11", "3.12", "3.13", "3.14")]
    [string]$PythonVersion = "",
    [string]$PythonCommand = "",
    [switch]$WithPlots,
    [switch]$WithPsychometrics,
    [switch]$WithML,
    [switch]$WithArrow,
    [switch]$WithPhysio,
    [switch]$WithDocs,
    [switch]$WithStan,
    [switch]$WithBayes,
    [switch]$WithGaze,
    [switch]$WithStreaming,
    [switch]$WithAllRecommended,
    [switch]$WithDev,
    [switch]$ForceReinstall,
    [switch]$UserInstall,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ScriptDir

function Get-PythonProbe {
    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [string[]]$PrefixArgs = @()
    )

    try {
        $raw = & $Exe @PrefixArgs -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0 -or -not $raw) {
            return $null
        }

        $versionText = [string]($raw | Select-Object -Last 1)
        $parts = $versionText.Trim().Split(".")
        if ($parts.Count -lt 2) {
            return $null
        }

        return [pscustomobject]@{
            Exe        = $Exe
            PrefixArgs = @($PrefixArgs)
            Version    = $versionText.Trim()
            Major      = [int]$parts[0]
            Minor      = [int]$parts[1]
        }
    }
    catch {
        return $null
    }
}

function Test-SupportedPython {
    param([Parameter(Mandatory=$true)]$Probe)

    if ($Probe.Major -ne 3) {
        return $false
    }
    return ($Probe.Minor -ge 11 -and $Probe.Minor -le 14)
}

function Test-RequestedVersion {
    param(
        [Parameter(Mandatory=$true)]$Probe,
        [string]$RequestedVersion
    )

    if (-not $RequestedVersion) {
        return $true
    }
    return ("{0}.{1}" -f $Probe.Major, $Probe.Minor) -eq $RequestedVersion
}

function Resolve-PythonPath {
    param([string]$RequestedCommand)

    if (-not $RequestedCommand) {
        return $null
    }

    if (Test-Path -LiteralPath $RequestedCommand -PathType Leaf) {
        return (Resolve-Path -LiteralPath $RequestedCommand).Path
    }

    $cmd = Get-Command $RequestedCommand -ErrorAction SilentlyContinue
    if ($cmd) {
        if ($cmd.Source) {
            return $cmd.Source
        }
        if ($cmd.Path) {
            return $cmd.Path
        }
    }

    return $null
}

function Resolve-Python {
    param(
        [string]$RequestedVersion,
        [string]$RequestedCommand
    )

    if ($RequestedCommand) {
        $explicitPath = Resolve-PythonPath -RequestedCommand $RequestedCommand
        if (-not $explicitPath) {
            throw "Python command/path '$RequestedCommand' was not found."
        }

        $probe = Get-PythonProbe -Exe $explicitPath
        if (-not $probe) {
            throw "The requested Python command '$RequestedCommand' could not execute Python."
        }
        if (-not (Test-SupportedPython -Probe $probe)) {
            throw "eyeprocesspy 0.1.0 manual validation supports Python 3.11-3.14. '$RequestedCommand' is Python $($probe.Version)."
        }
        if (-not (Test-RequestedVersion -Probe $probe -RequestedVersion $RequestedVersion)) {
            throw "Requested Python $RequestedVersion, but '$RequestedCommand' is Python $($probe.Version)."
        }
        return $probe
    }

    # Prefer the Windows py launcher when it exists, but do not require it.
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        $versions = @("3.14", "3.13", "3.12", "3.11")
        if ($RequestedVersion) {
            $versions = @($RequestedVersion)
        }

        foreach ($version in $versions) {
            $probe = Get-PythonProbe -Exe $pyCommand.Source -PrefixArgs @("-$version")
            if ($probe -and (Test-SupportedPython -Probe $probe) -and
                (Test-RequestedVersion -Probe $probe -RequestedVersion $RequestedVersion)) {
                return $probe
            }
        }
    }

    # PATH-based commands. This covers standard python.org installs, Conda shells,
    # many IDE terminals, and environments where the Windows py launcher is absent.
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $probe = Get-PythonProbe -Exe $cmd.Source
            if ($probe -and (Test-SupportedPython -Probe $probe) -and
                (Test-RequestedVersion -Probe $probe -RequestedVersion $RequestedVersion)) {
                return $probe
            }
        }
    }

    # Activated virtual/Conda environments.
    $candidatePaths = New-Object System.Collections.Generic.List[string]
    if ($env:VIRTUAL_ENV) {
        $candidatePaths.Add((Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))
    }
    if ($env:CONDA_PREFIX) {
        $candidatePaths.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }

    # Common per-user Conda locations.
    if ($HOME) {
        $candidatePaths.Add((Join-Path $HOME "miniconda3\python.exe"))
        $candidatePaths.Add((Join-Path $HOME "anaconda3\python.exe"))
    }

    # Common python.org per-user installations.
    if ($env:LOCALAPPDATA) {
        $pythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
        if (Test-Path -LiteralPath $pythonRoot -PathType Container) {
            $pythonDirs = Get-ChildItem -LiteralPath $pythonRoot -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending
            foreach ($dir in $pythonDirs) {
                $candidatePaths.Add((Join-Path $dir.FullName "python.exe"))
            }
        }
    }

    $seen = @{}
    foreach ($candidate in $candidatePaths) {
        if (-not $candidate -or $seen.ContainsKey($candidate)) {
            continue
        }
        $seen[$candidate] = $true

        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            continue
        }

        $probe = Get-PythonProbe -Exe $candidate
        if ($probe -and (Test-SupportedPython -Probe $probe) -and
            (Test-RequestedVersion -Probe $probe -RequestedVersion $RequestedVersion)) {
            return $probe
        }
    }

    $requestedText = ""
    if ($RequestedVersion) {
        $requestedText = " $RequestedVersion"
    }

    throw @"
No supported Python$requestedText was found.

eyeprocesspy 0.1.0 requires Python >=3.11 and this manual bundle is validated on Python 3.11-3.14.

If Python is already installed, rerun with its full path, for example:
  .\install_eyeprocesspy.ps1 -PythonCommand "C:\Path\To\python.exe" -WithAllRecommended

If Python is not installed, install Python 3.13 or another supported version, reopen PowerShell,
and run this installer again. The Windows 'py' launcher is optional.
"@
}

$Resolved = Resolve-Python -RequestedVersion $PythonVersion -RequestedCommand $PythonCommand
$script:PythonExe = $Resolved.Exe
$script:PythonPrefixArgs = @($Resolved.PrefixArgs)

function Invoke-Python {
    param([Parameter(Mandatory=$true)][string[]]$Arguments)

    $prefix = @($script:PythonPrefixArgs)
    Write-Host ("> {0} {1} {2}" -f $script:PythonExe, ($prefix -join " "), ($Arguments -join " ")) -ForegroundColor DarkGray
    & $script:PythonExe @prefix @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Python command failed with exit code $($exitCode): $($Arguments -join ' ')"
    }
}

function Ensure-Pip {
    $prefix = @($script:PythonPrefixArgs)
    & $script:PythonExe @prefix -m pip --version *> $null
    $pipExitCode = $LASTEXITCODE
    if ($pipExitCode -eq 0) {
        return
    }

    Write-Host "pip was not available; bootstrapping it with ensurepip..." -ForegroundColor Yellow
    Invoke-Python -Arguments @("-m", "ensurepip", "--upgrade")
}

$Wheel = Get-ChildItem -LiteralPath $ScriptDir -Filter "eyeprocesspy-0.1.0-py3-none-any.whl" -File |
    Select-Object -First 1
if (-not $Wheel) {
    $Wheel = Get-ChildItem -LiteralPath $ScriptDir -Filter "eyeprocesspy-*.whl" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}
if (-not $Wheel) {
    throw "No eyeprocesspy wheel was found beside this installer: $ScriptDir"
}

Write-Host ""
Write-Host "eyeprocesspy 0.1.0 manual installer" -ForegroundColor Cyan
Write-Host "Bundle:  $ScriptDir"
Write-Host "Wheel:   $($Wheel.Name)"
Write-Host "Python:  $($Resolved.Version)  [$($Resolved.Exe) $($Resolved.PrefixArgs -join ' ')]"
Write-Host ""

Ensure-Pip

if (-not $SkipPipUpgrade) {
    Write-Host "Updating pip..." -ForegroundColor Yellow
    Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip")
}

Write-Host "Installing eyeprocesspy core from the bundled wheel..." -ForegroundColor Yellow
$coreArgs = @("-m", "pip", "install", "--upgrade")
if ($ForceReinstall) {
    $coreArgs += "--force-reinstall"
}
if ($UserInstall) {
    $coreArgs += "--user"
}
$coreArgs += $Wheel.FullName
Invoke-Python -Arguments $coreArgs

$extraNames = New-Object System.Collections.Generic.List[string]

if ($WithAllRecommended -or $WithPlots)          { $extraNames.Add("plots") }
if ($WithAllRecommended -or $WithPsychometrics) { $extraNames.Add("psychometrics") }
if ($WithAllRecommended -or $WithML)             { $extraNames.Add("ml") }
if ($WithAllRecommended -or $WithArrow)          { $extraNames.Add("arrow") }
if ($WithAllRecommended -or $WithPhysio)         { $extraNames.Add("physio") }
if ($WithAllRecommended -or $WithDocs)           { $extraNames.Add("docs") }

if ($WithStan)      { $extraNames.Add("stan") }
if ($WithBayes)     { $extraNames.Add("bayes") }
if ($WithGaze)      { $extraNames.Add("gaze") }
if ($WithStreaming) { $extraNames.Add("streaming") }
if ($WithDev)       { $extraNames.Add("dev") }

$failedExtras = New-Object System.Collections.Generic.List[string]
foreach ($extra in $extraNames) {
    Write-Host "Installing optional extra: $extra" -ForegroundColor Yellow
    $requirement = "$($Wheel.FullName)[$extra]"
    $extraArgs = @("-m", "pip", "install", "--upgrade")
    if ($UserInstall) {
        $extraArgs += "--user"
    }
    $extraArgs += $requirement

    try {
        Invoke-Python -Arguments $extraArgs
    }
    catch {
        $failedExtras.Add($extra)
        Write-Warning "Optional extra '$extra' could not be installed. Core eyeprocesspy remains installed."
        Write-Warning $_.Exception.Message
    }
}

Write-Host "Running installation verification..." -ForegroundColor Yellow
$VerifyScript = Join-Path $ScriptDir "verify_eyeprocesspy.py"
if (Test-Path -LiteralPath $VerifyScript -PathType Leaf) {
    Invoke-Python -Arguments @($VerifyScript)
}
else {
    $verifyCode = "import eyeprocesspy as ep; print('eyeprocesspy', ep.__version__); print('R reference', ep.__r_reference_version__); print('API import OK')"
    Invoke-Python -Arguments @("-c", $verifyCode)
}

Write-Host ""
Write-Host "SUCCESS: eyeprocesspy core is installed and import verification passed." -ForegroundColor Green

if ($failedExtras.Count -gt 0) {
    Write-Warning ("These optional extras did not install: " + ($failedExtras -join ", "))
    Write-Host "You can retry any one later, e.g.:" -ForegroundColor Yellow
    Write-Host ('  .\install_eyeprocesspy.ps1 -WithPlots') -ForegroundColor DarkGray
}
else {
    if ($extraNames.Count -gt 0) {
        Write-Host ("Optional extras installed: " + ($extraNames -join ", ")) -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Verification command for this exact interpreter:" -ForegroundColor DarkGray
Write-Host ('  "{0}" {1} -c "import eyeprocesspy as ep; print(ep.__version__)"' -f $script:PythonExe, ($script:PythonPrefixArgs -join " ")) -ForegroundColor DarkGray
