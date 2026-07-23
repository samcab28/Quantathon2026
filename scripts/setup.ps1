<#
.SYNOPSIS
    Creates the project's virtual environment and installs pinned dependencies.

.DESCRIPTION
    Safe to run multiple times. It does not register or mutate a user-wide
    Jupyter kernel.

.EXAMPLE
    .\scripts\setup.ps1
    From anywhere:  powershell -File "C:\path\to\Quantathon\scripts\setup.ps1"
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

try {
    # $PSScriptRoot = folder containing THIS script (scripts\), independent of
    # the caller's current directory. Project root is always its parent.
    $Root       = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $VenvDir    = Join-Path $Root ".venv"
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $Requirements = Join-Path $Root "requirements.txt"

    Write-Host "Project root: $Root"

    if (-not (Test-Path $Requirements)) {
        throw "requirements.txt not found at $Requirements"
    }

    if (-not (Test-Path $VenvPython)) {
        Write-Host "No virtual environment found. Creating one at $VenvDir ..."

        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) {
            & python -m venv $VenvDir
        } else {
            $pyCmd = Get-Command py -ErrorAction SilentlyContinue
            if (-not $pyCmd) {
                throw "Neither 'python' nor 'py' was found on PATH. Install Python 3 first."
            }
            & py -3 -m venv $VenvDir
        }

        if (-not (Test-Path $VenvPython)) {
            throw "Virtual environment creation appears to have failed: $VenvPython not found."
        }
        Write-Host "Virtual environment created."
    } else {
        Write-Host "Virtual environment already exists at $VenvDir (skipping creation)."
    }

    # PyPI can be flaky (transient read timeouts); pip's default 15s timeout
    # is too short for slower connections, especially for larger packages
    # like pytket. Use a longer timeout instead of failing the whole setup.
    $PipTimeout = 120

    Write-Host "Upgrading pip..."
    & $VenvPython -m pip install --upgrade --default-timeout $PipTimeout pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)." }

    Write-Host "Installing dependencies from $Requirements ..."
    & $VenvPython -m pip install --default-timeout $PipTimeout -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed (exit $LASTEXITCODE)." }

    Write-Host ""
    Write-Host "Setup complete." -ForegroundColor Green
    Write-Host "  - Run tests: .\.venv\Scripts\python.exe -m pytest"
    Write-Host "  - Run smoke pipeline: .\scripts\run_smoke.ps1"
    Write-Host "  - Run full pipeline: .\scripts\run_all.ps1"
    Write-Host "  - To activate this venv interactively in your shell: . .\scripts\activate.ps1"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
