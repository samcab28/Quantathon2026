<#
.SYNOPSIS
    Creates the project's virtual environment (if missing) and installs
    requirements.txt, then registers a project-specific Jupyter kernel.

.DESCRIPTION
    Safe to run multiple times: skips venv creation if .venv already exists,
    and pip install is idempotent. All paths are resolved from this script's
    own location ($PSScriptRoot), so it works no matter what directory you
    are in when you call it.

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
    $KernelName = "quantathon-ch2"

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

    Write-Host "Registering Jupyter kernel '$KernelName' (points at this venv, not a global default)..."
    & $VenvPython -m ipykernel install --user --name $KernelName --display-name "Python (Quantathon Ch2 venv)"
    if ($LASTEXITCODE -ne 0) { throw "ipykernel install failed (exit $LASTEXITCODE)." }

    Write-Host ""
    Write-Host "Setup complete." -ForegroundColor Green
    Write-Host "  - Run the full notebook pipeline: .\scripts\run_notebooks.ps1"
    Write-Host "  - Or do both setup + run in one call: .\scripts\run_all.ps1"
    Write-Host "  - To activate this venv interactively in your shell: . .\scripts\activate.ps1"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
