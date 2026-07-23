<#
.SYNOPSIS
    Reproducible full experiment entry point.

.DESCRIPTION
    Creates/updates the local virtual environment and executes the immutable
    full run configured in configs/full.yaml. No user-wide Jupyter kernel is
    registered.

.EXAMPLE
    .\scripts\run_all.ps1
    .\scripts\run_all.ps1 -RunId my-final-run
#>
[CmdletBinding()]
param(
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

& (Join-Path $PSScriptRoot "setup.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Environment setup failed with exit code $LASTEXITCODE."
}

$arguments = @(
    "-m", "src.experiments.run",
    "--config", "configs/full.yaml"
)
if ($RunId) {
    $arguments += @("--run-id", $RunId)
}

& $VenvPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Full experiment failed with exit code $LASTEXITCODE."
}
