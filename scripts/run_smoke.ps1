<#
.SYNOPSIS
    Fast end-to-end validation using configs/smoke.yaml.
#>
[CmdletBinding()]
param(
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    & (Join-Path $PSScriptRoot "setup.ps1")
}

$arguments = @(
    "-m", "src.experiments.run",
    "--config", "configs/smoke.yaml"
)
if ($RunId) {
    $arguments += @("--run-id", $RunId)
}
& $VenvPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Smoke experiment failed with exit code $LASTEXITCODE."
}
