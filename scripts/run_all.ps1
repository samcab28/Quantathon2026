<#
.SYNOPSIS
    Single entry point: sets up the environment (if needed) and runs the
    full notebook pipeline, in order. This is the reproducibility entry
    point required by the challenge submission rules.

.DESCRIPTION
    Equivalent to running, in order:
        .\scripts\setup.ps1
        .\scripts\run_notebooks.ps1
    Stops immediately if either step fails.

.EXAMPLE
    .\scripts\run_all.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

try {
    Write-Host "=== Step 1/2: setup ===" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0) { throw "setup.ps1 failed (exit $LASTEXITCODE)." }

    Write-Host ""
    Write-Host "=== Step 2/2: run notebooks ===" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "run_notebooks.ps1")
    if ($LASTEXITCODE -ne 0) { throw "run_notebooks.ps1 failed (exit $LASTEXITCODE)." }

    Write-Host ""
    Write-Host "Pipeline complete. Check results\metrics\ and results\figures\ for outputs." -ForegroundColor Green
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
