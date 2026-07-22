<#
.SYNOPSIS
    Activates the project's virtual environment in your CURRENT shell.

.DESCRIPTION
    This script must be dot-sourced (note the leading ". "), otherwise the
    activation only affects a child scope that disappears when the script
    ends and your shell will look unchanged.

.EXAMPLE
    . .\scripts\activate.ps1
#>
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ActivateScript = Join-Path $Root ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $ActivateScript)) {
    Write-Error "Virtual environment not found at $(Join-Path $Root '.venv'). Run .\scripts\setup.ps1 first."
    return
}

. $ActivateScript
Write-Host "Activated venv at $(Join-Path $Root '.venv')" -ForegroundColor Green
python --version
