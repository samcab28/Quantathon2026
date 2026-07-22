<#
.SYNOPSIS
    Executes every numbered notebook under notebooks\ (01_*, 02_*, ...) in
    order, in place, using this project's venv — the single reproducibility
    entry point for the challenge.

.DESCRIPTION
    - Auto-discovers notebooks matching notebooks\NN_*.ipynb and sorts them
      numerically, so newly added notebooks (03_quantum_kernel, 04_...) are
      picked up automatically without editing this script.
    - Always uses the venv's own python.exe and a pinned kernel name
      ("quantathon-ch2", registered by setup.ps1) instead of relying on
      whatever "python3" kernel jupyter happens to find first on this
      machine — this avoids silently running against a teammate's different
      Python/Anaconda install.
    - Stops at the first notebook that fails (either nbconvert itself
      reports a non-zero exit, or a cell error is found in the resulting
      .ipynb), so a partial/broken pipeline never looks "done".

.EXAMPLE
    .\scripts\run_notebooks.ps1
#>
[CmdletBinding()]
param(
    [int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"

try {
    $Root         = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $VenvPython   = Join-Path $Root ".venv\Scripts\python.exe"
    $NotebooksDir = Join-Path $Root "notebooks"
    $CheckScript  = Join-Path $PSScriptRoot "check_notebook_errors.py"
    $KernelName   = "quantathon-ch2"

    if (-not (Test-Path $VenvPython)) {
        throw "Virtual environment not found at $VenvPython. Run .\scripts\setup.ps1 first."
    }
    if (-not (Test-Path $NotebooksDir)) {
        throw "Notebooks directory not found at $NotebooksDir"
    }

    $notebooks = Get-ChildItem -Path $NotebooksDir -Filter "*.ipynb" |
        Where-Object { $_.Name -match '^\d+_' } |
        Sort-Object Name

    if ($notebooks.Count -eq 0) {
        Write-Warning "No numbered notebooks (NN_*.ipynb) found in $NotebooksDir"
        exit 0
    }

    Write-Host "Found $($notebooks.Count) notebook(s) to run, in order:"
    $notebooks | ForEach-Object { Write-Host "  - $($_.Name)" }
    Write-Host ""

    foreach ($nb in $notebooks) {
        Write-Host "=== Running $($nb.Name) ===" -ForegroundColor Cyan

        & $VenvPython -m jupyter nbconvert --to notebook --execute --inplace `
            --ExecutePreprocessor.kernel_name=$KernelName `
            --ExecutePreprocessor.timeout=$TimeoutSeconds `
            $nb.FullName

        if ($LASTEXITCODE -ne 0) {
            throw "$($nb.Name) failed during execution (nbconvert exit code $LASTEXITCODE)."
        }

        & $VenvPython $CheckScript $nb.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "$($nb.Name) ran but contains a cell error output. Inspect it before continuing."
        }

        Write-Host "=== OK: $($nb.Name) ===" -ForegroundColor Green
        Write-Host ""
    }

    Write-Host "All $($notebooks.Count) notebook(s) executed successfully." -ForegroundColor Green
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
