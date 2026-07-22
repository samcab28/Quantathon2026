<#
.SYNOPSIS
    Deletes every generated artifact so the pipeline can be run again from
    a clean slate, as many times as needed.

.DESCRIPTION
    Removes (keeping each folder's README.md):
      - data\processed\*        (imputed/scaled/balanced train-test splits)
      - data\quantum_subset\*   (16/32/64-sample subsets for Part 3)
      - results\metrics\*       (classical_baseline.json, classical_optuna.json, ...)
      - results\figures\*       (confusion matrices, decision boundaries, ...)
    and clears all cell outputs/execution counts in notebooks\*.ipynb
    (via `jupyter nbconvert --clear-output`), so the notebooks look
    unexecuted too.

    Never touches:
      - data\raw\water_potability.csv (manually downloaded from Kaggle —
        this script cannot re-download it, so it is never deleted)
      - src\, notebooks\*.ipynb SOURCE CODE, requirements.txt, README.md files
      - .venv\ (unless -RemoveVenv is passed explicitly)

.PARAMETER Force
    Skip the confirmation prompt.

.PARAMETER RemoveVenv
    Also delete .venv\ entirely (forces a full reinstall next time
    scripts\setup.ps1 runs). Off by default because recreating the venv is
    slow and needs network access.

.EXAMPLE
    .\scripts\reset.ps1
    .\scripts\reset.ps1 -Force
    .\scripts\reset.ps1 -Force -RemoveVenv
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$RemoveVenv
)

$ErrorActionPreference = "Stop"

try {
    $Root       = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $VenvDir    = Join-Path $Root ".venv"
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $NotebooksDir = Join-Path $Root "notebooks"

    $TargetsToClear = @(
        (Join-Path $Root "data\processed"),
        (Join-Path $Root "data\quantum_subset"),
        (Join-Path $Root "results\metrics"),
        (Join-Path $Root "results\figures")
    )

    Write-Host "This will permanently delete generated files under:" -ForegroundColor Yellow
    foreach ($t in $TargetsToClear) { Write-Host "  $t" }
    Write-Host "  (each folder's README.md is kept)"
    Write-Host "Notebook cell outputs under $NotebooksDir will also be cleared."
    if ($RemoveVenv) {
        Write-Host "  .venv\ will ALSO be removed entirely (-RemoveVenv)." -ForegroundColor Yellow
    }
    Write-Host "data\raw\water_potability.csv is NEVER touched by this script."

    if (-not $Force) {
        $confirm = Read-Host "Type 'yes' to continue"
        if ($confirm -ne "yes") {
            Write-Host "Aborted. Nothing was deleted."
            exit 0
        }
    }

    foreach ($dir in $TargetsToClear) {
        if (-not (Test-Path $dir)) {
            Write-Host "Skipping (does not exist): $dir"
            continue
        }
        Get-ChildItem -LiteralPath $dir -File |
            Where-Object { $_.Name -ne "README.md" } |
            ForEach-Object {
                Write-Host "Removing $($_.FullName)"
                Remove-Item -LiteralPath $_.FullName -Force
            }
    }

    if (Test-Path $VenvPython) {
        if (Test-Path $NotebooksDir) {
            Get-ChildItem -LiteralPath $NotebooksDir -Filter "*.ipynb" | ForEach-Object {
                Write-Host "Clearing outputs in $($_.Name)"
                & $VenvPython -m jupyter nbconvert --clear-output --inplace $_.FullName
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Failed to clear outputs for $($_.Name) (exit $LASTEXITCODE)"
                }
            }
        }
    } else {
        Write-Warning "Virtual environment not found at $VenvPython; skipped clearing notebook outputs. Run .\scripts\setup.ps1 first if you need that too."
    }

    if ($RemoveVenv -and (Test-Path $VenvDir)) {
        Write-Host "Removing virtual environment at $VenvDir"
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }

    Write-Host ""
    Write-Host "Reset complete." -ForegroundColor Green
    Write-Host "  Run .\scripts\run_all.ps1 (or setup.ps1 + run_notebooks.ps1) to regenerate everything."
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
