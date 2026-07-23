$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    py -3.12 -m venv (Join-Path $root ".venv")
    & $python -m pip install -r (Join-Path $root "requirements.txt")
}

Push-Location $root
try {
    & $python main.py
}
finally {
    Pop-Location
}
