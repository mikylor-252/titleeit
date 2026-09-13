$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$app = Join-Path $root "backend\app.py"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run: python -m venv .venv"
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg is required but was not found on PATH."
}

Write-Host "Starting SubSync at http://localhost:5000"
& $python $app