param(
    [string]$Python = "py",
    [string]$VenvPath = ".venv"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Run from repo root (main.py not found)."
}

if (-not (Test-Path $VenvPath)) {
    & $Python -3 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtualenv at $VenvPath" }
}

$venvPy = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    throw "Virtualenv Python not found at $venvPy"
}

& $venvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

if (Test-Path "requirements-dev.txt") {
    & $venvPy -m pip install -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw "requirements-dev install failed" }
}

Write-Host "Install complete. Activate with: .\$VenvPath\Scripts\Activate.ps1"
