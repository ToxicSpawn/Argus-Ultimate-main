param(
    [string]$Manifest = "docs/hardware/R740_PREBUILD_MANIFEST.yaml",
    [string]$OutputRoot = "deploy/r740_bundle"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "scripts/r740_prepare_bundle.py")) {
    throw "Run from repo root (scripts/r740_prepare_bundle.py not found)."
}

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python not found." }

if ($py -eq "py") {
    py -3 scripts/r740_prepare_bundle.py --manifest $Manifest --output-root $OutputRoot
} else {
    python scripts/r740_prepare_bundle.py --manifest $Manifest --output-root $OutputRoot
}
if ($LASTEXITCODE -ne 0) {
    throw "R740 prep bundle generation failed."
}

Write-Host "[r740-prep] bundle generated"
