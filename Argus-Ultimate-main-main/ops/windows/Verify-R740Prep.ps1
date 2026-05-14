param(
    [string]$Bundle = "",
    [string]$BundleRoot = "deploy/r740_bundle",
    [string]$Output = "reports/infra/r740_bundle_check_latest.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "scripts/r740_bundle_check.py")) {
    throw "Run from repo root (scripts/r740_bundle_check.py not found)."
}

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python not found." }

$argsList = @("scripts/r740_bundle_check.py", "--bundle-root", $BundleRoot, "--output", $Output)
if ($Bundle -and $Bundle.Trim()) {
    $argsList += @("--bundle", $Bundle)
}

if ($py -eq "py") {
    py -3 @argsList
} else {
    python @argsList
}
if ($LASTEXITCODE -ne 0) {
    throw "R740 prep bundle validation failed."
}

Write-Host "[r740-prep] bundle validation passed"
