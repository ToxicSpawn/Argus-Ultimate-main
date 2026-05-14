param(
    [string]$Config = "unified_config.yaml",
    [string]$Profile = "",
    [string]$Manifest = "docs/hardware/R740_PREBUILD_MANIFEST.yaml",
    [string]$BundleRoot = "deploy/r740_bundle",
    [string]$BundleCheckOutput = "reports/infra/r740_bundle_check_latest.json",
    [string]$SuiteOutput = "reports/infra/r740_prebuild_suite_latest.json",
    [switch]$RunReadiness
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "scripts/r740_prebuild_suite.py")) {
    throw "Run from repo root (scripts/r740_prebuild_suite.py not found)."
}

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python not found." }

$argsList = @(
    "scripts/r740_prebuild_suite.py",
    "--config", $Config,
    "--manifest", $Manifest,
    "--bundle-root", $BundleRoot,
    "--bundle-check-output", $BundleCheckOutput,
    "--suite-output", $SuiteOutput
)
if ($Profile -and $Profile.Trim()) {
    $argsList += @("--profile", $Profile.Trim())
}
if ($RunReadiness) {
    $argsList += "--run-readiness"
}

if ($py -eq "py") {
    py -3 @argsList
} else {
    python @argsList
}
if ($LASTEXITCODE -ne 0) {
    throw "R740 prebuild suite failed."
}

Write-Host "[r740-prebuild] suite passed"
