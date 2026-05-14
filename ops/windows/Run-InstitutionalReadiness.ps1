param(
    [string]$Config = "unified_config.yaml",
    [string]$Profile = "",
    [string]$Manifest = "docs/institutional/evidence_manifest.json",
    [string]$Output = "reports/institutional_readiness_latest.json",
    [switch]$AllowManualUnverified,
    [switch]$SkipPreLive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "make.ps1")) {
    throw "Run from repo root (make.ps1 not found)."
}

Write-Host "[institutional] running readiness gate"
$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py -3" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python not found." }

$argsList = @("scripts/institutional_readiness_check.py", "--config", $Config, "--manifest", $Manifest, "--output", $Output)
if ($Profile -and $Profile.Trim()) { $argsList += @("--profile", $Profile) }
if ($AllowManualUnverified) { $argsList += "--allow-manual-unverified" }
if ($SkipPreLive) { $argsList += "--skip-pre-live" }

if ($py -eq "py -3") {
    py -3 @argsList
} else {
    python @argsList
}
if ($LASTEXITCODE -ne 0) {
    throw "Institutional readiness gate failed."
}

Write-Host "[institutional] done"
