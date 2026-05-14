param(
    [string]$Config = "unified_config.yaml",
    [string]$Profile = "restricted_live",
    [string]$Output = "reports/pre_live_audit_latest.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$cmd = @(
    "py", "-3", "scripts/pre_live_audit.py",
    "--config", $Config,
    "--output", $Output
)
if ($Profile -and $Profile.Trim().Length -gt 0) {
    $cmd += @("--profile", $Profile)
}

Write-Host "Running pre-live audit..." -ForegroundColor Cyan
Write-Host ($cmd -join " ") -ForegroundColor DarkGray
& $cmd[0] $cmd[1..($cmd.Length-1)]
exit $LASTEXITCODE

