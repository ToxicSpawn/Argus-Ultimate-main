param(
    [string]$Config = "unified_config.yaml",
    [string]$Profile = "restricted_live_soak",
    [double]$Capital = 1000.0,
    [int]$Cycles = 0,
    [double]$CycleSeconds = 0.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

Write-Host "[restricted-live-soak] validate profile" -ForegroundColor Cyan
py -3 main.py validate --config $Config --profile $Profile
if ($LASTEXITCODE -ne 0) {
    throw "Profile validation failed."
}

Write-Host "[restricted-live-soak] starting paper soak (Ctrl+C to stop)" -ForegroundColor Cyan
py -3 main.py paper --config $Config --profile $Profile --capital $Capital --cycles $Cycles --cycle-seconds $CycleSeconds --no-multilang
exit $LASTEXITCODE
