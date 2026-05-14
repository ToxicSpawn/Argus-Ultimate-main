param(
    [string]$Config = "unified_config.yaml",
    [string]$Profile = "",
    [double]$Capital = 1000.0,
    [int]$Cycles = 3,
    [switch]$Offline = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "make.ps1")) {
    throw "Run from repo root (make.ps1 not found)."
}

Write-Host "[smoke] validate config"
.\make.ps1 -Target validate -Config $Config -Profile $Profile

Write-Host "[smoke] run paper smoke"
.\make.ps1 -Target paper_smoke -Config $Config -Profile $Profile -Capital $Capital -SmokeCycles $Cycles -CycleSeconds 0 -OfflineSmoke:$Offline

Write-Host "[smoke] done"
