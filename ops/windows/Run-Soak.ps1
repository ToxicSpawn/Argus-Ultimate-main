param(
    [string]$Config = "unified_config.yaml",
    [string]$Profile = "",
    [double]$Capital = 1000.0,
    [int]$Cycles = 1800,
    [double]$CycleSeconds = 1.0,
    [int]$MaxRestarts = 3,
    [switch]$Offline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "make.ps1")) {
    throw "Run from repo root (make.ps1 not found)."
}

New-Item -ItemType Directory -Path "logs" -Force | Out-Null
New-Item -ItemType Directory -Path "reports" -Force | Out-Null
$soakStartTs = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

if (Test-Path "logs\argus_production.log") {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Move-Item "logs\argus_production.log" "logs\argus_production_$stamp.log" -Force
}

$attempt = 0
$ok = $false
while ($attempt -le $MaxRestarts -and -not $ok) {
    $attempt += 1
    Write-Host ("[soak] attempt {0}/{1}" -f $attempt, ($MaxRestarts + 1))
    try {
        .\make.ps1 -Target paper_soak -Config $Config -Profile $Profile -Capital $Capital -SoakCycles $Cycles -CycleSeconds $CycleSeconds -OfflineSmoke:$Offline
        $ok = $true
    }
    catch {
        Write-Warning ("[soak] attempt failed: {0}" -f $_.Exception.Message)
        Start-Sleep -Seconds 5
    }
}

if (-not $ok) {
    throw "Soak failed after $($MaxRestarts + 1) attempts."
}

Write-Host "[soak] generating daily report"
py -3 scripts/daily_report.py --db data/unified_trades.db --output-dir reports
if ($LASTEXITCODE -ne 0) {
    throw "Daily report generation failed"
}

Write-Host "[soak] evaluating promotion gate"
if ($Profile -and $Profile.Trim()) {
    py -3 scripts/soak_gate.py --config $Config --profile $Profile --db data/unified_trades.db --start-ts $soakStartTs --output-dir reports
} else {
    py -3 scripts/soak_gate.py --config $Config --db data/unified_trades.db --start-ts $soakStartTs --output-dir reports
}
if ($LASTEXITCODE -ne 0) {
    throw "Soak promotion gate failed"
}

Write-Host "[soak] done"
