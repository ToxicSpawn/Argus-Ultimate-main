# ARGUS Ultimate — Start with Self-Healing Watchdog
# Starts Argus paper trading with the watchdog monitoring it.
#
# Usage:
#   .\scripts\start_with_watchdog.ps1                  # paper mode (default)
#   .\scripts\start_with_watchdog.ps1 -Mode live       # live mode
#   .\scripts\start_with_watchdog.ps1 -Interval 60     # custom check interval

param(
    [string]$Mode = "paper",
    [int]$Interval = 30,
    [string]$HealthUrl = "http://localhost:8080/health",
    [int]$MaxFailures = 3
)

$ErrorActionPreference = "Stop"

# Navigate to project root
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $ProjectRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ARGUS Ultimate — Self-Healing Watchdog"     -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Mode:           $Mode"
Write-Host "  Check interval: ${Interval}s"
Write-Host "  Health URL:     $HealthUrl"
Write-Host "  Max failures:   $MaxFailures"
Write-Host ""

# Ensure data directory exists
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

# Check Python availability
try {
    $pyVersion = & py --version 2>&1
    Write-Host "  Python: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: 'py' not found. Install Python." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting watchdog..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# Start the watchdog (it will auto-start Argus if not running)
& py -B -m ops.watchdog `
    --mode $Mode `
    --check-interval $Interval `
    --health-url $HealthUrl `
    --max-failures $MaxFailures `
    --project-root $ProjectRoot
