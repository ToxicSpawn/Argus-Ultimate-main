# ARGUS Ultimate - One-command paper trading start (Windows)
#
# Usage:
#   .\scripts\start_paper.ps1
#   .\scripts\start_paper.ps1 -Capital 500
#   .\scripts\start_paper.ps1 -SkipClean
#
# Prerequisites:
#   - Python accessible via `py` launcher
#   - unified_config.yaml present at repo root
#   - .env file with API keys (even paper mode needs exchange connectivity)

param(
    [int]$Capital = 1000,
    [switch]$SkipClean,
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ARGUS ULTIMATE - Paper Trading Startup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clean paper state
if (-not $SkipClean) {
    Write-Host "[1/3] Cleaning paper state..." -ForegroundColor Yellow
    py -B scripts/clean_paper_state.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Clean step had issues (non-fatal)" -ForegroundColor Yellow
    }
    Write-Host ""
} else {
    Write-Host '[1/3] Skipping clean (-SkipClean)' -ForegroundColor DarkGray
}

# Step 2: Validate config
if (-not $SkipValidation) {
    Write-Host "[2/3] Validating config..." -ForegroundColor Yellow
    py -B scripts/validate_config.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FATAL: Config validation failed. Fix errors above." -ForegroundColor Red
        exit 1
    }
    Write-Host ""
} else {
    Write-Host '[2/3] Skipping validation (-SkipValidation)' -ForegroundColor DarkGray
}

# Step 3: Start paper trading
Write-Host "[3/3] Starting paper trading (capital: $Capital AUD)..." -ForegroundColor Green
Write-Host ""
py -B main.py paper --capital $Capital
