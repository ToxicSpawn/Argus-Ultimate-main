param(
    [string]$DbPath = "data/unified_trades.db",
    [string]$PingHost = "1.1.1.1",
    [int]$PingCount = 100,
    [int]$PingTimeoutMs = 1000,
    [string]$ExchangeHost = "api.kraken.com",
    [string]$ExchangePath = "/0/public/Time",
    [int]$ExchangeAttempts = 8,
    [double]$ExchangeTimeoutSeconds = 5.0,
    [string]$OutputDir = "reports/benchmark",
    [switch]$Quick,
    [switch]$SkipNetworkProbes,
    [switch]$SkipSpeedtest,
    [switch]$FailOnChecks
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$cmd = @(
    "py", "-3", "scripts/network_benchmark.py",
    "--db", $DbPath,
    "--ping-host", $PingHost,
    "--ping-count", "$PingCount",
    "--ping-timeout-ms", "$PingTimeoutMs",
    "--exchange-host", $ExchangeHost,
    "--exchange-path", $ExchangePath,
    "--exchange-attempts", "$ExchangeAttempts",
    "--exchange-timeout-seconds", "$ExchangeTimeoutSeconds",
    "--output-dir", $OutputDir
)

if ($Quick) { $cmd += "--quick" }
if ($SkipNetworkProbes) { $cmd += "--skip-network-probes" }
if ($SkipSpeedtest) { $cmd += "--skip-speedtest" }
if ($FailOnChecks) { $cmd += "--fail-on-checks" }

Write-Host ("Running: {0}" -f ($cmd -join " "))
& $cmd[0] $cmd[1..($cmd.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "Network benchmark failed with exit code $LASTEXITCODE"
}

