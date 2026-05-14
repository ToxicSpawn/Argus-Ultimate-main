param(
    [string]$PaperCommand = "py -3 main.py paper --capital 1000 --cycles 0 --cycle-seconds 10 --no-multilang",
    [int]$RestartDelaySeconds = 5,
    [switch]$Once
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $repoRoot "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -Path $logsDir -ItemType Directory -Force | Out-Null
}

$statsLog = Join-Path $logsDir "soak_runtime_stats.jsonl"
$textLog = Join-Path $logsDir "soak_runner.log"
$restartCount = 0

while ($true) {
    $start = Get-Date
    $startUtc = (Get-Date).ToUniversalTime().ToString("o")
    $startMsg = "[{0}] soak start restart_count={1} command=""{2}""" -f $startUtc, $restartCount, $PaperCommand
    Write-Host $startMsg
    Add-Content -Path $textLog -Value $startMsg -Encoding UTF8

    $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c $PaperCommand" -WorkingDirectory $repoRoot -NoNewWindow -PassThru -Wait
    $end = Get-Date
    $uptime = [Math]::Round((New-TimeSpan -Start $start -End $end).TotalSeconds, 3)
    $exitCode = [int]$proc.ExitCode
    $record = [ordered]@{
        ts_utc = $end.ToUniversalTime().ToString("o")
        command = $PaperCommand
        exit_code = $exitCode
        uptime_seconds = $uptime
        restart_count = $restartCount
        restart_delay_seconds = [int]$RestartDelaySeconds
    }
    ($record | ConvertTo-Json -Compress) | Add-Content -Path $statsLog -Encoding UTF8
    $endMsg = "[{0}] soak stop exit_code={1} uptime_seconds={2}" -f $record.ts_utc, $exitCode, $uptime
    Write-Host $endMsg
    Add-Content -Path $textLog -Value $endMsg -Encoding UTF8

    if ($Once) {
        break
    }

    $restartCount += 1
    Write-Host ("[{0}] restart scheduled in {1}s" -f ((Get-Date).ToUniversalTime().ToString("o"), $RestartDelaySeconds))
    Start-Sleep -Seconds $RestartDelaySeconds
}
