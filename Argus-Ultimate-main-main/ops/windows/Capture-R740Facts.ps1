param(
    [string]$Output = "reports/infra/r740_host_facts_latest.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "scripts/r740_capture_host_facts.py")) {
    throw "Run from repo root (scripts/r740_capture_host_facts.py not found)."
}

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python not found." }

$argsList = @("scripts/r740_capture_host_facts.py", "--output", $Output)
if ($py -eq "py") {
    py -3 @argsList
} else {
    python @argsList
}
if ($LASTEXITCODE -ne 0) {
    throw "Host facts capture failed."
}

Write-Host "[r740-facts] capture complete"
