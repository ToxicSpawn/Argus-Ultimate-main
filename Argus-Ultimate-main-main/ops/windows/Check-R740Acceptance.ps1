param(
    [string]$Spec = "docs/hardware/R740_ACCEPTANCE_SPEC.yaml",
    [string]$Facts = "docs/hardware/R740_HOST_FACTS_SAMPLE.json",
    [string]$Output = "reports/infra/r740_acceptance_latest.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "scripts/r740_hardware_acceptance.py")) {
    throw "Run from repo root (scripts/r740_hardware_acceptance.py not found)."
}

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python not found." }

$argsList = @(
    "scripts/r740_hardware_acceptance.py",
    "--spec", $Spec,
    "--facts", $Facts,
    "--output", $Output
)
if ($py -eq "py") {
    py -3 @argsList
} else {
    python @argsList
}
if ($LASTEXITCODE -ne 0) {
    throw "R740 acceptance check failed."
}

Write-Host "[r740-acceptance] check passed"
