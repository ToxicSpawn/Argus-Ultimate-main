param(
    [ValidateSet("validate", "paper_smoke", "paper_soak", "paper_edge_eval", "infra_latency_report", "export_audit_bus", "r740_prep_bundle", "r740_bundle_check", "r740_prebuild_suite", "r740_capture_facts", "r740_acceptance_check", "institutional_readiness", "evidence_pipeline")]
    [string]$Target = "validate",
    [string]$Config = "unified_config.yaml",
    [string]$Profile = "",
    [double]$Capital = 1000.0,
    [int]$SmokeCycles = 3,
    [int]$SoakCycles = 180,
    [double]$CycleSeconds = 1.0,
    [switch]$OfflineSmoke,
    [string]$EdgeOutput = "data/paper_results.json",
    [string]$InfraReportOutputDir = "reports/infra",
    [string]$AuditBusOutput = "logs/audit_bus_latest.jsonl",
    [int]$AuditBusLimit = 5000,
    [string]$R740Manifest = "docs/hardware/R740_PREBUILD_MANIFEST.yaml",
    [string]$R740OutputRoot = "deploy/r740_bundle",
    [string]$R740Bundle = "",
    [string]$R740CheckOutput = "reports/infra/r740_bundle_check_latest.json",
    [string]$R740SuiteOutput = "reports/infra/r740_prebuild_suite_latest.json",
    [switch]$R740SuiteRunReadiness,
    [string]$R740FactsOutput = "reports/infra/r740_host_facts_latest.json",
    [string]$R740AcceptanceSpec = "docs/hardware/R740_ACCEPTANCE_SPEC.yaml",
    [string]$R740AcceptanceFacts = "docs/hardware/R740_HOST_FACTS_SAMPLE.json",
    [string]$R740AcceptanceOutput = "reports/infra/r740_acceptance_latest.json",
    [string]$ReadinessManifest = "docs/institutional/evidence_manifest.json",
    [string]$ReadinessOutput = "reports/institutional_readiness_latest.json",
    [switch]$AllowManualUnverified,
    [switch]$SkipPreLiveCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    throw "Python launcher not found (py/python)."
}

function Get-ProfileArgs {
    if ($Profile -and $Profile.Trim()) {
        return @("--profile", $Profile.Trim())
    }
    return @()
}

function Invoke-ArgusMain {
    param([string[]]$MainArgs)
    $py = Get-PythonCommand
    & $py ".\main.py" @MainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $py .\main.py $($MainArgs -join ' ')"
    }
}

function Invoke-PythonScript {
    param([string]$ScriptPath, [string[]]$ScriptArgs)
    $py = Get-PythonCommand
    if ($py -eq "py") {
        & $py "-3" $ScriptPath @ScriptArgs
    }
    else {
        & $py $ScriptPath @ScriptArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $py $ScriptPath $($ScriptArgs -join ' ')"
    }
}

function Invoke-Validate {
    $mainArgs = @("validate", "--config", $Config) + (Get-ProfileArgs)
    Invoke-ArgusMain -MainArgs $mainArgs
}

function Invoke-PaperSmoke {
    $mainArgs = @(
        "paper",
        "--config", $Config
    ) + (Get-ProfileArgs) + @(
        "--capital", "$Capital",
        "--cycles", "$SmokeCycles",
        "--cycle-seconds", "$CycleSeconds",
        "--no-multilang"
    )
    # OfflineSmoke retained for backward compatibility with older scripts.
    # Current main.py has no --offline-smoke flag, so this is intentionally a no-op.
    Invoke-ArgusMain -MainArgs $mainArgs
}

function Invoke-PaperSoak {
    $mainArgs = @(
        "paper",
        "--config", $Config
    ) + (Get-ProfileArgs) + @(
        "--capital", "$Capital",
        "--cycles", "$SoakCycles",
        "--cycle-seconds", "$CycleSeconds",
        "--no-multilang"
    )
    # OfflineSmoke retained for backward compatibility with older scripts.
    # Current main.py has no --offline-smoke flag, so this is intentionally a no-op.
    Invoke-ArgusMain -MainArgs $mainArgs
}

function Invoke-PaperEdgeEval {
    $scriptArgs = @(
        "--config", $Config,
        "--output", $EdgeOutput
    )
    $scriptArgs += Get-ProfileArgs
    Invoke-PythonScript -ScriptPath ".\scripts\evaluate_paper_edge.py" -ScriptArgs $scriptArgs
}

function Invoke-InfraLatencyReport {
    $scriptArgs = @(
        "--db", "data/unified_trades.db",
        "--output-dir", $InfraReportOutputDir
    )
    Invoke-PythonScript -ScriptPath ".\scripts\infra_latency_report.py" -ScriptArgs $scriptArgs
}

function Invoke-ExportAuditBus {
    $scriptArgs = @(
        "--db", "data/unified_trades.db",
        "--output", $AuditBusOutput,
        "--limit", "$AuditBusLimit"
    )
    Invoke-PythonScript -ScriptPath ".\scripts\export_audit_bus.py" -ScriptArgs $scriptArgs
}

function Invoke-R740PrepBundle {
    $scriptArgs = @(
        "--manifest", $R740Manifest,
        "--output-root", $R740OutputRoot
    )
    Invoke-PythonScript -ScriptPath ".\scripts\r740_prepare_bundle.py" -ScriptArgs $scriptArgs
}

function Invoke-R740BundleCheck {
    $scriptArgs = @(
        "--bundle-root", $R740OutputRoot,
        "--output", $R740CheckOutput
    )
    if ($R740Bundle -and $R740Bundle.Trim()) {
        $scriptArgs += @("--bundle", $R740Bundle.Trim())
    }
    Invoke-PythonScript -ScriptPath ".\scripts\r740_bundle_check.py" -ScriptArgs $scriptArgs
}

function Invoke-R740PrebuildSuite {
    $scriptArgs = @(
        "--config", $Config,
        "--manifest", $R740Manifest,
        "--bundle-root", $R740OutputRoot,
        "--bundle-check-output", $R740CheckOutput,
        "--suite-output", $R740SuiteOutput
    )
    $scriptArgs += Get-ProfileArgs
    if ($R740SuiteRunReadiness) {
        $scriptArgs += "--run-readiness"
    }
    Invoke-PythonScript -ScriptPath ".\scripts\r740_prebuild_suite.py" -ScriptArgs $scriptArgs
}

function Invoke-R740CaptureFacts {
    $scriptArgs = @(
        "--output", $R740FactsOutput
    )
    Invoke-PythonScript -ScriptPath ".\scripts\r740_capture_host_facts.py" -ScriptArgs $scriptArgs
}

function Invoke-R740AcceptanceCheck {
    $scriptArgs = @(
        "--spec", $R740AcceptanceSpec,
        "--facts", $R740AcceptanceFacts,
        "--output", $R740AcceptanceOutput
    )
    Invoke-PythonScript -ScriptPath ".\scripts\r740_hardware_acceptance.py" -ScriptArgs $scriptArgs
}

function Invoke-Readiness {
    $scriptArgs = @(
        "--config", $Config,
        "--manifest", $ReadinessManifest,
        "--output", $ReadinessOutput
    )
    $scriptArgs += Get-ProfileArgs
    if ($AllowManualUnverified) {
        $scriptArgs += "--allow-manual-unverified"
    }
    if ($SkipPreLiveCheck) {
        $scriptArgs += "--skip-pre-live"
    }
    Invoke-PythonScript -ScriptPath ".\scripts\institutional_readiness_check.py" -ScriptArgs $scriptArgs
}

switch ($Target) {
    "validate" {
        Invoke-Validate
    }
    "paper_smoke" {
        Invoke-PaperSmoke
    }
    "paper_soak" {
        Invoke-PaperSoak
    }
    "paper_edge_eval" {
        Invoke-PaperEdgeEval
    }
    "infra_latency_report" {
        Invoke-InfraLatencyReport
    }
    "export_audit_bus" {
        Invoke-ExportAuditBus
    }
    "r740_prep_bundle" {
        Invoke-R740PrepBundle
    }
    "r740_bundle_check" {
        Invoke-R740BundleCheck
    }
    "r740_prebuild_suite" {
        Invoke-R740PrebuildSuite
    }
    "r740_capture_facts" {
        Invoke-R740CaptureFacts
    }
    "r740_acceptance_check" {
        Invoke-R740AcceptanceCheck
    }
    "institutional_readiness" {
        Invoke-Readiness
    }
    "evidence_pipeline" {
        Write-Host "[evidence] validate"
        Invoke-Validate
        Write-Host "[evidence] paper_smoke"
        Invoke-PaperSmoke
        Write-Host "[evidence] paper_edge_eval"
        Invoke-PaperEdgeEval
        Write-Host "[evidence] r740_prep_bundle"
        Invoke-R740PrepBundle
        Write-Host "[evidence] r740_bundle_check"
        Invoke-R740BundleCheck
        Write-Host "[evidence] r740_prebuild_suite"
        Invoke-R740PrebuildSuite
        Write-Host "[evidence] r740_acceptance_check"
        Invoke-R740AcceptanceCheck
        Write-Host "[evidence] infra_latency_report"
        Invoke-InfraLatencyReport
        Write-Host "[evidence] paper_soak"
        Invoke-PaperSoak
        Write-Host "[evidence] soak_gate"
        $soakArgs = @("--config", $Config, "--db", "data/unified_trades.db", "--output-dir", "reports") + (Get-ProfileArgs)
        Invoke-PythonScript -ScriptPath ".\scripts\soak_gate.py" -ScriptArgs $soakArgs
        Write-Host "[evidence] institutional_readiness"
        Invoke-Readiness
    }
}

Write-Host "Target '$Target' completed successfully."
