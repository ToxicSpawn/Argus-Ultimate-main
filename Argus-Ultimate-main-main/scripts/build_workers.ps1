# build_workers.ps1 — Pre-compile all compiled language workers
# Run: powershell -ExecutionPolicy Bypass -File scripts/build_workers.ps1
#
# Compiles Go, Rust, Haskell, Java, Kotlin, C#, F# workers to binary form.
# Interpreted workers (Ruby, R, Erlang, Elixir, Clojure, Scala) need no compilation.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$workers = Join-Path $root "multilang\workers"
$bin = Join-Path $workers "bin"

# Ensure bin directory exists
if (-not (Test-Path $bin)) { New-Item -ItemType Directory -Path $bin -Force | Out-Null }

$results = @{}

function Compile-Worker {
    param([string]$Name, [string]$Command, [string[]]$Args, [string]$WorkDir)
    Write-Host "[$Name] Compiling..." -ForegroundColor Cyan
    try {
        $proc = Start-Process -FilePath $Command -ArgumentList $Args -WorkingDirectory $WorkDir `
            -NoNewWindow -Wait -PassThru -RedirectStandardError "$bin\${Name}_stderr.log" 2>$null
        if ($proc.ExitCode -eq 0) {
            Write-Host "[$Name] OK" -ForegroundColor Green
            $results[$Name] = "OK"
        } else {
            $err = ""
            if (Test-Path "$bin\${Name}_stderr.log") { $err = Get-Content "$bin\${Name}_stderr.log" -Raw | Select-Object -First 3 }
            Write-Host "[$Name] FAILED (exit $($proc.ExitCode)): $err" -ForegroundColor Red
            $results[$Name] = "FAILED"
        }
    } catch {
        Write-Host "[$Name] ERROR: $_" -ForegroundColor Red
        $results[$Name] = "ERROR"
    }
    # Clean up stderr log on success
    if ($results[$Name] -eq "OK" -and (Test-Path "$bin\${Name}_stderr.log")) {
        Remove-Item "$bin\${Name}_stderr.log" -Force 2>$null
    }
}

# ── Go ────────────────────────────────────────────────────────────────────
$goSrc = Join-Path $workers "go_worker.go"
$goBin = Join-Path $bin "go_worker.exe"
if (Test-Path $goSrc) {
    if (Get-Command go -ErrorAction SilentlyContinue) {
        Compile-Worker -Name "go" -Command "go" -Args @("build", "-o", $goBin, $goSrc) -WorkDir $workers
    } else { Write-Host "[go] SKIP (go not in PATH)" -ForegroundColor Yellow; $results["go"] = "SKIP" }
} else { Write-Host "[go] SKIP (source not found)" -ForegroundColor Yellow; $results["go"] = "SKIP" }

# ── Rust ──────────────────────────────────────────────────────────────────
$rustSrc = Join-Path $workers "rust_worker.rs"
$rustBin = Join-Path $bin "rust_worker.exe"
if (Test-Path $rustSrc) {
    if (Get-Command rustc -ErrorAction SilentlyContinue) {
        Compile-Worker -Name "rust" -Command "rustc" -Args @("-O", $rustSrc, "-o", $rustBin) -WorkDir $workers
    } else { Write-Host "[rust] SKIP (rustc not in PATH)" -ForegroundColor Yellow; $results["rust"] = "SKIP" }
} else { Write-Host "[rust] SKIP (source not found)" -ForegroundColor Yellow; $results["rust"] = "SKIP" }

# ── Haskell ───────────────────────────────────────────────────────────────
$hsSrc = Join-Path $workers "haskell_worker.hs"
$hsBin = Join-Path $bin "haskell_worker.exe"
$ghc = $null
@("C:\tools\ghc-9.8.2\bin\ghc.exe", "ghc") | ForEach-Object {
    if (-not $ghc -and (Test-Path $_ -ErrorAction SilentlyContinue)) { $ghc = $_ }
    if (-not $ghc -and (Get-Command $_ -ErrorAction SilentlyContinue)) { $ghc = $_ }
}
if ((Test-Path $hsSrc) -and $ghc) {
    $ghcTmp = Join-Path $bin "ghc_tmp"
    if (-not (Test-Path $ghcTmp)) { New-Item -ItemType Directory -Path $ghcTmp -Force | Out-Null }
    Compile-Worker -Name "haskell" -Command $ghc -Args @("-O2", "-o", $hsBin, $hsSrc, "-outputdir", $ghcTmp) -WorkDir $workers
} else { Write-Host "[haskell] SKIP (ghc or source not found)" -ForegroundColor Yellow; $results["haskell"] = "SKIP" }

# ── Java ──────────────────────────────────────────────────────────────────
$javaSrc = Join-Path $workers "java_worker\ArgusWorker.java"
if (Test-Path $javaSrc) {
    if (Get-Command javac -ErrorAction SilentlyContinue) {
        Compile-Worker -Name "java" -Command "javac" -Args @("-d", (Join-Path $workers "java_worker"), $javaSrc) -WorkDir $workers
    } else { Write-Host "[java] SKIP (javac not in PATH)" -ForegroundColor Yellow; $results["java"] = "SKIP" }
} else { Write-Host "[java] SKIP (source not found)" -ForegroundColor Yellow; $results["java"] = "SKIP" }

# ── Kotlin ────────────────────────────────────────────────────────────────
$ktSrc = Join-Path $workers "kotlin\ArgusWorker.kt"
$ktJar = Join-Path $bin "kotlin_worker.jar"
$kotlinc = $null
@("C:\kotlin\bin\kotlinc.bat", "kotlinc") | ForEach-Object {
    if (-not $kotlinc -and (Test-Path $_ -ErrorAction SilentlyContinue)) { $kotlinc = $_ }
    if (-not $kotlinc -and (Get-Command $_ -ErrorAction SilentlyContinue)) { $kotlinc = $_ }
}
if ((Test-Path $ktSrc) -and $kotlinc) {
    Compile-Worker -Name "kotlin" -Command $kotlinc -Args @($ktSrc, "-include-runtime", "-d", $ktJar) -WorkDir $workers
} else { Write-Host "[kotlin] SKIP (kotlinc or source not found)" -ForegroundColor Yellow; $results["kotlin"] = "SKIP" }

# ── C# ────────────────────────────────────────────────────────────────────
$csproj = Join-Path $workers "csharp_worker\csharp_worker.csproj"
$csBinDir = Join-Path $bin "csharp"
if (Test-Path $csproj) {
    if (Get-Command dotnet -ErrorAction SilentlyContinue) {
        Compile-Worker -Name "csharp" -Command "dotnet" `
            -Args @("publish", "-c", "Release", "-o", $csBinDir, "--nologo", "-v", "q") `
            -WorkDir (Join-Path $workers "csharp_worker")
    } else { Write-Host "[csharp] SKIP (dotnet not in PATH)" -ForegroundColor Yellow; $results["csharp"] = "SKIP" }
} else { Write-Host "[csharp] SKIP (.csproj not found)" -ForegroundColor Yellow; $results["csharp"] = "SKIP" }

# ── F# ────────────────────────────────────────────────────────────────────
$fsproj = Join-Path $workers "fsharp_worker\fsharp_worker.fsproj"
$fsBinDir = Join-Path $bin "fsharp"
if (Test-Path $fsproj) {
    if (Get-Command dotnet -ErrorAction SilentlyContinue) {
        Compile-Worker -Name "fsharp" -Command "dotnet" `
            -Args @("publish", "-c", "Release", "-o", $fsBinDir, "--nologo", "-v", "q") `
            -WorkDir (Join-Path $workers "fsharp_worker")
    } else { Write-Host "[fsharp] SKIP (dotnet not in PATH)" -ForegroundColor Yellow; $results["fsharp"] = "SKIP" }
} else { Write-Host "[fsharp] SKIP (.fsproj not found)" -ForegroundColor Yellow; $results["fsharp"] = "SKIP" }

# ── Summary ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Build Summary ===" -ForegroundColor White
$ok = ($results.Values | Where-Object { $_ -eq "OK" }).Count
$skip = ($results.Values | Where-Object { $_ -eq "SKIP" }).Count
$fail = ($results.Values | Where-Object { $_ -ne "OK" -and $_ -ne "SKIP" }).Count
foreach ($kv in $results.GetEnumerator() | Sort-Object Name) {
    $color = switch ($kv.Value) { "OK" { "Green" } "SKIP" { "Yellow" } default { "Red" } }
    Write-Host ("  {0,-10} {1}" -f $kv.Name, $kv.Value) -ForegroundColor $color
}
Write-Host ""
Write-Host "Compiled: $ok  Skipped: $skip  Failed: $fail" -ForegroundColor White
