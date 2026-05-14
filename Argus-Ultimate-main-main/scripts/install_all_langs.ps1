# Install and fix all language runtimes needed for Argus 25-language services
# Run as: powershell.exe -File install_all_langs.ps1

$ErrorActionPreference = 'Continue'

Write-Host "=== Installing missing/broken language runtimes ===" -ForegroundColor Cyan

# Node.js LTS (needed for JS/TS services)
Write-Host "`n[1/6] Installing Node.js LTS..." -ForegroundColor Yellow
choco install nodejs-lts -y --force 2>&1 | Select-String -Pattern "already|installed|error|fail" | Write-Host

# Ruby 3.4 (fresh install, not old 2.3)
Write-Host "`n[2/6] Installing Ruby 3.4..." -ForegroundColor Yellow
choco install ruby --version 3.4.8.1 -y --force 2>&1 | Select-String -Pattern "already|installed|error|fail" | Write-Host

# Kotlin
Write-Host "`n[3/6] Installing Kotlin..." -ForegroundColor Yellow
choco install kotlin -y --force 2>&1 | Select-String -Pattern "already|installed|error|fail" | Write-Host

# Crystal (Windows build)
Write-Host "`n[4/6] Installing Crystal..." -ForegroundColor Yellow
choco install crystal -y --force 2>&1 | Select-String -Pattern "already|installed|error|fail" | Write-Host

# Fix Julia - reinstall to correct path
Write-Host "`n[5/6] Reinstalling Julia..." -ForegroundColor Yellow
choco install julia -y --force 2>&1 | Select-String -Pattern "already|installed|error|fail" | Write-Host

# wasmtime for WebAssembly
Write-Host "`n[6/6] Installing wasmtime..." -ForegroundColor Yellow
choco install wasmtime -y 2>&1 | Select-String -Pattern "already|installed|error|fail" | Write-Host

Write-Host "`n=== Refreshing environment PATH ===" -ForegroundColor Cyan
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "`n=== Verification ===" -ForegroundColor Cyan
$tools = @{
    "node"     = "node --version"
    "ruby"     = "ruby --version"
    "kotlinc"  = "kotlinc -version"
    "crystal"  = "crystal --version"
    "julia"    = "julia --version"
    "wasmtime" = "wasmtime --version"
    "Rscript"  = "Rscript --version"
    "elixir"   = "elixir --version"
    "ghc"      = "ghc --version"
    "dotnet"   = "dotnet --version"
    "java"     = "java --version"
    "scala"    = "scala --version"
    "go"       = "go version"
    "erl"      = "erl -version"
    "clojure"  = "clojure --version"
    "cabal"    = "cabal --version"
    "cargo"    = "cargo --version"
}
foreach ($name in $tools.Keys | Sort-Object) {
    try {
        $out = cmd /c "$($tools[$name]) 2>&1" 2>&1
        $first = ($out | Where-Object { $_ -match '\S' } | Select-Object -First 1) -replace '\r?\n',''
        Write-Host "  OK  $name -> $first" -ForegroundColor Green
    } catch {
        Write-Host "  MISS $name" -ForegroundColor Red
    }
}
