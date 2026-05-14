# Direct installer script for all missing Argus language runtimes
# Run as Administrator: powershell.exe -ExecutionPolicy Bypass -File install_runtimes.ps1

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'  # faster downloads

function Download-And-Install {
    param($url, $dest, $args_str)
    Write-Host "  Downloading $dest..." -ForegroundColor Gray
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    if (Test-Path $dest) {
        Write-Host "  Installing..." -ForegroundColor Gray
        if ($args_str) { Start-Process $dest -ArgumentList $args_str -Wait -NoNewWindow }
        else { Start-Process $dest -Wait -NoNewWindow }
        Remove-Item $dest -Force -ErrorAction SilentlyContinue
        Write-Host "  Done." -ForegroundColor Green
    } else {
        Write-Host "  Download failed." -ForegroundColor Red
    }
}

$tmp = "$env:TEMP\argus_installs"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# ── Node.js 22 LTS ─────────────────────────────────────────────────────────
Write-Host "`n[1/5] Node.js 22 LTS" -ForegroundColor Yellow
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Download-And-Install `
        "https://nodejs.org/dist/v22.14.0/node-v22.14.0-x64.msi" `
        "$tmp\node.msi" `
        "/quiet /norestart ADDLOCAL=ALL"
} else { Write-Host "  Already installed: $(node --version)" -ForegroundColor Green }

# ── Ruby 3.4 ────────────────────────────────────────────────────────────────
Write-Host "`n[2/5] Ruby 3.4" -ForegroundColor Yellow
if (-not (Get-Command ruby -ErrorAction SilentlyContinue)) {
    Download-And-Install `
        "https://github.com/oneclick/rubyinstaller2/releases/download/RubyInstaller-3.4.4-1/rubyinstaller-3.4.4-1-x64.exe" `
        "$tmp\ruby.exe" `
        "/verysilent /allusers /dir=C:\Ruby34-x64 /tasks=modpath,assocfiles,noridkinstall"
} else { Write-Host "  Already installed: $(ruby --version)" -ForegroundColor Green }

# ── Kotlin ───────────────────────────────────────────────────────────────────
Write-Host "`n[3/5] Kotlin (via SDKMAN zip)" -ForegroundColor Yellow
$kotlinDir = "C:\kotlin"
if (-not (Test-Path "$kotlinDir\bin\kotlinc.bat")) {
    Write-Host "  Downloading Kotlin 2.1.0 compiler zip..." -ForegroundColor Gray
    Invoke-WebRequest -Uri "https://github.com/JetBrains/kotlin/releases/download/v2.1.0/kotlin-compiler-2.1.0.zip" `
        -OutFile "$tmp\kotlin.zip" -UseBasicParsing
    if (Test-Path "$tmp\kotlin.zip") {
        Expand-Archive "$tmp\kotlin.zip" -DestinationPath "C:\" -Force
        Rename-Item "C:\kotlinc" $kotlinDir -ErrorAction SilentlyContinue
        # Add to system PATH
        $syspath = [Environment]::GetEnvironmentVariable("Path","Machine")
        if ($syspath -notlike "*$kotlinDir\bin*") {
            [Environment]::SetEnvironmentVariable("Path", "$syspath;$kotlinDir\bin", "Machine")
        }
        Remove-Item "$tmp\kotlin.zip" -Force -ErrorAction SilentlyContinue
        Write-Host "  Installed to $kotlinDir" -ForegroundColor Green
    }
} else { Write-Host "  Already installed at $kotlinDir" -ForegroundColor Green }

# ── Crystal ──────────────────────────────────────────────────────────────────
Write-Host "`n[4/5] Crystal" -ForegroundColor Yellow
$crystalDir = "C:\crystal"
if (-not (Test-Path "$crystalDir\crystal.exe")) {
    Write-Host "  Downloading Crystal 1.15.1 for Windows..." -ForegroundColor Gray
    Invoke-WebRequest -Uri "https://github.com/crystal-lang/crystal/releases/download/1.15.1/crystal-1.15.1-windows-x86_64-msvc-unsupported.zip" `
        -OutFile "$tmp\crystal.zip" -UseBasicParsing
    if (Test-Path "$tmp\crystal.zip") {
        Expand-Archive "$tmp\crystal.zip" -DestinationPath "C:\" -Force
        # Rename the extracted folder
        $extracted = Get-ChildItem "C:\" -Directory -Filter "crystal-1*" | Select-Object -First 1
        if ($extracted) { Rename-Item $extracted.FullName $crystalDir -ErrorAction SilentlyContinue }
        $syspath = [Environment]::GetEnvironmentVariable("Path","Machine")
        if ($syspath -notlike "*$crystalDir*") {
            [Environment]::SetEnvironmentVariable("Path", "$syspath;$crystalDir", "Machine")
        }
        Remove-Item "$tmp\crystal.zip" -Force -ErrorAction SilentlyContinue
        Write-Host "  Installed to $crystalDir" -ForegroundColor Green
    }
} else { Write-Host "  Already installed at $crystalDir" -ForegroundColor Green }

# ── Fix Julia path ────────────────────────────────────────────────────────────
Write-Host "`n[5/5] Fix Julia PATH" -ForegroundColor Yellow
$juliaSearch = Get-ChildItem "C:\Users\hinge\AppData\Local\Programs" -Filter "julia*" -Directory -ErrorAction SilentlyContinue |
               Sort-Object Name -Descending | Select-Object -First 1
if (-not $juliaSearch) {
    $juliaSearch = Get-ChildItem "C:\Julia*" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
}
if ($juliaSearch) {
    $juliaExe = Join-Path $juliaSearch.FullName "bin\julia.exe"
    if (-not (Test-Path $juliaExe)) {
        # Download fresh Julia
        Write-Host "  Julia exe missing, downloading Julia 1.11.3..." -ForegroundColor Gray
        Invoke-WebRequest -Uri "https://julialang-s3.julialang.org/bin/winnt/x64/1.11/julia-1.11.3-win64.exe" `
            -OutFile "$tmp\julia.exe" -UseBasicParsing
        if (Test-Path "$tmp\julia.exe") {
            Start-Process "$tmp\julia.exe" -ArgumentList "/S /D=C:\julia-1.11.3" -Wait -NoNewWindow
            Remove-Item "$tmp\julia.exe" -Force -ErrorAction SilentlyContinue
        }
    }
    # Fix choco shim to point to real julia
    $realJulia = Get-ChildItem "C:\" -Filter "julia.exe" -Recurse -Depth 5 -ErrorAction SilentlyContinue |
                 Where-Object { $_.FullName -notlike "*choco*" } | Select-Object -First 1
    if ($realJulia) {
        Write-Host "  Julia found at: $($realJulia.FullName)" -ForegroundColor Green
        $syspath = [Environment]::GetEnvironmentVariable("Path","Machine")
        $juliaDir = Split-Path $realJulia.FullName
        if ($syspath -notlike "*$juliaDir*") {
            [Environment]::SetEnvironmentVariable("Path", "$syspath;$juliaDir", "Machine")
        }
    }
} else {
    Write-Host "  Downloading Julia 1.11.3..." -ForegroundColor Gray
    Invoke-WebRequest -Uri "https://julialang-s3.julialang.org/bin/winnt/x64/1.11/julia-1.11.3-win64.exe" `
        -OutFile "$tmp\julia.exe" -UseBasicParsing
    Start-Process "$tmp\julia.exe" -ArgumentList "/S /D=C:\julia-1.11.3" -Wait -NoNewWindow
    Remove-Item "$tmp\julia.exe" -Force -ErrorAction SilentlyContinue
}

# ── Fix R PATH ────────────────────────────────────────────────────────────────
Write-Host "`nFixing R PATH..." -ForegroundColor Yellow
$rscript = "C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
if (Test-Path $rscript) {
    $rDir = Split-Path $rscript
    $syspath = [Environment]::GetEnvironmentVariable("Path","Machine")
    if ($syspath -notlike "*R-4.5.2*") {
        [Environment]::SetEnvironmentVariable("Path", "$syspath;$rDir", "Machine")
        Write-Host "  Added R to PATH" -ForegroundColor Green
    } else { Write-Host "  R already in PATH" -ForegroundColor Green }
}

# ── Fix Elixir PATH ────────────────────────────────────────────────────────────
Write-Host "Fixing Elixir PATH..." -ForegroundColor Yellow
$elixirDir = "C:\ProgramData\chocolatey\lib\elixir\tools\bin"
if (Test-Path $elixirDir) {
    $syspath = [Environment]::GetEnvironmentVariable("Path","Machine")
    if ($syspath -notlike "*elixir*tools*bin*") {
        [Environment]::SetEnvironmentVariable("Path", "$syspath;$elixirDir", "Machine")
        Write-Host "  Added Elixir to PATH" -ForegroundColor Green
    } else { Write-Host "  Elixir already in PATH" -ForegroundColor Green }
}

# ── Fix GHC PATH ────────────────────────────────────────────────────────────
Write-Host "Fixing GHC PATH..." -ForegroundColor Yellow
$ghcDir = "C:\tools\ghc-9.8.2\bin"
if (Test-Path $ghcDir) {
    $syspath = [Environment]::GetEnvironmentVariable("Path","Machine")
    if ($syspath -notlike "*ghc-9.8.2*") {
        [Environment]::SetEnvironmentVariable("Path", "$syspath;$ghcDir", "Machine")
        Write-Host "  Added GHC to PATH" -ForegroundColor Green
    } else { Write-Host "  GHC already in PATH" -ForegroundColor Green }
}

# ── Fix Clojure PATH ──────────────────────────────────────────────────────────
Write-Host "Fixing Clojure PATH..." -ForegroundColor Yellow
$clojurePossible = @(
    "C:\ProgramData\chocolatey\lib\clojure\tools\clojure\bin",
    "C:\tools\Clojure"
)
foreach ($d in $clojurePossible) {
    if (Test-Path $d) {
        $syspath = [Environment]::GetEnvironmentVariable("Path","Machine")
        if ($syspath -notlike "*$d*") {
            [Environment]::SetEnvironmentVariable("Path", "$syspath;$d", "Machine")
            Write-Host "  Added Clojure from $d to PATH" -ForegroundColor Green
        }
        break
    }
}

# ── Refresh current session PATH ──────────────────────────────────────────────
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")

Write-Host "`n=== Final Verification ===" -ForegroundColor Cyan
foreach ($cmd in @("node","ruby","kotlinc","crystal","julia","Rscript","elixir","ghc","clojure","wasmtime")) {
    $loc = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($loc) { Write-Host "  OK   $cmd -> $($loc.Source)" -ForegroundColor Green }
    else       { Write-Host "  MISS $cmd" -ForegroundColor Red }
}
