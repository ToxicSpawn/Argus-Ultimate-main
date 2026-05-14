# =============================================================================
# ARGUS Ollama Setup Script (Windows / Workstation)
# =============================================================================
# Installs Ollama and pulls the LLM model for local inference on RTX 5080.
# Run from PowerShell as Administrator.
#
# Usage:
#   .\scripts\setup_ollama.ps1
#   .\scripts\setup_ollama.ps1 -Model "llama3.3:70b-instruct-q4_K_M"
# =============================================================================

param(
    [string]$Model = "llama3.1:8b-instruct-q4_K_M",  # Fast default; change to 70b when ready
    [switch]$LargeModel  # Pull the 70B model (requires ~16GB VRAM)
)

$ErrorActionPreference = "Stop"

Write-Host "=== ARGUS Ollama Setup ===" -ForegroundColor Cyan

# --- 1. Check if Ollama already installed ---
$ollamaExe = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaExe) {
    Write-Host "[OK] Ollama already installed: $(ollama --version)" -ForegroundColor Green
} else {
    Write-Host "[INFO] Downloading Ollama installer..." -ForegroundColor Yellow
    $installerUrl = "https://ollama.com/download/OllamaSetup.exe"
    $installerPath = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "[INFO] Installing Ollama..." -ForegroundColor Yellow
    Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait
    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
    Write-Host "[OK] Ollama installed" -ForegroundColor Green
}

# --- 2. Start Ollama service ---
Write-Host "[INFO] Starting Ollama service..." -ForegroundColor Yellow
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 3

# --- 3. Select model ---
if ($LargeModel) {
    $Model = "llama3.3:70b-instruct-q4_K_M"  # ~42GB download, fits in 16GB VRAM at q4
    Write-Host "[INFO] Large model selected: $Model" -ForegroundColor Yellow
    Write-Host "[WARN] This will download ~42GB. RTX 5080 16GB VRAM required." -ForegroundColor Yellow
}

Write-Host "[INFO] Pulling model: $Model" -ForegroundColor Yellow
Write-Host "       (this may take 10-60 minutes depending on connection speed)" -ForegroundColor Gray
ollama pull $Model

# --- 4. Test inference ---
Write-Host "[INFO] Testing inference..." -ForegroundColor Yellow
$testPrompt = "Reply with only: OK"
$response = ollama run $Model $testPrompt 2>&1
if ($response -match "OK") {
    Write-Host "[OK] Inference test passed" -ForegroundColor Green
} else {
    Write-Host "[WARN] Inference test response: $response" -ForegroundColor Yellow
}

# --- 5. Configure ARGUS ---
Write-Host ""
Write-Host "=== Configure ARGUS to use local LLM ===" -ForegroundColor Cyan
Write-Host "Add to unified_config.yaml:" -ForegroundColor White
Write-Host ""
Write-Host "  llm_signal:" -ForegroundColor Yellow
Write-Host "    provider: ollama" -ForegroundColor Yellow
Write-Host "    model: $Model" -ForegroundColor Yellow
Write-Host "    base_url: http://localhost:11434" -ForegroundColor Yellow
Write-Host "    enabled: true" -ForegroundColor Yellow
Write-Host ""

# --- 6. GPU check ---
Write-Host "=== GPU Status ===" -ForegroundColor Cyan
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1
    Write-Host "[GPU] $gpuInfo" -ForegroundColor Green
} catch {
    Write-Host "[WARN] nvidia-smi not found — confirm NVIDIA drivers installed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Ollama Setup Complete ===" -ForegroundColor Green
Write-Host "Ollama API: http://localhost:11434" -ForegroundColor Cyan
Write-Host "Model:      $Model" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run ARGUS with local LLM:" -ForegroundColor White
Write-Host "  py main.py paper" -ForegroundColor Yellow
