param(
    [ValidateSet("sdxl", "qwen")][string]$Backend = "qwen",
    [string]$Python = "",
    [string]$Proxy = "",
    [switch]$ImportGenerated,
    [switch]$DownloadModel
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "runtime\logs") | Out-Null
$transcript = Join-Path $projectRoot ("runtime\logs\setup-" + $Backend + "-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".log")
Start-Transcript -Path $transcript | Out-Null
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Install Git for Windows before setup." }
    $nvidia = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object Name -Match 'NVIDIA')
    $nvidiaDetected = $nvidia.Count -gt 0
    if (-not $nvidiaDetected) {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($smi) { & $smi.Source --query-gpu=name --format=csv,noheader | Out-Host; $nvidiaDetected = $LASTEXITCODE -eq 0 }
    }
    if (-not $nvidiaDetected) { throw "NVIDIA GPU/driver not detected. Run this installer on the RTX desktop after installing its NVIDIA driver." }
    if ($Proxy) { $env:HTTP_PROXY = $Proxy; $env:HTTPS_PROXY = $Proxy }
    $env:HF_HOME = Join-Path $projectRoot "runtime\huggingface"
    $env:PIP_CACHE_DIR = Join-Path $projectRoot "runtime\pip-cache"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    $trainingPython = & (Join-Path $PSScriptRoot "bootstrap-python.ps1") -EnvironmentName ".venv-$Backend" -Python $Python -Proxy $Proxy
    & $trainingPython scripts/install_environment.py --backend $Backend
    if ($LASTEXITCODE -ne 0) { throw "Environment installation failed; see $transcript" }
    & $trainingPython train.py doctor --backend $Backend
    if ($LASTEXITCODE -ne 0) { throw "GPU verification failed; see $transcript" }
    & $trainingPython train.py init
    if ($LASTEXITCODE -ne 0) { throw "Directory initialization failed" }
    if ($ImportGenerated) {
        & $trainingPython train.py import-generated
        if ($LASTEXITCODE -ne 0) { throw "Generated dataset import failed" }
    }
    if ($DownloadModel) {
        if ($Backend -eq "sdxl") { & $trainingPython train.py download --backend sdxl --model sdxl-base }
        else { & $trainingPython train.py download --backend qwen }
        if ($LASTEXITCODE -ne 0) { throw "Model download failed; existing files are never overwritten" }
    }
    Write-Host "Environment ready. Next: .\scripts\train-style.ps1 -Backend $Backend -Style cartoon -Mode smoke"
} finally { Stop-Transcript | Out-Null }
