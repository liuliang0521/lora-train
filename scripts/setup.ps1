param(
    [ValidateSet("sdxl", "qwen")][string]$Backend = "sdxl",
    [string]$Python = "python",
    [string]$Proxy = "http://127.0.0.1:7890"
)
$ErrorActionPreference = "Stop"
$trainRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $trainRoot
if ($Proxy) { $env:HTTP_PROXY = $Proxy; $env:HTTPS_PROXY = $Proxy }
& $Python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), 'Use Python 3.10-3.12; pass -Python with its executable path'"
if ($LASTEXITCODE -ne 0) { throw "Unsupported Python" }
$environmentDirectory = Join-Path $trainRoot ".venv-$Backend"
$trainingPython = Join-Path $environmentDirectory "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $trainingPython)) {
    & $Python -m venv $environmentDirectory
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
& $trainingPython scripts/install_environment.py --backend $Backend
if ($LASTEXITCODE -ne 0) { throw "Environment installation failed" }
& $trainingPython train.py doctor --backend $Backend
if ($LASTEXITCODE -ne 0) { throw "GPU environment check failed" }
Write-Host "Ready. Use: $trainingPython train.py init"
