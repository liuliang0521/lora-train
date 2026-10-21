param(
    [ValidateSet("cartoon", "doubao-3d", "miyazaki", "jojo", "van-gogh", "sand-art", "all")][string]$Style = "cartoon",
    [ValidateSet("sdxl", "qwen")][string]$Backend = "qwen",
    [ValidateSet("validate", "prepare", "preview", "cache", "smoke", "pilot", "train")][string]$Mode = "smoke",
    [string]$Resume = "",
    [switch]$Restart,
    [string]$Proxy = ""
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$trainingPython = Join-Path $projectRoot ".venv-$Backend\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $trainingPython)) { throw "Run scripts/setup.ps1 -Backend $Backend on the RTX desktop first." }
if ($Resume -and ($Mode -ne "train" -or $Style -eq "all")) { throw "Resume requires -Mode train and one explicit style." }
if ($Restart -and ($Mode -ne "train" -or $Resume)) { throw "Restart requires -Mode train and cannot be combined with -Resume." }
if ($Proxy) { $env:HTTP_PROXY = $Proxy; $env:HTTPS_PROXY = $Proxy }
$env:HF_HOME = Join-Path $projectRoot "runtime\huggingface"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$styles = @($Style)
if ($Style -eq "all") { $styles = @("cartoon", "doubao-3d", "miyazaki", "jojo", "van-gogh", "sand-art") }
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "runtime\logs") | Out-Null
$transcript = Join-Path $projectRoot ("runtime\logs\launch-" + $Style + "-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".log")
Start-Transcript -Path $transcript | Out-Null
try { foreach ($selected in $styles) {
    $arguments = @("train.py", "train", "--backend", $Backend, "--style", $selected)
    if ($Mode -in @("validate", "prepare", "cache")) { $arguments[1] = $Mode }
    if ($Mode -eq "preview") { $arguments += "--dry-run" }
    if ($Mode -eq "smoke") { $arguments += "--smoke" }
    if ($Mode -eq "pilot") { $arguments += "--pilot" }
    if ($Resume) { $arguments += @("--resume", $Resume) }
    if ($Restart) { $arguments += "--restart" }
    Write-Host "[$selected] $Mode ($Backend)"
    & $trainingPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped at $selected. Read the error and its runs/ log before continuing." }
} } finally { Stop-Transcript | Out-Null }
