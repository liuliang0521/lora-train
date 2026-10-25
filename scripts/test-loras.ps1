param(
    [string]$Style = "all",
    [string]$Checkpoints = "final",
    [string]$Strengths = "default",
    [int]$Images = 3,
    [int]$Size = 512,
    [int]$Steps = 20,
    [string]$Url = "http://127.0.0.1:8188",
    [switch]$Baseline,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$trainingPython = Join-Path $projectRoot '.venv-qwen\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$arguments = @('-u', (Join-Path $PSScriptRoot 'test_loras.py'), '--style', $Style,
    '--checkpoints', $Checkpoints, '--strengths', $Strengths, '--images', "$Images",
    '--size', "$Size", '--steps', "$Steps", '--url', $Url)
if ($Baseline) { $arguments += '--baseline' }
if ($DryRun) { $arguments += '--dry-run' }
& $trainingPython @arguments
exit $LASTEXITCODE
