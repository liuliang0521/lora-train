param(
    [ValidateSet(".venv-sdxl", ".venv-qwen", ".venv-tools")][string]$EnvironmentName = ".venv-sdxl",
    [string]$Python = "",
    [string]$Proxy = ""
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$environmentDirectory = Join-Path $projectRoot $EnvironmentName
$environmentPython = Join-Path $environmentDirectory "Scripts\python.exe"
if ($Proxy) { $env:HTTP_PROXY = $Proxy; $env:HTTPS_PROXY = $Proxy }
if (Test-Path -LiteralPath $environmentPython) {
    & $environmentPython -c "import struct,sys; assert (3,10) <= sys.version_info[:2] <= (3,12), 'Existing environment requires Python 3.10-3.12'; assert struct.calcsize('P') == 8, '64-bit Python required'"
    if ($LASTEXITCODE -ne 0) { throw "Existing environment has an unsupported Python; preserve it and use a compatible environment." }
    return $environmentPython
}
if (Test-Path -LiteralPath $environmentDirectory) { throw "An incomplete environment already exists: $environmentDirectory. Inspect/rename it before retrying." }
if ($Python) {
    & $Python -c "import struct,sys; assert (3,10) <= sys.version_info[:2] <= (3,12), 'Pass Python 3.10-3.12 x64'; assert struct.calcsize('P') == 8, '64-bit Python required'"
    if ($LASTEXITCODE -ne 0) { throw "Unsupported Python" }
    & $Python -m venv $environmentDirectory
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    if (-not [Environment]::Is64BitProcess -or [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() -ne "X64") { throw "Run 64-bit PowerShell on Windows x64." }
    $settings = Get-Content -LiteralPath (Join-Path $projectRoot "config\bootstrap.json") -Raw | ConvertFrom-Json
    $toolDirectory = Join-Path $projectRoot ("runtime\bootstrap\uv-" + $settings.uv_version)
    New-Item -ItemType Directory -Force -Path $toolDirectory | Out-Null
    $archive = Join-Path $toolDirectory "uv.zip"
    if (-not (Test-Path -LiteralPath $archive)) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $request = @{Uri=$settings.uv_wheel_url; OutFile=($archive + ".part"); UseBasicParsing=$true}
        if ($Proxy) { $request.Proxy = $Proxy }
        Invoke-WebRequest @request | Out-Null
        if ((Get-FileHash -LiteralPath ($archive + ".part") -Algorithm SHA256).Hash.ToLowerInvariant() -ne $settings.uv_wheel_sha256) { throw "uv download checksum mismatch" }
        Move-Item -LiteralPath ($archive + ".part") -Destination $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $settings.uv_wheel_sha256) { throw "Cached uv checksum mismatch" }
    $unpacked = Join-Path $toolDirectory "unpacked"
    Expand-Archive -LiteralPath $archive -DestinationPath $unpacked -Force
    $uv = @(Get-ChildItem -LiteralPath $unpacked -Recurse -File -Filter "uv.exe")
    if ($uv.Count -ne 1) { throw "Expected one uv.exe in the verified wheel" }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot "runtime\python"
    $env:UV_CACHE_DIR = Join-Path $projectRoot "runtime\uv-cache"
    & $uv[0].FullName venv --python $settings.python --managed-python --seed $environmentDirectory | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Managed Python/venv installation failed" }
}
& $environmentPython -c "import struct,sys; assert struct.calcsize('P') == 8; print(sys.version, file=sys.stderr)"
if ($LASTEXITCODE -ne 0) { throw "64-bit Python required" }
return $environmentPython
