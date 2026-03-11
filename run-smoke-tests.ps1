$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $projectRoot "smoke_test_billing.py"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment Python not found at $pythonExe"
}

if (-not (Test-Path $scriptPath)) {
    throw "Smoke test script not found at $scriptPath"
}

& $pythonExe $scriptPath @args
