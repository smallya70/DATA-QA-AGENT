$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$devDir = Join-Path $projectRoot ".dev"
$backendPidFile = Join-Path $devDir "backend.pid"
$frontendPidFile = Join-Path $devDir "frontend.pid"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment Python not found at $pythonExe"
}

if (-not (Test-Path $devDir)) {
    New-Item -ItemType Directory -Path $devDir | Out-Null
}

$stopScript = Join-Path $projectRoot "stop-dev.ps1"
if (Test-Path $stopScript) {
    & $stopScript | Out-Null
}

$backendProc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8092" `
    -WorkingDirectory $projectRoot `
    -PassThru

Start-Sleep -Seconds 1

$frontendProc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501" `
    -WorkingDirectory $projectRoot `
    -PassThru

Set-Content -Path $backendPidFile -Value $backendProc.Id
Set-Content -Path $frontendPidFile -Value $frontendProc.Id

Write-Host "Started FastAPI on http://127.0.0.1:8092 (PID $($backendProc.Id))" -ForegroundColor Green
Write-Host "Started Streamlit on http://127.0.0.1:8501 (PID $($frontendProc.Id))" -ForegroundColor Green
