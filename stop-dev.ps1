$ErrorActionPreference = "SilentlyContinue"

$projectRoot = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$devDir = Join-Path $projectRoot ".dev"
$backendPidFile = Join-Path $devDir "backend.pid"
$frontendPidFile = Join-Path $devDir "frontend.pid"

$stopped = 0
$stoppedPids = @{}

function Stop-DevProcessByPidFile {
    param(
        [string]$PidFilePath
    )

    if (-not (Test-Path $PidFilePath)) {
        return
    }

    $pidText = (Get-Content -Path $PidFilePath -Raw).Trim()
    if ($pidText -match "^\d+$") {
        $pidValue = [int]$pidText
        if (-not $script:stoppedPids.ContainsKey($pidValue)) {
            try {
                taskkill /PID $pidValue /T /F | Out-Null
                $script:stoppedPids[$pidValue] = $true
                $script:stopped++
            }
            catch {
            }
        }
    }

    Remove-Item -Path $PidFilePath -Force -ErrorAction SilentlyContinue
}

Stop-DevProcessByPidFile -PidFilePath $backendPidFile
Stop-DevProcessByPidFile -PidFilePath $frontendPidFile

$projectRootEscaped = [Regex]::Escape($projectRoot)
$targets = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -and
        $_.CommandLine -match $projectRootEscaped -and
        (
            $_.CommandLine -like "*uvicorn app.main:app*" -or
            $_.CommandLine -like "*uvicorn app.main:app --port 8099*" -or
            $_.CommandLine -like "*streamlit run app/streamlit_app.py*" -or
            $_.CommandLine -like "*streamlit run app\streamlit_app.py*"
        )
    }

foreach ($proc in $targets) {
    if (-not $stoppedPids.ContainsKey($proc.ProcessId)) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            $stoppedPids[$proc.ProcessId] = $true
            $stopped++
        }
        catch {
        }
    }
}

$listenerPorts = @(8092, 8094, 8099, 8501)
foreach ($port in $listenerPorts) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $listenerPid = $listener.OwningProcess
        if ($stoppedPids.ContainsKey($listenerPid)) {
            continue
        }

        $listenerProc = Get-CimInstance Win32_Process -Filter "ProcessId = $listenerPid" -ErrorAction SilentlyContinue
        if ($listenerProc -and $listenerProc.Name -eq "python.exe") {
            try {
                taskkill /PID $listenerPid /T /F | Out-Null
                $stoppedPids[$listenerPid] = $true
                $stopped++
            }
            catch {
            }
        }
    }
}

for ($attempt = 1; $attempt -le 5; $attempt++) {
    $activePids = @()
    foreach ($port in $listenerPorts) {
        $activePids += Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess
    }

    $activePids = $activePids | Where-Object { $_ } | Select-Object -Unique
    if (-not $activePids) {
        break
    }

    foreach ($activePid in $activePids) {
        if (-not $stoppedPids.ContainsKey($activePid)) {
            try {
                taskkill /PID $activePid /T /F | Out-Null
                $stoppedPids[$activePid] = $true
                $stopped++
            }
            catch {
            }
        }
    }

    Start-Sleep -Milliseconds 300
}

if ($stopped -eq 0) {
    Write-Host "No matching FastAPI/Streamlit dev processes found." -ForegroundColor Yellow
    exit 0
}

Write-Host "Stopped $stopped dev process(es)." -ForegroundColor Green
