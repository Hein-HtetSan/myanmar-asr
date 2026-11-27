<#
.SYNOPSIS
  Run a training notebook unattended and resume it automatically after power cuts.

.DESCRIPTION
  Designed for a Windows GPU machine with unreliable mains power.

  Each attempt executes the notebook top to bottom with nbconvert. The notebook
  itself picks the newest checkpoint that passes validation and reuses the saved
  MLflow run, so rerunning it is how training resumes. The script:

    * exits at once if the run already finished (test_metrics.json exists);
    * allows only one copy per model (a second launch exits);
    * starts Docker Desktop and the notebook MLflow service and waits for it;
    * keeps Windows from sleeping while training runs;
    * retries a failed attempt (CUDA hiccup, MLflow not ready) up to -MaxAttempts;
    * writes a transcript and each executed notebook to notebook/runs/autorun/.

  With -Install it registers a Scheduled Task that runs this script at logon,
  so training restarts by itself when power returns. For that to happen with
  nobody at the machine, also enable in BIOS/UEFI "Restore on AC Power Loss"
  (= Power On) and Windows automatic sign-in. See notebook/README.md.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\infra\autoresume_train.ps1 -Model seamless
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\infra\autoresume_train.ps1 -Model seamless -Install
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\infra\autoresume_train.ps1 -Model seamless -Uninstall
#>
[CmdletBinding()]
param(
    [ValidateSet('seamless', 'whisper-turbo', 'dolphin')]
    [string]$Model = 'seamless',
    # Must match RUN_LABEL in the notebook; full runs only (QUICK_RUN = False).
    [string]$RunLabel = 'baseline',
    # Python of the training environment (the one with torch/transformers/jupyterlab).
    [string]$Python = '',
    [int]$MaxAttempts = 20,
    [int]$RetryDelaySeconds = 120,
    [switch]$Install,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Python) {
    $Python = @('venv\Scripts\python.exe', '.venv-training\Scripts\python.exe') |
        ForEach-Object { Join-Path $Root $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path $Python)) {
    throw "Training Python not found. Pass -Python <path to python.exe>."
}
$TaskName = "myanmar-asr-autoresume-$Model"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task $TaskName"
    return
}

if ($Install) {
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Model $Model " +
                 "-RunLabel $RunLabel -Python `"$Python`" -MaxAttempts $MaxAttempts"
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $trigger.Delay = 'PT2M'  # Let Docker Desktop and drivers settle after boot.
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -DontStopOnIdleEnd -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Description "Resume Myanmar ASR $Model training after power loss" -Force | Out-Null
    Write-Host "Registered scheduled task $TaskName (runs at logon)."
    Write-Host "Also enable BIOS 'Restore on AC Power Loss' and Windows automatic sign-in."
    return
}

$Notebook = Join-Path $Root "notebook\$Model\train.ipynb"
$RunDir = Join-Path $Root "notebook\runs\$Model\full\$RunLabel"
$Done = Join-Path $RunDir 'test_metrics.json'
$LogDir = Join-Path $Root 'notebook\runs\autorun'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Start-Transcript -Path (Join-Path $LogDir "$Model-$stamp.log") -Append | Out-Null

# One copy per model: a scheduled start and a manual start must not train together.
$mutex = New-Object System.Threading.Mutex($false, "Global\myanmar-asr-train-$Model")
if (-not $mutex.WaitOne(0)) {
    Write-Host "Another $Model training run is already active; exiting."
    Stop-Transcript | Out-Null
    return
}

# Keep the PC awake while this process lives (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]'0x80000001') | Out-Null

function Wait-Until([scriptblock]$Condition, [int]$TimeoutSeconds, [string]$What) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Condition) { return }
        Start-Sleep -Seconds 5
    }
    throw "Timed out waiting for $What"
}

function Test-Docker {
    & docker info *> $null
    return $LASTEXITCODE -eq 0
}

function Start-MLflow {
    if (-not (Test-Docker)) {
        $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
        if (Test-Path $desktop) { Start-Process $desktop }
        Wait-Until { Test-Docker } 600 'Docker'
    }
    & docker compose -f (Join-Path $Root 'services\mlflow-notebooks.compose.yml') up -d
    if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed' }
    $port = if ($env:MLFLOW_NOTEBOOK_PORT) { $env:MLFLOW_NOTEBOOK_PORT } else { '5050' }
    if (-not $env:MLFLOW_TRACKING_URI) { $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:$port" }
    Wait-Until {
        try { (Invoke-WebRequest "$env:MLFLOW_TRACKING_URI/health" -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200 }
        catch { $false }
    } 300 'MLflow'
    Write-Host "MLflow ready at $env:MLFLOW_TRACKING_URI"
}

try {
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        if (Test-Path $Done) {
            Write-Host "Run already finished: $Done"
            break
        }
        Write-Host "=== $Model attempt $attempt/$MaxAttempts at $(Get-Date -Format s) ==="
        try {
            Start-MLflow
            $output = "$Model-$(Get-Date -Format 'yyyyMMdd-HHmmss')-attempt$attempt.ipynb"
            # kernel_name=python3 runs the notebook in this same training Python.
            & $Python -m jupyter nbconvert --to notebook --execute $Notebook `
                --output-dir $LogDir --output $output `
                --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Notebook finished. Executed copy: $(Join-Path $LogDir $output)"
                break
            }
            Write-Warning "Notebook failed (exit $LASTEXITCODE). See $(Join-Path $LogDir $output)"
        } catch {
            Write-Warning "Attempt $attempt failed: $_"
        }
        if ($attempt -lt $MaxAttempts) {
            Write-Host "Retrying in $RetryDelaySeconds s..."
            Start-Sleep -Seconds $RetryDelaySeconds
        } else {
            Write-Warning 'Giving up after the maximum number of attempts.'
        }
    }
} finally {
    [Win32.Power]::SetThreadExecutionState([uint32]'0x80000000') | Out-Null
    $mutex.ReleaseMutex()
    Stop-Transcript | Out-Null
}
