<# Weekly Tune — PowerShell wrapper for Windows Task Scheduler
   Optimizes confidence floors and model parameters from backtest data.
   Scheduled: Sundays at 4:00 AM Pacific.
#>
$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir

Push-Location $ProjectRoot
try {
    $venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        & $venvActivate
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting weekly tune..."
    python scripts/weekly_tune.py @args
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Weekly tune completed successfully."
    } elseif ($exitCode -eq 1) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Weekly tune completed with warnings (exit $exitCode)."
    } else {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Weekly tune FAILED (exit $exitCode)."
    }

    exit $exitCode
} finally {
    Pop-Location
}
