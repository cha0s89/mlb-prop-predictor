<# Nightly Cycle — PowerShell wrapper for Windows Task Scheduler
   Auto-grades, computes metrics, updates weights, checks drift.
   Scheduled: daily at 11:45 PM Pacific.
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

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting nightly cycle..."
    python scripts/nightly_cycle.py @args
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Nightly cycle completed successfully."
    } elseif ($exitCode -eq 1) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Nightly cycle completed with warnings (exit $exitCode)."
    } else {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Nightly cycle FAILED (exit $exitCode)."
    }

    exit $exitCode
} finally {
    Pop-Location
}
