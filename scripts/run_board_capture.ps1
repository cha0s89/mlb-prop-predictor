<# Board Capture — PowerShell wrapper for Windows Task Scheduler
   Runs the headless board-build pipeline.
   Scheduled: every 2 hours, 8 AM - 6 PM Pacific daily.
#>
$ErrorActionPreference = "Stop"

# Resolve project root (parent of scripts/)
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir

Push-Location $ProjectRoot
try {
    # Activate venv if it exists
    $venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        & $venvActivate
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting board capture..."
    python scripts/board_capture.py @args
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Board capture completed successfully."
    } elseif ($exitCode -eq 1) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Board capture completed with warnings (exit $exitCode)."
    } else {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Board capture FAILED (exit $exitCode)."
    }

    exit $exitCode
} finally {
    Pop-Location
}
