<# One-shot preseason backtest + tuning wrapper.
   Rebuilds the historical backtest, then runs the offline tuner.
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

    $env:PYTHONUTF8 = "1"
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting preseason training..."
    python scripts/preseason_train.py @args
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Preseason training completed successfully."
    } elseif ($exitCode -eq 1) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Preseason training completed with warnings (exit $exitCode)."
    } else {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Preseason training FAILED (exit $exitCode)."
    }

    exit $exitCode
} finally {
    Pop-Location
}
