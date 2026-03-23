<#
.SYNOPSIS
    Creates or updates Windows Task Scheduler tasks for MLB Prop Predictor automation.
#>
param(
    [switch]$Remove,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir
$PowerShellExe = (Get-Command powershell.exe).Source

$tasks = @(
    @{
        Name = 'MLB-BoardCapture'
        Description = 'MLB Prop Predictor: headless board capture'
        Wrapper = Join-Path $ScriptDir 'run_board_capture.ps1'
        Args = @('/SC','HOURLY','/MO','2','/ST','08:00','/ET','18:00')
    },
    @{
        Name = 'MLB-NightlyCycle'
        Description = 'MLB Prop Predictor: nightly auto-grade + metrics'
        Wrapper = Join-Path $ScriptDir 'run_nightly.ps1'
        Args = @('/SC','DAILY','/ST','23:45')
    },
    @{
        Name = 'MLB-WeeklyTune'
        Description = 'MLB Prop Predictor: weekly offline model tuning'
        Wrapper = Join-Path $ScriptDir 'run_weekly_tune.ps1'
        Args = @('/SC','WEEKLY','/D','SUN','/ST','04:00')
    }
)

function Invoke-Schtasks([string[]]$Arguments, [string]$Display) {
    if ($DryRun) {
        Write-Host "  [DRY RUN] schtasks $Display"
        return
    }
    & schtasks.exe @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks failed: $Display"
    }
}

foreach ($task in $tasks) {
    Write-Host '---'
    Write-Host "Task: $($task.Name)"
    Write-Host "  $($task.Description)"
    Write-Host "  Wrapper: $($task.Wrapper)"

    if ($Remove) {
        Invoke-Schtasks -Arguments @('/Delete','/F','/TN',$task.Name) -Display "/Delete /F /TN $($task.Name)"
        continue
    }

    $taskCommand = '"' + $PowerShellExe + '" -ExecutionPolicy Bypass -File "' + $task.Wrapper + '"'
    $args = @('/Create','/F','/TN',$task.Name,'/TR',$taskCommand) + $task.Args
    Invoke-Schtasks -Arguments $args -Display ($args -join ' ')
}

Write-Host ''
Write-Host "Verify with: Get-ScheduledTask -TaskName 'MLB-*' | Format-Table TaskName,State,LastRunTime,NextRunTime"
