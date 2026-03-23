<#
.SYNOPSIS
    Creates or updates Windows Task Scheduler tasks for MLB Prop Predictor automation.

.DESCRIPTION
    Sets up three scheduled tasks:
    1. MLB-BoardCapture   — every 2 hours, 8 AM - 6 PM Pacific daily
    2. MLB-NightlyCycle   — daily at 11:45 PM Pacific
    3. MLB-WeeklyTune     — Sundays at 4:00 AM Pacific

    Run this script once from an elevated (Admin) PowerShell prompt.
    Re-running updates the tasks in place.

.NOTES
    Requires: Administrator privileges (for schtasks / Register-ScheduledTask)
    Times are specified in Pacific. Task Scheduler converts to local time.
#>

param(
    [switch]$Remove,       # Remove all tasks instead of creating them
    [switch]$DryRun,       # Show what would happen without making changes
    [string]$PythonPath    # Override: full path to python.exe (auto-detected if omitted)
)

$ErrorActionPreference = "Stop"

# ── Resolve paths ────────────────────────────────────────────────────────────
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir

# Auto-detect Python
if (-not $PythonPath) {
    $venvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $PythonPath = $venvPython
    } else {
        $PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
        if (-not $PythonPath) {
            Write-Error "Cannot find python.exe. Pass -PythonPath or activate your venv."
            exit 1
        }
    }
}

Write-Host "Project root : $ProjectRoot"
Write-Host "Python       : $PythonPath"
Write-Host ""

# ── Task definitions ─────────────────────────────────────────────────────────
# NOTE: Pacific times — Task Scheduler handles DST automatically if the
# system time zone is Pacific. If you're in a different zone, adjust the
# trigger times or use the /TN option to set the timezone explicitly.

$Tasks = @(
    @{
        Name        = "MLB-BoardCapture"
        Description = "MLB Prop Predictor: headless board capture (every 2h, 8AM-6PM PT)"
        Script      = Join-Path $ScriptDir "board_capture.py"
        # Task Scheduler "daily trigger" + repetition interval approach:
        # Start at 8:00 AM, repeat every 2 hours for 10 hours (= 6 PM last run)
        TriggerTime = "08:00"
        RepeatInterval = "PT2H"
        RepeatDuration = "PT10H"
        DaysOfWeek  = $null  # daily
    },
    @{
        Name        = "MLB-NightlyCycle"
        Description = "MLB Prop Predictor: nightly auto-grade + metrics (11:45 PM PT)"
        Script      = Join-Path $ScriptDir "nightly_cycle.py"
        TriggerTime = "23:45"
        RepeatInterval = $null
        RepeatDuration = $null
        DaysOfWeek  = $null  # daily
    },
    @{
        Name        = "MLB-WeeklyTune"
        Description = "MLB Prop Predictor: weekly offline model tuning (Sun 4 AM PT)"
        Script      = Join-Path $ScriptDir "weekly_tune.py"
        TriggerTime = "04:00"
        RepeatInterval = $null
        RepeatDuration = $null
        DaysOfWeek  = "Sunday"
    }
)

# ── Remove mode ──────────────────────────────────────────────────────────────
if ($Remove) {
    foreach ($task in $Tasks) {
        $existing = Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue
        if ($existing) {
            if ($DryRun) {
                Write-Host "[DRY RUN] Would remove task: $($task.Name)"
            } else {
                Unregister-ScheduledTask -TaskName $task.Name -Confirm:$false
                Write-Host "Removed task: $($task.Name)"
            }
        } else {
            Write-Host "Task not found (skipping): $($task.Name)"
        }
    }
    exit 0
}

# ── Create / update tasks ───────────────────────────────────────────────────
foreach ($task in $Tasks) {
    Write-Host "---"
    Write-Host "Task: $($task.Name)"
    Write-Host "  $($task.Description)"

    # Build the action: run python with the script
    $action = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument "`"$($task.Script)`"" `
        -WorkingDirectory $ProjectRoot

    # Build the trigger
    if ($task.DaysOfWeek) {
        $trigger = New-ScheduledTaskTrigger `
            -Weekly -DaysOfWeek $task.DaysOfWeek `
            -At $task.TriggerTime
    } else {
        $trigger = New-ScheduledTaskTrigger `
            -Daily -At $task.TriggerTime
    }

    # Add repetition for board capture (every 2h for 10h)
    if ($task.RepeatInterval) {
        $trigger.Repetition.Interval = $task.RepeatInterval
        $trigger.Repetition.Duration = $task.RepeatDuration
        $trigger.Repetition.StopAtDurationEnd = $true
    }

    # Settings
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

    # Principal: run as current user, highest privileges not needed
    $principal = New-ScheduledTaskPrincipal `
        -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType S4U `
        -RunLevel Limited

    $existing = Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue

    if ($DryRun) {
        $verb = if ($existing) { "update" } else { "create" }
        Write-Host "  [DRY RUN] Would $verb task at $($task.TriggerTime)"
        if ($task.RepeatInterval) {
            Write-Host "  [DRY RUN] Repeats every $($task.RepeatInterval) for $($task.RepeatDuration)"
        }
        continue
    }

    if ($existing) {
        # Update in place
        Set-ScheduledTask `
            -TaskName $task.Name `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal | Out-Null
        Write-Host "  Updated existing task."
    } else {
        Register-ScheduledTask `
            -TaskName $task.Name `
            -Description $task.Description `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal | Out-Null
        Write-Host "  Created new task."
    }
}

Write-Host ""
Write-Host "All tasks configured. Verify with:"
Write-Host "  Get-ScheduledTask -TaskName 'MLB-*' | Format-Table TaskName, State, LastRunTime"
Write-Host ""
Write-Host "To run a task manually:"
Write-Host "  Start-ScheduledTask -TaskName 'MLB-BoardCapture'"
