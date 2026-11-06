# Register only research archive tasks. Run after prospective_credentials.py setup.
# Task Scheduler invokes the guard frequently; only the five fixed XNYS cuts
# can create snapshots. This avoids hard-coded Mexico/New York DST conversions.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$researchPython = Join-Path $projectRoot 'output\historical_replay\finbert_research_env\Scripts\pythonw.exe'
$runner = Join-Path $projectRoot 'scripts\prospective_credentials.py'
$vault = Join-Path $projectRoot 'output\prospective_archive\secrets\market_data_current_user.dpapi'
if (-not (Test-Path -LiteralPath $researchPython)) { throw 'No existe el entorno FinBERT congelado.' }
if (-not (Test-Path -LiteralPath $vault)) { throw 'Falta la bóveda DPAPI. Ejecuta prospective_credentials.py setup primero.' }

$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$start = (Get-Date).AddMinutes(1)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 12)
$jobs = @(
    @{ Name='STOCK-Prospective-Collector'; Every=1; Task='collector'; Description='Solo cinco cortes XNYS; sin predicciones retrospectivas' },
    @{ Name='STOCK-Prospective-Labeler'; Every=15; Task='labeler'; Description='Etiqueta 1h/1D solo tras vencimiento XNYS y velas completas' },
    @{ Name='STOCK-Prospective-Coverage'; Every=60; Task='coverage'; Description='Reporte local de cobertura prospectiva firmado' }
)
foreach ($job in $jobs) {
    $action = New-ScheduledTaskAction -Execute $researchPython -Argument ('"' + $runner + '" run ' + $job.Task) -WorkingDirectory $projectRoot
    $trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes $job.Every) -RepetitionDuration (New-TimeSpan -Days 3650)
    Register-ScheduledTask -TaskName $job.Name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description $job.Description -Force | Out-Null
    Write-Host ('Registrada: ' + $job.Name + ' (cada ' + $job.Every + ' min; el código filtra XNYS)')
}

# A repeated minute trigger does NOT wake Windows from sleep. Register one
# independent wake-only task three minutes before each possible NY cut in the
# host's local time zone. The union covers both DST regimes; extra wake markers
# never collect data. The calendar guard in the collector remains authoritative.
$nyZone = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
$localZone = [TimeZoneInfo]::Local
$localWakeTimes = [System.Collections.Generic.HashSet[string]]::new()
foreach ($offset in 0..400) {
    $day = [datetime]::Today.AddDays($offset)
    foreach ($nyClock in @('10:00', '11:00', '12:30', '14:00', '15:30')) {
        $nyWall = [datetime]::SpecifyKind($day.Add([TimeSpan]::Parse($nyClock)), [DateTimeKind]::Unspecified)
        $asUtc = [TimeZoneInfo]::ConvertTimeToUtc($nyWall, $nyZone)
        $localWall = [TimeZoneInfo]::ConvertTimeFromUtc($asUtc, $localZone)
        [void]$localWakeTimes.Add($localWall.AddMinutes(-3).ToString('HH:mm'))
    }
}
$wakeTriggers = @($localWakeTimes | Sort-Object | ForEach-Object {
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
        -At ([datetime]::Today.Add([TimeSpan]::Parse($_)))
})
$wakeScript = Join-Path $projectRoot 'scripts\prospective_wake.py'
$wakeAction = New-ScheduledTaskAction -Execute $researchPython -Argument ('"' + $wakeScript + '"') -WorkingDirectory $projectRoot
$wakeSettings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
Register-ScheduledTask -TaskName 'STOCK-Prospective-Cut-Wake' -Action $wakeAction `
    -Trigger $wakeTriggers -Settings $wakeSettings -Principal $principal `
    -Description 'Despierta Windows antes de cortes XNYS; no descarga ni firma predicciones' -Force | Out-Null
Write-Host ('Registrada: STOCK-Prospective-Cut-Wake (WakeToRun; horas locales: ' + (($localWakeTimes | Sort-Object) -join ', ') + ')')
