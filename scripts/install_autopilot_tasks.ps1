# Creates the three GBM forward tasks and the encrypted daily backup task.
# Does not change Windows timezone or store credentials in task XML.
[CmdletBinding()]
param(
    [switch]$Preview,
    [switch]$Unattended,
    [switch]$Replace,
    [string[]]$Symbols = @("SMCI", "NVDA")
)
$ErrorActionPreference = "Stop"
$projectPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$pythonPath = Join-Path $projectPath ".venv\Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Missing project Python: $pythonPath" }
foreach ($symbol in $Symbols) {
    if ($symbol -cnotmatch '^[A-Z][A-Z0-9.\-]{0,14}$') { throw "Invalid symbol: $symbol" }
}
$taskUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$taskCredential = $null
if ($Unattended -and -not $Preview) {
    $taskCredential = Get-Credential -UserName $taskUser -Message "Windows credentials for the three GBM tasks (never written to project files)"
    if ($null -eq $taskCredential) { throw "Cancelled: no tasks installed." }
    $taskUser = $taskCredential.UserName
    $account = [Security.Principal.NTAccount]::new($taskUser)
    $taskSid = $account.Translate([Security.Principal.SecurityIdentifier]).Value
}
function Escape-Xml([string]$Value) { return [Security.SecurityElement]::Escape($Value) }
$mode = if ($Unattended) { "Password" } else { "InteractiveToken" }
$taskDefinitions = @(
    @{ Name = "GBM_Forward_Collector"; Script = "daily_auto_collector.py"; Hours = @("15:00:00", "16:00:00"); Arguments = "--scheduled --symbols " + ($Symbols -join " ") },
    @{ Name = "GBM_Forward_Resolver"; Script = "auto_resolver.py"; Hours = @("21:00:00", "22:00:00"); Arguments = "--scheduled --symbols " + ($Symbols -join " ") },
    @{ Name = "GBM_Forward_Catchup"; Script = "boot_catchup.py"; Hours = @("13:05:00", "14:05:00"); Arguments = "--scheduled --symbols " + ($Symbols -join " ") },
    @{ Name = "GBM_Backup_Daily"; Script = "github_backup.py"; Hours = @("22:00:00", "23:00:00"); Arguments = "--encrypt" }
)
$generated = @()
foreach ($definition in $taskDefinitions) {
    $triggerXml = ""
    foreach ($hour in $definition.Hours) {
        # Two UTC candidates handle US DST independently of Windows local zone.
        # Python --scheduled admits only the correct New York wall-clock window.
        $triggerXml += @"
<CalendarTrigger><StartBoundary>2026-01-05T$($hour)Z</StartBoundary><Enabled>true</Enabled>
<ScheduleByWeek><WeeksInterval>1</WeeksInterval><DaysOfWeek>
<Monday/><Tuesday/><Wednesday/><Thursday/><Friday/>
</DaysOfWeek></ScheduleByWeek></CalendarTrigger>
"@
    }
    if ($definition.Name -eq "GBM_Forward_Catchup") {
        if ($Unattended) {
            $triggerXml += "<BootTrigger><Enabled>true</Enabled><Delay>PT5M</Delay></BootTrigger>"
        } else {
            $triggerXml += "<LogonTrigger><Enabled>true</Enabled><Delay>PT5M</Delay><UserId>$(Escape-Xml $taskSid)</UserId></LogonTrigger>"
        }
    }
    $scriptPath = Join-Path $PSScriptRoot $definition.Script
    if (-not (Test-Path -LiteralPath $scriptPath)) { throw "Missing script: $scriptPath" }
    $arguments = '"' + $scriptPath + '" ' + $definition.Arguments
    $xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
<RegistrationInfo><Description>GBM evidence/backup automation. No trading orders.</Description></RegistrationInfo>
<Triggers>$triggerXml</Triggers>
<Principals><Principal id="Author"><UserId>$(Escape-Xml $taskSid)</UserId><LogonType>$mode</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
<Settings>
<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
<AllowHardTerminate>true</AllowHardTerminate>
<StartWhenAvailable>true</StartWhenAvailable>
<RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
<AllowStartOnDemand>true</AllowStartOnDemand>
<Enabled>true</Enabled><Hidden>true</Hidden>
<WakeToRun>false</WakeToRun>
<ExecutionTimeLimit>PT15M</ExecutionTimeLimit>
<Priority>7</Priority>
<RestartOnFailure><Interval>PT5M</Interval><Count>3</Count></RestartOnFailure>
</Settings>
<Actions Context="Author"><Exec>
<Command>$(Escape-Xml $pythonPath)</Command>
<Arguments>$(Escape-Xml $arguments)</Arguments>
<WorkingDirectory>$(Escape-Xml $projectPath)</WorkingDirectory>
</Exec></Actions>
</Task>
"@
    [xml]$parsed = $xml
    $generated += @{ Name = $definition.Name; Xml = $xml }
}
if ($Preview) {
    $previewPath = Join-Path $projectPath "data\autopilot\task_definitions"
    New-Item -ItemType Directory -Force -Path $previewPath | Out-Null
    foreach ($item in $generated) {
        [IO.File]::WriteAllText((Join-Path $previewPath ($item.Name + ".xml")), $item.Xml, [Text.Encoding]::Unicode)
    }
    Write-Output "Preview only: $previewPath. No scheduled tasks were registered."
    return
}
# Preflight all names before installing anything. Never silently overwrite tasks.
foreach ($item in $generated) {
    $existing = Get-ScheduledTask -TaskName $item.Name -ErrorAction SilentlyContinue
    if ($existing -and -not $Replace) { throw "Task exists: $($item.Name). Inspect it; use -Replace only to replace these named tasks." }
}
foreach ($item in $generated) {
    $registration = @{ TaskName = $item.Name; Xml = $item.Xml; Force = [bool]$Replace }
    if ($Unattended) {
        # Credential reaches the Windows API in memory, never XML/logs/arguments.
        $registration.User = $taskUser
        $registration.Password = $taskCredential.GetNetworkCredential().Password
    }
    try {
        Register-ScheduledTask @registration | Select-Object TaskName, State
    } finally {
        $registration.Remove("Password")
    }
}
Write-Output "Installed. Collector 11:00 NY; resolver 17:00 NY; catch-up 09:05 NY + boot/logon; backup 18:00 NY."
if (-not $Unattended) { Write-Output "Interactive mode: the Windows user must be logged on. Use -Unattended for real boot execution." }
Write-Output "No task can start a powered-off PC. StartWhenAvailable recovers a missed backup after the next logon."
Write-Output "See logs\collector.log, logs\resolver.log, logs\catchup.log and Task Scheduler history."
