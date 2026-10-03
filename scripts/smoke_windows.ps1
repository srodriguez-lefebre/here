param(
    [Parameter(Mandatory)][string]$BundlePath,
    [string]$InstallerPath,
    [string]$EvidenceDirectory,
    [string]$AppId = '{94BE3378-6B95-48A1-BD13-C534746C7319}',
    [string]$Group = 'here'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $IsWindows) { throw 'Windows runtime acceptance requires Windows' }
$taskBundle = (Resolve-Path -LiteralPath $BundlePath).Path
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $EvidenceDirectory) { $EvidenceDirectory = Join-Path $taskRoot 'dist/windows/evidence' }
$taskEvidence = [IO.Path]::GetFullPath($EvidenceDirectory)
New-Item -ItemType Directory -Force -Path $taskEvidence | Out-Null
$taskOwned = Join-Path ([IO.Path]::GetTempPath()) ('here-acceptance-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskOwned | Out-Null
$taskData = Join-Path $taskOwned 'synthetic data'
New-Item -ItemType Directory -Path $taskData | Out-Null
$taskCanary = Join-Path $taskData 'preserve.txt'
[IO.File]::WriteAllText($taskCanary, 'owned synthetic user data must survive')
$taskCanaryHash = (Get-FileHash -LiteralPath $taskCanary).Hash
$taskEnvironment = @{}
foreach ($taskVariable in @('HERE_DATA_DIR','HERE_ENV_FILE','HERE_SETTINGS_FILE','OPENAI_API_KEY','PYTHONPATH','QT_QPA_PLATFORM')) {
    $taskEnvironment[$taskVariable] = [Environment]::GetEnvironmentVariable($taskVariable, 'Process')
}
$env:HERE_DATA_DIR = $taskData
$env:HERE_ENV_FILE = Join-Path $taskOwned 'absent.env'
$env:HERE_SETTINGS_FILE = Join-Path $taskOwned 'preferences.ini'
Remove-Item Env:OPENAI_API_KEY, Env:PYTHONPATH, Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
$taskResult = [ordered]@{schema=1; os=[Environment]::OSVersion.VersionString; bundle=$taskBundle;
    installer=$InstallerPath; app_id=$AppId; group=$Group; owned_root=$taskOwned;
    status='failed'; checks=[ordered]@{} }
$taskGui = $null
$taskInstalled = $false
$taskProgram = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) ('Programs/here-acceptance-' + [Guid]::NewGuid().ToString('N'))
$taskRegistry = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\' + $AppId + '_is1'
$taskStart = Join-Path ([Environment]::GetFolderPath('Programs')) $Group

function Invoke-Process([string]$Path, [string[]]$Arguments, [int]$Timeout = 60) {
    $taskProcess = Start-Process -FilePath $Path -ArgumentList $Arguments -WorkingDirectory $taskOwned -PassThru -WindowStyle Hidden
    if (-not $taskProcess.WaitForExit($Timeout * 1000)) {
        $taskProcess.Kill()
        $taskProcess.WaitForExit()
        throw "Owned acceptance process timed out: $Path"
    }
    return $taskProcess.ExitCode
}

function Assert-Canary {
    if (-not (Test-Path -LiteralPath $taskCanary) -or (Get-FileHash -LiteralPath $taskCanary).Hash -ne $taskCanaryHash) {
        throw 'Synthetic user-data canary changed'
    }
}

function Assert-NoAutomaticStartup {
    $taskRun = Get-Item -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -ErrorAction SilentlyContinue
    if ($taskRun -and $null -ne $taskRun.GetValue('here', $null)) { throw 'Unexpected here automatic startup entry' }
    $taskStartup = [Environment]::GetFolderPath('Startup')
    if (Test-Path -LiteralPath (Join-Path $taskStartup 'here.lnk')) { throw 'Unexpected here Startup shortcut' }
}

function Check-Bundle([string]$Path, [string]$Label) {
    foreach ($taskExecutable in @('here.exe','here-cli.exe')) {
        $taskReport = Join-Path $taskEvidence ($Label + '-' + $taskExecutable + '.json')
        $taskExit = Invoke-Process (Join-Path $Path $taskExecutable) @('--smoke-check', ('"' + $taskReport + '"'))
        if ($taskExit -ne 0) { throw "Frozen startup failed for $taskExecutable, exit $taskExit" }
        $taskSmoke = Get-Content -Raw -LiteralPath $taskReport | ConvertFrom-Json
        if ($taskSmoke.status -ne 'ok' -or -not $taskSmoke.frozen -or $taskSmoke.checks.provider_configured -or $taskSmoke.checks.hardware_opened -or $taskSmoke.checks.recording_started -or $taskSmoke.checks.editable_finder) { throw 'Frozen smoke violated startup contract' }
        if ($taskExecutable -eq 'here.exe' -and -not $taskSmoke.checks.windowed_stdio_absent) { throw 'GUI unexpectedly has Python console stdio' }
    }
    $taskResult.checks[$Label] = $true
}

try {
    Check-Bundle $taskBundle 'offrepo-no-key'
    Assert-Canary
    if ($InstallerPath) {
        $InstallerPath = (Resolve-Path -LiteralPath $InstallerPath).Path
        # Stable delivered AppId is safe only when all affected entries are absent.
        # This preflight also prevents local tests from overwriting another user's
        # program directory or Start entries. No real installation is uninstalled.
        if (Test-Path -LiteralPath $taskRegistry) { throw 'Existing installation AppId: refusing local acceptance collision' }
        $taskOtherRegistry = 'HKCU:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\' + $AppId + '_is1'
        if (Test-Path -LiteralPath $taskOtherRegistry) { throw 'Existing 32-bit AppId: refusing collision' }
        foreach ($taskMachineRegistry in @('HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\', 'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\')) {
            if (Test-Path -LiteralPath ($taskMachineRegistry + $AppId + '_is1')) { throw 'Existing machine installation AppId: refusing collision' }
        }
        if (Test-Path -LiteralPath $taskStart) { throw 'Existing Start group: refusing collision' }
        if (Test-Path -LiteralPath $taskProgram) { throw 'Owned program directory collision' }
        Assert-NoAutomaticStartup
        $taskInstallArguments = @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/CURRENTUSER',('/DIR="' + $taskProgram + '"'),('/LOG="' + (Join-Path $taskEvidence 'install.log') + '"'))
        if ((Invoke-Process $InstallerPath $taskInstallArguments 120) -ne 0) { throw 'Current-user installation failed' }
        $taskInstalled = $true
        if (-not (Test-Path -LiteralPath $taskRegistry)) { throw 'Missing current-user uninstall entry' }
        $taskRegistration = Get-ItemProperty -LiteralPath $taskRegistry
        if ($taskRegistration.InstallLocation.TrimEnd('\') -ne $taskProgram.TrimEnd('\')) { throw 'Unexpected registered install location' }
        $taskShortcut = Join-Path $taskStart 'here.lnk'
        $taskUninstallShortcut = Join-Path $taskStart 'Uninstall here.lnk'
        if (-not (Test-Path -LiteralPath $taskShortcut) -or -not (Test-Path -LiteralPath $taskUninstallShortcut)) { throw 'Missing current-user Start entries' }
        $taskShell = New-Object -ComObject WScript.Shell
        $taskLink = $taskShell.CreateShortcut($taskShortcut)
        if ($taskLink.TargetPath -ne (Join-Path $taskProgram 'here.exe')) { throw 'Unexpected shortcut target' }
        [Runtime.InteropServices.Marshal]::ReleaseComObject($taskShell) | Out-Null
        $taskResult.checks.current_user_entries = $true
        Assert-NoAutomaticStartup
        $taskResult.checks.no_automatic_startup = $true
        Check-Bundle $taskProgram 'installed-no-key'
        $taskPayload = Get-Content -Raw -LiteralPath (Join-Path $taskBundle '_internal/notices/bundle-inventory.json') | ConvertFrom-Json
        foreach ($taskFile in $taskPayload.files) {
            $taskInstalledFile = Join-Path $taskProgram $taskFile.path
            if ((Get-FileHash -LiteralPath $taskInstalledFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskFile.sha256) { throw "Installed payload differs: $($taskFile.path)" }
        }
        $taskResult.checks.installer_payload_matches_bundle = $true
        Assert-Canary
        # Actually launch via the current user's Start shortcut. The process is
        # identified only by the synthetic install's exact executable path.
        # This is the interactive product GUI under test, not a background helper.
        # SW_HIDE on a shortcut can suppress its first native window and invalidate
        # the real Start/visible-window acceptance check.
        Start-Process -FilePath $taskShortcut -WorkingDirectory $taskOwned -WindowStyle Normal
        $taskDeadline = [DateTime]::UtcNow.AddSeconds(20)
        while ([DateTime]::UtcNow -lt $taskDeadline) {
            $taskGui = Get-Process -Name 'here' -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq (Join-Path $taskProgram 'here.exe') } | Select-Object -First 1
            if ($taskGui) { $taskGui.Refresh(); if ($taskGui.MainWindowHandle -ne [IntPtr]::Zero) { break } }
            Start-Sleep -Milliseconds 100
        }
        if (-not $taskGui -or $taskGui.MainWindowHandle -eq [IntPtr]::Zero) { throw 'Start shortcut did not open the real GUI' }
        $taskResult.checks.start_shortcut_gui_opened = $true
        $taskBefore = (Get-FileHash -LiteralPath (Join-Path $taskProgram 'here.exe')).Hash
        $taskBlockedArgs = $taskInstallArguments + @('/LOG="' + (Join-Path $taskEvidence 'blocked-update.log') + '"')
        $taskBlockedExit = Invoke-Process $InstallerPath $taskBlockedArgs 30
        if ($taskBlockedExit -eq 0 -or $taskGui.HasExited -or (Get-FileHash -LiteralPath (Join-Path $taskProgram 'here.exe')).Hash -ne $taskBefore) { throw 'Installer did not refuse active application replacement' }
        $taskResult.checks.active_update_refused_exit = $taskBlockedExit
        if (-not $taskGui.CloseMainWindow() -or -not $taskGui.WaitForExit(10000)) { throw 'Owned idle GUI did not exit safely' }
        $taskResult.checks.idle_gui_exit = $true
        $taskGui = $null
        Assert-Canary
        if ((Invoke-Process $InstallerPath ($taskInstallArguments + @('/LOG="' + (Join-Path $taskEvidence 'reinstall.log') + '"')) 120) -ne 0) { throw 'Reinstallation failed' }
        Assert-Canary
        Check-Bundle $taskProgram 'reinstalled-no-key'
        $taskResult.checks.reinstall_preserves_data = $true
        $taskUninstaller = Join-Path $taskProgram 'unins000.exe'
        if ((Invoke-Process $taskUninstaller @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',('/LOG="' + (Join-Path $taskEvidence 'uninstall.log') + '"')) 120) -ne 0) { throw 'Owned current-user uninstall failed' }
        $taskInstalled = $false
        if ((Test-Path -LiteralPath (Join-Path $taskProgram 'here.exe')) -or (Test-Path -LiteralPath $taskRegistry) -or (Test-Path -LiteralPath $taskStart)) { throw 'Owned program/registration/Start entry remains after uninstall' }
        Assert-Canary
        Assert-NoAutomaticStartup
        $taskResult.checks.uninstall_preserves_data = $true
    }
    $taskResult.status = 'ok'
} catch {
    $taskResult.error = $_.Exception.Message
    throw
} finally {
    if ($taskGui -and -not $taskGui.HasExited) { $taskGui.CloseMainWindow() | Out-Null; if (-not $taskGui.WaitForExit(10000)) { $taskGui.Kill(); $taskGui.WaitForExit() } }
    # Failure cleanup may only uninstall the exact synthetic directory that this
    # script registered after its preflight. Preserve data/evidence for inspection.
    if ($taskInstalled -and (Test-Path -LiteralPath $taskRegistry)) {
        $taskOwnedRegistration = Get-ItemProperty -LiteralPath $taskRegistry
        if ($taskOwnedRegistration.InstallLocation.TrimEnd('\') -eq $taskProgram.TrimEnd('\')) {
            Invoke-Process (Join-Path $taskProgram 'unins000.exe') @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART') 120 | Out-Null
        }
    }
    $taskResult | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskEvidence 'acceptance.json') -Encoding utf8
    foreach ($taskVariable in $taskEnvironment.Keys) { [Environment]::SetEnvironmentVariable($taskVariable, $taskEnvironment[$taskVariable], 'Process') }
}
