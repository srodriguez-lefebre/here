param(
    [string]$Python = '3.13.15',
    [string]$CompilerPath,
    [string]$OutputDirectory,
    [string]$TestAppId,
    [string]$TestGroup
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $IsWindows) { throw 'Build here on Windows x64' }
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $taskRoot 'dist/windows' }
$taskOutput = [IO.Path]::GetFullPath($OutputDirectory)
if ($taskOutput -eq $taskRoot -or -not $taskOutput.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Build output must be a child of this checkout'
}
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskBuild = Join-Path $taskRoot 'build/windows'
New-Item -ItemType Directory -Force -Path $taskBuild | Out-Null
$taskVenv = Join-Path $taskBuild 'venv'
$taskPython = Join-Path $taskVenv 'Scripts/python.exe'

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit $LASTEXITCODE" }
}

Push-Location $taskRoot
try {
    Invoke-Checked uv @('venv','--python',$Python,'--no-python-downloads','--allow-existing',$taskVenv)
    Invoke-Checked uv @('export','--frozen','--no-dev','--no-default-groups','--group','build','--no-emit-project',
        '--format','requirements.txt','--output-file',"$taskBuild/requirements.txt",'--quiet')
    Invoke-Checked uv @('pip','sync','--python',$taskPython,'--require-hashes','--only-binary',':all:',"$taskBuild/requirements.txt")
    Invoke-Checked $taskPython @('-I','-B','-c',"import sys,platform,importlib.util; assert sys.version_info[:3]==(3,13,15); assert platform.machine()=='AMD64'; assert importlib.util.find_spec('here') is None")
    Invoke-Checked uv @('build','--wheel','--no-build-isolation','--python',$taskPython,'--out-dir',"$taskBuild/wheel",'--quiet')
    $taskWheels = @(Get-ChildItem -LiteralPath "$taskBuild/wheel" -Filter 'here-0.2.0-*.whl')
    if ($taskWheels.Count -ne 1) { throw 'Expected exactly one versioned build wheel' }
    $taskWheel = $taskWheels[0]
    Invoke-Checked uv @('pip','install','--python',$taskPython,'--no-deps',$taskWheel.FullName)
    Invoke-Checked $taskPython @('-I','-B','-c',"import here,sys,importlib.metadata; assert 'site-packages' in here.__file__; assert importlib.metadata.version('here')=='0.2.0'; assert not any(n.startswith('__editable__') for n in sys.modules)")
    Invoke-Checked $taskPython @('-B',"$taskRoot/scripts/windows_notices.py",'prepare',"$taskBuild/notices",'--wheel',$taskWheel.FullName)
    $env:HERE_BUILD_NOTICES = "$taskBuild/notices"
    # Native dependency resolution must not borrow DLLs from arbitrary developer
    # PATH entries (for example an unrelated image/codec tool's ICU/UCRT build).
    $taskOriginalPath = $env:PATH
    $taskRuntimeBase = (& $taskPython -I -B -c 'import sys; print(sys.base_prefix)').Trim()
    $env:PATH = @((Join-Path $taskVenv 'Scripts'), $taskRuntimeBase,
        (Join-Path $taskRuntimeBase 'DLLs'), (Join-Path $env:SystemRoot 'System32'), $env:SystemRoot) -join ';'
    try {
        Invoke-Checked $taskPython @('-I','-B','-m','PyInstaller','--clean','--noconfirm','--distpath',$taskOutput,
            '--workpath',"$taskBuild/pyinstaller","$taskRoot/packaging/windows.spec")
    } finally { $env:PATH = $taskOriginalPath }
    Invoke-Checked $taskPython @('-B',"$taskRoot/scripts/windows_notices.py",'audit',"$taskOutput/here",'--wheel',$taskWheel.FullName)
    if (-not $CompilerPath) {
        $taskCompilerRoot = Join-Path $taskBuild 'inno-6.7.0'
        New-Item -ItemType Directory -Force -Path $taskCompilerRoot | Out-Null
        $taskCompilerDownload = Join-Path $taskBuild 'innosetup-6.7.0.exe'
        Invoke-WebRequest -Uri 'https://github.com/jrsoftware/issrc/releases/download/is-6_7_0/innosetup-6.7.0.exe' -OutFile $taskCompilerDownload
        if ((Get-FileHash -LiteralPath $taskCompilerDownload -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'f45c7d68d1e660cf13877ec36738a5179ce72a33414f9959d35e99b68c52a697') { throw 'Inno compiler checksum mismatch' }
        $taskSignature = Get-AuthenticodeSignature -LiteralPath $taskCompilerDownload
        if ($taskSignature.Status -ne 'Valid' -or $taskSignature.SignerCertificate.Subject -notmatch 'Pyrsys B.V.') { throw 'Inno compiler signature mismatch' }
        $taskCompilerProcess = Start-Process -FilePath $taskCompilerDownload -ArgumentList @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/CURRENTUSER',('/DIR="' + $taskCompilerRoot + '"')) -Wait -PassThru -WindowStyle Hidden
        if ($taskCompilerProcess.ExitCode -ne 0) { throw 'Inno compiler extraction failed' }
        $CompilerPath = Join-Path $taskCompilerRoot 'ISCC.exe'
    }
    $CompilerPath = (Resolve-Path -LiteralPath $CompilerPath).Path
    # The Inno 6 ISCC PE resource reports 0.0.0.0. installer.iss checks the
    # authoritative preprocessor/compiler engine VER against exact 6.7.0.
    $taskCompilerArgs = @('/Qp',('/DBundleDir=' + "$taskOutput/here"),('/DOutputDir=' + $taskOutput),'/DAppVersion=0.2.0')
    if ($TestAppId) { $taskCompilerArgs += '/DHereAppId={{' + ([Guid]::Parse($TestAppId)).ToString().ToUpperInvariant() + '}' }
    if ($TestGroup) {
        if ($TestGroup -notmatch '^[A-Za-z0-9][A-Za-z0-9 -]{0,63}$') { throw 'Test Start group must be a simple owned name' }
        $taskCompilerArgs += '/DHereGroup=' + $TestGroup
    }
    $taskCompilerArgs += "$taskRoot/packaging/installer.iss"
    Invoke-Checked $CompilerPath $taskCompilerArgs
    Copy-Item -LiteralPath $taskWheel.FullName -Destination $taskOutput
    Invoke-Checked $taskPython @('-B',"$taskRoot/scripts/windows_notices.py",'checksums',$taskOutput,'--compiler',$CompilerPath)
} finally {
    Remove-Item Env:HERE_BUILD_NOTICES -ErrorAction SilentlyContinue
    Pop-Location
}
