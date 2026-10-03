# Building and verifying the Windows delivery

The target is Windows 11 x64, here 0.2.0. The artifacts include Python; source
development still uses the repository's normal wheel/uv commands.

From PowerShell 7 at the repository root:

```powershell
uv python install 3.13.15
./scripts/build_windows.ps1 -Python (uv python find --managed-python 3.13.15)
```

The build exports the frozen runtime/build groups with hashes, installs wheels
into `build/windows/venv` without the project, then builds and installs a real
noneditable project wheel. PyInstaller resolves DLLs using that environment and
Windows system directories. The build group pins PyInstaller 6.22.3, hooks 2026.8
and their tool dependencies. The official Inno Setup 6.7.0 download is checked
against its pinned SHA-256 and valid Pyrsys B.V. signature. An existing verified
compiler may be supplied with `-CompilerPath C:/path/to/ISCC.exe`.

Output is in `dist/windows`: the installer, onedir ZIP, wheel, full native-file
inventory, runtime/source/tool provenance and `SHA256SUMS.txt`. Full license and
copyright texts remain under `here/_internal/notices`. The inventory excludes
its own embedded copy to avoid a circular hash; external checksums cover it.
Internal codec/libffi/liblzma/mpdecimal versions that cannot be observed remain
unknown. Candidate source versions label notice provenance rather than binary
versions. Application signing is not performed without a configured certificate.

Run the actual frozen startup and isolated installer lifecycle:

```powershell
./scripts/smoke_windows.ps1 -BundlePath ./dist/windows/here `
  -InstallerPath ./dist/windows/here-0.2.0-windows-x64-setup.exe
```

The script first refuses collisions with the chosen existing AppId and Start
group. It uses a unique current-user program directory and synthetic data root,
launches outside the repository without a key, checks both executable entries,
compares installed payload hashes, launches the real Start shortcut, verifies
active-app replacement refusal, closes the idle GUI, reinstalls and uninstalls.
Synthetic data must survive; real installations/data are never removed. Reports
and installer logs are retained in `dist/windows/evidence` with the owned paths.
The direct `--smoke-check REPORT_PATH` entry writes a bounded local report without
opening devices, recording, provider requests or existing sessions/preferences.

A clean Windows CI runner exercises the actual delivered stable-AppId installer.
A developer with an existing installation can instead compile a clearly labeled
local variant with `-TestAppId <new-guid> -TestGroup 'here acceptance test'` and
pass the same AppId/Group to the smoke script. That variant alone is not evidence
for the exact delivered installer. Do not uninstall an existing real installation
to make an acceptance run pass.

The Windows packaging workflow uploads inspectable artifacts and evidence. Its
runner reports its actual OS; a Windows Server runner is not called Windows 11.
Local Windows 11 and clean hosted installation observations remain separate.
Ubuntu is optional manual-only and is not run as an M1 release gate.

The user excluded the two-hour capture/resource run and long generated-audio
transcription/timestamp/diarization run. These are omitted, not passing tests.
Existing short native capture and synthetic-provider evidence retains its own
source provenance. See [M1 evidence](M1_EVIDENCE.md) for the delivery audit; M2–M5
are outside this implementation.
