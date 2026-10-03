# Windows distribution and capture acceptance

## Intent

Deliver an installable Windows 11 x64 application that includes its Python runtime,
opens without a console or developer environment, uses user data paths, appears in
Start and uninstalls its program files without removing meetings. Finish M1 acceptance
with recorded evidence rather than equating unit tests with hardware certification.

## Decisions

Use PyInstaller onedir and a per-user Inno Setup installer. Onedir keeps Qt DLL/plugin
deployment inspectable and makes user-owned data separate from the installation.
The GUI executable is windowed; the CLI executable is a separate console entry sharing
the same packaged application. Build from the locked non-editable wheel layout, not
setuptools' editable alias from src to here. Build-tool dependencies are a dedicated
locked group. The spec includes the Qt modules used by M1, soundfile runtime, PyAudioWPatch
and provider/configuration modules without bundling `.env`, personal audio, caches or
workspace scratch. Include third-party notices and dynamic library license files.

Compiler is the official signed Inno Setup 6.7.0 release, pinned/checksummed when
downloaded for CI. A developer-provided compiler path can override download. Nothing
requires administrator privileges. Installer uses a stable AppId, install root under
the current user's Programs directory, Start entries for here and uninstaller, and no
automatic startup/capture. It checks for running application before replacing files.
No uninstall-delete action targets the configured data root. The app is unsigned unless
an actual signing certificate is configured; never imply Authenticode signing occurred.

Add a deterministic `--smoke-check REPORT_PATH` packaged entry that initializes local
configuration and Qt/plugin resources, checks application/memory imports as available,
and writes a safe report without recording, provider calls or opening real sessions.
This is build validation, not a product user flow. Normal GUI startup keeps implementation
details out of the UI.

CI tests the locked project on Windows only, builds Windows executables/installer,
checks their startup without the repo/venv on import paths, records SHA-256 checksums,
and uploads installable artifacts. Installer acceptance in a clean Windows runner is
separate from local hardware capture acceptance. Exercise install, Start shortcut,
launch, reinstall/update and uninstall under an isolated per-user program directory;
synthetic user-data canaries survive each step. A real user's installation/data is never
overwritten or uninstalled for testing.

User scope update on 2026-10-03: finish M1 and stop. Ubuntu validation is retained
apart as an optional manual workflow, is not run in this delivery, and is not a
release gate. The M2-M5 roadmap remains outside the current implementation run.

The user's latest explicit instruction on 2026-10-03 omits both the two-hour
capture/pause/recovery/resource run and long generated-audio transcription with
timestamps/diarization. These cases are excluded from this delivery, not passed.
Complete the remaining Windows distribution, actual runtime/installation checks,
notices/checksums, reviewed PRs and evidence audit; then close M1 under this revised
scope and stop. Existing short hardware/fault/provider observations retain their
actual provenance. M2-M5 remain outside this run.

QtMultimedia/playback belongs to future navigation work and is not imported by
current M1 source. Do not add playback functionality or force unused QtMultimedia/
FFmpeg into this delivery. Include notices for the actual shipped dependencies.

## Acceptance

- Build commands work on Windows from a fresh locked environment.
- Bundle and installer contain no credential or personal recording canaries.
- Executable launched outside repo/venv opens Qt and local settings without API key.
- Current-user installation registers Start/uninstall entries and opens without console.
- Reinstall/update retains synthetic data, and uninstall removes only that install.
- DLL/plugin/runtime license notices accompany the delivered installer.
- CI artifacts and checksums are inspectable; version/build provenance is recorded.
- Audit the remaining M1 matrix against actual source and installed-runtime proof;
  preserve prior short capture, pause, recovery and provider evidence with its source.
- Mark long capture/resource and long transcription explicitly omitted by the user.
  Do not execute or claim them as passing. Other unexecuted cases remain identified.

## Sources

- [PyInstaller spec files](https://pyinstaller.org/en/latest/spec-files.html).
- [PyInstaller operating modes](https://pyinstaller.org/en/stable/operating-mode.html).

Compiler verification on 2026-10-03: official `jrsoftware/issrc` release `is-6_7_0`,
asset `innosetup-6.7.0.exe`, SHA-256
`f45c7d68d1e660cf13877ec36738a5179ce72a33414f9959d35e99b68c52a697`.
Local download matched the release asset digest and Windows Authenticode reported
Valid, signer `Pyrsys B.V.`. Silent current-user compiler installation returned 0;
`ISCC.exe` exists in ignored `.superpowers/tools/inno-6.7.0`. This certifies the
compiler preparation, not a built or installed here application.
- [Inno Setup current-user mode](https://jrsoftware.org/ishelp/topic_admininstallmode.htm).
- [Stable Inno AppId](https://jrsoftware.org/ishelp/topic_setup_appid.htm).
