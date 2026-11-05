# M5 release and acceptance specification

Status: written 2026-11-05, not in execution (ruling 15). Implementation starts only on
an explicit user instruction after M5 group B merges. Governing design:
`PRODUCT_DESIGN.md` §7 "Diagnósticos y distribución", §8 and §9, a taken decision
(ruling 3); the delivered M1 pipeline in `M1_DISTRIBUTION_SPEC.md` and
`WINDOWS_DISTRIBUTION.md`. Spanish plan and acceptance criteria:
`../MILESTONE_5_PLAN.md`, criteria 14–20.

## Intent

Ship version 1.0.0 as an installable Windows 11 x64 application that contains
everything M2 to M5 added, proves it on the clean machine the user makes available,
upgrades the real 0.2.0 installation without losing the library, and closes the
product acceptance matrix honestly, with the two-hour run recorded as omitted.

## Decisions

### Build pipeline

Keep PyInstaller onedir (`packaging/windows.spec`), the per-user Inno Setup installer
(`packaging/installer.iss`), `scripts/build_windows.ps1`, `scripts/smoke_windows.ps1`
and `.github/workflows/windows-package.yml`. Changes:

- Remove `PySide6.QtMultimedia` from `excludes` and `/multimedia/` from the plugin
  filter in `windows.spec`, keeping only the backend and plugins M3's player uses;
  justify every re-added binary in the inventory. If M3 already did this, verify.
- Add an FTS5 and QtMultimedia probe to the packaged `--smoke-check` report
  (`src/distribution.py`), without opening devices or sessions.
- Read the version from `pyproject.toml` once and pass it to the build script, the
  installer (`AppVersion`), the wheel glob and the smoke script; no hard-coded
  `0.2.0` remains.
- Keep the stable `AppId`, current-user install root, Start entries, no automatic
  startup, `PrivilegesRequired=lowest`, and the refusal to replace a running
  application (`AppMutex`, `CloseApplications=no`). The installer's running-app
  message tells the user to exit through `Salir`, which stops and saves.
- Keep the locked build group, notices tree, native inventory, build and compiler
  provenance and `SHA256SUMS.txt`; extend the notices audit to any new dependency.
- No Authenticode signing unless a certificate is configured; the About screen and
  docs never claim it.

### About screen and update procedure

`Acerca de` in the Qt window shows version, build provenance (commit, date, tool
versions from the bundled `build-provenance.json`), and a link to the official channel
read from `[project.urls]` in `pyproject.toml`. The build verifies the URL responds;
if no release with binaries is published there, the screen states that and shows the
manual install procedure from `WINDOWS_DISTRIBUTION.md`. No version check, download or
execution happens from the application.

### Upgrade and data preservation

Installing 1.0.0 over 0.2.0 keeps `%LOCALAPPDATA%\here` and any configured data
root, runs the M2 migration with a managed backup at first start, and opens the
library. Uninstall preserves data; a separate explicit "delete my data" action in the
settings page is the only path that removes it. The N→N+1 test uses the real
`here-0.2.0-windows-x64-setup.exe` from M1 CI as N; if that artifact has expired, N
is rebuilt from commit `beeedac` with the same pipeline and its hash is recorded as a
reconstruction.

### Acceptance matrix

`ACCEPTANCE.md` gains the M5 rows and the final product rows. Each row names its
run. Rows certifiable with the available clean machine and credential (ruling 5):
installed startup from Start, both default sources recorded, WAV playback with seek,
local and provider intelligence smoke with synthetic audio and text, export and
import, upgrade N→N+1, uninstall with data preservation, separate data deletion. Rows
omitted by ruling 5: the two-hour capture with resource measurement, recorded as
"omitted by user instruction", never as passed.

### Qt tests, visual inspection and accessibility

Every new page (first run, settings, library, detail, ask, about) has offscreen Qt
tests for keyboard traversal, accessible names and visible focus; contrast is checked
against the theme tokens. A visual inspection on the clean machine records
screenshots of each page in light and dark Windows themes as evidence files.

## Acceptance

- `./scripts/build_windows.ps1` from a fresh locked environment produces installer,
  onedir ZIP, wheel, inventory, provenance and `SHA256SUMS.txt` for version 1.0.0 with
  no hard-coded version left in scripts.
- Packaged `--smoke-check` reports FTS5 available and QtMultimedia loadable, with no
  device opened, no recording, no provider call and no session read.
- Inventory lists the re-added multimedia binaries with justification; excluded
  M1 plugins stay excluded.
- `scripts/smoke_windows.ps1` passes on the clean machine: install, Start launch,
  payload hashes, active-replacement refusal, idle close, reinstall, uninstall with
  synthetic data preserved.
- Upgrade test: 0.2.0 installed with a synthetic library → 1.0.0 installed → data
  present, migration backup present, library opens, search and playback work.
- Separate uninstall test preserves data; separate data-deletion test removes only
  the managed root after explicit confirmation.
- About screen shows version and provenance; link responds or limitation shown.
- Recording both default sources, playback with seek and a local-mode question work
  on the clean machine from the installed application.
- Accessibility and visual inspection evidence recorded for every new page.
- `ACCEPTANCE.md` M5 rows name their runs; the two-hour row reads "omitted by ruling 5".
- Full Windows CI suite and the Windows package workflow pass at the release head.

## Boundaries

No auto-update, no silent download, no Authenticode without certificate, no public
release upload unless the user performs it, no machine-wide install, no change to
product behavior beyond packaging and the About screen.

## Technical references

- [PyInstaller spec files](https://pyinstaller.org/en/latest/spec-files.html) and
  [hooks for PySide6](https://pyinstaller.org/en/stable/hooks.html).
- [Inno Setup AppMutex](https://jrsoftware.org/ishelp/topic_setup_appmutex.htm) and
  [current-user mode](https://jrsoftware.org/ishelp/topic_admininstallmode.htm).
- [SQLite compile options](https://www.sqlite.org/compile.html): FTS5 presence probe.
