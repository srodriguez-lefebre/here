# M5 settings and runtime specification

Status: written 2026-11-05, not in execution (ruling 15). Implementation starts only on
an explicit user instruction after M4 closes. Governing design: `PRODUCT_DESIGN.md`
§7 "Ajustes, primera ejecución y privacidad" and "Segundo plano e integración
Windows", a taken decision (ruling 3). Spanish plan and acceptance criteria:
`../MILESTONE_5_PLAN.md`, criteria 1–9 and 13.

## Intent

Make the application explainable and controllable without a credential: one versioned
settings file, a first run that says where data lives and what uses the network,
redacted logs, an explicit diagnostics export, a data-root migration that cannot lose
the library, and the Windows integration the daily product needs (optional tray,
single instance, capture lock, opt-in startup, safe exit). Nothing here adds a
service, a resident process by default, or any network traffic.

## Decisions

### Settings file

Non-secret settings live in `<data_dir>/settings.json` with `schema_version`,
`data_dir` (null means the default from `here.config.paths.get_data_dir()`), `accent`,
`close_to_tray=false`, `launch_at_login=false`, `intelligence_mode="local"`,
`intelligence_model` (null until M4's ruling names one), `log_level="INFO"` and
`retention="manual"`. Writes use temporary file + flush + fsync + replace in the same
directory (`src/output/paths.py` helpers). A corrupt or future-version file yields a
typed `SettingsUnreadable` error, keeps the previous file untouched and starts with
defaults for the session; nothing overwrites it silently.

`here.config.settings.Settings` keeps reading `.env` and process environment for the
credential and the model names; the settings file never stores `OPENAI_API_KEY`, and
no code path copies a credential into settings, `QSettings`, exports or diagnostics.
`VisualPreferences` in `src/ui/preferences.py` reads `accent` from the settings
service; the existing `QSettings` value migrates once on first M5 start and is then
unused. The settings service lives in `src/config/preferences.py`, imports no Qt.

### First run and explanation

A first-run page in the Qt window (shown when no settings file exists) states the
data root, that recording and local recovery need no network, that transcription and
provider-mode intelligence send audio or text to OpenAI only on explicit actions, and
how to place a credential in the user `.env`. It shows "credential configured" or
"not configured", never the value. Dismissing it writes the settings file.

### Data root migration

Changing `data_dir` is an explicit use case, `MigrateDataRoot`, not a side effect of
a text box. It refuses while any capture, processing, import, export or deletion runs;
copies the library and managed meetings to the new root with SHA-256 verification;
switches the setting only after verification; keeps the old root intact until the
user confirms deletion through the ordinary integral deletion of M2. Any failure
leaves the old root current. Installed program directories, network and synced
folders are allowed only with a visible warning.

### Disk usage and deletion

The settings page shows library size, managed audio size and backup size, and offers
"delete selected meetings" that calls `MemoryService.delete_meeting` from M2. No new
deletion path.

### Logs and diagnostics

The GUI and CLI entrypoints (`src/entrypoints.py`) configure one `loguru` sink at
`<data_dir>/logs/here.log`, rotation 10 MiB, retention three files, level from
settings. Log records carry operation ids, stages, error types and aggregate metrics
only. A `redact()` helper sanitizes exception messages and causes before logging;
existing `str(exc)` logging in `src/cli.py`, `src/application/` and `src/ui/` is
audited and routed through it. Diagnostics export is an explicit action that writes a
ZIP of logs, settings with `data_dir` only, versions and device names, lists its
contents on screen first, and never includes sessions, `.env`, credentials or
transcript text.

### Tray, single instance, capture lock, startup, exit

- Tray is opt-in (`close_to_tray`). When on, closing hides to a tray icon with Open,
  New recording and Exit; the overlay still marks active work. When off, M1 semantics
  stay: idle close exits, active close hides, terminal indicator then exit.
- Single GUI instance through `QLocalServer`/`QLocalSocket` in `src/ui/app.py`: a
  second launch asks the first to restore its window and exits with code 0.
- Capture lock at `<data_dir>/.locks/capture.lock` using an OS file lock
  (`msvcrt.locking` on Windows) in `src/application/capture_lock.py`; GUI and CLI
  acquire it before opening devices and release it after capture closes. Query and
  memory commands never take it.
- Startup opt-in writes one `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`
  value named `here` with the quoted frozen executable path and `--minimized`;
  turning it off removes the value; a non-frozen build refuses with a message.
  `scripts/smoke_windows.ps1` already asserts no startup entry exists after install;
  that assertion stays.
- Exit with active work runs the existing `Salir` stop-and-save path; tray Exit uses
  the same path.

## Acceptance

- Settings round-trip through the service with atomic writes; corrupt and
  future-version files raise `SettingsUnreadable`, leave the file unchanged and start
  with defaults.
- A canary credential in `.env` and environment never appears in `settings.json`,
  `QSettings`, exports, diagnostics ZIP or logs after a full synthetic run.
- First-run page appears only without a settings file; dismissing it writes the file;
  the credential status is shown without its value.
- `MigrateDataRoot` copies and verifies; a fault injected after copying some files
  leaves the old root current and the new root incomplete but harmless; it refuses
  while a fake capture is active.
- Log rotation produces at most three files of at most 10 MiB; a run seeded with
  canary phrases in transcript, titles, names and questions leaves no canary in logs.
- Diagnostics ZIP contains exactly the listed items.
- Tray on/off behaviors, second-instance restore, capture lock contention between a
  CLI process and the GUI, and startup registry value creation/removal are covered by
  Qt offscreen and subprocess tests on Windows; the dev build refuses startup opt-in.
- Full Windows CI suite passes with a synthetic key, temporary roots and no display.

## Boundaries

No credential editor UI (configuration by `.env` remains; a status-only view is
allowed). No automatic retention. No Windows service. No tray by default. No
telemetry. No change to capture, overlay, catalog, search or intelligence behavior.

## Technical references

- [QLocalServer](https://doc.qt.io/qtforpython-6/PySide6/QtNetwork/QLocalServer.html):
  single-instance IPC.
- [loguru rotation and retention](https://loguru.readthedocs.io/en/stable/api/logger.html#file):
  size-based rotation and file count.
- [Run registry key](https://learn.microsoft.com/windows/win32/setupapi/run-and-runonce-registry-keys):
  per-user startup without privileges.
