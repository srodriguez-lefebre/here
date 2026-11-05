# M5 settings and runtime implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The author merges after Copilot review, CI and independent subagent review (rulings 16 and 17). Steps use checkbox
> syntax. **Status: not in execution (ruling 15).** Start only on an explicit user
> instruction after M4 closes.

**Goal:** Versioned non-secret settings, honest first run, redacted logs and
diagnostics, safe data-root migration, and opt-in Windows integration (tray, single
instance, capture lock, startup) without any resident default or network traffic.
**Architecture:** A Qt-free settings service under `src/config/`; a capture lock in the
application layer; Qt adapters for first run, settings page, tray and single instance;
`loguru` sinks configured once in the entrypoints.
**Tech Stack:** Python 3.13, Pydantic, PySide6 (QtWidgets, QtNetwork), `loguru`,
`msvcrt`, `winreg`, pytest, pytest-qt; no new runtime dependency.
**Spec:** `docs/development/M5_SETTINGS_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_5_PLAN.md`.

## Global constraints

- Everything works with `OPENAI_API_KEY` unset; no code path copies a credential.
- `src/config/`, `src/application/` and `src/memory/` never import Qt or Typer.
- Tray, startup and residency are opt-in and default off; M1 close semantics are the
  default and stay tested.
- Deletion from settings calls M2's `delete_meeting`; no second deletion path.
- Tests use temporary roots, synthetic `.env` files, fake registry roots (a test hive
  under `HKCU\Software\here-test`) and never the real user data or `.env`.
- Touched files leave ruff clean and formatted; no new mypy errors in them.

## Review focus

- Credential leakage into settings, `QSettings`, logs, diagnostics or exports.
- `MigrateDataRoot` fault injection after partial copy and while work is active.
- Second-instance race: two launches within the same second.
- Capture lock held by a CLI process when the GUI starts recording, and vice versa.
- Startup registry value with spaces in the executable path; dev build refusal.
- Log redaction of exception messages that embed transcript or file paths.

### Task 1: Settings, first run, migration, logs and diagnostics

**Files:** `src/config/preferences.py` (new), `src/config/settings.py`,
`src/config/paths.py`, `src/application/migration.py` (new),
`src/application/diagnostics_export.py` (new), `src/logging_setup.py` (new),
`src/entrypoints.py`, `src/cli.py`, `src/ui/preferences.py`, `src/ui/first_run.py`
(new), `src/ui/settings.py` (new), `src/ui/main_window.py`, `pyproject.toml` if a
new package is added; tests in `tests/test_config_preferences.py`,
`tests/test_application_migration.py`, `tests/test_logging_redaction.py`,
`tests/test_diagnostics_export.py`, `tests/test_ui_first_run.py`,
`tests/test_ui_settings.py`.

**Interfaces:** `PreferencesService.load() -> Preferences`,
`PreferencesService.save(Preferences) -> None` (atomic), `SettingsUnreadable`;
`MigrateDataRoot.run(new_root: Path, progress, cancellation) -> MigrationReport`;
`redact(exc: BaseException) -> str`; `configure_logging(data_dir, level)`;
`DiagnosticsExport.describe() -> list[str]` and `.write(destination) -> Path`.
`VisualPreferences` takes its accent from `PreferencesService`.

- [ ] Write failing tests for atomic round-trip, corrupt file, future version, default
  `data_dir` null, and one-time `QSettings` accent migration.
- [ ] Implement `PreferencesService` with `src/output/paths.py` write helpers; wire
  `VisualPreferences`.
- [ ] Write failing canary tests: credential in `.env` and environment never lands in
  settings, `QSettings`, diagnostics or logs after a synthetic full run.
- [ ] Implement `configure_logging` with rotation 10 MiB × 3 and `redact`; audit and
  route every `str(exc)` log call in `src/cli.py`, `src/application/`, `src/ui/`.
- [ ] Write failing migration tests: refuse while active, verify hashes, fault after
  partial copy leaves the old root current, warning for network/synced/program paths.
- [ ] Implement `MigrateDataRoot` on a background worker; settings switch only after
  verification.
- [ ] Write failing Qt tests: first-run page only without settings file, credential
  status without value, settings page shows disk usage and calls `delete_meeting`.
- [ ] Implement first-run and settings pages; diagnostics export with content listing.
- [ ] Run focused tests and the full offscreen suite; self-review; commit as
  "settings: versioned preferences, first run, migration, redacted logs".

### Task 2: Tray, single instance, capture lock, startup and safe exit

**Files:** `src/ui/app.py`, `src/ui/tray.py` (new), `src/ui/main_window.py`,
`src/ui/gui.py`, `src/application/capture_lock.py` (new),
`src/application/controller.py`, `src/recording/service.py`, `src/cli.py`,
`src/windows_startup.py` (new), `src/entrypoints.py`; tests in
`tests/test_ui_tray.py`, `tests/test_ui_single_instance.py`,
`tests/test_capture_lock.py`, `tests/test_windows_startup.py`,
`tests/test_ui_lifecycle.py`.

**Interfaces:** `SingleInstance.try_claim() -> bool` and `.on_restore(callback)` over
`QLocalServer`; `CaptureLock.acquire(timeout) -> CaptureLock` and `.release()`;
`StartupRegistration.enable(executable: Path)`, `.disable()`, `.is_enabled()`,
`NotFrozen` error; controller `start` acquires the capture lock before opening devices
and releases on terminal state; tray actions call the existing application commands.

- [ ] Write failing Qt tests: with `close_to_tray=false` M1 semantics unchanged; with
  `true`, close hides to tray, tray Open restores, tray New recording starts, tray
  Exit with active work stops and saves before quitting.
- [ ] Implement `tray.py` and the `close_to_tray` branch in `main_window.py`.
- [ ] Write failing single-instance tests: second launch restores first and exits 0;
  two launches racing within one second leave exactly one server.
- [ ] Implement `SingleInstance` in `app.py`; `gui.py` exits early on restore.
- [ ] Write failing lock tests with a real subprocess: CLI recording blocks GUI start
  with a clear error and vice versa; query commands run concurrently; a dead holder's
  stale lock is reclaimed.
- [ ] Implement `CaptureLock` and wire controller and CLI.
- [ ] Write failing startup tests against a test registry hive: quoted path with
  spaces, enable/disable idempotent, dev build raises `NotFrozen`.
- [ ] Implement `windows_startup.py` with `winreg`; add `--minimized` handling.
- [ ] Run focused tests and the full offscreen suite; self-review; independent agent
  review; commit as "runtime: tray, single instance, capture lock and startup opt-in".

## Closing the group

- Update `README.md` (settings file, tray, startup, data root), `ARCHITECTURE.md`
  (settings service, lock, single instance), `docs/MILESTONE_5_PLAN.md` (criteria 1–9
  and 13 status), `ACCEPTANCE.md` (M5 settings rows with runs) and `WHATS_NEW.md`.
- Open one PR with the two task commits, self-review and independent agent review in
  the description, CI green. The author merges per rulings 16 and 17. Then group B starts from
  `M5_PORTABILITY_PLAN.md`.
