# Windows desktop reliability implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development, task-by-task.
> User delegated autonomous decisions and execution.

**Goal:** A recoverable Windows capture experience that works across process restarts.
**Architecture:** Per-user settings and disk journals feed application recovery and Qt
adapters. Background diagnostics stay separate from capture; explicit exit waits for save.
**Tech Stack:** Python, Pydantic settings, soundfile, PySide6, pytest.
**Spec:** `docs/development/M1_DESKTOP_SPEC.md`.


Execution status: both implementation tasks and their corrective reviews completed
in PR #13, merged at 2026-10-03T20:41:23Z as `7ed78afc551c09729ebd886c27089b9c989bbafe`.
Final source `843279c`; reviewed PR head `cb4fa9d`;
hosted CI passed 746 Windows tests
(0 skips), plus wheel/source builds.
Ubuntu validation is retained separately as manual-only and was not dispatched.
Copilot review/dispositions and independent acceptance are recorded in
`M1_DESKTOP_EVIDENCE.json`. Distribution and revised-scope closure remain open.
The user explicitly omitted long capture and long transcription acceptance.
Historical observations retain their original source provenance.

## Global Constraints

- Ordinary idle close exits; ordinary active close hides; explicit exit stops/saves.
- No automatic recording or provider retry on startup.
- No fabricated timestamps, source names or speaker identity.
- Data paths come from per-user configuration; journal paths are validated below root.
- No real credentials/private recordings in tests, artifacts or logs.

## Review Focus

- Crash between journal creation, WAV flush, normalization and pending metadata.
- Full exit requested before the capture factory reports ready.
- Missing API key while opening GUI or listing local recovery candidates.
- Tampered journal uses absolute paths, symlinks or `..`.
- Diagnostic callback arriving after recording began/window shutdown.

### Task 1: Paths, optional credentials and restart-safe capture

**Files:** `src/config/paths.py`, `src/config/settings.py`, `src/recording/journal.py`,
`src/recording/shared.py`, `src/recording/windows.py`, `src/recording/service.py`,
`src/application/recovery.py`, `src/application/controller.py`,
`src/application/processing.py`, `src/output/session_writer.py`,
`src/output/metadata.py`, `src/transcription/client.py`, `src/cli.py`, `pyproject.toml`, tests.

**Interfaces:** `get_data_dir() -> Path`; optional API key settings plus explicit
provider credential validation; `CaptureJournal` with checkpoint/finalization;
`RecoveryService(root: Path).discover() -> list[RecoveryCandidate]` and explicit local
recovery materialization. Capture/session fields are additive to M1_CAPTURE interfaces;
the journal UUID is preserved in optional `SessionMetadata.meeting_id` for M2 reuse.

- [x] Write failing path/key tests for normal, frozen and HERE_DATA_DIR/HERE_ENV_FILE.
- [x] Implement path precedence and defer key validation to provider boundaries.
- [x] Write failing subprocess-kill/idempotent recovery tests with production writer
  still open, checkpoint handshake and expected mono/multichannel sample values.
- [x] Implement owned journal/raw source paths, flush/checkpoint, pending processing
  publication and startup recovery with stable mapping before destination creation and
  atomic final metadata committed last. Preserve raw material until commit succeeds.
- [x] Test each crash boundary including explicit retry, leftover completed journal,
  malformed/path-traversal/symlink journal/session/destination and destructive cancel.
- [x] Cover existing CLI file-transcription reuse: validate neighboring metadata/chunks/
  errors and advertised references before reading/provider work, with external canaries
  and zero provider calls for unsafe sessions. Explicit ordinary audio inputs outside
  a session remain legitimate. Consume the shared artifact validation/staging helpers.
- [x] Consolidate the retained private CLI `_save_transcription` helper with shared
  SessionProcessor persistence. Regression: fail normalization of a real tiny WAV,
  inspect copied local capture_sources.audio_file and original cause, then retry
  successfully from that session. Keep compatibility while eliminating duplicate
  failure-persistence logic; current recording commands already use the controller.
- [x] Test stop/cancel/pause inside a large catch-up silence burst; honor intent between
  blocks, checkpoint and keep persisted/live frame timelines identical. This addresses
  the existing suspend-gap limitation parked by the first M1 branch review.
- [x] Verify credential precedence and real production/default live factory fails locally
  before hardware/temp/provider allocation for missing/empty/whitespace key.
- [x] Run focused tests and full Qt-offscreen suite; commit paths and durable recovery
  in separate cohesive commits; write red/green report.

### Task 2: Device diagnostics, recovery actions and graceful exit

**Files:** `src/application/diagnostics.py`, `src/application/models.py`,
`src/ui/main_window.py`, `src/ui/app.py`, `src/ui/contract.py`, `src/ui/bridge.py`,
`src/ui/gui.py`, `src/application/contracts.py`, `src/application/controller.py`,
`src/recording/control.py`, capture adapters, test doubles and Qt tests.

**Interfaces:** `AudioDiagnosticsService` returns existing `AudioDeviceInfo` and
`SignalTestResult`; UI adapter invokes shared recovery/diagnostic use cases. Actual opened
devices are immutable application snapshot fields/events, not parsed logs.

- [x] Write failing Qt tests: missing-key startup, actual names, async no-signal/error,
  startup recovery selection/retry and exit in each lifecycle state.
- [x] Add background diagnostic worker and truthful results; refresh opened device
  names from immutable ready-handle descriptors through the shared snapshot. Keep
  default capture IDs unpinned. Expose background recovery selector and explicit retry.
- [x] Test stalled enumeration/read, timeout cleanup ownership and stale result after
  capture/shutdown; isolated diagnostic helpers have bounded termination/cleanup.
- [x] Add explicit exit intent waiting for stop/save/persisted terminal state, including
  preparation via controller atomic stop latch; require worker completion acknowledgement
  after persisted terminal state. Test barrier races/opening timeout/delayed cleanup.
  Retain ordinary-close and overlay semantics; never synchronously wait in Qt.
- [x] Run focused Qt/application tests plus full suite, self-review and create commits.
- [x] Record Windows device/signal/close/recovery smoke evidence using synthetic or
  explicitly bounded audio, without capturing private conversations for tests.
