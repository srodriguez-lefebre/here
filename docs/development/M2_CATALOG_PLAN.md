# M2 immutable catalog implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The author merges after Copilot review, CI and independent subagent review (rulings 16 and 17). Steps use checkbox
> syntax. **Status: not in execution.** Start only on an explicit user instruction
> (ruling 1).

**Goal:** Durable meeting identity, immutable transcript revisions and one
application-owned persistence path, rebuildable from files and usable without a key.
**Architecture:** SQLite catalog as authority for identity and relations; audio and
transcript snapshots as immutable files under a managed root; staging plus an
operations journal make file and database publication reconcilable after a crash.
**Tech Stack:** Python 3.13, `sqlite3` (FTS5 when available), Pydantic, soundfile,
pytest; no new runtime dependency.
**Spec:** `docs/development/M2_CATALOG_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_2_PLAN.md`.

## Global constraints

- No provider call in any catalog operation; everything works with `OPENAI_API_KEY` unset.
- Audio is never deleted by this group except in the existing destructive recording
  cancellation; catalog failure never touches the WAV.
- No invented timestamps, speakers, device names or global human identities.
- `application/`, `recording/`, `transcription/`, `output/` and the new `memory/` do
  not import Qt or Typer.
- Every path derived from data is validated under the managed root before any read,
  write or delete; existing helpers in `src/output/paths.py` are reused, not reimplemented.
- Tests use temporary roots and databases; never the real `.env`, private sessions or
  the user's library.
- Register `here.memory` in `pyproject.toml` under `[tool.setuptools]` and
  `[tool.setuptools.package-dir]`.
- Touched files leave ruff clean and formatted; no new mypy errors in them (baseline is
  not clean; see `CLAUDE.md` §6).

## Review focus

- Crash between staging, rename, commit and operation completion: one identity, no
  orphan files, no half-active revision.
- Retry failure after the new revision's files exist but before commit.
- A v1 session imported twice, and the same UUID with different bytes.
- Segment timing: absent, contradictory, non-finite, and the five-minute pause case.
- Path traversal, symlinks and junctions in journal, session and destination entries.
- Catalog failure injected while a recording is being published.
- Busy database from a second connection during publication.

### Task 1: Models, migrations and repository

**Files:** `src/memory/__init__.py`, `src/memory/models.py`, `src/memory/database.py`,
`src/memory/repository.py`, `src/memory/migrations/0001_initial.sql` (or a Python
migration list in `database.py`), `pyproject.toml`; tests in `tests/test_memory_database.py`,
`tests/test_memory_repository.py`.

**Interfaces:** `open_library(data_dir: Path) -> LibraryConnection` with per-connection
pragmas; `migrate(connection) -> MigrationReport` with managed backup and checks;
`UnsupportedSchema`, `IntegrityFailure`, `Busy` typed errors; immutable DTOs
`Meeting`, `AudioAsset`, `TranscriptRevision`, `Segment`, `IngestSource`,
`PendingOperation`; repository methods for insert/get by id and by legacy id, revision
activation, and pending-operation state transitions. Enum `MeetingStatus` with the
seven durable states.

- [ ] Write failing tests: fresh database schema matches the spec's tables and
  constraints; FK violation rejected; `foreign_key_check`/`integrity_check` pass
  after migration; a future-version database raises `UnsupportedSchema`; a managed
  backup exists before migration and is listed as managed data.
- [ ] Implement connection factory, pragmas, migration runner and backup via the
  SQLite backup API.
- [ ] Write failing repository tests: insert meeting/asset/revision/segments in one
  transaction; activate revision only inside the transaction; `BEGIN IMMEDIATE`
  conflict from a second connection returns `Busy` within `busy_timeout`.
- [ ] Implement repository and typed errors; register the package in `pyproject.toml`.
- [ ] Run focused tests and the full offscreen suite; self-review; commit as
  "memory: schema, migrations and repository".

### Task 2: Segments and raw text on every transcript path

**Files:** `src/transcription/service.py`, `src/live_processing.py`,
`src/application/processing.py`, `src/output/session_writer.py`,
`src/output/metadata.py`, `src/cli.py`; tests in `tests/test_transcription_service.py`,
`tests/test_application_processing.py`, `tests/test_output_session_writer.py`,
`tests/test_segments.py`.

**Interfaces:** no new public API. Guarantee that `TranscriptionResult.segments` is
populated on live, offline, fallback, file-transcription and retry paths, and that
`SessionProcessor` writes `segments.json` and raw text on all of them. Add
`timing_source` to `TranscriptSegmentMetadata` as an additive optional field.

- [ ] Write failing tests that assert `segments.json` and raw text exist after each of
  the five paths with a fake provider returning timed, untimed, contradictory and
  non-finite segments; assert `timing_source` values; assert the five-minute-pause
  offset case against a synthetic WAV.
- [ ] Fix every path that drops segments; validate bounds once at the artifact boundary.
- [ ] Consolidate the three direct `write_session_artifacts` calls in `src/cli.py`
  behind `SessionProcessor` so file transcription and retry produce the same artifacts
  as recording; keep the existing compatibility regressions passing.
- [ ] Run focused tests and the full suite; self-review; commit as "transcription:
  carry segments and raw text on every path".

### Task 3: Ingestion, identity and revisions

**Files:** `src/memory/service.py`, `src/memory/artifacts.py`,
`src/application/controller.py`, `src/application/processing.py`,
`src/application/recovery.py`, `src/application/models.py`, `src/cli.py`; tests in
`tests/test_memory_service.py`, `tests/test_memory_artifacts.py`,
`tests/test_application_controller.py`, `tests/test_cli.py`.

**Interfaces:** `MemoryService.ingest_session(session_dir: Path) -> IngestResult`;
`MemoryService.reconcile() -> RecoveryReport`; `IngestResult` with
`inserted|unchanged|revised|rejected`, `meeting_id`, warnings. Controller publishes a
`MEMORY_INGESTED` event (additive `EventKind`) after persistence with the result; a
catalog failure is reported as a recoverable warning, never as a recording failure.
CLI: `here memory ingest PATH`, `here memory reconcile`, `here memory show ID --json`
(exit codes 0/2/3/4 as in `PRODUCT_DESIGN.md` §5).

- [ ] Write failing tests: ingest a v1 session folder (no `meeting_id`) → UUID
  assigned, `legacy_session_id` kept, `ingest_sources` fingerprint stored; second
  ingest → `unchanged`; same UUID different content → readable conflict; identical
  audio in two folders → two meetings.
- [ ] Implement `artifacts.py` (v1 reading, fingerprint, hash, path validation through
  `src/output/paths.py`) and `service.py` ingestion inside the publication sequence of
  the spec, with `pending_operations` rows.
- [ ] Write failing fault-injection tests that kill a subprocess at each publication
  boundary and assert one identity, one active revision and no orphans after
  `reconcile()`; assert reconciliation issues zero provider calls.
- [ ] Write failing retry tests: revision `n+1` created, activated only after commit,
  `n` historical and readable; failed retry leaves `n` active.
- [ ] Wire the controller, `here trans` and retry to `ingest_session`; inject a
  catalog failure during a recording and assert audio plus `session.json` survive and
  the pending-recovery message is reported.
- [ ] Create the FTS5 table in sync with the active revision inside the publication
  transaction; probe FTS5 availability and degrade with a typed warning.
- [ ] Add the three CLI commands with JSON output and exit codes.
- [ ] Run focused tests and the full offscreen suite; self-review; independent agent
  review; commit as "memory: ingestion, identity and revisions".

## Closing the group

- Update `README.md` (new commands, data root), `ARCHITECTURE.md` (memory package and
  publication sequence), `docs/MILESTONE_2_PLAN.md` (criteria 1–10, 13–15 status),
  `ACCEPTANCE.md` (M2 rows with the run that closes them) and `WHATS_NEW.md`.
- Open one PR with the three task commits, self-review and independent agent review in
  the description, CI green. The author merges per rulings 16 and 17. Then write group B's SPEC and PLAN.
