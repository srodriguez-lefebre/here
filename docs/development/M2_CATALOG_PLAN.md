# M2 immutable catalog implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The author merges after Copilot review, CI and independent subagent review (rulings 16 and 17). Steps use checkbox
> syntax. **Status: in execution since 2026-11-05 (ruling 20).** Task 1 independent
> review accepted after its Busy correction. Task 2 SPEC/quality review passed;
> local suite evidence is recorded below. Task 3 and current-head Windows CI remain
> separate gates.

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

- [x] Write failing tests: fresh database schema matches the spec's tables and
  constraints; FK violation rejected; `foreign_key_check`/`integrity_check` pass
  after migration; a future-version database raises `UnsupportedSchema`; a managed
  backup exists before migration and is listed as managed data.
- [x] Implement connection factory, pragmas, migration runner and backup via the
  SQLite backup API.
- [x] Write failing repository tests: insert meeting/asset/revision/segments in one
  transaction; activate revision only inside the transaction; `BEGIN IMMEDIATE`
  conflict from a second connection returns `Busy` within `busy_timeout`.
- [x] Implement repository and typed errors; register the package in `pyproject.toml`.
- [x] Run focused tests and the full offscreen suite; self-review; commit as
  "memory: schema, migrations and repository".

Local evidence, 2026-11-05: dedicated RED 27 failures; root RED confirmation 27
failures in 3.32s; initial focused GREEN 26 passed/1 symlink skip in 2.30s; initial
full suite 768 passed/9 skipped in 174.94s. Independent review found untranslated
locked reads under supported rollback journaling. Eight real second-connection
EXCLUSIVE-lock regressions failed, then the corrected focus passed 34 tests with
one symlink-privilege skip in 4.70s. Scoped independent re-review accepted the fix;
no actionable task 1 findings remain. Final run `M2A-T1-FINAL-20261105`: 778 passed,
7 symlink-privilege skips in 173.68s, including both noneditable-wheel smokes.
Limits are recorded in `ACCEPTANCE.md`; task source commit `9a80060` records the
implementation. Current-head Windows CI for M2 A remains pending.
No file publication, producer integration, FTS or M2 completion is claimed here.

### Task 2: Segments and raw text on every transcript path

**Files:** `src/transcription/service.py`, `src/transcription/client.py`, `src/live_processing.py`,
`src/application/processing.py`, `src/output/session_writer.py`,
`src/output/metadata.py`, `src/cli.py`; tests in `tests/test_transcription_service.py`,
`tests/test_application_processing.py`, `tests/test_output_session_writer.py`,
`tests/test_segments.py`, `tests/test_cli.py`, `tests/test_desktop_cli_recovery.py`,
`tests/test_capture_recovery.py`.
The client coercion seam and existing CLI/recovery fault-injection targets are
authorized additions: text-only provider output must retain untimed evidence and
the application-owned publication path must preserve compatibility under faults.
The capture-publication fault wrapper forwards additive `raw_text` while retaining
its exact process-kill phases.

**Interfaces:** existing CLI commands remain compatible.
`SessionProcessor.transcribe_audio_path(...) -> SessionArtifactPaths` owns the
former direct file-transcription writes; `SessionArtifactPaths.raw_transcript_path`
is additive and optional. Guarantee that `TranscriptionResult.segments` is
populated on live, offline, fallback, file-transcription and retry paths, and that
`SessionProcessor` writes `segments.json` and raw text on all of them. Add
`timing_source` to `TranscriptSegmentMetadata` as an additive optional field.
Raw evidence is persisted separately as `raw_transcript.txt` (ruling 21).

- [x] Write failing tests that assert `segments.json` and raw text exist after each of
  the five paths with a fake provider returning timed, untimed, contradictory and
  non-finite segments; assert `timing_source` values; assert the five-minute-pause
  offset case against a synthetic WAV.
- [x] Fix every path that drops segments; validate bounds once at the artifact boundary.
- [x] Consolidate the three direct `write_session_artifacts` calls in `src/cli.py`
  behind `SessionProcessor` so file transcription and retry produce the same artifacts
  as recording; keep the existing compatibility regressions passing.
- [x] Run focused tests and the full suite; self-review; commit as "transcription:
  carry segments and raw text on every path".

Source `037fc036ef43467136bcf7b8e95c6f8395d1aa5e`, integrated as `1cdaadb`.
Independent SPEC/quality review: PASS, no findings. RED: 82 failures/153 passes/
1 skip in 21.18s, plus two raw rollback regressions failing before the writer fix.
Focused GREEN: 310 passed/1 skip in 24.60s. The first full run exposed eight
publication-hook wrappers rejecting the additive `raw_text` keyword (775 other
passes/8 skips); narrow forwarding fixes preserved all original assertions and kill
phases. Capture/recovery then passed all 68 cases in 25.64s. Final named run
`M2A-T2-FINAL-20261105`: 783 passed/8 skipped in 160.71s on the reviewed task source.
The skips are six existing symlink-privilege cases and two noneditable-wheel smokes
without their opt-in environment. All 13 touched files pass Ruff/format checks;
mypy comparison reports 60 baseline/58 current diagnostics with no new messages,
not a clean mypy base. Full evidence/limits: `ACCEPTANCE.md` and
`M2_CATALOG_EVIDENCE.json`. This task does not close catalog ingestion, immutable
revision publication, group hosted CI or M2.

### Task 3: Ingestion, identity and revisions

**Files:** `src/memory/service.py`, `src/memory/artifacts.py`,
`src/application/controller.py`, `src/application/processing.py`,
`src/application/recovery.py`, `src/application/models.py`, `src/cli.py`; tests in
`tests/test_memory_service.py`, `tests/test_memory_artifacts.py`,
`tests/test_application_controller.py`, `tests/test_cli.py`.

**Interfaces:** `MemoryService.ingest_session(session_dir: Path, *,
revision_of: str | None = None) -> IngestResult`;
`MemoryService.reconcile() -> RecoveryReport`; `IngestResult` with
`inserted|unchanged|revised|rejected`, `meeting_id`, warnings. Controller publishes a
`MEMORY_INGESTED` event (additive `EventKind`) after persistence with the result; a
catalog failure is reported as a recoverable warning, never as a recording failure.
CLI: `here memory ingest PATH`, `here memory reconcile`, `here memory show ID --json`
(exit codes 0/2/3/4 as in `PRODUCT_DESIGN.md` §5).
The additive optional `revision_of` is the expected active revision ID supplied
only by a trusted explicit retry path (ruling 22). Ordinary ingestion omits it;
same-UUID imports with changed content remain conflicts. Task 3 RED tests must
cover both distinctions, stale expected IDs and failed-publication preservation
before this planned contract is claimed implemented.

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
