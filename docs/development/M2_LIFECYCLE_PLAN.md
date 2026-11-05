# M2 lifecycle and deletion implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The author merges after Copilot review, CI and independent subagent review (rulings 16 and 17). Steps use checkbox
> syntax. **Status: not in execution (ruling 15).** Starts only after
> `M2_CATALOG_PLAN.md` (group A) is merged and only on an explicit user instruction.

**Goal:** Participants, reversible speaker assignment, optimistic edits, startup
reconciliation and deletion that leaves nothing derived behind.
**Architecture:** The group A catalog and `pending_operations` journal are reused;
deletion is a two-phase persisted operation (rows hidden first, files second, rows
removed last). Reconciliation is the single recovery path for both publication and
deletion. Services stay free of Qt and Typer.
**Tech Stack:** Python 3.13, `sqlite3` (FTS5), Pydantic, pytest; no new dependency.
**Spec:** `docs/development/M2_LIFECYCLE_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_2_PLAN.md`.

## Global constraints

- No provider call in any operation of this group; everything works with no key.
- Audio is deleted only by the confirmed deletion use case, and only under the managed
  root; the WAV of a `recording`/`processing` meeting is never touched.
- No inferred human identity; `speaker_key` scope is the request unless the user links it.
- Every path from `pending_operations`, journals or sessions is validated with the
  helpers in `src/output/paths.py` before any read, write or delete.
- Tests use temporary roots and databases; never the real `.env`, private sessions or
  the user's library.
- Touched files leave ruff clean and formatted; no new mypy errors in them (baseline
  not clean, `CLAUDE.md` §6).

## Review focus

- Crash after the deletion commit and before the file phase; crash during the file
  phase with a held handle; repeated reconciliation after each.
- Deleted content surviving in FTS, the `-wal` file or a managed backup.
- Stale `expected_version` on concurrent edits from CLI and GUI.
- `speaker_0` across two requests being merged by accident.
- `deleted_origins` preventing re-import while allowing an explicit new import of a
  different origin with identical audio.
- Reconciliation touching anything outside the managed root.

### Task 4: Participants, assignment, editing and reconciliation

**Files:** `src/memory/models.py`, `src/memory/repository.py`, `src/memory/service.py`,
`src/application/controller.py`, `src/application/recovery.py`, `src/cli.py`;
tests in `tests/test_memory_participants.py`, `tests/test_memory_reconcile.py`,
`tests/test_cli.py`.

**Interfaces:** `MemoryService.update_meeting(meeting_id, patch: MeetingPatch,
expected_version: int) -> MeetingDetail`; `MemoryService.assign_speaker(meeting_id,
speaker_key, participant_id | None, scope) -> MeetingDetail`;
`MemoryService.add_participant(display_name) -> Participant`;
`MemoryService.reconcile() -> RecoveryReport` extended with status marking and asset
verification; typed errors `Conflict`, `NotFound`, `InvalidInput`. CLI:
`here memory rename ID "Título"`, `here memory participant ID "Nombre" [--speaker KEY]`,
`here memory reconcile --json`.

- [ ] Write failing tests: participant creation and linking; assignment and removal
  keep the original `speaker_key`; two `speaker_0` labels stay separate; stale
  `expected_version` yields `Conflict` and no change; success increments
  `row_version` and writes a `lifecycle_events` row.
- [ ] Implement participants, assignments and optimistic `update_meeting`.
- [ ] Write failing reconciliation tests: stale `processing` meeting becomes
  `interrupted` with zero provider calls; `prepared` publication operation completes
  or reverts idempotently across repeated runs; missing or tampered asset marked
  `missing`/`corrupt` without deletion; operation path outside root refused while
  siblings run.
- [ ] Implement reconciliation and wire it to application startup (after the existing
  `RecoveryService.discover()`), never blocking the Qt thread.
- [ ] Add the three CLI commands with JSON output and exit codes 0/2/3/4.
- [ ] Run focused tests and the full offscreen suite; self-review; commit as
  "memory: participants, optimistic edits and reconciliation".

### Task 5: Transactional deletion and FTS purge

**Files:** `src/memory/service.py`, `src/memory/repository.py`,
`src/memory/database.py` (backup regeneration helper), `src/memory/models.py`,
`src/application/controller.py`, `src/cli.py`; tests in
`tests/test_memory_delete.py`, `tests/test_memory_fts.py`, `tests/test_cli.py`.

**Interfaces:** `MemoryService.delete_meeting(meeting_id) -> DeletionResult` with
`completed | pending | rejected` and the reason; a `before_delete` hook list the UI
registers (M3 adds the player stop); `MemoryService.reindex() -> ReindexReport`.
CLI: `here memory delete ID --yes [--json]`, `here memory reindex`.

- [ ] Write failing tests: deletion of an idle meeting removes rows, FTS hits, files
  under `meetings/<uuid>/`, and records `deleted_origins`; a subprocess killed after
  the deletion commit and before the file phase is finished by `reconcile()`; a held
  handle leaves `deleting` + `pending` and completes on a later reconciliation;
  `processing` meeting returns `Busy`; imported meeting leaves the origin folder
  byte-identical and is not re-imported.
- [ ] Implement the two-phase deletion with a `pending_operations` row of kind
  `delete`, owned-path validation, WAL checkpoint and managed-backup regeneration.
- [ ] Write failing residue tests: after deletion, scan every `.sqlite3` and backup
  under the data root for the meeting's canary phrase and token; assert none; cover
  the FTS5 `secure-delete` probe path and the fallback when it is unavailable.
- [ ] Implement FTS maintenance inside revision activation and deletion transactions,
  plus `reindex` from active complete revisions.
- [ ] Wire the confirmation flow: GUI confirms once with the list of what is removed;
  CLI requires `--yes`; both call the same use case.
- [ ] Run focused tests and the full offscreen suite; self-review; independent agent
  review; commit as "memory: transactional deletion and FTS purge".

## Closing the group and the milestone

- Update `README.md` (new commands), `ARCHITECTURE.md` (lifecycle, reconciliation and
  deletion sequence), `docs/MILESTONE_2_PLAN.md` (criteria 11–14 and the remaining
  rows), `ACCEPTANCE.md` (M2 rows with the runs that close them), `WHATS_NEW.md`.
- Bump `pyproject.toml` to 0.3.0 (ruling 9) and the installer name in
  `scripts/build_windows.ps1`, `scripts/smoke_windows.ps1` and
  `.github/workflows/windows-package.yml`.
- Open one PR with the two task commits, self-review and independent agent review in
  the description, CI green. The author merges per rulings 16 and 17. Then M3 starts from
  `docs/MILESTONE_3_PLAN.md` and `M3_SEARCH_PLAN.md`, task 1.
