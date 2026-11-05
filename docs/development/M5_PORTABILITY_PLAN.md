# M5 portability implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The author merges after Copilot review, CI and independent subagent review (rulings 16 and 17). Steps use checkbox
> syntax. **Status: not in execution (ruling 15).** Start only on an explicit user
> instruction after M5 group A merges.

**Goal:** Export selected meetings to a validated `.here.zip` and import such packages
into any library root without losing identity, timing, citations or audio integrity,
and without ever writing outside staging or executing package content.
**Architecture:** A Qt-free `PortabilityService` in `src/memory/portability.py` that
snapshots through the M2 repository, streams files, validates in staging and publishes
through M2's recoverable operations; Qt and Typer adapters only call it.
**Tech Stack:** Python 3.13, `zipfile`, `hashlib`, Pydantic, pytest, pytest-qt; no new
runtime dependency.
**Spec:** `docs/development/M5_PORTABILITY_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_5_PLAN.md`.

## Global constraints

- The SQLite file, WAL, backups, `.env`, credentials, logs and settings never enter a
  package; a canary test enforces it.
- Package entry names are validated before any extraction; extraction streams into
  `<data_dir>/.staging/` only; nothing from a package is executed or followed.
- Publication reuses M2's `pending_operations` and ingestion identity rules; no
  second write path into the catalog.
- `src/memory/` never imports Qt or Typer.
- Tests use synthetic libraries in temporary roots and generated hostile ZIPs; never
  the user's library or private recordings.
- Touched files leave ruff clean and formatted; no new mypy errors in them.

## Review focus

- Entry-name normalization: NFC, case folding, trailing dots and spaces, ADS colon,
  reserved device names, backslashes.
- Zip bomb and entry-count limits enforced before extraction, from the central
  directory only.
- Snapshot consistency: a meeting edited between snapshot and publish.
- Partial publication after a kill: one meeting committed, others absent, staging gone
  after reconciliation.
- Answers whose citations cross the package boundary.

### Task 1: Export

**Files:** `src/memory/portability.py` (new), `src/memory/models.py`,
`src/memory/repository.py`, `src/output/markdown.py`, `src/ui/library.py`,
`src/ui/workers.py` (or the M3 worker module), `src/cli.py`; tests in
`tests/test_portability_export.py`, `tests/test_ui_library_export.py`,
`tests/test_cli_meetings_export.py`.

**Interfaces:** `PortabilityService.export(ids: tuple[str, ...], destination: Path,
include_audio: bool, progress, cancellation) -> ExportResult`; `Manifest` Pydantic
model with `format_version = 1`; `ExportResult` with path, included and omitted
answers, sizes; `Conflict` and `InsufficientSpace` typed errors.

- [ ] Write failing tests: manifest hashes and sizes match; archive appears only after
  completion (temporary name during write); cancellation removes the temporary file;
  no database, WAL, backup, `.env`, log or settings canary inside the archive.
- [ ] Write failing tests: free-space refusal with audio; `Conflict` when a meeting's
  `row_version` changes between snapshot and publish; answers crossing the boundary
  are omitted and listed.
- [ ] Implement `Manifest`, snapshot read, streamed audio copy, temp + rename.
- [ ] Write failing Qt and CLI tests: Export selected with audio checkbox on a worker
  with progress and cancel; `here meetings export` exit codes and `--no-audio`.
- [ ] Implement the adapters and the single-meeting text/Markdown export from detail.
- [ ] Run focused tests and the full offscreen suite; self-review; commit as
  "portability: validated export of meetings".

### Task 2: Import

**Files:** `src/memory/portability.py`, `src/memory/artifacts.py`,
`src/memory/service.py`, `src/ui/library.py`, `src/cli.py`; tests in
`tests/test_portability_import.py`, `tests/test_portability_hostile.py`,
`tests/test_ui_library_import.py`, `tests/test_cli_meetings_import.py`.

**Interfaces:** `PortabilityService.inspect(source: Path) -> PackageSummary` (central
directory only); `PortabilityService.import_archive(source, progress, cancellation)
-> ImportReport` with per-meeting `inserted|unchanged|conflict|rejected`;
`InvalidPackage`, `UnsupportedSchema` typed errors; `validate_entry_name(name) ->
PurePosixPath` shared helper.

- [ ] Write failing hostile-corpus tests generating ZIPs with absolute paths, `..`,
  UNC, drive prefix, ADS, `CON`, NFC/case duplicates, symlink entry, future manifest,
  extreme ratio, 10,001 entries, unlisted file and hash mismatch; assert rejection
  before extraction or before publication, nothing outside `.staging/`, staging
  removed.
- [ ] Implement `inspect` and `validate_entry_name`; enforce limits from the central
  directory.
- [ ] Write failing round-trip tests: export → import into a fresh root preserves ids,
  legacy ids, revisions, segments, participants, assignments, answers, audio hashes;
  search and seek work on the imported meeting; identical reimport is `unchanged`;
  modified revision is `conflict` with the library untouched.
- [ ] Implement streamed extraction with hash verification, relation validation in
  staging, and publication through M2 `pending_operations` one meeting per
  transaction, recording the package fingerprint in `ingest_sources`.
- [ ] Write failing kill tests at "after extraction" and "after first commit"; assert
  M2 reconciliation leaves the expected state and removes staging.
- [ ] Write failing Qt and CLI tests: Import package with progress and cancel; report
  shown per meeting; `here meetings import --json`.
- [ ] Implement the adapters.
- [ ] Run focused tests and the full offscreen suite; self-review; independent agent
  review; commit as "portability: validated import through recoverable operations".

## Closing the group

- Update `README.md` (export and import commands and limits), `ARCHITECTURE.md`
  (portability service and staging flow), `docs/MILESTONE_5_PLAN.md` (criteria 10–12
  status), `ACCEPTANCE.md` (M5 portability rows with runs) and `WHATS_NEW.md`.
- Open one PR with the two task commits, self-review and independent agent review in
  the description, CI green. The author merges per rulings 16 and 17. Then group C starts from
  `M5_RELEASE_PLAN.md`.
