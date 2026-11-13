# M5 portability implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. Root owns dispatch, file leases, integration
> and merges under rulings 20/35/64. Steps use checkbox syntax. **Status: preparation
> draft, no implementation started.** Core work may start after M2 B merges,
> the required normal contract-document PR and the real B5 handoff; this draft
> does not grant implementation readiness. Claims/answers wait for M4 B.

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

## Preparation checkpoint and execution split

Foundation: M2 A merge `53712c5beedafc2d40e2abf10c13362a1c3c9f78`,
contract-document merge `73d3cb384bd422de6899d8c7b761f3f3159ea3f1`.
M2 B merge/B5 accepted handoff, M4 B, final M3 library/CLI/citation resolver and
M5 A settings handoffs are PENDING. CLAUDE §11 and R35 supersede the old blanket
M5 A-first ordering for the core only; they do not close any milestone criteria.

Before root dispatches a source front, its brief must record the actual M2 B merge
SHA, accepted SPEC/PLAN and document-PR head, B5 signatures/ownership and evidence,
base branch/worktree, exclusive/shared files, pending adapters and verifiable DoD.
Inspect the actual handoff before selecting implementation seams; if it contradicts
this draft, revise/review the plan before coding. Recheck the snapshot and final
rename lock order, per-meeting and native leases, exact suppression and source
matching, operation/rebuild/delete inventory, and audio roles against the accepted
head. Record the verified B5 SHA and contracts in the document PR. The concrete
Windows proof binding for the transient destination sibling remains an M5 B
implementation check. The static comparison uses reviewed M2 B source `ec57204`.

| Portion | Allowed dependency boundary | Evidence still required |
|---|---|---|
| Archive/meeting core | After M2 B MERGE, accepted contract PR and root handoff/GO | Hostile ZIP checks, snapshot/version refusal, identity/hash round-trip and M2 crash reconciliation |
| Claims and answers | M4 B actual accepted data/verification/publication contract | Included and omitted answers, claims and verified citations; no empty substitute |
| Search and seek after import | Actual M3 search and library/citation resolver handoff | Imported revisions searchable and exact linked audio seekable |
| Qt/CLI and Markdown detail adapters | Actual M3 library, worker and meetings-command ownership | Existing UI/CLI progress, cancel, reports and exit-code tests |
| Settings/root integration | M5 A actual root/settings handoff where used | Same configured MemoryService/root and safe root-change integration |

The task checkboxes below retain their complete original obligations. Core-only
work may cover the independent portions of a mixed checkbox, but must leave that
checkbox open until its dependent assertions/adapters are verified. Neither core
completion nor a partial PR completes group B, starts final release certification,
or changes the full-suite, privacy, p95/heartbeat or milestone acceptance gates.

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

## Inherited implementation boundaries

- Use R50 read internals under the catalog lock for the detached export view.
  `_operation_locked` belongs to writer setup and may migrate. Preserve B's
  explicit native roots and native-before-catalog owner order where applicable.
  R62's native session marker, reservation and receipt path remains a separate
  operation from portable group publication.
- Match R42 UUID and exact origin/fingerprint before adding a new source row.
  Bind the local group and `import-*` roles in both startup and deletion recovery,
  including catalog loss, and extend the managed purge inventory. Clean up only
  proven export temporaries before deletion completion. Keep the R51 derived-row
  hook limited to SQL and bind claims after the M4 B handoff.

- Serialize the real schema-2 revision IDs/numbers, active revision, segment timing
  and exact revision/audio links. Preserve null dates/timing, participant IDs,
  roles and nullable speaker assignments from R46/R48. Do not replace links with
  newest-audio selection or infer a date from a directory or file timestamp.
- Reuse M2 receipt-3/marker-3 durability and managed operation ownership through its
  accepted service; do not ingest a portable archive as an old session or invent
  receipt replay, a second catalog writer or an FTS/rebuild implementation.
  No package-provided path, pending marker or suppression record grants authority.
- R42 applies to exposure, aliases, reimport and recovery. R50's approved
  `memory.service.catalog_read_view(library)` owns the existing lock and read
  transaction on a caller-owned fresh connection; caller closes it. Consume its
  immutable `SuppressionSnapshot` IDs/fingerprint, including exact resolved aliases.
  Do not duplicate its parser, filter suppression after LIMIT or nest BEGIN/locks.
- The proposed private export context in `MemoryService` reuses R50's locked
  snapshot internals for both reads, retaining per-meeting leases from snapshot
  through streaming and close. Compare the detached rows as well as meeting
  versions and suppression at publish. End the final read transaction before the
  rename, still holding the catalog lock. Streaming and file hashing stay outside
  the read transaction and lock.
- A destination temporary has a distinct operation scope. Create the exact sibling
  exclusively and durably bind its operation, selected meetings, file and parent
  identity before writing package bytes. Use the proven transient file for publish
  or cleanup. Startup and deletion retry under the existing ownership guards and
  retain unverified material with cleanup pending. Preserve M2's managed-root purge.
- The proposed private import adapter takes a complete validated batch and runs
  destination collision and suppression checks before publishing any meeting.
  Compare the full portable projection, including the no-audio comparison, and
  retain the exact historical IDs and relationships. Extend existing publication
  and reconciliation to recognize the local import stage and commit each new
  meeting's complete history through one entity transaction. A locally registered
  role inventory and OS lease protect extraction. Uncertain staging stays intact
  with a warning. Bind each imported receipt to its exact locally registered group
  before first visibility. Reconcile or rebuild the whole verified group only
  after complete local proof, preserving ordinary receipt and retry behavior.
- M3 owns R45 search corpus generation and `corpus_changed` behavior after imports;
  normal listing keeps its keyset insertion behavior. No new search/index writer,
  persistent cache, dependency, migration or performance-threshold relaxation.
- R51's private derived-row purgers run trusted SQL on the same connection at the
  start of the final delete transaction, before core/FTS cleanup. An error rolls
  back the transaction and keeps deletion pending. M4 B owns claims/answer storage
  and verification. Their export, verification and import adapters and integration
  with deletion wait for M4 B's accepted contracts. M3 owns citation resolution,
  UI and CLI seams. M5 A owns settings/root changes. All shared-file edits require
  root's lease after their actual handoffs.

## Review focus

- Entry-name normalization: NFC, case folding, trailing dots and spaces, ADS colon,
  reserved device names, backslashes.
- Zip bomb and entry-count limits enforced before extraction, from the central
  directory only.
- Snapshot consistency: a meeting edited between snapshot and publish.
- Relationship-only change and suppression alias change between snapshot and
  publish, no interval between final validation and rename outside the catalog lock,
  and no database read transaction during audio streaming.
- Owned destination sibling: real process kill before rename, startup and deletion
  cleanup, replacement of the file or parent, and valid publication including a
  kill before retiring the temporary registration. Verify destinations on the same
  and a different volume from the library when supported by the Windows setup.
- Partial publication after a kill: one meeting committed, others absent, staging gone
  after reconciliation.
- Catalog loss and rebuild before receipts, between member receipts, after the last
  receipt and before group proof, and after group proof. Verify zero restored
  meetings from a partial group, full history from the completed group, and
  continued rebuild of ordinary M2 receipts and resistance to retry spoofing.
- Preflight conflicts across the whole batch, no-audio reimport equivalence,
  complete histories on recovery and deletion, and retained unknown stage content.
- Answers whose citations cross the package boundary.

### Task 1: Export

**Files:** `src/memory/portability.py` (new), `src/memory/models.py`,
`src/memory/repository.py`, `src/memory/service.py`, `src/output/markdown.py`,
`src/ui/library.py`, `src/ui/workers.py` (or the M3 worker module), `src/cli.py`; tests in
`tests/test_portability_export.py`, `tests/test_ui_library_export.py`,
`tests/test_cli_meetings_export.py`.

**Interfaces:** `PortabilityService.export(ids: tuple[str, ...], destination: Path,
include_audio: bool, progress, cancellation) -> ExportResult`; `Manifest` Pydantic
model with `format_version = 1`; `ExportResult` with path, included and omitted
answers, sizes; `Conflict` and `InsufficientSpace` typed errors.

The facade receives the composed `MemoryService`. Repository queries materialize
detached export rows in one caller-owned read transaction. The service owns
`_portable_export_view(ids: tuple[str, ...]) -> ContextManager[_PortableExportView]`.
Within that view, `_open_export_destination(destination: Path) ->
ContextManager[_OwnedExportTemporary]` owns registration and cleanup of the app's
temporary sibling. The view publishes with `publish(temporary: _OwnedExportTemporary,
*, cancelled: Callable[[], bool]) -> None`. The facade owns cancellation, progress
and the allowlisted serializer. It streams to the scoped output and consumes only
locally validated managed audio references.

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
`src/memory/service.py`, `src/memory/models.py`, `src/memory/repository.py`,
`src/ui/library.py`, `src/cli.py`; tests in
`tests/test_portability_import.py`, `tests/test_portability_hostile.py`,
`tests/test_ui_library_import.py`, `tests/test_cli_meetings_import.py`.

**Interfaces:** `PortabilityService.inspect(source: Path) -> PackageSummary` (central
directory only); `PortabilityService.import_archive(source, progress, cancellation)
-> ImportReport` with per-meeting `inserted|unchanged|conflict|rejected`;
`InvalidPackage`, `UnsupportedSchema` typed errors; `validate_entry_name(name) ->
PurePosixPath` shared helper.

The facade validates names, sizes, hashes and detached entities. The service and
artifact validators use `_portable_import_stage(inventory: _PortableImportInventory)
-> ContextManager[_PortableImportStage]` to register the exact staged roles
before extraction and own their lease and cleanup. The facade calls
`_publish_portable_import(batch: _ValidatedPortableImport, *, cancelled:
Callable[[], bool]) -> tuple[_PortableMeetingOutcome, ...]` inside that scope with
the validated values and registered stage handle. The service reports per-meeting
results with existing typed errors, records local package provenance and uses
the repository inside the one entity transaction per meeting. The stage context
releases ownership on exit and reuses the existing reconciliation, receipt, state
and deletion validators. Tasks 1 and 2 share the service, repository, models
and facade and run their source edits in sequence under root's file leases.

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
- Root coordinates the group PR with both tasks complete, self-review and independent
  agent review. Open the PR against `main` with evidence and the independent review
  in its description. R64 omits the Copilot step for pending PRs and preserves
  existing reviews and replies. After Windows CI passes on the final head, root
  uses the merge commit with `--admin --match-head-commit` and records it in
  `DELIVERY.md` under CLAUDE §7. Partial core evidence keeps dependent work and
  acceptance rows pending. Group C follows `M5_RELEASE_PLAN.md` only with its full
  real dependencies. Advancing this core alone does not certify group B or M5.
