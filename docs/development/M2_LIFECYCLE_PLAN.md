# M2 lifecycle and deletion implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. Root merges after received/responded
> current-head Copilot, green Windows CI and independent review (rulings 16/17/20).
> Steps use checkbox syntax. **Status: task 4 implemented, locally verified and
> independently reviewed; distinct docs/evidence complete. Task 5 PLANNED;
> group B/M2 OPEN, version 0.2.0.**
> Execution base `73d3cb384bd422de6899d8c7b761f3f3159ea3f1`; accepted A merge
> `53712c5beedafc2d40e2abf10c13362a1c3c9f78` is an ancestor. Exact source/tests,
> historical failures and current gates: `M2_LIFECYCLE_EVIDENCE.json`.

**Goal:** Participants, reversible speaker assignment, optimistic edits, startup
reconciliation and deletion that leaves nothing derived behind.
**Architecture:** Reuse group A's catalog/journal and single reconciliation path.
Deletion first makes minimal suppression durable, then hides SQL rows, purges owned
files and removes rows; only verified file/DB/WAL/backup purge completes its
tombstone. Shared B/M3 reads honor that barrier even after catalog loss. Services
stay free of Qt and Typer.
**Tech Stack:** Python 3.13, `sqlite3` (FTS5), Pydantic, pytest; no new dependency.
**Spec:** `docs/development/M2_LIFECYCLE_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_2_PLAN.md`.

## Global constraints

- No provider call in any operation of this group; everything works with no key.
- Audio is deleted only by the confirmed deletion use case, and only under the managed
  root; the WAV of a `recording`/`processing` meeting is never touched.
- No inferred human identity; `speaker_key` scope is the request unless the user links it.
- Every path from `pending_operations`, journals or sessions is validated with the
  helpers in `src/output/paths.py` and group A's nested managed-path helper in
  `src/memory/artifacts.py` before any read, write or delete.
- Reuse the 250 ms bounded OS locks from ruling 23: `.memory.lock` serializes local
  publication/reconciliation; `.owners/<uuid>.lock` proves actual live ownership.
  Native `.capture_owners/<capture_id>.lock` starts before journal publication and
  stays through actual closure/final persistence under the existing job scope.
  Neither a PID, a timestamp nor an unlocked residual file is ownership evidence.
- `ingest_session(session_dir, *, revision_of=None)` keeps ruling 22's trusted retry
  boundary. Lifecycle operations do not turn ordinary identity conflicts into retries.
- Reuse `PendingOperation.phase` (`prepared|complete|reverted`) and its
  `staging_path`, `destination_path`, `payload_json` fields. Do not add an imaginary
  `state` column or a second filesystem publication protocol.
- Tests use temporary roots and databases; never the real `.env`, private sessions or
  the user's library.
- Touched files leave ruff clean and formatted; no new mypy errors in them (baseline
  not clean, `CLAUDE.md` §6).
- Root serializes migrations, `pyproject.toml`, `uv.lock`, version and global docs;
  preserve migration 1/2 checksums. Reserve exclusive windows for deadline suites,
  benchmarks and heavy builds, without competing tests/builds/mypy.

## Contract reconciliation before execution

Accepted foundation: PR #21 merged as `53712c5beedafc2d40e2abf10c13362a1c3c9f78`
at 2026-11-06T03:22:06Z, from reviewed head
`8a498e207812592ff84a0acc5ba4e9f392e02373`. Product source
`c59ae331ebf124400ef3e476e5f0fcc029e83999` retains schema 2, marker 3,
receipt 3 and R36–38, with R40/R43/R44 corrections inherited below. Independent
reviews accepted C1–C5; the final C5 verdict approved source c59ae33 and committed
documentation 3fdbd05. Received Copilot review 5795325438 on exact head 8a498e2
confirmed no open issues; all five observations have individual replies/resolved
threads. Complete Windows CI 42155415514 passed 980 cases/two wheel-environment
skips in 307.46s; package CI 42155415501 passed on that same head. Fresh local
installed-wheel distribution checks passed all four cases in 7.51s. These are
automatic synthetic evidence, not hardware/provider/clean-install certification.
This contract documentation still needs its own independent review, received and
answered current-head Copilot, exact-head Windows CI and guarded merge before
dispatch. Root records its actual reviewed head/merge and execution bases in
DELIVERY/STATUS; neither future chat exists yet. Planned B/M3 APIs below are not
A implementations and do not become accepted integration through this docs PR.

Preserve these inherited contracts:

- Ruling 31's agreed migration-2 `revision_audio_assets` relation and the
  `MemoryService.show`/JSON list of `{meeting_id, revision_id, asset_id, ordinal,
  source_file}`. Composite FKs enforce the same meeting; ordinal/source name are
  unique per revision. Receipt v2/v3 holds only consumed assets plus links and rebuild
  restores A→B→A reuse exactly. Implemented/scoped-reviewed in A; never edit
  migration 1 or its checksum, and never infer links from an ambiguous old receipt.
- Ownership acquisition before producer/recovery mutation, with a recheck at
  materialization after discovery; validate missing-metadata UUID before journal
  join/probe. Preserve native/catalog leases, provider-call and byte-retention guards.
- Receipt 3's exact token/revision/parent/origin/fingerprint/source-kind proof and
  restored complete operation; receipt 2 grants no invented provenance. Marker 3
  has token/source kind, ingest/status intent and an exact `hashes` seal; marker
  1/2 remains compatible without fabricating that seal.
  Fresh identical retry creates a revision; replay needs exact committed proof.
- R37's canonical audio-only reservation with origin revision NULL, zero historical
  revisions and durable NULL capture dates across retries/receipt/rebuild.
- Registered `.publication.json` inventory and validated receipt/state roles;
  contained paths or UUID directories alone never authorize read/purge.
- `IngestResult.error_category` (`invalid_input|conflict|io`), `OwnershipBusy(Busy)`
  and committed-success warnings; CLI maps categories to exits 2/3/4, not prose.
  Existing `RecoveryReport` contains warnings only; B extends it compatibly.
- Accepted R38 `hashes` maps `session.json` to the actual staged manifest hash and
  includes declared/locally consumed role-validated artifact hashes. The writer
  seals immediately before final metadata replace; validation precedes catalog
  mutation. Sealed pending/changed output stays uncertain, with existing recovery
  `error_summary` visibility and no writer claim/old-output ingest/provider call.
  The real controller's private capture-failure deferral reseals accepted cancel
  under the same operation before one publication; direct/injected contracts stay.
  Receipt envelope validation isolates malformed siblings. These are accepted A
  behavior; B adds no second fix/protocol/public producer API.
- R40 keeps ordinary callback rollback/cause and failed-rollback backups, with
  BaseException/process-death behavior unchanged. Default processor and controller
  share the actual MemoryService/custom root; explicit processors stay compatible.
  Rejected completed replay is a nonretryable warning; successful/no-marker
  completed sessions stay hidden.
- R43/R44 retain valid metadata as a bounded nonretryable library warning for a
  malformed marker or unresolved legacy catalog ownership (corrupt/ambiguous),
  without media/credentials/provider work, mutation or invented identity/parent.
  Unknown ownership stops marker/error-document/audio inspection. Native/catalog
  Busy and path guards remain fail-closed, including stale materialization when
  ownership is unknown. Preserve these A behaviors; add no new recovery flow.
- R41 keeps persistent FTS active-complete only; M3 reads opted-in partials through
  its literal/rank-0 branch. R42's durable minimal suppression precedes delete SQL
  and all read/reimport/rebuild exposure; B5 supplies the one shared guard to M3.
  Both are planned B/M3 work, not current A implementations.

## Front scopes and leases

| Scope | Owner/dependency after root dispatch | Exclusive write rule |
|---|---|---|
| Lifecycle participants/edits/recovery/delete/reindex and their new tests | M2 B tasks 4/5 | Task 5 follows task 4 where files overlap; one B writer per file. |
| `src/memory/search.py`, new `test_memory_search*.py` | M3 A query/list/filter/fallback, independent of B | May overlap B without editing B-owned persistence. Shared DTO/read-helper writes wait for leases. |
| `src/memory/models.py`, `repository.py`, `database.py` | Both fronts need accepted A contracts | Root grants an exclusive per-file lease; migrations have one owner. |
| `src/memory/service.py`, `artifacts.py`, application processing/recovery | B extends accepted A persistence | M3 consumes accepted services; any required common-file edit returns to root for a lease. |
| `MeetingPatch`, `update_meeting`, indexed title consistency | B task 4 supplies; M3 title editor consumes | Real title/conflict integration waits for accepted B4, never a duplicate mutation/fake acceptance. |
| `MemoryService.reindex() -> ReindexReport` | B task 5 supplies; M3 facade delegates | One reindexer; actual equivalence integration waits for accepted B5. |
| R42 minimal suppression/read guard | B task 5 supplies; B/M3 reads, ordinary ingest and existing rebuild consume | Root approves private adapter; no duplicate registry parser. M3 real integration waits for accepted B5. |
| `src/ui/main_window.py`, `app.py`, `bridge.py`, `contract.py`, fake/preview and common UI tests; `src/application/controller.py`, `src/cli.py` | B confirmation/application adapters and M3 library integration | Root grants exact file leases before either front writes shared integration files. |
| Milestone/global docs, migrations, dependency metadata, version | Root coordinates all fronts | Serialized; no duplicate persistence, reindexer or alternate layout to avoid ownership. |

Keep prepared synthetic fixtures under their existing ownership; transfer accepted
files once through root instead of regenerating them. Resource leases apply to
verification as well as source files; no benchmark/deadline evidence under load.

## Review focus

- Crash after the deletion commit and before the file phase; crash during the file
  phase with a held handle; repeated reconciliation after each.
- Deleted content surviving in FTS, the `-wal` file or a managed backup.
- Stale `expected_version` on concurrent edits from CLI and GUI.
- `speaker_0` across two requests being merged by accident.
- `deleted_origins` preventing re-import while allowing an explicit new import of a
  different origin with identical audio.
- Reconciliation touching anything outside the managed root.
- A live owner being marked interrupted or its prepared publication being reverted;
  versioned `memory_pending.json` intent being mistaken for provider or external-path
  authority; a stale `revision_of` becoming an implicit revision overwrite.
- Raw text or historical snapshot residue surviving deletion and later catalog
  reconstruction, independently of the derived FTS index.
- Exact revision audio associations surviving edits, status changes and rebuild;
  shared/reused meeting assets being purged with all their owning revisions.
- Ownership changing after discovery but before materialization; malformed marker
  envelopes aborting sibling recovery; a successful unchanged-evidence retry later
  being mistaken for interrupted work.
- A committed mutation being mislabeled rejected after housekeeping fails, or an
  incomplete physical purge being mislabeled completed; category loss in adapters.
- Unknown children under otherwise valid UUID directories being recursively removed.
- A durable delete barrier followed by Busy being reported rejected, a lost SQL
  catalog bypassing suppression, or surviving receipts restoring deleted content.
- Partial opt-in mistakenly writing persistent FTS or disappearing after B5 reindex.

### Task 4: Participants, assignment, editing and reconciliation

**Files:** `src/memory/models.py`, `src/memory/repository.py`, `src/memory/service.py`,
`src/memory/artifacts.py`, `src/application/controller.py`,
`src/application/processing.py`, `src/application/recovery.py`, `src/cli.py`;
tests in `tests/test_memory_participants.py`, `tests/test_memory_reconcile.py`,
`tests/test_cli.py`.

**Implemented B4 interfaces (added after A):** `MemoryService.update_meeting(meeting_id, patch: MeetingPatch,
expected_version: int) -> MeetingDetail`; `MemoryService.assign_speaker(meeting_id,
speaker_key, participant_id | None, scope) -> MeetingDetail`;
`MemoryService.add_participant(display_name) -> Participant`;
`MemoryService.reconcile() -> RecoveryReport` extended with status marking and asset
verification while retaining existing warnings and publication recovery; typed errors
`Conflict`, `NotFound`, `InvalidInput`. `PendingOperation` already has `id`, `kind`,
`phase`, `meeting_id`, `staging_path`, `destination_path`, `payload_json`.
Group A's `publish_revision` payload fields include `revision_id`, `hashes`
(artifact basename to SHA-256) and exact `publication` proof when present;
its immutable destination is `meetings/<uuid>/revisions/<revision-uuid>/`.
CLI:
`here memory rename ID "Título"`, `here memory participant ID "Nombre" [--speaker KEY]`,
`here memory reconcile --json`.

- [x] Write failing tests: participant creation and linking; assignment and removal
  keep the original `speaker_key`; two `speaker_0` labels stay separate; stale
  `expected_version` yields `Conflict` and no change; success increments
  `row_version` and writes a `lifecycle_events` row.
  Reuse the final A state/receipt mechanism so reconstruction preserves the edit
  and exact revision/audio associations without rewriting historical snapshots.
  Bind those checks to accepted schema-2 `revision_audio_assets` and receipt-2/3
  contracts. Preserve R37 NULL capture dates/audio-only origin without fabricating
  historical revisions; update indexed title in the same optimistic edit transaction.
- [x] Implement participants, assignments and optimistic `update_meeting`.
- [x] Write failing reconciliation tests: stale `processing` meeting becomes
  `interrupted` with zero provider calls; `prepared` publication operation completes
  or reverts idempotently across repeated runs; missing or tampered asset marked
  `missing`/`corrupt` without deletion; operation path outside root refused while
  siblings run. Include a held OS owner lock, an unlocked residual lock file and an
  unsafe owner path: never mark live or unverifiable work stale. Versioned pending
  markers retry local ingestion/state outcomes only and reject invalid identity,
  stale revision, array/null envelopes and non-string fields while valid siblings
  continue. Recheck both ownership leases after discovery and before materialization.
  Reuse accepted A replay tests for exact proof/new identical retry, sealed marker-3
  matching and uncertain output, invalid-receipt sibling isolation and consistent
  accepted capture-failure cancellation; do not reimplement A fixes here.
- [x] Extend group A's existing reconciliation and startup integration, retaining its
  OS ownership checks, prepared-publication recovery and local pending-marker retry.
  Add asset verification and detailed completed/reverted/marked/skipped outcomes
  with reasons to `RecoveryReport`; never duplicate startup jobs or block the Qt thread.
  Use the final A postcommit/result categories in CLI and application reporting.
  B5 extends this same path to restore validated R42 suppression/tombstones/delete
  state before ordinary receipt reconstruction or ingest, not a second rebuild.
- [x] Add the three CLI commands with JSON output and exit codes 0/2/3/4.
- [x] Run focused tests and the full offscreen suite; self-review; commit as
  "memory: participants, optimistic edits and reconciliation".

Task 4 source is frozen at `4e1b2cead486563516b69e09fb45543ee85b53aa`:
source tree `4ca55bbe7e88ff59f79dee47e8740fef2253718e`, tests tree
`bf4f471a87ad84df8ac41f8da375f8fd1025f42e`. R46 canonical error identity,
R48 typed coherent reads/actual controller service and R49 durable existing-journal
repair are detailed in the SPEC checkpoint. Independent scoped review marks R1
ADDRESSED/SPEC and quality APPROVED. Current mapped mypy 79→79/delta empty;
corrected full passes1043/9skips/390.03s (1052 collected,0F/0E), exit0.
Distinct docs/evidence is recorded in this checkpoint; hosted/group gates remain open.
Linked attribution reconstructs from state2; unlinked global participants are
SQL-only. No hardware/provider/wheel/hosted-CI certification is claimed.

### Task 5: Transactional deletion and FTS purge (PLANNED)

**Files:** `src/memory/service.py`, `src/memory/repository.py`, `src/memory/artifacts.py`,
`src/memory/database.py` (backup regeneration helper), `src/memory/models.py`,
`src/application/controller.py`, `src/cli.py`; tests in
`tests/test_memory_delete.py`, `tests/test_memory_fts.py`, `tests/test_cli.py`.

**Planned B5 interfaces (not implemented in A):** `MemoryService.delete_meeting(meeting_id) -> DeletionResult` with
`completed | pending | rejected` and the reason; a `before_delete` hook list the UI
registers (M3 adds the player stop); `MemoryService.reindex() -> ReindexReport`.
CLI: `here memory delete ID --yes [--json]`, `here memory reindex`.
Reuse group A's `segments_fts` publication maintenance and runtime capability probe.
At frozen source `c59ae33`, `fts_available`/`FtsUnavailable` live in `service.py`; reuse their
accepted location. Publish the accepted reindex result/adapter contract to M3 A
before its integration, rather than requiring another index writer.
Catalog reconstruction from managed snapshots stays with the group A ingestion path;
this task owns its deletion/tombstone integration and the separate FTS `reindex` use case.
R42's registry/guard is planned under the existing data root, retaining only UUID,
normalized-origin/fingerprint pairs and pending/complete control. Root approves
private fields/guard adapter; no transcript/title/audio/participant data is stored.

- [ ] Write failing tests: deletion of an idle meeting removes rows, FTS hits, files
  under `meetings/<uuid>/`, and records `deleted_origins`; a subprocess killed after
  the deletion commit and before the file phase is finished by `reconcile()`; a held
  handle leaves `deleting` + `pending` and completes on a later reconciliation;
  `processing` meeting returns `Busy`; imported meeting leaves the origin folder
  byte-identical and is not re-imported. Prepare an A→B→A synthetic audio/revision
  history and verify exact associations before deletion and after reconstruction;
  after deletion no receipt, state snapshot or reused asset restores that meeting.
  Exercise every row of the SPEC's nine-boundary R42 crash matrix with subprocess
  failures and DB removal/rebuild: before barrier; barrier before SQL; SQL Busy;
  delete commit before removal; mid-file/unknown child; WAL/backup failure;
  complete-write failure; completed tombstone plus DB loss; invalid registry.
  Assert pre-barrier rejection only, post-barrier pending/hidden, origins restored
  before receipts, no FTS resurrection, safe idempotent replay and typed fail-closed
  exposure errors. Different explicit origin/new identity with identical audio is
  allowed; tombstoned ordinary origin is blocked and external bytes stay unchanged.
- [ ] Implement journaled deletion with a `pending_operations` row of kind
  `delete` and `phase='prepared'`. Define/validate its minimal `payload_json` of
  owned relative paths and origin fingerprints by operation kind, reusing the
  serialized publication/reconciliation lock and retained real owner leases.
  Validate confirmation/identity/inventory before atomically flushing/replacing
  the R42 pending barrier under `<data_dir>/.deletions/`, BEFORE delete SQL. This
  is logical acceptance: later Busy/rollback/file/backup failures return pending,
  never rejected. Restore registry origins/delete state before the one existing
  receipt rebuild/import path; surviving receipts prove cleanup inventory only.
  Share one validated coherent suppression guard with B/M3 reads, ordinary ingest
  and FTS activation, applied before cursor/LIMIT; invalid/unreadable registry
  produces typed recoverable fail-closed exposure, never invented purge authority.
  Invalidate older UI callbacks on deletion through the existing request sequence.
  Remove every
  immutable revision directory, including `raw_transcript.txt`, presentation text
  and segment documents and group A's committed snapshot receipts. Safely remove
  only owned local markers/inactive per-meeting locks; preserve the shared lock and
  external origins. Uncommitted prepared trees never establish a completed revision.
  Purge `revision_audio_assets` rows with their owning revisions/assets, validating
  audio roles and managed locations even when a reference reuses an earlier
  revision's WAV. Preserve migration 1; any later schema work stays serialized.
  Apply the registered inventory and allowed roles before removal; unknown children
  remain with a pending-cleanup warning. Checkpoint WAL and remove or
  regenerate managed backups returned by `LibraryConnection.managed_backup_paths()`.
  Mark operation/registry complete only after all file, DB/WAL and backup checks;
  failure writing final durable complete remains pending. Retain the minimal
  complete tombstone for future `deleted_origins` reconstruction, no deleted text.
- [ ] Write failing residue tests: after deletion, scan every `.sqlite3` and backup
  under the data root for the meeting's canary phrase and token; assert none; cover
  the FTS5 `secure-delete` probe path and the fallback when it is unavailable.
  Include recognized receipts/state snapshots and an unrelated unknown canary;
  preserve the unknown material and do not claim completed deletion while it blocks purge.
  Inspect registry fields to prove only authorized minimal metadata persists; loss
  or restoration of DB/backups cannot bypass suppression before ordinary exposure.
- [ ] Extend group A's transactional FTS maintenance for deletion, retaining its
  activation path, plus `reindex` from unsuppressed active complete revisions only.
  No partial FTS insertion occurs on opt-in. Integration with the actual M3 R41
  literal-partial branch must preserve results before/after reindex with partials
  disabled/enabled, including mixed global ranks/date-NULL/LIMIT and FTS missing;
  no second persistent index/reindexer is created by those queries.
  With FTS unavailable, preserve its typed warning and continue lifecycle work.
  Ensure group A's snapshot reconstruction honors deletion tombstones, so rebuilding
  either catalog or index cannot restore deleted raw text or historical revisions.
- [ ] Wire the confirmation flow: GUI confirms once with the list of what is removed;
  CLI requires `--yes`; both call the same use case.
- [ ] Run focused tests and the full offscreen suite; self-review; independent agent
  review; commit as "memory: transactional deletion and FTS purge".

## Closing the group and the milestone

- Update `README.md` (new commands), `ARCHITECTURE.md` (lifecycle, reconciliation and
  deletion sequence), `docs/MILESTONE_2_PLAN.md` (criteria 11–14 and the remaining
  rows), `ACCEPTANCE.md` (M2 rows with the runs that close them), `WHATS_NEW.md`.
- After implementation, make the milestone version bump in one serialized change:
  `pyproject.toml` to 0.3.0 (ruling 9), the local `here` package version in `uv.lock`,
  the fallback `AppVersion` in `packaging/installer.iss`, every wheel filename/version
  assertion and `/DAppVersion` argument in `scripts/build_windows.ps1`, and installer
  path/artifact names in `.github/workflows/windows-package.yml`. No dependency changes.
  `scripts/smoke_windows.ps1` takes `-InstallerPath` and has no hardcoded product
  version to edit; invoke it with the new installer path. Check the wheel metadata,
  compiler override and fallback all report 0.3.0, then build/smoke the actual artifacts.
- Open one PR with the two task commits, self-review and independent agent review in
  the description. Root merges only after exact-head Windows CI and received/responded
  Copilot plus independent review; no gate is waived by a local pass or delay.
  M3 A's independently dispatched query work may already run under ruling 35; hand
  off accepted B4 title/B5 reindex contracts for its remaining integration.
- After B merges, root may advance M5 B portability with its claims hook pending
  until M4 B. Other groups retain STATUS dependencies. This plan does not create
  chats/worktrees or start that future work; M2 closes only with all original DoD
  and acceptance requirements, not by overlap or fixture preparation.

## R50 acceptance additions for planned task 5

Task 5 starts only after R1 resolution, actual corrected-full acceptance and the
distinct task-4 docs/evidence checkpoint. The SPEC's R50 section fixes the future
catalog_read_view ownership and exact two-field SuppressionSnapshot contract;
private canonical entries must not escape as a third public field.

- [ ] Independent RED: fresh caller-owned connection/close-once, lock-before-BEGIN,
  one read transaction, WAL/rollback Busy, rejected nested/foreign transaction use
  and owned-resource exception cleanup; no repair/second connection/provider work.
- [ ] Pending/complete UUIDs plus exact-origin-pair aliases after reconstruction,
  permitted explicit new origin/new identity with identical audio, absent versus
  invalid/inaccessible/unsafe registry; retain all nine R42 crash/purge boundaries.
- [ ] Stable framed canonical digest across fresh connections; alias mapping changes
  affect fingerprint even with unchanged registry bytes; resolved sorted unique IDs.
- [ ] Actual M3 shared context and SQL-before-keyset/LIMIT for empty/large sets,
  ENTIRE complete-FTS generation with suppression fingerprint/R45, R41 partial
  equivalence before/after the single reindexer. Fakes alone do not close handoff.
- [ ] Quiet-host unchanged p95/heartbeat measure real lock wait, validation, alias
  resolution, canonicalization/generation and full page fetch. No performance
  claim follows from static contract review or preparation.
