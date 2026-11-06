# M2 immutable catalog implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The author merges after Copilot review, CI and independent subagent review (rulings 16 and 17). Steps use checkbox
> syntax. **Status: in execution since 2026-11-05 (ruling 20).** Task 1 independent
> review accepted after its Busy correction. Task 2 SPEC/quality review passed;
> local suite evidence is recorded below. Source `8087e25` includes scoped fixes
> through round 3: original F1–F7, NB1 and NB2 addressed. The final whole-group
> source/docs review requests changes for GR1–GR3; their single combined fix is
> integrated at `7bfd0cb`, with final group/recovery focus 26 passed/34.46s.
> Full 2's 940/seven skips belongs to historical `8087e25`. Full 3 at `7bfd0cb`
> has eight test-wrapper failures/945 passes/seven skips, retained historically.
> Test-only correction `ef70f78` passes focus13 and full4:953/seven skips/358.94s.
> Final scoped re-review remains pending; GR1–GR3 stay OPEN
> until review. No task/group approval
> or current-head Windows/Copilot gate is complete.

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

Authorized narrow additions: `src/recording/isolated.py` exposes only the journal
meeting ID; `src/ui/main_window.py` renders the existing detail fallback and uses
absolute imports; `src/memory/database.py` adds the volume capability/warning
policy without changing the migration. Additional synthetic tests cover memory
CLI, recovery ownership, storage policy, rebuild, import privacy, processing
ownership, legacy retry, dates, unknown cleanup and warning UI. The reusable
`tests/memory_fixtures.py` is synthetic. Existing desktop CLI identity assertions
preserve an existing UUID and require a new pending UUID to survive retry.
`tests/test_isolated_capture.py` has a separately reviewed test-only bootstrap/
measurement correction (ruling 29), retaining all product deadlines and kill cases.

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
before final task verification. Additional contracts in rulings 23–30 cover actual
owner locks, local ingest/status markers, visible nonfatal warnings, allowed source
roles, post-commit receipts/state snapshots, trusted legacy lineage and registered
cleanup. Aware dates normalize to UTC; naive dates remain unspecified with warning.

Ruling 31 authorized migration-2 `revision_audio_assets` and the receipt
v2/show/rebuild contract for exact ordered revision-to-audio links, preserving
migration 1 and warning on ambiguous old evidence. The initial reviewed `6c422f0`
candidate lacked it; the round-1 fix and scoped re-review are recorded below.
The round-1 schema writer was authorized to extend `src/memory/database.py`, `models.py` and
`repository.py` for that approved relation; four separate
`tests/test_memory_review1_*.py` files provide independent RED for the seven
findings. This authorization does not alter migration 1 or allow parallel schema writers.

Current source implements that relation/schema 2, receipt v2 and typed error
categories. Native capture ownership now uses shared `src/output/ownership.py`,
`recording/journal.py`, `isolated.py` and `models.py`, with a controller job scope
under rulings 32/34; original protocol/deadlines remain unchanged. Trusted unmapped
legacy preimport before mutation/provider follows ruling 33. Rulings 36/37 are
implemented in rounds 2/3 as recorded below; aggregate task acceptance remains open.

- [x] Write failing tests: ingest a v1 session folder (no `meeting_id`) → UUID
  assigned, `legacy_session_id` kept, `ingest_sources` fingerprint stored; second
  ingest → `unchanged`; same UUID different content → readable conflict; identical
  audio in two folders → two meetings.
- [ ] Implement `artifacts.py` (v1 reading, fingerprint, hash, path validation through
  `src/output/paths.py`) and `service.py` ingestion inside the publication sequence of
  the spec, with `pending_operations` rows.
- [x] Write failing fault-injection tests that kill a subprocess at each publication
  boundary and assert one identity, one active revision and no orphans after
  `reconcile()`; assert reconciliation issues zero provider calls.
- [x] Write failing retry tests: revision `n+1` created, activated only after commit,
  `n` historical and readable; failed retry leaves `n` active.
- [ ] Wire the controller, `here trans` and retry to `ingest_session`; inject a
  catalog failure during a recording and assert audio plus `session.json` survive and
  the pending-recovery message is reported.
- [ ] Create the FTS5 table in sync with the active revision inside the publication
  transaction; probe FTS5 availability and degrade with a typed warning.
- [ ] Add the three CLI commands with JSON output and exit codes.
- [ ] Run focused tests and the full offscreen suite; self-review; independent agent
  review; commit as "memory: ingestion, identity and revisions".

**Original frozen-candidate checkpoint (superseded by the integration note below):**
Task 3 was **changes requested / round 1 pending**. Candidate-only
commits in `m2-ingestion`: test fixture `efb73a92ea89bc53318ce1a811eaf61129a76fbf`,
task source `6c422f0b8671d2f736645abb6835d351783cbcc7`. All 27 file hashes match
the independent review manifest; these commits freeze FIX_BASE and are not review
approval or root integration. Original RED 31 failures/18 passes in 2.50s;
CLI RED eight failures in 5.94s; root combined confirmation 39 failures/18 passes
in 7.33s. Scoped GREEN covers initial service/artifacts/controller (49 passed),
CLI/processing (50), ownership/storage (8), rebuild/service/artifacts/ownership
(34), warning UI/controller (27), privacy/preflight/capture recovery (70), combined
memory focus (48), processing/recovery/CLI/controller (85), legacy retry/processing
(5) and UTC (2). These are distinct source-state scopes, not a summed suite result.

The first diagnostic full run was 34 failed/822 passed/9 skipped in 363.35s while
source evolved and other checks overlapped. Later stable isolated capture failed
17/33 cases; a fully source-isolated pre-task baseline also failed 15/33. Measured
diagnostics and the narrow test-only correction are recorded in `ACCEPTANCE.md`;
the corrected fixture passed its four-case focus and all 33 isolated cases, with
independent fixture SPEC/quality review PASS. Post-amendment focus passed 122 tests
in 108.36s; receipt/pending/state role guards then passed 46 cases in 66.62s (earlier
related focus 31/34.78s). All 27 candidate files pass static checks and the resolved
mypy comparison remains 79/79 with no new touched-file diagnostics.

The first stable full gave 6 failed/877 passed/9 skipped in 486.12s. The RED author
scoped the old cleanup fault counter to the real cleanup delegate without changing
its assertions: 10 passed in 16.76s, transferred Green rerun 10/17.10s. The second
stable run `M2A-T3-CANDIDATE-FULL-20261105` passed 883 with nine skips in 598.56s;
log and JUnit are retained separately from the first failure. This verifies the
candidate's covered software paths, not the seven static-review contracts.

Review `task-3-review-report.md` requests F1 revision/audio association (P1), F2
legacy CLI identity, F3 identical-retry outcome, F4 postcommit result, F5 producer/
recovery owner exclusion, F6 malformed-marker sibling isolation and F7 typed CLI
input/conflict/I/O categories (P2). All seven were open at that checkpoint.
The old missing-role guards
were corrected before this reviewed snapshot and are not a current finding.
At that checkpoint round-1 RED/fixes/re-review and a corrected stable suite were
required; no task/group completion, root integration, hosted CI or M2 close was claimed.

### Historical checkpoint after round-1 integration

The paragraph above is the original candidate checkpoint. Green fixes `16263c4`
and `c28472b` are now integrated at root `1f1bc88513840b84c388a61d9c077907cf2fba2e`.
Root/Green src and tests trees match exactly; source/config/scripts/workflow diff
is empty. The 40 independent RED cases pass together in 69.70s; Ruff/format on
20 files and canonical mypy 79/79/no new diagnostics are supplied author evidence.
`task-3-fix-1-review-report.md` marks original F1–F7 ADDRESSED, not task acceptance.

Important NB1: distinguish a fresh retry from repetition of an already committed
publication. Actual controller+processor currently double-publish, and restart
replay of a retained postcommit origin marker becomes a stale-parent conflict.
Round-2 RED in `tests/test_memory_publication_replay.py` is pending delivery; no
result or fix is claimed. Preserve fresh identical retries and reject arbitrary
stale parents; do not clear uncertain markers without durable proof.

Root full `M2A-INTEGRATED-FULL1-20261105`: **5 failed/920 passed/7 skipped in
553.39s**, 932 XML cases, zero errors. Three capture-recovery snapshot differences
include new lock files; CLI sf.info stub lacks subtype; audio-only legacy retry
requires missing session.json. Each awaits diagnosis/fix, with assertions intact.
Both noneditable-wheel smokes pass. Fresh wheel/sdist and off-repo `-I` probe verify
65 byte-identical Python files, schema2/FTS5/FK/integrity; local automatic evidence
only. New suite, re-review and whole-group approval remain required before PR gates.

### Historical source checkpoint after rounds 2/3

Root `8087e252112fd2e6de1efeff1d0f3f531d4be77d` matches Green
`4a3cb7ced77a73046098e20cb34aeb00af0e6f61`: src tree
`948efa342c9569ca751cffb4718d82f3dd872de6`, tests tree
`30c62e6f7e9e0a818e125a6c7762db91335b47f1`; runtime/config/scripts/workflow diff
is empty. R36/R37 add no migration/dependency/version/process/protocol/deadline.

Marker 2 carries publication UUID and `source_kind` in addition to original optional
meeting ID, expected parent and ingest/status intent. Receipt 3 and the operation
carry exact token/parent/origin/fingerprint/committed-revision proof. Rebuild restores
the proven complete operation; receipt 2 stays compatible without fabricated proof.
Private single-consumption controller handoff avoids a second real-processor callback,
while durable replay independently covers actual child crashes with DB present/rebuilt.
Fresh identical retries retain distinct tokens/revisions. Audio-only reservation
uses canonical UUID/lease, NULL origin revision and zero historical revisions;
durable source kind preserves NULL capture dates through retry/rebuild.

Round-2 review marks NB1 and the integration regressions ADDRESSED; 61 cases pass
in 68.06s plus existing controller/processing/UI 69 in 23.02s. The intermediate
unchanged 30-second child timeout remains evidence: identical isolated case passed
in 5.02s and final combined call in 2.72s, with cause unproven and no timeout change.
Round-3 review `task-3-fix-3-review-report.md` marks NB2 ADDRESSED with no new
Critical/Important breakage in its diff: capture UUID is checked before missing-
metadata journal join/probe while opaque existing-manifest identity is preserved.
Foci 3/2.40s and 27/20.86s; Ruff/format clean, mypy baseline/current79/79/no new.

Full 2 started alone at 2026-11-06T00:01:58Z on root `8087e25`, log/XML
`.superpowers/m2a-integrated-final-2.*`: **940 passed/7 skipped/zero failures in
325.09s**, XML 947 cases/zero failures or errors/internal 325.029s. Both wheel
smokes pass; seven skips need symlink privilege.
Fresh wheel/sdist and off-repo noneditable `-I` probe on that source verify 65
byte-identical Python files/schema2/FTS5/FK/integrity. Previous full/probes remain
historical. At that checkpoint, whole-group source review was in progress with three provisional
Important paths: completed metadata before the first durable pending marker, and
non-object receipt JSON aborting valid rebuild siblings, and accepted capture-failure
cancellation diverging between cancelled manifest/UI and failed catalog with conflict/
stuck marker. `group-review-report.md` records GR1/GR2/GR3 Important/P2 OPEN,
source verdict changes requested; final all-source/docs verdict awaits committed
docs supplement. Cover all three with coordinated RED/GREEN/re-review; preserve
one meeting, original capture-error/audio evidence and ordinary identity-conflict
rules. Do not restart already addressed NB1/NB2.
Closing checkboxes remain unchecked pending aggregate results/review and hosted gates.

### Current source checkpoint — combined GR1–GR3 fix wave

Root `7bfd0cbf477a42aacc44b83b3d2da1e438801756` integrates Green
`ec55b9564d3af01247d3367675a340a361f6b6ef`; source tree
`2b4036f7deb905efb0a8836f378d267a066564b4`, tests tree
`a4310416bd10dc922b49e010d8495a98bfdfd585`. Source/tests/config/scripts/workflow
diff is empty. The single final whole-group review of `763a891..8087e25`, including
docs supplement `e00dd08`, requests changes for GR1/GR2/GR3 Important/P2. GD1/GD2
minor checkpoint corrections are in `7be93af`; no second whole-source review is
planned. GR1–GR3 remain OPEN until one scoped independent re-review of this wave.

R38 is implemented: every final producer output seals marker 3 in the writer's
internal pre-metadata-commit callback after subordinate artifacts exist. Its exact
`hashes` include prospective final manifest bytes and role-validated artifacts,
alongside the same token, expected parent, intent and source kind. Pending or
unmatched final evidence stays byte-preserved with a warning and no provider or
old-result revision; discovery uses existing `RecoveryCandidate.error_summary`.
Markers 1/2 and receipts 2/3 remain compatible; DB stays schema 2. The real
controller's private capture-failure deferral seals accepted cancellation before
the first catalog publication, preserving one canonical/optional identity, one
partial revision and original capture error/audio. Direct calls/injected processors
keep their contracts. Receipt container validation isolates malformed meeting
trees, preserving them and rebuilding good siblings with warning/JSON results.

Independent baseline RED: 12 cases, 11 failures/1 interrupted-attempt control PASS,
four sequential logs totaling 27.48s. The thirteenth control was authored without
baseline execution; on the new candidate it first failed only the missing visible
pending-v3 uncertainty warning (1 failure/5 deselected/3.93s), while preserving
R1/interrupted status/source bytes/marker. After the narrow warning correction,
final group plus recovery focus passes **26 cases in 34.46s**, covering all 13 group
cases and malformed-marker/legacy-ownership/NB2 controls. First 12 GREEN passed
in 21.67s. Earlier affected compatibility focus passed **404/1 symlink skip in
159.03s**, before only the final pending-v3 warning correction; do not relabel its
source as the final post-correction source. Ruff/format pass on nine touched files;
canonical comparative mypy against `8087e25` is 79 errors/20 files/65 source files
in both baseline/current, with no new diagnostic messages, not a clean base.

Root fresh wheel/sdist build and isolated noneditable probe have passed for
`7bfd0cb`: 65 byte-identical source files, schema 2/FTS5 and root-reported FK/
integrity checks. Full 3 reports **8 failed/945 passed/7 skipped in 364.23s**,
XML 960 cases/8 failures/0 errors/7 skips/internal 362.276s. Both wheel smokes pass.
All eight traces identify two existing capture-recovery monkeypatch wrappers
rejecting additive `before_metadata_commit`. The independently authored narrow
test-only correction must forward that optional callback through both wrappers,
retaining every assertion/crash cutpoint; no production change is diagnosed.
Correction author `c3a3ccbcf2928502cbf0d0881fdfee3d7f86d84a` is integrated as
`ef70f78cfc9c8f2910e77b2b2818ce2e6b4d71ae`; only the two wrapper signatures/
delegations change, 114 assertions and all crash cutpoints intact. Its focus passes
13 in 24.82s (6 real kills/7 simulated), and current root full 4 passes **953/
7 symlink skips in 358.94s**, XML 960/0 failures/0 errors/7 skips/internal 358.621s.
Both wheel smokes pass. Full 4 began alone at 2026-11-06T00:54:42.9822634Z;
`.superpowers/m2a-integrated-final-4.*`. Source tree remains
`2b4036f7deb905efb0a8836f378d267a066564b4`; tests tree now
`30c88848b3cf42c9ce4443328e3c8e0100e2af44`. The existing fresh `7bfd0cb` wheel/probe
is reused because product source/config/scripts/workflows are unchanged; no
redundant build or fabricated new artifact hash. Final scoped docs verdict pending.
Historical full 2 at
`8087e25` is valid 940/7/325.09s evidence for its own source, not this head. All
earlier failed suites and the unchanged/unexplained child timeout remain above and
in ACCEPTANCE/DELIVERY/evidence. No source-level fix verification closes a review
finding or accepts Task 3/group A/M2. PR/Copilot/exact-head Windows CI/guarded merge
remain pending; version is 0.2.0, with no new certification or future front started.

## Closing the group

- Update `README.md` (new commands, data root), `ARCHITECTURE.md` (memory package and
  publication sequence), `docs/MILESTONE_2_PLAN.md` (criteria 1–10, 13–15 status),
  `ACCEPTANCE.md` (M2 rows with the run that closes them) and `WHATS_NEW.md`.
- Open one PR with the three task commits, self-review and independent agent review in
  the description, CI green. The author merges per rulings 16/17/20. Before group B
  implementation, deliver its contract-adjusted SPEC/PLAN in the documentation PR
  required by ruling 14; that next-plan work does not close M2 A or bypass its gates.
- Ruling 35: finish/merge A and required contract documentation PRs first. Then root
  creates separate project chats for M2 B and M3 A, each with its own branch/worktree
  and complete brief pinned to the actual merged SHA. IDs/paths/bases remain pending
  in STATUS; this PLAN creates no chat and starts no future group. Root owns all
  leases/integration/merges; later M5 B overlap waits for M2 B and keeps claims gated
  on M4 B, with other dependencies and quality/privacy/acceptance gates unchanged.
