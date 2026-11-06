# M2 lifecycle and deletion specification

Status: planned, not in execution. Second PR group of milestone 2; the roadmap is
authorized by ruling 20. Ruling 35 requires group A accepted/merged and the necessary
contract documentation PRs merged before root creates the separate M2 B/M3 A
project chats/worktrees. One cohesive docs PR may reconcile both contracts; no
fixed PR count is required. Final merged-A base SHA: `53712c5beedafc2d40e2abf10c13362a1c3c9f78`. Based on
`M2_CATALOG_SPEC.md`.
Governing design: `PRODUCT_DESIGN.md` §4 "Estados, publicación y recuperación" and
"Borrado íntegro", a taken decision (ruling 3). Spanish plan and acceptance criteria:
`../MILESTONE_2_PLAN.md`.

## Intent

Make a meeting a thing the user can name, attribute and remove. Participants are
explicit, speaker labels are associated by hand and reversibly, edits are optimistic
and versioned, startup reconciliation finishes whatever a crash interrupted, and
deleting a meeting leaves nothing derived from it anywhere the application manages.

## Decisions

### Contracts inherited from group A

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

`MemoryService.ingest_session(session_dir, *, revision_of=None)` publishes immutable
revisions. Only a trusted explicit retry supplies the expected active revision ID
as `revision_of` (ruling 22); an ordinary import with the same UUID and different
content remains a conflict. Lifecycle edits, startup and deletion do not manufacture
retry authority from source-folder contents.
An imported legacy session may retain `meeting_id=None` in its source manifest;
the canonical catalog identity is resolved by the trusted origin/revision context
(ruling 28), without rewriting that optional legacy identity.

The existing `PendingOperation` fields are `id`, `kind`, `phase`, `meeting_id`,
`staging_path`, `destination_path`, `payload_json`; `phase` is `prepared`, `complete`
or `reverted`, not a separate `state` column. Publication uses kind
`publish_revision`, paths `.staging/<operation-id>` and
`meetings/<meeting-id>/revisions/<revision-id>`, and payload fields `revision_id`,
`hashes` (artifact basename to SHA-256) and, for proven explicit publication, a
`publication` object. Revision IDs in paths are UUIDs; revision numbers
remain catalog metadata. This group extends the same journal for deletion rather
than adding a second publication/recovery protocol.

New transcript snapshots retain `raw_transcript.txt` independently of presentation text, segment
documents and audio (ruling 21). Historical snapshots stay immutable. Catalog
reconstruction from managed snapshots and revision publication belong to group A;
this group's reconciliation and deletion extend that path with tombstone and purge
handling. Rebuilding the catalog and rebuilding its derived FTS index are distinct
operations; deleting a meeting must prevent either from resurrecting it.
R42's planned durable suppression registry below takes precedence over both paths,
including when the SQL catalog and its pending-delete rows have been lost.
Preserve group A's distinction between committed snapshot completion receipts and
uncommitted prepared files: a directory alone never proves a revision committed.
Ruling 26 fixes `catalog.json` as the postcommit receipt and immutable state
snapshots under `meetings/<uuid>/states/`. Reuse the final A receipt/state contract
for lifecycle changes and purge; do not infer completion from prepared files.

Each revision must retain an explicit relationship to the exact audio assets used
by its segment timeline, including reused assets. A cumulative meeting asset list
does not establish that relationship. Preserve its ordered asset identity/hash
references through lifecycle reads and catalog reconstruction; never choose audio
by filename, recency or trial matching. Ruling 31 fixes migration 2's
`revision_audio_assets` relation: `meeting_id`, `revision_id`, `asset_id`, `ordinal`,
`source_file`, with same-meeting composite foreign keys and unique source name and
ordinal within a revision. Migration 1 and its checksum remain unchanged.
`MemoryService.show` and its JSON expose `revision_audio_assets` as a list of those
five-field objects. Version-2 receipts contain those links and only the assets
consumed by that revision; reconstruction restores audio A→B→A exactly, including
reuse. An older ambiguous receipt grants no inferred association. This contract is
implemented in the accepted A source; the scoped F1 re-review marks it addressed.
That R31 contract remains accepted; the checkpoint above distinguishes later
correction review and hosted/merge gates.

Receipt 3 preserves the same consumed-asset links and additionally records the
operation ID and exact publication proof: `publication_id`, `revision_id`, expected
`revision_of`, normalized `origin_path`, measured `origin_fingerprint` and
`source_kind`. Rebuild restores the proven complete `pending_operations` row,
including same-meeting parent validation. Receipt 2 stays readable without invented
proof. An already committed publication replays unchanged only when its exact
proof matches; a fresh explicit retry receives a new token and revision even for
identical evidence (ruling 36). Lifecycle/recovery must not accept arbitrary stale
parents or remove uncertain intent. No schema migration accompanies these formats.

For WAV-only legacy retry, A reserves one canonical UUID and actual owner before
mutation/provider work, with `ingest_sources.revision_id=NULL` and zero historical
revisions. Durable `source_kind=legacy_audio_only` keeps capture
`started_at`/`completed_at` NULL through retries and rebuild, even if later v1
compatibility metadata has inferred dates. Processing timestamps never substitute
for unknown recording dates (ruling 37). Preserve optional/opaque source IDs on the
trusted legacy path; ordinary import retains strict UUID validation.

R40 preserves ordinary premetadata-callback exception rollback and its original
cause/retained failed-rollback backups; process termination/BaseException behavior
is unchanged. The controller's default processor shares its actual MemoryService,
including custom roots; explicit processors retain their contract. Rejected
completed replay remains an existing nonretryable recovery warning, while
successful replay and completed sessions without a marker stay hidden (C1–C3).

### Participants and speaker assignment

`participants` rows are created manually with a display name; a name never implies
biometric identity. `meeting_participants` links a participant to a meeting with an
optional role. `speaker_assignments` maps a `speaker_key` within its scope (meeting,
revision or provider request) to an optional participant. Assignment is explicit,
reversible and never inferred across requests: `speaker_0` from two chunks stays two
unassigned labels until the user links each one. Removing an assignment keeps the
segment's original `speaker_key`.
Use the stored request-scoped speaker key from group A; equal provider labels alone
are not an assignment key. Preserve the request scope and original label in segment
evidence rather than deriving a new identity from display text.

### Optimistic editing

Planned B4 contract: `MeetingPatch` carries optional title and user notes only;
no free SQL fields. These edit/result DTOs are not A implementations.
`update_meeting(meeting_id, patch, expected_version)` compares `meetings.row_version`
inside `BEGIN IMMEDIATE`, increments it on success and returns `Conflict` when the
stored version differs. The CLI and the GUI both pass the version they displayed.
Edits never touch transcript revisions or audio.

### Durable status transitions

Group A introduced `meetings.status` and capture ownership. This group extends the
existing startup path: a `recording` or `processing` meeting becomes `interrupted`
only after ownership is shown to be absent, with its available material retained;
`deleting` is entered by the deletion use case and left only by completion.
No transition issues a network call. `lifecycle_events` records every transition with
`occurred_at`, `recorded_duration_ms` and a validated payload without secrets.

Publication and reconciliation are serialized by `<data_dir>/.memory.lock`, while
actual capture/processing ownership is an OS file lock at
`<data_dir>/.owners/<meeting-id>.lock` (ruling 23). Lock acquisition is bounded to
250 ms and returns `Busy`; neither PID, timestamp nor a residual lock filename
proves activity or inactivity. A held owner lock preserves live work; an unsafe or
unverifiable owner path is reported and skipped rather than marked interrupted.
These locks do not replace short database transactions, and file I/O and hashing
remain outside `BEGIN IMMEDIATE`.
Native capture additionally owns
`<sessions_root>/.capture_owners/<capture_id>.lock` before its journal is published.
The existing controller job scope retains that OS lease through real helper closure
and final persistence, including catalog failure; direct wait/open-failure releases
after actual closure (rulings 32/34). Both native and catalog leases are checked.
Ownership must be acquired before compatibility manifests, source copies or audio
are changed, including recovery materialization. Discovery is not an ownership
lease: recheck before mutation because an earlier candidate may now have a live
owner. Missing-metadata recovery validates the canonical capture UUID before joining
or probing its journal; existing manifests preserve their optional/opaque IDs.

### Reconciliation

`MemoryService.reconcile() -> RecoveryReport` runs at startup and on demand. It
first validates the R42 suppression registry, restores its origin tombstones and
pending-delete state, then uses the existing group-A reconcile/rebuild path. It
never reads a suppressed receipt as authority to restore a meeting or FTS rows;
such receipts can only prove owned cleanup inventory. It walks `pending_operations`
with `phase='prepared'` and either completes or reverts each
one idempotently, using only paths validated below the managed root; it marks stale
`recording`/`processing` meetings `interrupted`; it verifies `audio_assets` presence
and hash and sets `available`/`missing`/`corrupt`; it never deletes unrecognized
files, never converts a malformed journal into a completed meeting, and never imports
anything whose origin fingerprint is in `deleted_origins`. The report lists what it
completed, reverted, marked and skipped, with reasons, retaining group A's warnings.
It preserves prepared operations belonging to a live owner and reuses group A's
idempotent publication reconciliation before adding deletion handling.

The application-owned `memory_pending.json` marker 3 carries `schema_version`, the
original optional `meeting_id`, expected `revision_of`, `intent` (`ingest|status`),
`publication_id`, `source_kind` (`session|legacy_audio_only`) and a `hashes` mapping.
Markers 1/2 remain readable without invented seal/proof. Ruling 27 also permits a local state-outcome intent
with its expected revision, so a provider failure changes durable status without
fabricating a revision from old text. It contains no provider payload, secrets or external
path authority. Validate it against its owning session and the ownership contract
before retrying local ingestion; a malformed marker never authorizes a revision or
turns live work stale. Validate the envelope and field types before using them;
an array, null or a non-string revision must not abort valid sibling recovery.
Opening the application never retries transcription.

R43/R44 retain valid source metadata as an existing `RecoveryCandidate` with
`can_retry=False` and a fixed bounded library warning when its marker is malformed
or its legacy catalog owner cannot be resolved (including corruption or ambiguous
origin mappings). Source identity/status and evidence remain intact; the warning
grants no media, credential, provider, mutation or invented-parent authority.
Unknown ownership stops marker/error-document/audio inspection. Actual native or
catalog owner Busy and unsafe paths still fail closed; stale materialization also
fails closed when ownership is unknown. These are inherited A safeguards, not a
new lifecycle recovery flow.

The accepted R38 writer callback runs after subordinate artifacts publish and
immediately before final `session.json` replace. Marker-3 `hashes` contains the
hash of the writer's actual staged `session.json` bytes plus the declared and
locally consumed role-validated artifacts; there is no separate manifest-hash
field. `publication_hashes`/`validate_pending_evidence` in `memory/artifacts.py`
measure and compare that exact set before catalog mutation. Pending metadata or
changed/uncommitted evidence cannot satisfy a seal. Recovery retains uncertain
intent; a sealed pending manifest exposes its warning through existing
`RecoveryCandidate.error_summary`, without a writer claim, old-output ingestion or
provider call. Marker 1/2, receipt 2/3 and schema 2 remain compatible.

For GR3, the real controller uses the private `_capture_failure_publication`
scope to defer its first capture-failure ingest through the existing cancellation
cutover. Accepted cancellation reseals under the same operation and publishes once
from SESSION_PERSISTED. Canonical and supported None source identities keep one
meeting/revision, cancelled UI/manifest/catalog, audio and original capture cause;
direct persistence and injected processors retain their contracts. GR2 validates
receipt container/hash types before mapping or data-derived reads, retaining
malformed evidence and warnings while valid meeting siblings rebuild. These fixes
are technically accepted A contracts; B extends them rather than duplicating them.

### Deletion

The GUI confirms once what will be removed; the CLI requires `--yes`. The service
rejects deletion of a meeting with capture, processing, import or export in progress
(`Busy`). Playback, if any, is stopped and its file handle released before the
operation starts (M3 wires the player; this group exposes the hook).

R42 extends the existing journal/reconcile mechanism with a versioned minimal
registry under `<data_dir>/.deletions/`, outside the meeting tree being purged.
It retains only canonical meeting UUID, normalized origin/fingerprint pairs and
pending/complete protocol control, with no text, audio, title, participant or
credential. This is the durable representation of existing `deleted_origins`
suppression under the same root/policy; the exact private filename/DTO/guard adapter
is B's internal choice approved by root, not an existing A API.

Sequence, under `.memory.lock` and applicable per-meeting owner leases: validate
the confirmed request, identity, actual ownership and registered inventory first.
Before a valid barrier exists, Busy/preflight failure may reject with unchanged
visibility. Flush and atomically replace the minimal pending suppression record
BEFORE the first delete transaction. Its durability is logical acceptance of the
confirmed deletion: from then on any DB Busy, rollback, file or backup failure
returns `pending`, never rejected/rolled back. The shared B/M3 read guard suppresses
the meeting/origin before exposing rows, ordinary import, receipt reconstruction
or FTS activation, even while old SQL rows survive. Queries apply the coherent
validated suppression snapshot in SQL before cursor/LIMIT; deletion invalidates
older UI request callbacks. Invalid/inaccessible registry fails catalog exposure,
ordinary import and reconstruction closed with a typed recoverable error, retains
all material and never grants inferred purge authority.

Next, a short transaction sets `status=deleting`, removes FTS rows, registers/upserts
the existing `pending_operations(kind=delete, phase=prepared)` with validated owned
relative paths, and commits. A rebuilt DB restores this pending state and
`deleted_origins` from the registry BEFORE ordinary A receipt rebuild/import.
Existing registered receipt/state inventory can reconstruct cleanup authority;
inability to prove it leaves pending. No second rebuild/index protocol is added.
The file phase then removes, in order, owned transcript snapshots (including every
historical raw and presentation artifact), segment documents, audio assets, temporary
exports, caches and snapshot completion receipts under `meetings/<uuid>/`, refusing
any path that resolves outside the managed root or through a link. Clean up only
validated meeting-owned local markers and inactive per-meeting lock files; never
remove the shared publication lock or alter an external import origin. Then a final
transaction removes rows (revisions, segments,
assignments, events, assets, FTS remnants, M4 answers and claims that cite the meeting
once those tables exist) and the meeting itself, records the origin fingerprint in
`deleted_origins`, and checkpoints the WAL where possible. Managed backups that
contain the meeting are removed or regenerated. Only after owned-file, DB/WAL and
backup residue checks pass does the operation become complete and the minimal
suppression record atomically become durable `complete`. Failure of that last
record write still reports pending. The minimal tombstone remains available to
restore `deleted_origins` after future DB loss; completed records retain no
deleted content and do not authorize reading the preserved external origin.
Purge the meeting's `revision_audio_assets` relationships together with its owned
revisions/assets. Validate referenced and reused WAV roles/locations under the
managed root before access; a link to an earlier immutable revision's asset does
not authorize an unknown file or external source. Receipt-2/3 purge must not leave
a consumed-asset link or older receipt capable of restoring the deleted meeting.
Use group A's registered publication inventory, including `.publication.json`,
and its validated receipt/state roles to establish file ownership (rulings 25/30).
A UUID directory does not authorize recursive deletion of unknown children. Keep
unrecognized material and report pending cleanup; never call that purge complete.

If Windows denies a file removal, the operation stays `prepared`, the meeting stays
`deleting` and hidden, the UI shows "deletion pending", and the next reconciliation
retries locally. Deletion never claims success it did not achieve. Deleting an
imported meeting never touches the external origin folder; the UI says so when the
meeting came from an import. The application does not promise forensic erasure on
SSDs or deletion of copies the user made elsewhere.
Suppression uses identity and normalized-origin/fingerprint pairs, never audio
hash alone. A confirmed explicit different origin with a new identity and identical
audio remains allowed; ordinary tombstoned-origin reimport remains blocked.

The existing deletion/reconstruction criteria include this required crash matrix:

| Boundary | Required result/restart behavior |
|---|---|
| Before valid barrier replace, including temporary-write failure | No logical acceptance; reject without visibility change. Retain/report unknown temporary material, never infer purge authority. |
| Barrier durable, before delete SQL | Accepted/pending; guard hides old rows/receipts and restores delete state even after DB loss. |
| Barrier durable, SQL Busy/rollback | Pending with reason; never rejected while authorized deletion will later resume. |
| Delete SQL committed, before first removal, then DB lost | Restore suppression/origins/delete state before ordinary receipt rebuild; no completed-row or FTS resurrection. |
| Mid-file purge, locked file or unknown child | Pending/hidden; retain unknowns and external origins, resume validated cleanup only. |
| Rows/files gone, WAL/backup purge or regeneration fails | Pending/hidden; restored backups also honor suppression before exposure. |
| Purge verified, final complete-record write fails | Pending; repeated reconciliation verifies and finishes idempotently. |
| Complete record durable, DB later rebuilt | Restore `deleted_origins` before ingestion; old receipt/origin cannot resurrect the meeting. |
| Registry invalid, unsafe or inaccessible | Typed recoverable error; fail exposure/import/rebuild closed, preserve material, no inferred deletion. |

Retain the final A distinction between precommit rejection, committed results with
recoverable housekeeping warnings, invalid input, identity conflict and storage
I/O failure. Lifecycle adapters use explicit outcomes/categories, not warning prose.
A committed edit is not reported as rolled back when later receipt work is pending;
deletion still requires its defined file/backup purge before reporting completion.
Observed A uses `IngestResult(status, meeting_id, warnings, error_category)` with
`invalid_input`, `conflict` and `io` categories mapped to CLI exits 2/3/4;
`OwnershipBusy` subclasses catalog `Busy`, and `RecoveryReport` currently carries
`warnings` only. B's detailed recovery, edit and deletion result types are planned
extensions below, not existing A APIs. Pin their adapters to the final accepted A
contract, preserving `ApplicationSnapshot.memory_warning` separately from real
capture errors.

### FTS consistency

The `segments_fts` FTS5 index created by group A is updated in the same transactions that
activate a revision or delete a meeting. `reindex` rebuilds it from unsuppressed
active complete revisions only. R41 keeps partials out of persistent FTS even
when selected: M3's read-only literal branch supplies opted-in active-partial hits,
rank 0, unioned with complete BM25 matches under global R39 ordering. Without FTS,
both branches are literal/rank 0. Reindex changes no partial rows or query branch
and equivalence covers `include_partial` false and true. B5 remains the one index
maintenance authority; queries do not create another index writer.
Reuse group A's runtime FTS capability warning and publication maintenance; this
group owns the deletion purge and the explicit `reindex` use case, not a second
revision activation path. A missing FTS5 capability does not block local lifecycle
operations. Managed migration backups are discovered through
`LibraryConnection.managed_backup_paths()` under `<data_dir>/backups/`; deletion
removes or regenerates those containing the meeting and checks the WAL too.
At the accepted source, `fts_available(connection)` and
`FtsUnavailable(code='fts_unavailable')` live in `src/memory/service.py`. B task 5
owns the single planned `MemoryService.reindex() -> ReindexReport`; M3 A's facade
delegates to that accepted operation. B task 4 owns `MeetingPatch`/`update_meeting`
and optimistic title editing, including keeping indexed titles consistent in the
edit transaction. M3 consumes those contracts rather than implementing mutations.
B5 also supplies the shared R42 suppression guard used by B/M3 reads and ordinary
ingest/rebuild. Its exact private adapter is finalized with root before integration;
neither service duplicates registry parsing or treats the future guard as an A API.

Root grants exclusive leases for shared memory/UI files and serializes schema,
dependency/lockfile, version and global documentation writes. Independent M3
`search.py`/search-test work can overlap B after dispatch; common models/repository/
database and UI integration cannot have two writers. Do not add another persistence
path, reindexer or module layout to avoid leases. Detailed front scopes are in both
implementation plans. Existing synthetic fixtures transfer only after acceptance;
do not regenerate or duplicate them for another front.

## Acceptance

- Creating a participant, linking it to a meeting, assigning a `speaker_key` in one
  request scope, and removing the assignment leaves the segment's original label
  intact and the participant row present.
- Two chunks with `speaker_0` each remain two independent unassigned labels; no
  automatic merge by equal label or similar name.
- `update_meeting` with a stale `expected_version` returns `Conflict` and changes
  nothing; with the current version it increments `row_version` and records a
  lifecycle event.
- Killing the process after the deletion transaction commits but before files are
  removed, then restarting, finishes the deletion: no rows, no FTS hits, no files
  under `meetings/<uuid>/`, origin fingerprint present in `deleted_origins`.
- Killing the process during the file phase with one file held open (simulated
  permission denial) leaves the meeting `deleting` and hidden, reports "deletion
  pending", and a later reconciliation completes it once the handle is released.
- A deletion attempt while the meeting is `processing` returns `Busy`.
- Deleting an imported meeting leaves the external origin folder byte-identical, and
  reconciliation does not re-import it because of `deleted_origins`.
- A managed backup taken before migration that contains the deleted meeting is removed
  or regenerated; inspection of every `.sqlite3` and `.bak` under the data root finds
  no row, FTS token or text of the deleted meeting.
- Startup with a stale `processing` meeting and no pending operation marks it
  `interrupted`, lists its available audio, and issues zero provider calls.
- Startup with a held OS owner lock leaves the meeting and its prepared operation
  untouched; a residual unlocked file does not prevent interruption. Unsafe owner
  paths are reported without treating their meeting as stale.
- Startup with a `prepared` publication operation from group A completes or reverts
  it idempotently on repeated runs.
- An asset whose file is missing or whose hash changed is marked `missing`/`corrupt`
  without deleting anything and without hiding the meeting's text.
- A pending local-ingestion marker retries no provider work and grants no retry
  authority beyond its validated expected revision; a stale revision is rejected.
- Paths in `pending_operations` that resolve outside the root or through a symlink or
  junction are refused and reported; sibling valid operations still run.
- Full Windows CI suite passes with `QT_QPA_PLATFORM=offscreen`, a synthetic
  `OPENAI_API_KEY`, temporary roots and no real `.env` or private recordings.

## Boundaries

No listing, search UI, playback, export, import, answers or settings UI. No automatic
retention or purge. No deletion of anything outside the managed root. No global
speaker identity. No provider call. The M4 `answers`/`claims` purge is a hook this
group defines and M4 fills; this group's tests use a synthetic table in its place.

## Technical references

- [SQLite FTS5 secure-delete](https://www.sqlite.org/fts5.html#the_secure_delete_configuration_option):
  distinct from the core `secure_delete` pragma; probe at runtime before relying on it.
- [SQLite WAL checkpoint](https://www.sqlite.org/wal.html#ckpt): required so deleted
  content does not linger in the `-wal` file.
- [Windows file deletion semantics](https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-deletefilew):
  a file opened without `FILE_SHARE_DELETE` cannot be removed; retry on reconciliation.
