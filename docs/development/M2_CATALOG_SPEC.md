# M2 immutable catalog specification

Status: group A CLOSED and merged in PR #21 at
`53712c5beedafc2d40e2abf10c13362a1c3c9f78` (2026-11-06T03:22:06Z).
Implemented and automatically verified: complete Windows CI on reviewed head
`8a498e207812592ff84a0acc5ba4e9f392e02373` passes980/two wheel skips/307.46s;
package success and received Copilot5795325438 with all five threads resolved.
Final sourcec59ae33/docs3fdbd05 scoped independent review APPROVED. Fresh exact-source
installed-wheel distribution4/7.51s supplies the two cases omitted in hosted CI.
No new hardware/provider/clean-install certification; M2 milestone/B remain open,
version0.2.0. Contract documentation for B/M3A is the next gate, before chat launch.

Historical implementation checkpoints (retained with their original source):
Tasks 1/2 have local verification and independent reviews. Original F1–F7, NB1
and NB2 are addressed by scoped re-reviews. The final whole-group source/docs
review of `8087e25` requests changes for GR1–GR3; minor GD1/GD2 checkpoint fixes
are committed. Their combined source fix is integrated at `7bfd0cb`, identical
to Green `ec55b95` source/tests. Final sealed markers use version 3; receipts
remain version 3 and legacy audio-only lineage is preserved. The final
group/recovery focus passes 26 cases/34.46s. The 940-pass/seven-skip full belongs
to historical `8087e25`. New full 3 at `7bfd0cb` reports eight failures/945 passes/
seven skips in 364.23s, caused by two existing capture-recovery test wrappers
rejecting the additive precommit callback; both wheel smokes pass. Narrow fixture
forwarding is integrated test-only at `ef70f78`, its 13-case focus passes and full 4
passes 953/seven skips in 358.94s with both wheel smokes on unchanged product
source. Final scoped independent technical review APPROVED source, fixture and
docs `8845877`; GR1–GR3/GD1/GD2 are ADDRESSED. Task 3 is locally implemented,
automatically verified and independently accepted at that checkpoint; hosted gates were then open and are now closed above.
The prior five-failure full remains historical evidence. Governing design:
`PRODUCT_DESIGN.md` §2–§4, which is a taken decision (ruling 3). Spanish plan and
acceptance criteria: `../MILESTONE_2_PLAN.md`.

## Intent

Current C5 source is `c59ae33` (Green `e8bd181`) under R44. Only actual catalog
ownership lookup errors (LibraryError or ambiguous-origin ValueError) become a
private signal. Validated legacy metadata remains visible as a fixed bounded
nonretryable warning without inspecting marker/error/audio/native material,
changing source/catalog bytes or inventing identity/parent. Actual OwnershipBusy
and UnsafeSessionPath guards remain strict; stale materialization fails closed.
Independent RED covers six distinct cases (three failures/three controls across
two runs); GREEN6/4.24s, affected169/114.05s and fresh installed-wheel distribution4/
7.51s pass. Final scoped source/docs review APPROVED and complete exact-head
Windows CI980/2/307.46s/package success satisfy R43/R44. Copilot5795325438 confirms
C5 resolved on8a; guarded merge53712c5 closes A. No duplicate local full7 or new
hardware/provider/clean-install certification; milestone M2 remains open.

Historical C4 correction source is `56af525` (Green `0a28842`) under R43. Malformed
marker ValueError is handled only around `read_pending_intent`; UnsafeSessionPath
is re-raised. A valid session retains its real identity/status as an existing
nonretryable warning candidate before WAV inspection, with fixed bounded text
that echoes no payload/path. Bytes, owners, valid siblings and ordinary failed
retry controls are preserved. Local8-case GREEN,124-case affected focus and fresh
installed-wheel smokes pass. Final scoped review APPROVED with docs58176ce;
Windows CI5b passes974/two wheel skips/349.23s. Copilot5795057762 confirms C4
resolved and opens C5 above. Those passes prove5b, not corrected C5 source.

Historical PR #21 correction source was `c3cdf08` (Green `61fb8e2`) under R40.
C1 restores previous raw/segments/obsolete evidence for ordinary callback failure
before propagating the original exception, retaining failed-rollback backups;
BaseException/process termination is unchanged. C2 shares the controller's actual
MemoryService with its default processor, including custom roots, while preserving
explicit processors. C3 exposes rejected completed local publication through an
existing nonretryable RecoveryCandidate and readable warning, without audio probes,
provider retry or a new UI flow. Successful/no-marker controls remain unchanged.
Eight new cases and affected 325/one symlink skip pass; full 5 passes 961/seven skips
in 371.80s, both wheel smokes passing. Final scoped independent review remains
pending. Prior technical approval/full4 are historical for source 339.

Give every recorded meeting a durable identity, an immutable transcript revision
history and one application-owned persistence path, so that M3 can search and seek
and M4 can cite without reading folder names. The catalog must be rebuildable from
files, must never make a recording depend on indexing succeeding, and must work with
no provider credential.

## Decisions

### Storage and connections

SQLite through the standard `sqlite3` module at `<data_dir>/library.sqlite3`, with
`<data_dir>` from `here.config.paths.get_data_dir()`. Managed meeting material lives
under `<data_dir>/meetings/<uuid>/`; staging under `<data_dir>/.staging/`. The
historical `TRANSCRIPTIONS_DIR` stays an explicit import source and compatible output
during migration; nothing scans the whole disk.

One connection per worker or operation; never share a connection with the Qt thread.
Per connection: `PRAGMA foreign_keys=ON`, bounded `busy_timeout`, WAL journal on local
disks, `synchronous=FULL` for durable writes. Mutations use short `BEGIN IMMEDIATE`
transactions that never wait on the provider, copy audio or hash files inside. A busy
database returns a typed recoverable error, not a hang.

The Windows volume capability probe includes mapped remote drives; remote or
unproven-local storage uses rollback journaling with a visible warning instead of
assuming a drive-letter path is local (ruling 24). Publication and reconciliation
share a bounded OS lock at `.memory.lock`. Per-meeting `.owners/<uuid>.lock`
handles represent actual capture/provider-work ownership, rather than PID or
timestamp guesses; a residual filename alone does not prove a live worker (ruling 23).

### Schema and migrations

Tables and invariants exactly as `PRODUCT_DESIGN.md` §4 "Tablas mínimas":
`schema_migrations`, `meetings`, `audio_assets`, `transcript_revisions`, `segments`,
`participants`, `meeting_participants`, `speaker_assignments`, `lifecycle_events`,
`ingest_sources`, `pending_operations`, `deleted_origins`. This group creates all of
them so group B does not need a second migration to add lifecycle tables; group B may
add columns.

Ruling 31 authorizes a separate migration 2 for `revision_audio_assets`, preserving
migration 1 and its checksum. The relation records meeting/revision/asset identity,
source name and ordinal, with composite FKs preventing cross-meeting references and
unique per-revision name/ordinal. Schema 2 and receipt/show/rebuild references are
implemented in the round-1 source; independent re-review marks original F1 addressed.

Migrations are numbered, applied in a transaction, validated with
`foreign_key_check` and `integrity_check`, and preceded by a managed backup taken
through the SQLite backup API, never by copying the file while WAL is active. A single
migrator owns the schema; an older client refuses a newer schema with a typed
`UnsupportedSchema` error. Managed backups count as data to purge on deletion.

### Identity

`meetings.id` is the capture UUID that M1 already stores as
`SessionMetadata.meeting_id`. `legacy_session_id` keeps the date-based `session_id`.
A v1 session without `meeting_id` receives a UUID at import time; `ingest_sources`
maps normalized origin path plus origin fingerprint to that meeting so repeated
imports are idempotent. Same UUID with different content is a readable conflict, never
an overwrite. Two different meetings with identical audio stay two meetings.

`ingest_session(session_dir, *, revision_of=None)` accepts the expected active
revision ID only from a trusted explicit retry. A stale precondition cannot replace
a newer active revision. For an already imported v1 origin without a manifest
UUID, the application resolves its existing normalized-origin lineage and expected
revision without rewriting the external v1 identity; ambiguous lineage is rejected
(rulings 22/28). Ordinary import remains governed by origin plus fingerprint.

Aware source datetimes are normalized to UTC while preserving their instants.
Naive historical dates remain unspecified in the catalog with a warning, rather
than acquiring an invented timezone; original manifest bytes remain in evidence.

Every revision must retain explicit ordered references to the audio assets it
consumed, including reused assets. Trusted retries using audio A, B, then A again
must resolve as A/B/A independently of file recency or cumulative meeting assets.
The ruling 31 version-2 receipt carries those links and only the referenced assets;
rebuild and `show` JSON expose the same `revision_audio_assets`. An older receipt
without unambiguous proof stays uncertain with a warning; no relationship is
invented. Receipt 3 retains these same links and additionally proves explicit
publication provenance; receipt 2 remains readable without invented proof.
Historical local full4 passes953/seven skips/358.94s on its stated source. Final
review and exact-head Windows980/2/307.46s/package/Copilot/merge close A as recorded
above, without relabeling that historical local run.

### Segments and evidence

`TranscriptionResult.segments` already exists; this group makes sure every path that
produces a final transcript (live, offline, fallback, file transcription, retry)
carries it into persistence, and that `segments.json` keeps being written next to the
raw text. Segment rows store source text, nullable `start_ms`/`end_ms`, `ordinal`,
`timing_source` (`provider`, `imported_explicit`, `absent`) and nullable
`speaker_key` scoped per request. Floats are validated finite before conversion to
milliseconds; `end >= start` when both exist; contradictory bounds become null, not
reordered. Segments without time keep their ordinal; no time is invented. Offsets are
applied once on the recorded-audio timeline; wall-clock pauses never shift a seek.

### Revisions and publication

Every completed transcription is a `transcript_revisions` row with a per-meeting
sequential number, provider/model, origin, `content_hash` and `complete`/`partial`.
Retry creates a new revision and activates it only after its files are published and
its transaction commits; the previous revision stays readable and is flagged
historical. A failed retry never removes earlier text.

A trusted completed retry also completes its durable outcome and revision contract
when its evidence matches the previous result; ordinary idempotent import is a
separate branch. After a committed activation, receipt/state/operation-housekeeping
failure reports committed inserted/revised success with a recoverable warning,
not a precommit rejection. Original F3/F4 and NB1 are addressed by scoped review.
Each explicit producer invocation receives a new `publication_id` UUID, including
identical retries. The operation payload and receipt 3 retain `publication_id`,
`revision_id`, expected `revision_of`, normalized `origin_path`, measured
`origin_fingerprint` and `source_kind`. Replay must match that exact proof, content
and a genuinely committed revision. It returns unchanged for that publication;
a fresh identical retry still creates its new revision. Prepared-only operations,
arbitrary stale parents or mismatched tokens cannot establish success or authorize
uncertain-marker removal (ruling 36).

Publication sequence: stage under `.staging/` → validate files and hashes → insert a
`pending_operations` row in state `prepared` → rename into the immutable destination →
one transaction inserting entities, indexing and activating the revision → mark the
operation `complete`. A crash before commit leaves a reconcilable operation; a crash
after commit leaves files already published. Reconciliation at startup finishes or
reverts prepared operations idempotently and never issues network calls.

Before copying, each reserved stage writes and flushes a versioned
`.publication.json` containing its operation UUID and planned artifact hashes.
Cleanup requires this validated registration or a persisted operation's registered
names, validates every entry before removing siblings, and defers with a warning
when unknown files or subdirectories exist. A UUID directory alone is not deletion
authority (ruling 30).

Each committed revision has a post-commit `catalog.json` receipt containing
meeting/revision/segment IDs, asset hashes and origin mappings. Append-only
`meetings/<uuid>/states/<row_version>-<uuid>.json` snapshots preserve later durable
status and active-pointer changes without rewriting historical revisions (ruling 26).
Rebuild uses validated managed receipts and state snapshots only; it does not scan
external origins. With the live DB, reconciliation can finish a missing receipt
from committed entities. If both commit knowledge and its receipt are lost, the
uncertain tree is retained and reported rather than guessed complete.
Receipt 3 also records `operation_id` and its validated publication proof; rebuild
restores the complete operation, including same-meeting parent validation. Receipt
2 compatibility grants no synthetic token/proof. DB schema remains 2. Receipt
envelopes and nested mappings/arrays of objects are validated before mapping
methods or referenced reads. Every malformed meeting tree is retained with an
identified warning so valid siblings still rebuild; repeat reconcile/discovery and
maintenance JSON remain usable. GR2 is ADDRESSED by final independent scoped review.

Source artifact roles are explicit: known session JSON/text/Markdown plus validated
WAV references. Containment alone never authorizes opening, probing or hashing a
credential or unknown file. `session.json` is included even if a legacy output list
omits it. Hashes/paths in receipts and pending operations are untrusted data and
must receive the same role, identity and containment validation before access.

### Application-owned persistence

Introduce `MemoryService.ingest_session(session_dir, *, revision_of=None) -> IngestResult` (`inserted`,
`unchanged`, `revised`, `rejected`, with `meeting_id` and warnings) and route every
producer through the application layer: the controller after `SessionProcessor`
publishes, `here trans`, and retry. The three direct `write_session_artifacts` calls
in `src/cli.py` (`_save_transcription`, `_transcribe_audio_path`) are consolidated
behind that path. Catalog failure must not touch audio: the WAV and `session.json`
remain, the result reports "audio saved; library pending recovery", and the next
startup retries local tasks only.

Application-owned output folders first persist `memory_pending.json` version 2 with
`schema_version`, original optional `meeting_id`, expected `revision_of`,
`intent` (`ingest` or `status`), `publication_id` and `source_kind` (`session` or
`legacy_audio_only`). After subordinate final artifacts are published, the
writer's internal `before_metadata_commit` callback durably replaces that intent
with marker 3 before the final `session.json` replace. Marker 3 retains those
fields and adds `hashes`: exact lowercase SHA-256 values for the prospective
manifest bytes and its role-validated final artifact inventory. The reader accepts
markers 1/2 without inventing seals or proof. Startup validates that local intent and
replays catalog work only, never provider transcription. A failed/cancelled retry
with no new result retains the previous complete active revision and replays only
its durable outcome under the same expected-revision precondition (ruling 27).
The controller consumes the real processor's private, directory/manifest-bound
result once; unrelated injected processors retain the fallback. This avoids double
publication while durable proof independently supports restart replay. The scoped
NB1 review covers actual postcommit child crashes with existing and rebuilt DBs.
No completed metadata may become unrecoverable before its first durable local
publication intent. Sealed replay requires the manifest to be final and the entire
measured artifact map to match. A still-pending manifest or changed/incomplete
final evidence retains its exact marker/source bytes with a warning, never
imports the previous result as a new revision and never calls the provider.
Pending-manifest discovery exposes that uncertainty through the existing public
`RecoveryCandidate.error_summary`. GR1 is ADDRESSED by final independent scoped review.

For valid WAV-only legacy retry without `session.json`, reserve a canonical UUID
and actual lease before mutation/provider, with a measured origin mapping whose
revision is NULL and zero historical transcript revisions (ruling 37). Provider
input preserves the source's absent ID. Durable `source_kind=legacy_audio_only`
keeps capture `started_at`/`completed_at` NULL across retry/receipt/rebuild, even if
compatibility v1 metadata later contains inferred dates. Real processing timestamps
describe processing, never unknown recording time.

`MEMORY_INGESTED` follows `SESSION_PERSISTED` and precedes terminal completion.
The additive `ApplicationSnapshot.memory_warning` renders library warnings in the
existing detail area while real `last_error` retains priority; clean publication
or new work clears the warning. A completed recording remains completed when
only the catalog is unavailable (ruling 24).
Accepted cancellation while persisting a capture failure must preserve one meeting
and the capture-error/audio evidence with cancelled UI, compatibility metadata and
catalog outcome. The real controller privately defers the first catalog publication
through the existing cancellation cutover; any accepted cancellation reseals the
final outcome before that publication. Canonical and optional source identities
retain one meeting and one partial revision. Direct capture-failure persistence and
foreign injected processors preserve their existing contracts, as do ordinary
same-UUID conflict checks. GR3 is ADDRESSED by final independent scoped review.

### Durable meeting state

`meetings.status` is its own machine: `recording`, `processing`, `completed`,
`failed`, `cancelled`, `interrupted`, `deleting`. It is written by the application
layer from controller outcomes and from startup reconciliation; `ApplicationState`
and `SessionMetadata.status` are not reused as the meeting state. `paused` remains an
event, not a status. Destructive cancellation during recording deletes the
`recording` row together with the capture directory.

File transcription and explicit retry also hold actual owner locks throughout
provider work, with durable `processing` state and no open SQLite transaction
around that work. Reconciliation preserves live owners and marks abandoned
recording/processing rows interrupted only after ownership is no longer held.

A competing producer must acquire actual ownership before mutating session output
or calling the provider. Recovery discovery/materialization must exclude a live
owner before reading changing material or attempting persistence/cleanup, and
materialization rechecks stale candidates. Merely preserving the catalog row is
insufficient. Malformed marker envelope/field types reject that entry without
aborting valid siblings. Re-review marks original F5/F6 addressed, including the
native lease before journal publication. The authoritative native path is
`<sessions_root>/.capture_owners/<capture_id>.lock` (rulings 32/34), retained by a
controller job scope through real helper closure and final persistence even on
catalog failure. Direct wait/open-failure releases only after closure. No product
protocol or deadline changes. Existing-metadata inspection probes only existing
leases without creation; journal materialization upgrades ownership before mutation.
In the missing-metadata branch, canonical capture UUID validation precedes even
the journal path join/probe. Existing manifests retain optional/opaque source IDs.
Round-2/3 scoped reviews address the snapshot/audio-only regressions and NB2;
historical full 2 at `8087e25` passes 940 tests/seven skips. The combined fix at
`7bfd0cb` has a passing 26-case group/recovery focus; test-only `ef70f78` has the
historical full4 passing953/seven skips. Final review and exact-head hosted gates
are satisfied at merge53712c5; those previous runs keep their source-specific scope.

### FTS

Create the FTS5 virtual table over active complete revisions in this group, kept in
sync inside the publication transaction, with `unicode61 remove_diacritics 2`. No
product query uses it yet; it exists so group B can prove deletion purges it and M3
can build on it. If FTS5 is unavailable at runtime, record the capability and skip
indexing with a typed warning; do not fail ingestion.

The warning exposes code `fts_unavailable`. The derived `segments_fts` table has
meeting/revision/segment IDs, text and title; historical/partial revisions are not
indexed. Product search remains M3.

### Maintenance CLI

`here memory ingest PATH --json`, `here memory reconcile --json` and
`here memory show UUID --json` operate on the configured data root without a
provider key. Exit codes: 0 success, 2 invalid arguments/missing input, 3 rejected
identity or missing meeting, 4 unavailable/corrupt catalog. JSON success includes
the relevant result; errors include a nonempty `error` or rejected-result warnings.
An ordinary successful transcription with catalog-only failure retains its
existing successful exit and reports local recovery pending.

Typed service/result categories distinguish invalid input (2), identity conflict
(3) and source/publication I/O or integrity failure (4), without parsing warning
prose. Original F7 is addressed. The prior CLI sf.info regression was corrected
with an independently authored real PCM16 fixture, preserving outcome assertions
and production media validation; it is covered in the 61-case focus.

## Acceptance

- New SQLite database created from migrations, a v1 test database migrated with a
  managed backup present, and a future-version database rejected with `UnsupportedSchema`.
- `foreign_key_check` and `integrity_check` pass after every migration and after every
  fault-injection test.
- GUI recording, CLI recording, `here trans` and retry each end with exactly one
  `meetings` row keyed by the journal `meeting_id`, one `audio_assets` row whose
  SHA-256 matches the file, and one active revision with segments.
- Importing the same v1 session twice yields `inserted` then `unchanged`; the same
  UUID with different content yields a readable conflict; identical audio in two
  sessions yields two meetings.
- Retry produces revision `n+1`, activates it only after commit, leaves revision `n`
  readable and historical; a retry that fails mid-publication leaves revision `n`
  active and no orphan files outside `.staging/`.
- Two chunks separated by a five-minute pause: the second segment's `start_ms` seeks
  the WAV to the correct recorded position, not five minutes later.
- Segments with absent or contradictory timestamps keep text, speaker and ordinal;
  `timing_source=absent`; nothing reordered by invented time.
- Process kill at each publication boundary (before staging, after rename, before
  commit, after commit, before marking complete) followed by restart leaves one
  identity, one active revision and no orphans.
- Catalog failure injected during a recording leaves audio and `session.json`, reports
  the pending-recovery state, and the next startup reconciles without provider calls.
- Journal, session and destination paths that are absolute, contain `..`, or resolve
  through a symlink or junction outside the root are rejected before any read, write
  or delete; sibling valid entries still appear.
- A busy database returns the typed recoverable error within `busy_timeout`; no
  transaction holds the provider, audio copy or hashing.
- The full Windows CI suite passes with `QT_QPA_PLATFORM=offscreen`, a synthetic
  `OPENAI_API_KEY`, temporary data roots and no real `.env` or private recordings.

## Boundaries

No listing, filtering, search UI, playback, participants editing, deletion, portable
package export/import, intelligence or settings UI. Explicit v1 session ingestion is
part of this group. No change to capture, overlay or chunking beyond
carrying `meeting_id` and segments. No global speaker identity. No provider call in
any catalog operation.

## Technical references

- [SQLite FTS5](https://www.sqlite.org/fts5.html): derived index; `secure-delete`
  behavior differs from the core option and must be probed at runtime.
- [SQLite backup API](https://www.sqlite.org/backup.html): the only supported way to
  copy a database with WAL active.
- [SQLite WAL](https://www.sqlite.org/wal.html): local-disk requirement and checkpointing.
