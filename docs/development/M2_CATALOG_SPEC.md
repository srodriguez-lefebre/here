# M2 immutable catalog specification

Status: in execution under the explicit 2026-11-05 mandate (ruling 20).
Tasks 1/2 have local verification and independent reviews. Task 3 round-1 source
`c28472b` is integrated at root `1f1bc88`. Re-review marks F1–F7 ADDRESSED but
requires Important NB1 (repeat publication/postcommit marker replay). The root full
run has five failures/920 passes/seven skips; no task/group approval or hosted gates
are complete. Requirements below distinguish implemented fixes from pending NB1
and the integrated failures. Governing design:
`PRODUCT_DESIGN.md` §2–§4, which is a taken decision (ruling 3). Spanish plan and
acceptance criteria: `../MILESTONE_2_PLAN.md`.

## Intent

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
invented. Round-1 source implements this invariant; acceptance still waits for NB1,
the five integrated full failures and the remaining group gates.

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
not a precommit rejection. Re-review marks original F3/F4 addressed. Important NB1
requires distinguishing a fresh explicit retry from replay of a publication already
committed under its parent. Replay must not add another revision or leave a stale
marker; a fresh identical retry still creates its new revision. Neither arbitrary
stale-parent acceptance nor deletion of uncertain markers is an acceptable shortcut.

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

Application-owned output folders may persist a durable `memory_pending.json`
with only schema version, original optional meeting ID, expected `revision_of`
and `intent` (`ingest` or `status`). Startup validates that local intent and
replays catalog work only, never provider transcription. A failed/cancelled retry
with no new result retains the previous complete active revision and replays only
its durable outcome under the same expected-revision precondition (ruling 27).
The controller must not publish an already committed processor result again as a
fresh stale-parent retry. Committed origin-marker replay requires durable proof and
idempotent completion; this NB1 path is not yet accepted after round 1.

`MEMORY_INGESTED` follows `SESSION_PERSISTED` and precedes terminal completion.
The additive `ApplicationSnapshot.memory_warning` renders library warnings in the
existing detail area while real `last_error` retains priority; clean publication
or new work clears the warning. A completed recording remains completed when
only the catalog is unavailable (ruling 24).

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
protocol or deadline changes. The full run's legacy audio-only compatibility and
snapshot failures remain unresolved rather than excluded from this requirement.

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
prose. Round-1 source adds typed categories and original F7 is addressed. The full
run's CLI stub failure remains a separate unresolved integration result.

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
