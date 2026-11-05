# M2 immutable catalog specification

Status: written 2026-11-04, not in execution. Implementation starts only on an explicit
user instruction (ruling 1 in `DELIVERY.md`). Governing design:
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

### Schema and migrations

Tables and invariants exactly as `PRODUCT_DESIGN.md` §4 "Tablas mínimas":
`schema_migrations`, `meetings`, `audio_assets`, `transcript_revisions`, `segments`,
`participants`, `meeting_participants`, `speaker_assignments`, `lifecycle_events`,
`ingest_sources`, `pending_operations`, `deleted_origins`. This group creates all of
them so group B does not need a second migration to add lifecycle tables; group B may
add columns.

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

Publication sequence: stage under `.staging/` → validate files and hashes → insert a
`pending_operations` row in state `prepared` → rename into the immutable destination →
one transaction inserting entities, indexing and activating the revision → mark the
operation `complete`. A crash before commit leaves a reconcilable operation; a crash
after commit leaves files already published. Reconciliation at startup finishes or
reverts prepared operations idempotently and never issues network calls.

### Application-owned persistence

Introduce `MemoryService.ingest_session(session_dir) -> IngestResult` (`inserted`,
`unchanged`, `revised`, `rejected`, with `meeting_id` and warnings) and route every
producer through the application layer: the controller after `SessionProcessor`
publishes, `here trans`, and retry. The three direct `write_session_artifacts` calls
in `src/cli.py` (`_save_transcription`, `_transcribe_audio_path`) are consolidated
behind that path. Catalog failure must not touch audio: the WAV and `session.json`
remain, the result reports "audio saved; library pending recovery", and the next
startup retries local tasks only.

### Durable meeting state

`meetings.status` is its own machine: `recording`, `processing`, `completed`,
`failed`, `cancelled`, `interrupted`, `deleting`. It is written by the application
layer from controller outcomes and from startup reconciliation; `ApplicationState`
and `SessionMetadata.status` are not reused as the meeting state. `paused` remains an
event, not a status. Destructive cancellation during recording deletes the
`recording` row together with the capture directory.

### FTS

Create the FTS5 virtual table over active complete revisions in this group, kept in
sync inside the publication transaction, with `unicode61 remove_diacritics 2`. No
product query uses it yet; it exists so group B can prove deletion purges it and M3
can build on it. If FTS5 is unavailable at runtime, record the capability and skip
indexing with a typed warning; do not fail ingestion.

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

No listing, filtering, search UI, playback, participants editing, deletion, export,
import, intelligence or settings UI. No change to capture, overlay or chunking beyond
carrying `meeting_id` and segments. No global speaker identity. No provider call in
any catalog operation.

## Technical references

- [SQLite FTS5](https://www.sqlite.org/fts5.html): derived index; `secure-delete`
  behavior differs from the core option and must be probed at runtime.
- [SQLite backup API](https://www.sqlite.org/backup.html): the only supported way to
  copy a database with WAL active.
- [SQLite WAL](https://www.sqlite.org/wal.html): local-disk requirement and checkpointing.
