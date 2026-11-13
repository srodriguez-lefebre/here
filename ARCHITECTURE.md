# Architecture

## Overview

`here` records audio, transcribes it, and writes a structured session artifact
folder.

The main flow is:

1. An interface creates the application controller and starts a recording.
2. Recording backend captures microphone, system audio, or both.
3. Capture writes its primary WAV before offering blocks to live processing.
4. Live processing cuts chunks near silence, normalizes/mixes audio, and
   transcribes chunks in the background.
5. Once recording ends, output creates a recoverable session folder and
   materializes normalized/mixed `audio.wav`.
6. If live processing fails or exceeds its queue budget, the application layer falls back to offline session
   transcription using the recoverable audio.
7. Output writing creates completed or failed artifacts.

## Main Modules

- `application/`: interface-independent lifecycle, commands, events, processing,
  recovery, and the production composition root.
- `cli.py`: secondary input adapter. It delegates recording lifecycle to the
  application controller.
- `recording/`: platform-specific capture and Windows diagnostics.
- `live_processing.py`: live chunk generation, transcription workers, and
  live chunk metadata.
- `audio/`: mixing, chunk planning, silence boundary selection, and text merge.
- `transcription/`: OpenAI client, offline transcription service, and segment
  timeline merge.
- `output/`: Pydantic metadata models, Markdown rendering, and session artifact
  writing.
- `memory/`: immutable catalog DTOs, operation-owned SQLite connections, numbered
  migrations with managed backups, an explicit transaction repository, validated
  artifact ingestion and local revision publication/reconciliation. Scoped fixes
  through round 3 and the combined group-fix wave are integrated at `7bfd0cb`.
  The final 26-case group/recovery focus passes; the earlier 940/seven-skip full
  belongs to `8087e25`. New full 3 has eight capture-recovery test-wrapper failures,
  945 passes/seven skips, now historical. Test-only correction `ef70f78` passes
  its 13-case focus and full 4: 953 passed/seven skips in 358.94s, both wheel
  smokes passing on unchanged source. Final technical review APPROVED source,
  fixture and docs `8845877` at that historical checkpoint. Final sourcec59ae33
  adds the accepted Copilot corrections; exact-head Windows980/2/package and
  received Copilot close those hosted gates in merge53712c5 (PR21).
- `ui/`: Qt Widgets presentation, the live-logo overlay, and the single adapter
  from visual actions to the application contract.

## Windows Presentation Layer

Windows distribution starts with a locked noneditable 0.3.0 wheel installed into
a dedicated CPython 3.13.15 AMD64 build environment. PyInstaller creates windowed
`here.exe` and console `here-cli.exe` sharing one inspectable `_internal` directory.
The fixed audio child dispatch runs before Qt/provider bootstrap and uses the
console sibling with bounded pipes and process ownership. A Windows lifetime
mutex lets the per-user Inno Setup installer refuse replacement while here runs.

Build DLL discovery uses owned Python/wheel paths and Windows system paths.
Current M1 uses raster-painted Qt Widgets; unused multimedia, image/icon codecs,
QML/virtual keyboard, PDF/SVG and software OpenGL fallback are omitted. Actual
native files and full namespaced notices accompany the build; wheel/runtime,
source input and native hashes are recorded. The deterministic packaged smoke
creates temporary local settings, initializes the real desktop, drains owned jobs,
and checks rejected/timeout child IPC without hardware, sessions or provider calls.

The Windows interface and live logo run in the same process as the application
controller. `ApplicationEventBridge` marshals application callbacks onto the UI
event loop and distributes immutable snapshots and the reduced combined audio
level. Widgets never open an audio device, parse logs, inspect session files,
or invoke the CLI.

`ApplicationUiAdapter` is the narrow integration seam. It translates UI intent
into the stable application commands, including a combined-source `StartRequest`.
The main window and overlay depend on that seam, so controller implementation
changes do not leak into painting or interaction code. A fake implementation of
the real contract supports deterministic preview and headless tests.

The overlay is a transparent, frameless, always-on-top Qt tool window. Each new
active session positions it in the lower-right of the current screen's available
area with a 24-pixel margin; dragging changes only that instance and is never
saved. Its 104-pixel surface redraws at roughly 30 frames per second, independently
from capture callbacks. Audio updates are reduced to one normalized level and are
ignored outside the recording state.

Visual state decisions are intentionally explicit:

- recording uses a reactive perimeter in the configured accent color;
- pause is a static ring with a pause mark;
- preparation, stopping, and processing use an indeterminate spiral in that color;
- success and failure use fixed green-check and red-cross marks for three seconds;
- recoverable processing cancellation uses a neutral gray minus mark for 1.8
  seconds, never a success or error symbol; destructive recording cancellation
  disappears immediately because it leaves no session.

The accent preference is persisted through `QSettings`; overlay position is not.
Closing the main window while `snapshot.has_active_work` is true hides it while
the overlay keeps the process reachable. Closing while idle requests full process
exit. If work ends while the window is hidden, the process exits after the
terminal indicator finishes.

Failed or recoverably cancelled sessions expose the application controller's
`retry` command from the main window. Starting a new recording remains available
from terminal states without conflating it with retrying the preserved session.

## Output Layer

The output layer owns persisted artifacts.

`session.json` stores global session metadata, including the recording window
and the concrete device names captured for each source when the backend exposes
them. `chunks.json` stores chunk-level metadata only, without audio, transcript
text, or duplicated global fields.

Failed sessions are still persisted. They include `audio.wav`, `session.json`,
`chunks.json`, and `errors.json`, so API/provider failures can be retried without
re-recording.

`CaptureFailed` exposes a partial `RecordingSession` and its original cause after
writers close. Application persistence records failed capture without claiming a
completed transcript. When normalization fails, local `source_*.wav` copies and
their provenance remain available for retry; recovery validates each filename
and resolved path within the session. Destructive recording cancellation takes
precedence over recovery and leaves no session.

Catch-up silence is offered to live processing in disk-write order so both paths
share the recording timeline. Capture completion and failure recovery atomically
leave destructive-cancellation states; any later accepted cancellation preserves
the session and publishes matching cancelled metadata and application state.

The capture queue holds 256 blocks and the transcription queue eight pending jobs.
Both producers use nonblocking handoff. Overflow disables live work and requires
offline transcription from complete saved audio, including after partial live
success. Completion events replace queue sentinels, so full queues cannot block
shutdown. Abort waits at most two seconds for workers; an already-issued provider
request may finish later, and workspace cleanup waits for worker completion.

Each source buffer is limited to the live chunk duration plus silence search and
overlap. A stalled or missing source that exceeds that skew budget disables live
work; an incomplete final source set also requires offline fallback.

`segments.json` has a versioned provider-evidence envelope. Segment text and
nullable timestamps/speakers survive optional display cleanup. Chunk segments
receive recording-time offsets once and carry request-local `chunk_index` and
`speaker_scope`; repeated labels across requests do not establish a human
identity. Original overlapping provider segments stay in order even when the
display timeline merges them. Invalid timestamps become missing; valid zero
timestamps remain zero. `session.json.capture_sources` separately retains original
device names and raw audio properties when `sources` describes a normalized mix.

`coerce_audio_transcription` retains nonempty text-only responses as untimed,
speakerless evidence without mutating the provider-owned DTO. Live and offline
merging still use the plain-text overlap path for unstructured evidence. The
additive nullable `timing_source` field accepts `provider`, `imported_explicit` and
`absent`; historical documents may omit it. Newly written provider segments use
`provider` when at least one valid bound remains, otherwise `absent`. Reversed
bounds become null before recording offsets are applied; one-sided finite
measurements retain their measured bound.

All five transcript completion paths carry `TranscriptionResult.raw_text` to
`raw_transcript.txt` independently of cleaned `final_text`. The optional
`SessionArtifactPaths.raw_transcript_path` and `session.json.output_files` expose
that artifact. The writer preflights its managed entry and stages raw and segment
evidence before publishing metadata. A handled metadata-replace failure restores
previous raw bytes or removes a fresh unpublished raw file. This I/O rollback is
not an atomic filesystem transaction against process termination or concurrent
writers; the immutable catalog publication/reconciliation path below handles durable revisions.

`events.json` is written when lifecycle events exist. Pause and resume events
contain both wall-clock time and the duration of material actually recorded.
`session.json.total_paused_seconds` records the accumulated wall-clock pause;
no silence is inserted into `audio.wav`.

This split keeps artifacts useful for debugging now and ready for a later local
search/RAG layer.

## Catalog foundation (M2 A task 1)

`open_library(data_dir)` owns one connection per worker or operation to
`<data_dir>/library.sqlite3`; the connection is never shared with the Qt thread.
Connections enable foreign keys, FULL synchronous writes and a 250 ms busy timeout.
Proven-local Windows volumes use WAL; remote/mapped or unproven-local volumes use
rollback journaling with a warning. Future schema versions are
refused before persistent pragmas are changed. Managed database, sidecar and backup
paths reuse the output path validators.

Numbered Python migrations create all twelve catalog tables, record names and
checksums, and validate foreign keys and integrity within a `BEGIN IMMEDIATE`
transaction. Before upgrading an existing populated database, a separate read
connection takes a SQLite backup API snapshot under the migrator's write
reservation. Backups live under `<data_dir>/backups/` and are enumerated as managed
data for later lifecycle purge. Failed migrations roll back DDL and their ledger.

`MemoryRepository` requires an explicit owned transaction for mutations and revision
activation. Database constraints enforce same-meeting active revisions, unique
revision numbers and segment ordinals, and valid durable states and timing bounds.
Frozen DTOs carry stored values without introducing global speaker identities.
SQLite busy/locked errors from migration reads, repository reads and writes become
the recoverable `Busy` error, including the rollback-journal exclusive-lock case.

M2 A publication/producer integration is merged in PR21/53712c5. Independent
reviews cover original F1–F7, NB1/NB2, GR1–GR3 and Copilot C1–C5. Exact-head8a498e2
Windows CI passes980/two wheel skips/307.46s with package success; separate fresh
installed-wheel checks pass4/7.51s. This is automatic synthetic verification, not
new hardware/provider/clean-install certification. M2 lifecycle/deletion remains B.

## Catalog publication and local recovery (M2 A task 3)

Merged sourcec59ae33 includes PR21 Copilot corrections R40/R43/R44. Ordinary
precommit `Exception` triggers evidence rollback before propagation;
`BaseException`/process termination is unchanged, and failed rollback retains
backups/causal diagnostics. The controller shares its actual MemoryService with
the default processor while preserving injected processors. Rejected completed
replay and malformed pending intents use the existing nonretryable RecoveryCandidate
warning before retry-audio inspection. An actual catalog ownership lookup failure
(LibraryError or ambiguous-origin ValueError) becomes a private metadata-only
warning; no marker/error/audio/native material is inspected and no identity or
parent is invented while ownership is unknown. Real OwnershipBusy/UnsafeSessionPath
guards remain strict, stale materialize fails closed, and the adapter rejects
nonretryable candidates before credentials/provider retry. No new process,
capture deadline, schema, dependency or UI flow is introduced by those corrections.

`MemoryService.ingest_session(session_dir, *, revision_of=None)` interprets v1
artifacts without modifying the origin. Known artifact roles and explicit WAV
references are validated before referenced files are opened, hashed or probed;
`session.json` is always included. A contained `.env` or unknown manifest entry is
not authorized input. Nested managed paths reuse the shared path helpers.

Publication holds `.memory.lock` with bounded acquisition while using short DB
transactions. It reserves `.staging/<operation-uuid>`, durably writes the planned
file/hash registration in `.publication.json`, copies and verifies recognized
artifacts, records a prepared operation, then renames to
`meetings/<meeting-uuid>/revisions/<revision-uuid>`. One DB transaction inserts the
entities/evidence, updates active-complete FTS and activates the revision. Copies,
hashing and provider work remain outside SQLite transactions.

After DB commit, `catalog.json` records the revision's IDs, hashes, assets and
origin mapping; append-only `states/<row_version>-<uuid>.json` preserves the latest
meeting status and active pointer. The operation becomes complete afterward.
Reconciliation can finish receipts for known committed entities or revert an
uncommitted owned publication. Cleanup validates all entries first; only registered
protocol names may be removed, and unknown files/subdirectories defer cleanup with
a warning. A UUID directory alone never establishes ownership of its contents.

With a missing catalog, managed receipts and state snapshots supply reconstruction
evidence without reopening external origins. Unreceipted material is uncertain and
is retained/reported. If DB commit knowledge and the receipt are both lost, recovery
does not infer that the material was committed. Receipt/pending metadata are
untrusted. Receipt envelope objects, nested mappings and arrays of objects are
validated before mapping methods or referenced reads. A malformed meeting tree is
retained with an identified warning; valid siblings can rebuild and maintenance
JSON remains usable. Artifact roles precede reads/hashes, and generated state
names precede state reads. Earlier missing-role-guard observations describe
superseded source.

Origin plus fingerprint makes ordinary import idempotent. Trusted explicit retry
supplies the expected active revision under `revision_of`. Each explicit producer
invocation has a new publication UUID. Operation payload and receipt 3 bind it to
the committed revision, expected parent, normalized origin, measured fingerprint
and source kind. Exact proof/content/committed-revision matching makes that
publication's replay unchanged, including after rebuild restores the proven
complete operation. Prepared-only state cannot prove commit. Fresh identical
retries still create new revisions; receipt 2 is accepted without invented proof.
Covered GUI legacy
retry and existing-session CLI preserve the source optional identity independently
of canonical catalog ownership. Trusted legacy preparation resolves/preimports its
lineage before mutation/provider, including None/opaque M1 IDs; ordinary imports
retain strict UUID validation. Aware
dates preserve their instants in UTC; naive source dates remain unspecified with
a warning. FTS5 indexes active complete revisions using accent-folding unicode61;
missing capability returns typed `FtsUnavailable` while preserving publication.

The service exposes `.owners/<meeting-uuid>.lock` handles for capture/provider-work
lifetimes. Catalog reconciliation probes OS ownership and preserves live rows;
residual lock filenames or PIDs do not prove liveness. Competing application
producer and recovery paths now acquire/recheck owners before mutation. Native
capture ownership starts before publishing the journal at
`<sessions_root>/.capture_owners/<capture_id>.lock`. The controller job scope retains
that OS lease through real helper closure and final persistence, even if the
constructor or catalog fails; no deadline/protocol changes. Abandoned
recording/processing rows become interrupted. Remote/mapped/unproven-local Windows
volumes use rollback journaling with a visible warning; proven local storage uses WAL.

`SessionProcessor` registers a version-2 intent before explicit producer work.
Every final artifact path supplies the output writer's internal
`before_metadata_commit` callback: after subordinate artifacts are published but
before the final `session.json` replace, it durably writes marker 3. This preserves
the original optional meeting ID, expected revision, ingest/status intent,
publication UUID and `source_kind` (`session`/`legacy_audio_only`), and adds `hashes`.
The seal contains the prospective `session.json` bytes and the exact recognized
artifact inventory, using validated roles and lowercase SHA-256 values. Hashing
remains outside SQLite transactions. Readers retain marker 1/2 compatibility
without inventing a seal or provenance.

`publish_memory` validates a sealed final manifest and the entire evidence map
before catalog work. A pending manifest, unmatched inventory or changed bytes
cannot authorize importing the previous result or removing uncertain intent.
The marker and source bytes remain with a warning; pending-manifest discovery
adds that warning to the existing `RecoveryCandidate.error_summary`. Matching
final evidence replays locally without a provider call. Failed/cancelled retries
with no new transcript update only durable state and preserve the previous complete
active revision, including during busy replay.

The real controller privately defers capture-failure catalog publication through
the existing cancellation cutover. Any accepted cancellation reseals its final
manifest and original error/audio evidence before the first catalog publication,
preserving one meeting and one partial revision for canonical or optional source
IDs. Direct `persist_capture_failure` and foreign injected processors keep their
existing contracts. The private directory/manifest-bound result is still consumed
once; injected processors retain a fallback. This handoff is separate from durable
replay proof, and catalog failure keeps the saved audio/text.

`MEMORY_INGESTED` follows `SESSION_PERSISTED` before terminal completion. Nonfatal
catalog issues populate `ApplicationSnapshot.memory_warning`; the existing window
detail displays it only when no real `last_error` has priority. New work or clean
publication clears the warning, without changing a completed recording into failure.
Maintenance CLI ingest/reconcile/show use the same service and expose JSON results
with typed error categories. Postcommit housekeeping now reports committed success
with a recoverable warning instead of precommit rejection.

Schema 2 adds `revision_audio_assets` (meeting/revision/asset, source file and
ordinal) without changing migration 1. Version-2/3 receipts, rebuild and `show` JSON
retain ordered links, including reused assets in A→B→A retries; cumulative assets
cannot substitute for those references. Ambiguous older evidence is retained with
a warning rather than assigned by guess. Scoped review marks original F1–F7 addressed.

WAV-only legacy retry reserves a canonical identity/actual lease with origin revision
NULL and no invented historical transcript before provider work. Durable
`source_kind=legacy_audio_only` keeps unknown capture dates NULL across retries and
rebuild. Metadata-only recovery inspection does not create lock files; actual
materialization upgrades and retains the leases before mutation. Missing-metadata
journal candidates require canonical UUID before constructing/probing the path;
existing manifests retain their optional/opaque source identity.

Scoped reviews address NB1, NB2 and the five prior integrated failures. Historical
full 2 at `8087e25` passes 940 tests/seven skips in 325.09s. Its final whole-group
source/docs verdict is changes requested for GR1–GR3; GD1/GD2 minor checkpoint
corrections were committed separately. The combined R38 fix at `7bfd0cb` matches
Green `ec55b95` source/tests exactly. Independent baseline RED reported 11 failures
and one passing control; the final group/recovery focus passes 26 in 34.46s. An
additional pre-final-replace control first failed only its missing uncertainty
warning, then passed with the recovery correction. The single scoped re-review
marks GR1–GR3/GD1/GD2 ADDRESSED and technically APPROVES source, fixture and
documentation supplement `8845877`. Fresh root wheel/sdist and noneditable 65-file probe
pass on `7bfd0cb`; full 3 reports eight failures/945 passes/seven skips in 364.23s,
both wheel smokes passing. Two existing capture-recovery monkeypatch wrappers
reject `before_metadata_commit`; their narrow optional-keyword forwarding fix
preserves all assertions and crash cutpoints. It is integrated at `ef70f78`:
13 cases pass in 24.82s and **full 4 passes 953/seven symlink skips in 358.94s**,
XML 960 cases/0 failures/0 errors/7 skips/internal 358.621s. Both wheel smokes pass.
The original `7bfd0cb` wheel remains exact for its unchanged 65-file product source;
only the test wrappers changed. Hosted gates were pending at that checkpoint;
final exact-head Windows CI/Copilot/merge now close A as recorded above. Earlier
failed fulls and the unexplained child timeout
retain their original results; no new certification or group acceptance is claimed.

## Lifecycle and deletion

The memory service owns title and notes edits, participant links and reversible
request-scoped speaker attribution. Optimistic writes compare the expected row
version and preserve immutable transcript revisions and their audio links.
Reconcile uses the existing operation journal to repair committed catalog state
and lifecycle event snapshots without repeating a transcription request.

Deletion separates durable acceptance from cleanup. The service records a minimal
suppression entry before destructive work. One bounded read view resolves the
meeting UUID and exact origin aliases before rows are returned. Accepted deletion
stays hidden while cleanup is pending. Cleanup uses the verified inventory for
managed data and derivatives, including FTS, WAL and backups. Unknown material
is retained and reported. The final tombstone has identity and origin suppression
metadata for reconstruction. External import sources stay under the user's control.

Destructive operations require explicit native session context and revalidate
ownership before proceeding. Native and catalog leases protect the relevant
cleanup boundary. A caller's release acknowledgement runs outside the
publication lock and SQLite transaction. Recovery can use the existing private
staging rules for a verified orphan with no journal reference. Journaled staging
requires its own validation and ownership. Pending recovery publication leaves
the read view before acquiring the writer lock and revalidates identity and
authority before opening material or replacing its marker. The independent
group review checked the integrated source and the 110-case run, including
destination collisions and SQLite Busy warnings. The follow-up restored typed
SQLite errors in the read view while preserving caller error identity for other
exceptions.

Reindex uses the existing `segments_fts` index and rebuilds it from active complete
revisions visible through suppression. It runs locally and preserves historical
revisions and partial transcripts. Lifecycle logic stays in the memory core,
which has no dependency on Qt or Typer.

## Library search and desktop workers

The M3 A candidate's `SearchService` reads the same catalog used by the
application controller. Its connection factory opens one fresh connection per
read and closes it after the R50 guarded view. That view resolves R42 suppression
by meeting UUID and exact origin/fingerprint aliases. SQL applies suppression
and date, status and participant filters before keyset paging. An unavailable
suppression registry fails the read with a typed catalog error.

Empty normalized search uses the meeting list, ordered by descending start date
with unknown dates last and meeting ID as the tie-breaker. Complete transcript
search binds a quoted token expression to FTS5 and uses BM25 ranking. The optional
active-partial branch uses literal title/segment matching and rank zero in the
same result page.
When FTS5 is unavailable, both branches use the literal route and the search page
carries a degraded flag. Search returns plain-text snippets and preserves
nullable dates and segment bounds. Maintenance delegates once to
`MemoryService.reindex()` on the same root.

The opaque ranked cursor binds the normalized query, filters, backend and
corpus generation. The generation includes the logical FTS documents and
eligibility data, including the coherent suppression snapshot, in the same
transaction as the page. An obsolete ranked continuation raises
`InvalidInput(reason="corpus_changed")`. List continuations keep the date/ID
keyset.

`LibraryView` calls the application adapter through `BackgroundJobs`. A 200 ms
debounce coalesces search input. Request IDs and separate page, participant and
detail generations prevent older callbacks from changing the current view.
Closing the detail invalidates its outstanding reads. A corpus-change error on
continuation restarts the search at page one. Title edits send the displayed row version
to the memory service, which updates the catalog and indexed title together.

The desktop registers a deletion acknowledgement on the composed memory service.
A tracked worker sends a queued Qt ticket and waits for the view to invalidate
its reads. Replacement reads wait for the worker to settle, so a late result
cannot restore a row hidden by accepted deletion. Selecting a transcript segment
emits a reference with its meeting, revision and segment IDs. Playback and
authoritative audio target resolution are M3 B work.

## Existing Audio Transcription

`here trans {file}` builds the same recoverable session shape from an existing
audio file. When `{file}` is already `audio.wav` inside a failed session folder,
the command updates that session in place.

`SessionProcessor.transcribe_audio_path(...)` now owns file normalization, pending
publication, completion/failure artifacts and conservative postcommit cleanup.
CLI helpers inject their existing transcriber, normalizer and clock and delegate
to that application owner; no direct CLI artifact writer remains. Recording,
offline fallback, file transcription and retry therefore share application-owned
raw/segment persistence while retaining external input ownership and the existing
recoverable-audio filename and exception-cause guards.

## Diagnostics

Windows diagnostics are intentionally separate from transcription. They inspect
audio devices and measure signal without needing OpenAI credentials.

## Application contract

GUI and CLI code import `create_default_controller` from `here.application`.
The returned controller is asynchronous: `start()` returns after scheduling
preparation, state changes arrive through `subscribe()`, and capture or
processing never runs on the caller's UI thread.

The stable states are `idle`, `preparing`, `recording`, `paused`, `stopping`,
`processing`, `completed`, `failed`, and `cancelled`. Commands are `start`,
`pause`, `resume`, `stop`, `cancel`, and `retry`. Audio-level events expose only
bounded peak/RMS aggregates for the combined signal at no more than 15 updates
per second; raw audio never crosses the UI contract.

Cancellation semantics depend on the current state:

- cancelling while preparing, recording, or paused deletes raw temporary audio
  and produces no session;
- cancelling while stopping or processing preserves normalized `audio.wav`,
  writes a recoverable session with status `cancelled`, and permits `retry()`;
- an already-issued provider request may finish before the worker observes the
  cancellation flag, but no later retry or processing stage is started.

Only one job can be active in a controller. A fresh job can start from any
terminal state. Listener callbacks run on worker threads, so the Qt adapter must
bridge them to queued Qt signals before touching widgets.

## Programmatic Windows capture

`recording.start_recording()` opens selected or default Windows devices and
returns a controllable handle. Pausing drains device buffers without writing
frames, preventing both stale audio and synthetic silence on resume. Stopping
returns a `RecordingSession`; cancelling removes all temporary sources.

The first implementation remains a single process and supports Windows capture
only. Device and long-duration validation still require the real-hardware matrix
described in the milestone plan; unit tests use deterministic adapters and do
not claim hardware certification.

## Decisions made during implementation

- Application events are immutable values and listener failure is isolated so
  a UI subscriber cannot stop capture.
- Peak and RMS are combined using the maximum current source value. This favors
  an unmistakable recording indicator over a meter intended for mastering.
- Offline provider failures receive three total attempts with incremental waits
  of 0.5 and 1 second; the terminal failure is appended to recoverable metadata.
- Session metadata stays schema version 1 because all new fields are additive
  and have backward-compatible defaults. `events.json` has its own versioned
  envelope.
- Device identifiers are optional in `StartRequest`; absence means the current
  Windows default. Selection and remembered preferences remain interface and
  configuration concerns.
