# M3 playback, evidence navigation and CLI specification

Status: preparation only, held before dispatch. M2 B merged in PR #25 at
`b4b3d1beb9b05e33ae03ab8381e805aa6a23ffc8`. The interface comparison uses
integrated M3 A candidate `31488693152067ae58eeacec08f0b01f47085fe4`.
M3 A's PR is unopened and its final review and acceptance remain pending.
The measured R63 warm-search
p95 of 879.48 ms exceeds the 500 ms gate. That measurement predates this candidate.
Under CLAUDE §11 and rulings 20/35/56, root may dispatch B after reviewing the
actual M3 A head and opening its PR, accepting the interfaces and merging any
required contract-documentation PR. Rebase B after A merges. Root owns file
leases, integration and merges.
Governing design: `PRODUCT_DESIGN.md` §3 and §5, a taken decision (ruling 3).
Spanish plan and acceptance criteria: `../MILESTONE_3_PLAN.md`.

## Intent

Take a person from a list item, a search hit or, later, a citation to the exact
original moment: the segment in context and, when the segment has real timing, the
audio at that millisecond. Add the `here meetings` commands so the CLI offers the
same operations with stable JSON. Playback must never pretend precision it does not
have, never recapture itself through loopback, and never start on its own.

## Decisions

### Inherited foundation and pending integration

Merged M2 supplies immutable revisions, schema 2's ordered
`revision_audio_assets`, `MemoryService.show`, receipt 3 and marker 3. Keep
receipt-2/marker-1/2 compatibility, exact committed-publication proof and retained
uncertain evidence. Reading a target never authorizes ingestion, replay or retry.
Unknown capture dates and absent segment times remain unknown, including
WAV-only legacy lineage with no historical revision. Metadata-only nonretryable
catalog warnings grant no media/provider/mutation authority; existing ownership
and path guards remain fail-closed.

The candidate implements the R39 keysets, R41 partial/fallback routing and R45
search generations. Its frozen DTOs in `memory.models` include `MeetingQuery`
and `SearchQuery` with `text`, `from_utc`, `to_utc`, `statuses`,
`participant_id`, `limit=50`, `cursor` and `include_partial=False`.
`MeetingPage.items` holds `MeetingSummary(meeting, audio_available, partial)`.
`SearchPage.items` holds `SearchHit` with the meeting/revision/segment IDs,
snippet, title, nullable date/bounds, timing source and partial flag.
Both pages have `next_cursor` and search has `degraded`. Merged B4/B5 provide
editing, participants, suppression and reindex. Final integration uses those
services on the actual roots. Before dispatch, verify these bindings against
the reviewed M3 A PR head.

R53's `SearchService(connection_factory, *, reindexer=None).reindex()`
delegates exactly once to the callback on the same injected MemoryService, returns
the identical R51 ReindexReport and propagates canonical errors without retry,
wrapping or recounting. Construction only stores the callback; delegation acquires
no factory connection, snapshot, lock, probe or migration. Without it, only reindex
raises existing LibraryError; reads remain available. ApplicationUiAdapter keeps
an explicit query override and binds maintenance through a separate optional typed
capability on the real memory/root, not a second service or required read protocol.
Merged `MemoryService.reindex()` supplies the callback. Reindex rebuilds a
derived index and does not call `before_delete` or unload playback.

### Common citation target

R46 binds the passive reference to
`Citation(meeting_id, revision_id, segment_id, quote=None, start_char=None,
end_char=None)`. It grants no file or playback authority. The B resolver validates
any supplied quote and Unicode character span against the selected source segment,
including membership and bounds; byte positions or UI-supplied text are not source
evidence. Inconsistent references use the canonical `InvalidInput` contract.
A UI-supplied `CitationTarget` is not proof of its path, status or text: navigation
resolves authoritative memory and guards before exposing a playable target.
Optional quote/span fields do not fill null timing or change exact audio links.

`CitationTarget` exists in the candidate's `memory.models` with
`meeting_id`, `revision_id`, `segment_id`, `title`, nullable `started_at`,
`text`, nullable `speaker`, nullable `start_ms`/`end_ms`, `timing_source`,
nullable `audio_path: Path` and `status`. B implements
`MemoryService.resolve_citation(Citation) -> CitationTarget` from guarded database
evidence and validated managed files. The existing `LibraryView.meeting_selected`
emits a meeting ID. Its `segment_selected` emits a passive `CitationTarget` with
no audio path. Use its three IDs as reference input. Route selected list/detail
segments, search hits and later M4 citations through the same resolver. Opening
the meeting alone supplies no segment or seek position.

Validate meeting/revision/segment membership together. Resolve audio through the
selected revision's ordered `revision_audio_assets` links (`meeting_id`,
`revision_id`, `asset_id`, `ordinal`, `source_file`) and matching asset identity,
hash and managed path. A meeting-wide asset list, newest filename or trial match
cannot select the consumed audio; A→B→A reuse retains its explicit links. Missing
or ambiguous evidence provides no guessed playable path. Shared role/path guards
validate containment and redirects before media access; the UI consumes a DTO.

Consume the merged B5 R42 guard before exposing target text/audio paths, including
historical targets. Old receipts, a lost SQL catalog or stale selections cannot
bypass deletion suppression. Invalid/unavailable suppression fails exposure closed
with the accepted typed recoverable outcome, without inventing purge authority.
The resolver consumes the merged guard. Its Qt playback binding remains B work.

Merged B5 provides `memory.service.catalog_read_view(library)` as the R50 read
boundary. The caller opens and closes a fresh connection without an active
transaction. The guard acquires the lock
before BEGIN and owns the single read transaction through generation/fetch.
Consume its immutable `SuppressionSnapshot`: sorted `suppressed_meeting_ids`
and `fingerprint` cover registry UUIDs plus identities resolved from exact
normalized-origin/fingerprint aliases in the same DB view. Apply suppression
before exposure and before keyset/LIMIT; pending and complete both suppress.
Do not parse the registry again, nest locks/transactions or trust a stale digest.
Invalid/inaccessible suppression remains fail-closed with the existing LibraryError.

### Evidence navigation

Opening a target selects the segment, centers it in the detail pane and shows the
previous and next segments as context. With timing present the detail offers
"Reproducir desde mm:ss" built from `start_ms`; with `timing_source=absent` it shows
"Sin marca de tiempo" and no play-from-zero control. A `historical` target shows the
revision badge and a link to the current revision; a `missing` target shows what is
missing (audio file or revision) without a broken control.

Measured zero remains zero; nullable bounds remain absent. A timing label alone
cannot supply missing `start_ms`. Never infer seek from text, ordinal, chunk order,
wall-clock pauses or processing timestamps, and apply recording offsets only once.
Unknown capture dates remain null in targets/JSON. Preserve the accepted detail
request lifecycle so stale callbacks/pending seeks cannot select old media.

### Playback adapter

`src/ui/playback.py` wraps `QMediaPlayer` plus `QAudioOutput` behind a small
`PlaybackController` the UI drives with DTOs. It accepts only the validated local
path from the target. It waits for `mediaStatus` loaded and `isSeekable()` before
`setPosition(start_ms)`; a seek requested earlier is queued once and applied when
seekable. Controls: play/pause, position slider, current and total time, volume,
speed from a fixed moderate set. Changing meeting stops and unloads the previous
media. Codec and missing-file errors are shown as text in the detail without
affecting list or search. While the application snapshot reports recording, paused
or preparing, playback controls are disabled with the reason visible, to avoid
loopback recapture. The controller exposes no autoplay entry point; group A's
contract and M4 may only select a target, never start playback.

### Deletion and composition bindings (R51/R52)

Merged B5 provides the deletion phases, native ownership checks and SQL hook
below. The M3 A candidate carries the R52 composition and the ACK implementation
from `d8fb75b37691cc9c4e57fa84bb545190b3b9304b`. This is the baseline for B's
player release binding.

R51's `DeletionResult(status, meeting_id=None, reason='', warnings=(),
error_category=None)` distinguishes completed/pending/rejected. Before acceptance,
existing typed InvalidInput/NotFound/Conflict/Busy outcomes retain their mappings.
After a valid durable barrier, including one from a prior attempt, an ordinary
failure yields pending with the canonical meeting UUID, fixed reason code and
category io/exit 4; absent SQL rows do not mean rejection or NotFound. Only verified
complete purge plus durable complete permits completed/exit 0. Consumers use fixed
reason/category fields, never exception or transcript text. Invalid registry state
fails closed and does not prove that an earlier attempt was rejected.

Desktop composition registers `_DeleteAcknowledgement` once on the real
`MemoryService.before_delete`. It sends a queued Qt ticket from a tracked
`BackgroundJobs` worker, calls `LibraryView._pause_for_delete(request)` on the Qt
thread and waits up to 1 s in the worker. The view invalidates page, participant
and detail requests, suppresses replacement reads and refreshes only after the
worker settles. A currently drops the affected meeting ID. B forwards the canonical
`meeting_id` supplied by `before_delete` through the same ticket, alongside the
tracked worker request, to `LibraryView._pause_for_delete(request, meeting_id)`
on Qt. Stop and unload current media only if it belongs to that meeting.
Invalidate pending loads and seeks belonging to that same ID before successful ACK.
Playback of another meeting continues. The existing broad list and detail
invalidation can remain. Keep teardown and ticket cancellation on view close.
Never wait on the Qt thread or touch player state from the worker.
The callback runs without a transaction or catalog lock needed by the GUI.
After ACK or failure, check the current durable phase; then reacquire/revalidate
identity, ownership, phase and inventory before a new barrier. Under R52 as clarified
by `f09c0fc`, acquire and retain the applicable native leases and canonical lease
after this lock-free ACK/revalidation, from before the durable barrier through
cleanup completion. Pre-ACK preflight is not acceptance; an owner appearing during
ACK must be detected. Keep the existing bounded timeout and use B5's pending
outcome if an already-accepted deletion cannot complete.

R51's `_derived_row_purgers` is a private trusted-code SQL registry, not a citation
resolver or UI hook. It runs in order at the start of the same final cleanup
transaction, before core/FTS relationships are removed, on that connection only.
No nested transaction, commit, other connection, I/O or provider work is permitted;
failure rolls core/derived rows back together and leaves pending replay.

R52 supplies ownership roots only from existing configuration/composition and
explicit StartRequest.output_dir, registered before the worker on the same injected
MemoryService without replacing data_dir. CLI destructive work uses the existing
settings_operation and get_settings().TRANSCRIPTIONS_DIR; core never loads settings
or .env itself. Composition, CLI and retry reuse that service/root, not a second
default MemoryService. No durable root registry, origin/receipt/manifest/package
probing or external-root discovery is authorized. Destructive work without explicit
context fails closed with existing recoverable LibraryError/io4, not a Busy claiming
a proven owner; an existing barrier remains pending/hidden. Reads and non-destructive
repair remain available. Playback and CLI reuse the merged B5 cleanup and lease
lifetime. The candidate composes desktop recovery from the
controller's `memory_service` and explicit output directory.

### CLI

`src/cli.py` gains a `meetings` Typer group:

```text
here meetings list [--from ISO] [--to ISO] [--status ...] [--participant ID] [--json]
here meetings show ID [--json]
here meetings search "consulta" [--meeting ID] [--json]
here meetings rename ID "Título"
here meetings participant ID "Nombre" [--speaker KEY]
here meetings retry ID
here meetings play ID [--segment SEGMENT_ID]
here meetings import-session PATH
here meetings reindex
here meetings delete ID --yes
```

The merged `here memory` group already exposes ingest, reconcile, show, rename,
participant, delete and reindex. Keep that group compatible and reuse its
service bindings for `meetings`: `show`, `update_meeting(MeetingPatch,
expected_version)` on the resolved ID, atomic `add_meeting_participant(...,
speaker_key=...)`, `ingest_session`, `delete_meeting` and `reindex`.
The optional speaker key belongs to the active revision. Use
`SearchService.list_meetings` and `search` for query operations. The candidate
`SearchQuery` has no meeting-ID filter. Task 4 adds an optional `meeting_id`
scope to the canonical query DTOs under root's shared-file lease, after A is
frozen. Bind the resolved UUID in SQL before keyset and LIMIT, and include it in
the cursor scope. Apply it to all query routes, including empty search delegated
to listing. The scoped query
still uses corpus-wide BM25 and R45 generation. Apply meeting-ID resolution
under R42 before exposing candidates or operating on a canonical ID.

`ApplicationController.retry(session_dir)` and `wait_until_terminal()` provide
the application retry path. B resolves the selected meeting to one eligible
recovery candidate in the configured session root, using B5/R62 suppression,
lineage and ownership checks on the same MemoryService. Apply the recovery
candidate's `can_retry` and materialization guards before calling the controller.
Use a targeted lookup that reuses the accepted recovery checks. Full
`RecoveryService.discover()` can reconcile or replay unrelated sessions, so it
cannot serve as a read-only ID resolver. A path exposed by `show` or a revision
origin is insufficient authority to retry. Inspect the terminal snapshot for the
result. Its current `ApplicationError(stage, error_type, message, retryable, ...)`
and the CLI expose no canonical `ProviderUnavailable` type. B needs a structured
provider outcome from the application boundary before mapping retry exit 5.
Preserve catalog, ownership and I/O categories and keep other commands provider-free.
Exit codes: 0 success, 2 invalid input, 3 not found or
conflict, 4 integrity or I/O failure, 5 provider (only `retry` can return it).
`Busy`/storage uses exit 4. Honor `IngestResult.error_category` and the
phase-aware `DeletionResult` fields. Existing `memory` lifecycle errors carry
`error` and `error_category`; expose `InvalidInput.reason` for structured
`meetings` errors such as `corpus_changed`.
`--json` emits full IDs and ISO UTC dates; abbreviations are accepted only when the
resolution is unambiguous, otherwise exit 2 with the candidates. `delete` without
`--yes` exits 2 and prints what would be removed. `play` opens a minimal Qt window
with the same `PlaybackController`; when Qt multimedia is unavailable it exits with
an explanatory error and, with `--json`, prints the resolved target without claiming
playback.

### Search/list continuation inherited from M3 A

R47 empty normalized search delegates to listing with the same filters, limit and
cursor and returns `MeetingPage`; nonempty search returns `SearchPage`.
CLI consumes the actual returned page type without a fabricated `SearchHit`,
segment or empty MATCH. Empty search retains listing's date/ID continuation.
R46's canonical `memory.service.InvalidInput.reason` carries
`corpus_changed`; consumers preserve that structured reason rather than parsing
an error message or defining another error class.

Pass query/page cursors through the accepted SearchService without decoding them
in the CLI/player. R39 list order is `started_at DESC NULLS LAST, meeting_id ASC`;
ranked search adds `rank ASC` before these keys and `segment_id ASC` after them.
Preserve explicit NULL keys, query/filter/backend binding and filters before LIMIT.
R41 keeps persistent FTS active-complete only; opted-in active partials are read-only
literal/rank-0 hits with a badge. Missing FTS uses the accepted literal/degraded
route. This group adds no query/index maintenance implementation.

R45 binds ranked continuation to a reproducible corpus generation from the page's
same read snapshot and coherent R42 suppression. Nonmatching, filtered or suppressed
rows still in FTS can affect BM25. A changed generation rejects the cursor before
using saved ranks as existing `InvalidInput` with structured `corpus_changed`.
Preserve the accepted reason/JSON shape and existing CLI exit 2; never join stale
ranked pages or add a new exit code. A restart is a cursor-free request; Qt uses
its existing replacement/request path and rejects superseded callbacks. Ordinary
listing retains its original insert-between-pages date/ID contract and is not
invalidated by this search-only rule. No CLI flag or public flow is added here.

### Real-audio verification

Seek accuracy is certified on a real WAV with audible markers at known offsets:
request a seek to each marker and measure the audible position within 250 ms. The
run, machine and WAV hash are recorded in `ACCEPTANCE.md`. A fake player proves the
contract, not the criterion.

## Acceptance

- From a list item, a search hit and a synthetic citation, the same resolver returns
  the same `CitationTarget` and the same detail opens with the segment centered and
  its neighbors visible.
- Segment with timing: "Reproducir desde mm:ss" appears and the controller requests
  `setPosition(start_ms)` only after loaded and seekable; segment without timing:
  "Sin marca de tiempo" and no playback-from-zero control.
- Historical target shows the badge and the link; missing target names what is
  missing and renders no broken control.
- Changing meeting stops and unloads the previous media; a missing file and a bad
  codec each show a text error while search and list keep working.
- With the snapshot in recording, paused or preparing, controls are disabled with the
  reason; after completion they are enabled again.
- No code path starts playback without a user click; a selection event from a
  citation does not play.
- Real WAV with markers: every seek lands within 250 ms, recorded in `ACCEPTANCE.md`.
- Each `here meetings` command returns the documented exit code for success, invalid
  input, not found, conflict, integrity failure and, for `retry`, provider failure;
  `--json` carries full IDs and ISO UTC dates; ambiguous abbreviations exit 2 with
  candidates; `delete` without `--yes` changes nothing.
- `play` without multimedia exits with an explanatory error and prints the target in
  JSON mode.
- Keyboard operation of the player (play/pause, seek step, volume) and accessible
  names on every control; contrast in both themes.
- Full Windows CI suite passes offscreen with a synthetic key and temporary roots;
  the real-audio run is a separate recorded observation.

## Boundaries

No questions, answers or extractions (M4); no export, import, settings or tray (M5);
no transcript text editing; no autoplay; no audio formats beyond the WAV M1 writes;
no provider call except the existing explicit application-owned `retry` path,
which uses the accepted catalog publication service.

## Technical references

- [QMediaPlayer for PySide6](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QMediaPlayer.html):
  media status, seekability and `setPosition`.
- [QAudioOutput](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QAudioOutput.html):
  volume and device output.
- [Typer](https://typer.tiangolo.com/): command groups and exit codes, as already used
  in `src/cli.py`.
