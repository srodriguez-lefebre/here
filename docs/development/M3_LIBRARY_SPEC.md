# M3 playback, evidence navigation and CLI specification

Status: preparation only, not in execution. The M2–M5 roadmap is authorized by
ruling 20; M3 B may use the same-milestone overlap permitted by CLAUDE §11.

Preparation foundation: M2 A merge
`53712c5beedafc2d40e2abf10c13362a1c3c9f78`; lifecycle/search contract PR #22 merge
`73d3cb384bd422de6899d8c7b761f3f3159ea3f1`. The latter accepts planning contracts,
not M3 A or M2 B implementation. Actual frozen/reviewed M3 A code/DTO handoff
and verified PR head remain **pending**. Under CLAUDE §11, root may dispatch M3 B
on that branch while M3 A's PR is open, once the actual interfaces are accepted
and any necessary contract-documentation PR is merged. M3 A's eventual merge SHA
remains **pending** for rebase and final integration/provenance; it is not a
prerequisite for preparation, the formal contract PR or this permitted root GO.
R35 leaves root sole owner of leases, integration and merges. Rebase B after A
merges; this preparation starts no implementation or new chat/front.
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

Merged M2 A supplies immutable revisions, schema 2's ordered
`revision_audio_assets`, `MemoryService.show`, receipt 3 and marker 3. Keep
receipt-2/marker-1/2 compatibility, exact committed-publication proof and retained
uncertain evidence. Reading a target never authorizes ingestion, replay or retry.
Unknown capture dates and absent segment times remain unknown, including
WAV-only legacy lineage with no historical revision. Metadata-only nonretryable
catalog warnings grant no media/provider/mutation authority; existing ownership
and path guards remain fail-closed.

M3 A's accepted plan supplies query/page and citation DTO contracts, R39 keysets,
R41 partial/fallback routing and R45 search generations. Actual implementations,
error shapes and UI hooks remain an accepted handoff dependency. M2 B4 owns real
editing/participant adapters; B5 owns deletion suppression R42 and reindex.
Consume those accepted adapters without another persistence path, index writer
or registry parser. Final integration cannot be claimed from fakes.

Before the formal B contract PR/root dispatch, pin the frozen/reviewed M3 A PR head,
query/page/Citation/CitationTarget definitions, structured errors, library/request
hooks and B4/B5 action/guard adapters. The interfaces below are planned B contracts,
not functions verified present in the merged M2 A source.

R53 binds the formerly pending reindex facade, implemented/reviewed at frozen
`d8ec92b`: `SearchService(connection_factory, *, reindexer=None).reindex()`
delegates exactly once to the callback on the same injected MemoryService, returns
the identical R51 ReindexReport and propagates canonical errors without retry,
wrapping or recounting. Construction only stores the callback; delegation acquires
no factory connection, snapshot, lock, probe or migration. Without it, only reindex
raises existing LibraryError; reads remain available. ApplicationUiAdapter keeps
an explicit query override and binds maintenance through a separate optional typed
capability on the real memory/root, not a second service or required read protocol.
This closes the facade binding only; real B5/R52/UI handoffs, performance and group
acceptance remain their existing gates, not evidence supplied by this draft.

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

`CitationTarget` (planned in M3 A's `src/memory/models.py`, actual handoff pending)
carries meeting title and date, source text, visible speaker, optional
`start_ms`/`end_ms`, an optional
validated local audio path and a state `current`, `historical` or `missing`.
The planned `MemoryService.resolve_citation(Citation) -> CitationTarget` resolves
it from the database only: titles, dates, paths and times never come from free text. List,
search and M4 citations all produce a `Citation` and open the same detail through
the same resolver. Reconcile exact DTO/error spelling with the actual accepted
handoff instead of creating competing target types.

Validate meeting/revision/segment membership together. Resolve audio through the
selected revision's ordered `revision_audio_assets` links (`meeting_id`,
`revision_id`, `asset_id`, `ordinal`, `source_file`) and matching asset identity,
hash and managed path. A meeting-wide asset list, newest filename or trial match
cannot select the consumed audio; A→B→A reuse retains its explicit links. Missing
or ambiguous evidence provides no guessed playable path. Shared role/path guards
validate containment and redirects before media access; the UI consumes a DTO.

Consume the accepted B5 R42 guard before exposing target text/audio paths, including
historical targets. Old receipts, a lost SQL catalog or stale selections cannot
bypass deletion suppression. Invalid/unavailable suppression fails exposure closed
with the accepted typed recoverable outcome, without inventing purge authority.
Its exact adapter and deletion/playback handoff are pending B5/root acceptance.

R50 approves `memory.service.catalog_read_view(library)` as the shared read
boundary; B5 implementation and integration remain pending. The caller opens a
fresh connection without an active transaction and closes it; B acquires the lock
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

These are approved contracts, not verified B5/native implementations or final UI
signatures. Reconcile the actual accepted handoff in the normal documentation PR
before M3 B source work; neither this binding nor static review grants GO.

R51's `DeletionResult(status, meeting_id=None, reason='', warnings=(),
error_category=None)` distinguishes completed/pending/rejected. Before acceptance,
existing typed InvalidInput/NotFound/Conflict/Busy outcomes retain their mappings.
After a valid durable barrier, including one from a prior attempt, an ordinary
failure yields pending with the canonical meeting UUID, fixed reason code and
category io/exit 4; absent SQL rows do not mean rejection or NotFound. Only verified
complete purge plus durable complete permits completed/exit 0. Consumers use fixed
reason/category fields, never exception or transcript text. Invalid registry state
fails closed and does not prove that an earlier attempt was rejected.

The future `before_delete` adapter must complete stop/unload/invalidation on the
Qt thread and return only after a bounded worker-side ACK; merely queueing a signal
is insufficient. Do not block the Qt thread or touch the player from the worker.
The callback runs without a transaction or catalog lock needed by the GUI.
After ACK or failure, check the current durable phase; then reacquire/revalidate
identity, ownership, phase and inventory before a new barrier. Under R52 as clarified
by `f09c0fc`, acquire and retain the applicable native leases and canonical lease
after this lock-free ACK/revalidation, from before the durable barrier through
cleanup completion. Pre-ACK preflight is not acceptance; an owner appearing during
ACK must be detected. The concrete Qt timeout remains an implementation decision;
no new IPC or user flow is introduced.

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
repair remain available. Revalidate declared root/lock roles after ACK and consume
the actual B5/native cleanup lifetime handoff rather than implementing another owner
protocol in playback or CLI.

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

Commands delegate to accepted memory/search services and the application-owned
explicit retry path; the CLI implements no search, persistence or producer of its
own. Merged M2 A has catalog ingest/show/reconcile and application retry, not the
future search/lifecycle/citation APIs. Confirm actual facade/error/result adapters
at handoff; do not invent `MemoryService.retry` or parse warning prose.
Exit codes: 0 success, 2 invalid input, 3 not found or
conflict, 4 integrity or I/O failure, 5 provider (only `retry` can return it).
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
