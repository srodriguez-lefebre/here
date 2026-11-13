# M3 search and library actions specification

Status: planned, not in execution. The latest explicit user instruction authorizes
M3 group A alongside M2 group B after M2 group A closes with every review/CI/merge
gate and the necessary contract documentation PRs are merged (ruling 35). One
cohesive docs PR may reconcile both fronts; no fixed PR count is required. Root
coordinates the separate chats/worktrees, file leases and merges. The accepted merged-A base is
`53712c5beedafc2d40e2abf10c13362a1c3c9f78`. Root records this documentation
PR's reviewed head and merge before dispatch.

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
Technical acceptance does not certify hardware/provider/installer or implement B.

Governing design: `PRODUCT_DESIGN.md` §2, §3 and §5, a taken decision (ruling 3).
Spanish plan and acceptance criteria: `../MILESTONE_3_PLAN.md`.

## Intent

Let a person list, filter and search the M2 memory and reach the right meeting and
segment, from a Qt library view that stays responsive on a large library and from a
service that never exposes raw SQL or raw FTS syntax. This group delivers the
`SearchService`, the typed query and page DTOs, the library list and detail widgets,
and title editing. Playback, citation target resolution and the CLI are group B;
the shared citation target type is declared here.

## Decisions

### Concurrent M2 lifecycle dependency

Search/list/filter reads can build on the accepted M2 A catalog without waiting for
all of M2 B. Title editing consumes `MemoryService.update_meeting` and optimistic
`row_version` from M2 B task 4; M3 does not implement another editing service or
present that integration as complete before the B contract is accepted. Participant
filters use the catalog relations; participant mutation remains owned by M2 B.

M2 B and this group share `src/memory/models.py`, `repository.py`, `database.py`;
root grants exclusive leases for each write. Source overlap is serialized even
when independent tasks run concurrently. Root also serializes migrations,
`pyproject.toml`, `uv.lock` and version changes. Preserve migrations 1/2 and their
checksums; no alternate module layout or parallel migrator avoids this ownership.
Deadline-sensitive suites and benchmark runs have exclusive execution windows,
without competing tests/builds/mypy. These are coordination gates, not reductions
of acceptance or permission to skip verification.

Independent files are `src/memory/search.py` and its new search/fallback/benchmark
tests. Shared memory model/repository/database writes require root's exclusive
file lease, as do common UI/application integration files. B task 4 owns title
mutations/indexed-title consistency; B task 5 owns the one reindex operation that
the search facade delegates to. Query work can proceed before those integrations;
their acceptance waits for the real accepted B contracts. Do not add another
persistence path, reindexer or alternate layout for parallelism. Both implementation
plans contain the concrete scope/lease matrix. Global docs also have one writer;
prepared synthetic fixtures transfer after acceptance rather than being regenerated.
B5's planned R42 suppression guard is shared by B/M3 reads, ordinary ingest and
the existing rebuild path. It is a real-integration dependency like B5 reindex;
independent query/view preparation can proceed against A, but final reads must use
the accepted guard. M3 does not implement another registry parser or suppression
policy. Invalid/inaccessible registry gives a typed recoverable catalog error,
fails exposure closed and never authorizes inferred purge.

### Service placement and contracts

`src/memory/search.py` holds `SearchService`, built on the M2 repository and
connection factory; it imports neither Qt nor Typer. Contracts are the ones in
`PRODUCT_DESIGN.md` §3: `MeetingQuery` (optional title text, UTC from inclusive and
to exclusive, statuses, participant, `limit` 1–100, stable cursor), `SearchQuery`
(text plus the same filters, limit, cursor), `MeetingPage` and `SearchPage` with
`items` and `next_cursor`, reusing M2 B's `MeetingPatch` (title and user notes only)
for editing rather than declaring a competing patch type. Dates are
stored ISO UTC and presented in local time by the UI. An empty query lists meetings
through `MeetingQuery`; it never issues an empty `MATCH`.
Consume accepted A identities and nullable source dates without inventing missing
dates or speakers. Ruling 31's ordered `revision_audio_assets` association remains
the source for a revision's evidence audio; a cumulative meeting asset inventory
does not select its source. This group declares the target DTO but does not resolve
or play citations.
Schema 2's same-meeting `revision_audio_assets` composite FKs and receipt-2/3 links
are observed A contracts. WAV-only reservation can have an origin mapping with
revision NULL and zero historical revisions; `source_kind=legacy_audio_only` keeps
unknown capture dates NULL across retry/rebuild. Queries/UI preserve those unknowns
instead of treating processing timestamps as recording dates. Lifecycle mutations
and playback resolution remain their owning groups' work.

### FTS5 querying

The `segments_fts` index exists from M2 A over active complete revisions, tokenizer
`unicode61 remove_diacritics 2`, kept in sync inside the publication transaction and
purged by M2 B's deletion path. Publication activation remains owned by A; deletion
and lifecycle maintenance remain owned by B. Querying does not add a second index
writer. R41's opt-in partial route below is read-only and never changes persistent
FTS. User text is normalized (trim, collapse
whitespace, strip control characters) and turned into a safe FTS expression: each
token is double-quoted with internal quotes doubled, tokens are joined with implicit
AND, and no user character is interpreted as operator, prefix or column filter.
R39 pins listing order to `started_at DESC NULLS LAST, meeting_id ASC`; search
order is `rank ASC, started_at DESC NULLS LAST, meeting_id ASC, segment_id ASC`,
where FTS rank is `bm25()`. The equivalent SQLite keys are
`(started_at IS NULL) ASC, started_at DESC, meeting_id ASC`, preceded by rank and
followed by segment ID for search. No sentinel or inferred date replaces NULL.
Without date bounds, unknown dates remain included; any from/to bound excludes
NULL dates. From is inclusive (`started_at >= :from_utc`), to exclusive
(`started_at < :to_utc`). All query/filter/cursor values are bound parameters, and
all visibility filters including R42 suppression apply before the keyset and
`LIMIT`, never by filtering a returned page.

Cursors are versioned opaque strings containing the last typed keys (rank only
for search, explicit NULL flag and nullable date, meeting ID and segment ID for
search), backend/route/order identity and, in version 2, a fixed-size SHA-256
digest of the canonical normalized query and filters, including `include_partial`.
Valid version 1 cursors remain readable with their embedded scope.
A malformed or mismatched cursor is `InvalidInput`. The token encodes neither
raw SQL nor an invented timestamp.
The NULL flag is exactly `0` with a valid UTC date or `1` with a NULL date; reject
inconsistent types/flags and non-finite rank keys before issuing SQL.
Use lexicographic keyset comparisons with these mixed directions, expanded without
NULL comparison ambiguity. For a non-NULL listing cursor `(d,m)`, the next row has
`started_at IS NULL OR started_at < :d OR (started_at = :d AND meeting_id > :m)`.
For a NULL cursor, use `started_at IS NULL AND meeting_id > :m`.
For search, accept `rank > :r` or equal rank followed by that date comparison,
replacing the final meeting tie with
`meeting_id > :m OR (meeting_id = :m AND segment_id > :s)`. Parenthesize each
cursor branch before combining with the bound filters. Apply the same date/null
ordering and filtering in the literal fallback; its uniform rank key is `0`,
while the FTS branch retains BM25 ordering. Never use OFFSET paging.

R45 binds every `SearchQuery` cursor to a deterministic logical corpus generation
as well as the keys above. BM25 uses corpus-wide statistics, so even a nonmatching
insert can change existing ranks. Generation and page must come from the same
short explicit SQLite read transaction through generation, query and page fetch,
with the coherent validated B5 suppression snapshot. Cover the actual logical FTS
document corpus, including nonmatching, filtered and already-suppressed documents
still present in FTS and affecting its statistics,
plus the active partial branch when selected and all metadata/relations affecting
eligibility. The default is a read-only canonical digest of existing catalog
metadata: immutable active-revision identities (which bind segment content),
indexed titles, status/date/version and filter relations, with R42 suppression.
Metadata is sufficient only where immutable-revision and index-maintenance
invariants prove equivalent coverage of actual rank inputs; index-only changes
must not silently alter those inputs under an unchanged token. Use deterministic
canonical ordering/encoding, not process-local iteration order or randomized hashes.
No new schema, persistent result cache, long-lived read transaction or second
index writer is introduced. Do not substitute query-hit-only hashes, file mtimes
or cross-connection `PRAGMA data_version`; each operation opens its own connection.
Equivalent corpus inputs across fresh connections must produce the same token.

Before applying a saved rank, compare that cursor generation with the current
snapshot. A mismatch raises existing `InvalidInput` with a machine-readable
`corpus_changed` reason; return no mixed-generation page or continuation cursor.
Restart without a cursor to rank the current corpus. This applies consistently
to FTS, hybrid and literal search routes, retaining their existing bound mode and
R39 keys. No-op reindex with identical logical rank inputs preserves equivalence;
changed index semantics/corpus must not reuse an old generation. The performance
and quiet-window requirements remain unchanged. `MeetingQuery.list_meetings`
retains its date/ID keyset and original insert-between-pages acceptance; do not
invalidate ordinary listing cursors under this search-specific rule.

Snippets come from FTS5 `snippet()` with plain-text markers and a bounded length;
the UI renders them as text, never as HTML. Results carry `meeting_id`,
`revision_id`, `segment_id`, snippet, `start_ms`/`end_ms` when present and
`timing_source`.

### Fallback and reindex

At open, the service uses the accepted M2 runtime FTS capability probe and records
the capability; observed A places `fts_available`/`FtsUnavailable` in
`src/memory/service.py`, not `database.py`. Reconcile helper visibility/location
with the accepted A/B contract before implementation, without duplicating a probe
or moving it merely to avoid a file lease. Without FTS5, listing
and filtering work unchanged; search uses a documented literal fallback over the
same authorized title and segment-text fields as complete FTS. Each normalized
query token is bound as a literal substring pattern; escape the escape character,
`%` and `_` for `LIKE ... ESCAPE` before adding the wrapping `%`. AND the tokens,
with each token matching title OR segment text. User wildcards/operators never
change SQL meaning. All literal routes reuse this matching/escaping and the same
filters, suppression, keyset and limit. The page carries
a `degraded` flag the UI shows. Reuse the single reindex operation owned by M2 B
task 5 through the search facade's agreed adapter; finalize that delegation against
the accepted B API rather than implementing a parallel rebuild. `reindex()` rebuilds
the derived index from active
complete revisions in one transaction per meeting batch and reports counts; without
FTS5 it reports the limitation instead of failing. A search result is identical
before and after `reindex()`.

### Opt-in partial search without index mutation

Persistent FTS contains active COMPLETE revisions only, maintained by A activation
and B5 reindex/delete. With `include_partial=False`, query that complete route.
When true, SearchService UNION ALLs complete FTS matches (actual BM25 rank) with a
read-only literal SQL branch over ACTIVE PARTIAL segments (uniform rank 0, never
claimed as BM25). Each branch applies the same authorized title/text matching,
meeting/date/status/participant filters and suppression before union/global
cursor/LIMIT. The branches are disjoint by active revision status; historical,
deleting or suppressed meetings never qualify. No per-branch LIMIT may discard a
row that belongs in the globally R39-ranked page. Without FTS both branches use
literal matching/rank 0; opt-in still controls partials. Backend/route identity and
`include_partial` are cursor-bound so a cursor cannot silently change engines.

Literal partial/fallback snippets are plain windows of actual segment text bounded
to 160 Unicode characters, centered on a text match when present or from its
beginning for a title-only match. They invent no words, HTML or timestamp. Hits
carry the partial badge and a source-derived untimed indication from nullable
timing fields/`timing_source=absent`; no start at zero masquerades as a timestamp.
Reindex equivalence covers both opt-in states: the complete index is rebuilt while
the partial branch remains a read. There is no partial persistent index or second
reindexer. All of this is planned R41 behavior, not an implemented A search API.

### Visibility rules

Default results include only meetings not in `deleting` and only active complete
revisions. Partial revisions appear only when `SearchQuery.include_partial` is set
and are badged. Historical revisions are never searched; they are reachable from a
meeting detail only.

### Library view

`src/ui/library.py` adds a Biblioteca tab to `MainWindow` next to the existing
capture panel. Search field on top; filters for date range, status and participant;
a list with editable title, date, duration, status and an audio-available signal; a
detail pane with participants, the original transcript by segment, recoverable
errors and a placeholder for the playback controls group B adds. Empty states
distinguish "Aún no hay reuniones" (action: start a recording) from "No hay
coincidencias" (action: clear filters).

All queries run through `BackgroundJobs` in `src/ui/bridge.py`, off the Qt thread,
with a 200 ms debounce on text input and a monotonic request id; a result whose id
is older than the latest request is discarded. Closing a detail invalidates its
callbacks. A structured R45 `corpus_changed` result discards the old search cursor
and starts the current query from page one through the same request/loading path,
replacing rather than appending old pages. Superseded callbacks remain invalid;
no new control, persistent snapshot or warning-prose parsing is required.
Title editing calls `MemoryService.update_meeting` with the displayed
`row_version`; a `Conflict` is shown and the row is refreshed, never overwritten.
The real title-edit adapter and its acceptance depend on the accepted M2 B task-4
contract; independent query/worker/detail preparation may proceed beforehand.
Deletion integration uses B5's shared R42 guard before row/detail exposure and
invalidates older request callbacks, so an already queued pre-delete page cannot
restore a logically accepted deletion. Suppression remains authoritative after
SQL DB loss; surviving receipts cannot bypass the guard or reconstruct its rows.
Keyboard navigation, accessible names and contrast in both themes are acceptance,
not polish.

## Acceptance

- Listing with an empty query returns pages ordered by date then id; paging forward
  and back across two pages yields no duplicates or gaps even after an insert between
  calls; a malformed cursor returns `InvalidInput`.
- Accents, case, quotes, a two-letter token, an operator-looking token (`NEAR`,
  `*`, `-x`, `col:`) and a nonexistent term each return the expected result and never
  raise an FTS syntax error; the raw user string never reaches `MATCH`.
- With `limit=10` and 50 matches of which 5 pass the filters, exactly 5 are returned.
- No result comes from a `deleting` meeting or a historical revision; partial
  revisions appear only with `include_partial` and carry the badge flag.
- Synthetic fixture of 1,000 meetings and 100,000 segments: warm search p95 under
  500 ms on the machine documented in `ACCEPTANCE.md`; the Qt event loop never
  blocks longer than 50 ms during a query, measured with a heartbeat timer.
- Without FTS5 (simulated by a connection flag), the library lists and filters, search
  returns `degraded` results through the literal fallback, and `reindex()` reports the
  limitation.
- `reindex()` after deleting and re-adding revisions yields identical search results
  to the pre-reindex state.
- Qt test: typing five characters within 200 ms issues one query; a delayed older
  response does not replace the newer result; closing the detail drops its callback.
- Title edit with a stale `row_version` shows the conflict and keeps the stored
  title; with the current version it updates and the list refreshes.
- Both empty states render in the right case with their distinct actions.
- Tab order covers search, filters, list, detail and title editor; every control has
  an accessible name; contrast passes in light and dark themes.
- Full Windows CI suite passes offscreen with a synthetic key and temporary roots.

## Boundaries

R36's publication token/source kind and receipt 3 exact committed publication proof
are implemented in the accepted A source; scoped re-review addressed NB1/NB2. Fresh
identical retries still create revisions, and receipt-3 rebuild restores proven
complete operations; receipt 2 grants no invented proof. Native/catalog leases and
typed `IngestResult.error_category`/`OwnershipBusy(Busy)` remain A contracts.
R38/GR1–GR3 are technically accepted: marker-3 `hashes` contains actual staged
`session.json` bytes plus declared/locally consumed role-validated artifacts before
final replace, with no separate manifest-hash field. Marker-1/2/receipt-2/3 readers
and schema 2 stay compatible. Unmatched sealed pending evidence remains visible in
existing recovery `error_summary`, without old-output ingest/provider work;
malformed receipts do not abort valid siblings. Private real-controller deferral
preserves a single capture-failure publication after accepted cancellation's
cutover, one identity and original audio/error; direct/injected contracts remain.
R40 also preserves ordinary callback rollback/cause and failed-rollback backups,
BaseException/process-death behavior, and the default processor's shared actual
MemoryService/custom root; explicit processors remain compatible. Rejected
completed replay is a nonretryable warning; successful/no-marker completed
sessions stay hidden. R43/R44 keep malformed-marker or unresolved legacy-owner
warnings metadata-only and nonretryable, without media/credentials/provider work,
mutation or invented identity/parent. Unknown ownership stops marker/error/audio
inspection; native/catalog Busy and unsafe paths remain fail-closed, as does stale
materialization with unknown ownership. Search must preserve these inherited
safeguards, not implement another recovery flow.
Hosted gates and merge still block launch. Search starts from actual merged A and
does not implement these fixes, expose private producer handoffs, choose a retry
API or issue provider/replay work. B result DTOs/mutations/reindex remain planned;
real M3 actions wait for accepted B4/B5 contracts.

No playback, no citation target resolution, no CLI, no participant editing beyond
what M2 already exposes, no provider call, no semantic search, no transcript text
editing, no changes to capture or overlay. The citation target DTO is declared here
only as a type in `src/memory/models.py` so group B and M4 share it.

## Technical references

- [SQLite FTS5](https://www.sqlite.org/fts5.html): `MATCH` syntax, `bm25()`,
  `snippet()`, tokenizer options and the need to quote user tokens.
- [SQLite BM25](https://www.sqlite.org/fts5.html#the_bm25_function): corpus-wide
  rank statistics require generation-bound continuation after mutations.
- [SQLite data_version](https://www.sqlite.org/pragma.html#pragma_data_version):
  its connection-local values are not a cross-connection corpus generation.
- [Qt QThread and worker objects](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html):
  queued signals and the request-sequence pattern already used by `BackgroundJobs`.
