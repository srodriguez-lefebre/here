# M3 search and library actions implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. Root coordinates file leases and merges
> only after current-head received/responded Copilot, Windows CI and independent
> review. Steps use checkbox syntax. **Status: planned, not in execution.** The
> latest explicit user instruction permits M3 A alongside M2 B only after M2 A closes
> with all gates and the necessary contract documentation PRs are merged. One
> cohesive docs PR may cover both fronts. Root creates
> the separate project chats/worktrees afterward. Final merged-A base SHA: `53712c5beedafc2d40e2abf10c13362a1c3c9f78`.

**Goal:** A `SearchService` with typed queries, safe FTS5 access, stable pagination
and a fallback, plus a Qt library tab that lists, filters, searches and edits titles
without blocking the UI.
**Architecture:** `src/memory/search.py` reads the M2 catalog and index through the
repository; `src/ui/library.py` renders pages delivered by `BackgroundJobs` with
request ids and debounce. No SQL or FTS syntax leaves the memory package.
**Tech Stack:** Python 3.13, `sqlite3` with FTS5, Pydantic DTOs, PySide6, pytest,
pytest-qt; no new runtime dependency.
**Spec:** `docs/development/M3_SEARCH_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_3_PLAN.md`.

## Global constraints

- Accepted foundation: PR #21 merged as `53712c5beedafc2d40e2abf10c13362a1c3c9f78`
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
- `src/memory/` imports neither Qt nor Typer; `src/ui/` never runs SQL or reads
  session files.
- User text never reaches `MATCH` unquoted; every token is quoted and escaped.
- Deleting meetings, historical revisions and, by default, partial revisions never
  appear in results.
- No provider call anywhere; everything works with `OPENAI_API_KEY` unset.
- Tests use synthetic fixtures and temporary roots; the 1,000/100,000 fixture is
  generated, never recorded.
- Touched files leave ruff clean and formatted; no new mypy errors in them (baseline
  is not clean, see `CLAUDE.md` §6).
- Root alone assigns exclusive file leases and performs merges. Shared
  `src/memory/models.py`, `repository.py`, `database.py` writes are serialized with
  M2 B; no alternate layout or duplicate helper is introduced to bypass a lease.
- Migrations, global docs, `pyproject.toml`, `uv.lock` and version changes have one writer at a
  time. Preserve migration 1/2 checksums; M3 A does not perform a milestone version bump.
- Reserve isolated execution windows for deadline-sensitive suites and benchmarks;
  no competing tests, builds or mypy during those runs. Independent work must honor
  the root resource lease as well as file ownership.

## Dependencies and handoff before execution

After A is accepted/merged and necessary contract documentation PRs are merged, task 1's
query/list/filter/fallback work may proceed with M2 B in parallel. Root records the
actual merged-A SHA, per-front branch/worktree and file leases in the dispatch brief.
Do not create those chats or start implementation as part of this preparation.

- Bind query DTO/read helpers to the accepted A schema, including nullable dates,
  `row_version`, active revision visibility, `segments_fts` and R31's
  `revision_audio_assets` identity/ordering. Audio association is retained for the
  target DTO; actual citation resolution stays in M3 B.
  Audio-only reservation may have `ingest_sources.revision_id=NULL`, no historical
  revisions and NULL capture dates preserved by durable source kind; listing,
  sorting/filtering and UI must not invent recording dates from processing times.
  R39 fixes list `started_at DESC NULLS LAST, meeting_id ASC`; search is rank ASC,
  then the same date/id keys and `segment_id ASC`. Unknown dates are included
  without date bounds, excluded if either bound is present; from inclusive/to
  exclusive. Bind every filter/key and apply them in SQL before LIMIT. Implement
  the SPEC's exact mixed-direction keyset branches, versioned query/filter-bound
  cursor with explicit NULL flag/value and no sentinel. Literal fallback uses the
  same date/null rules with uniform rank 0; FTS retains BM25. UI keeps unknown dates.
- Reconcile the existing A FTS capability probe/warning with the B maintenance API;
  observed A defines it in `service.py`. `database.py` is not automatically a probe
  implementation task. A lease is required if the accepted helper needs changing.
- Keep one FTS rebuild owner: B task 5 supplies reindex. Finalize the search facade's
  delegation/result contract when that B contract is accepted; do not add a second
  index implementation while waiting. Reindex integration/equivalence acceptance
  waits for that dependency, without blocking independent query work.
- Title-edit UI and optimistic conflict tests consume accepted B task 4
  `update_meeting`; never implement the B mutation inside search or count a fake as
  the real title-edit integration. Other worker/list/detail work is independent.
- R41 keeps active-complete-only persistent FTS under A/B5 ownership; SearchService
  supplies active-partial opt-in through read-only literal SQL UNION ALL, rank 0,
  with unchanged complete BM25 and a single global R39 page. Missing FTS makes both
  routes literal/rank 0. Use matching title/text fields and escaped LIKE tokens;
  no partial-index writer or second reindexer.
- B5's planned R42 guard suppresses logically accepted deletions before every
  B/M3 read, ordinary reimport and existing rebuild. Final query/detail integration
  waits for its accepted private adapter; independent query work remains possible.
  Do not parse the registry separately or postfilter a page. Invalid/unreadable
  registry is a typed recoverable fail-closed exposure error, never purge authority.
- Preserve R36's fresh-token/new-revision versus exact committed-proof replay,
  receipt-3 operation restoration and receipt-2 compatibility. Marker 3 carries
  token/source kind and `hashes`, including the writer's actual staged `session.json`
  and declared/locally consumed role artifacts before final replace; marker 1/2
  stays compatible. Accepted R38 retains unmatched sealed pending intent with
  existing recovery `error_summary`, no writer claim/old-output ingest/provider.
  Receipt validation isolates invalid siblings; private real-controller deferral
  preserves one cancelled capture publication/identity/audio/error at the existing
  cutover. These are accepted A behavior, not search fixes/public producer APIs.
  R40 retains ordinary callback rollback/cause and failed-rollback backups,
  unchanged BaseException/process-death behavior and a shared default-processor
  MemoryService/custom root; explicit processors stay compatible. Rejected
  completed replay warns without retry; successful/no-marker completed stays hidden.
  R43/R44 preserve valid metadata as bounded nonretryable warnings for malformed
  markers or unresolved legacy catalog owners, without media/credentials/provider
  work, mutation or invented identity/parent. Unknown ownership stops marker/error/
  audio inspection; stale materialization remains fail-closed. These are inherited
  safeguards, not new search or recovery scope.
  Keep native/catalog Busy/path guards fail-closed, the existing leases and
  typed `IngestResult.error_category` (`invalid_input|conflict|io`) and
  `OwnershipBusy(Busy)`; query adapters never parse warning prose.

## Front scopes and leases

| Scope | Owner/dependency after root dispatch | Exclusive write rule |
|---|---|---|
| `src/memory/search.py`; `tests/test_memory_search.py`, `test_memory_search_fallback.py`, `test_memory_search_benchmark.py` | M3 A independent query/list/filter/fallback | May overlap M2 B using accepted A contracts; shared DTO/read-helper writes still wait for leases. |
| `src/memory/models.py`, `repository.py`, `database.py` | Both fronts | Root grants per-file exclusive leases; one migrator, preserved migration-1/2 checksums. |
| `src/memory/service.py`, `artifacts.py`, application processing/recovery and B tests | M2 B persistence/lifecycle | M3 consumes accepted APIs; no copied persistence/probe or alternate module layout. |
| `MeetingPatch`, `update_meeting`, optimistic version/title-index consistency | B task 4 supplies; M3 title editor consumes | Real editor/conflict integration waits for accepted B4; fakes are preview/worker evidence only. |
| `MemoryService.reindex() -> ReindexReport` | B task 5 supplies; search facade delegates | One reindexer; integrate/equivalence-test the accepted B5 operation. |
| R42 suppression guard | B5 supplies; B/M3 read adapters consume | Accepted B5 adapter before real integration; no duplicate parser. Apply before cursor/LIMIT and invalidate pre-delete UI callbacks. |
| `src/ui/library.py`, `tests/test_ui_library.py` | M3 A independent view preparation | Lease common adapters before wiring; B acceptance dependencies still apply. |
| `src/ui/main_window.py`, `app.py`, `bridge.py`, `contract.py`, fake/preview, common UI tests, `src/application/controller.py`, `src/cli.py` | Shared integration or adapters | Root grants exact file leases before either front writes; B confirmation and M3 library wiring are serialized where shared. |
| Global/milestone docs, schema, dependency metadata and versions | Root coordination | Serialized across fronts; M3 A has no milestone bump. |

Reuse accepted prepared synthetic fixtures through one root-owned transfer; do not
regenerate or duplicate fixtures in each front. No new index/persistence/layout is
introduced for concurrency. Deadline suites, benchmarks/heartbeat and heavy builds
need root's quiet resource window without competing tests/builds/mypy.

## Review focus

- Operator-looking user input (`NEAR`, `*`, `^`, `-x`, `title:x`, unbalanced quotes).
- Cursor stability when a meeting is inserted or deleted between pages.
- Filters applied after `LIMIT` by mistake.
- A late response from a superseded request replacing the current result.
- Conflict handling on title edit with a stale `row_version`.
- FTS5-missing path leaving the library unusable.
- NULL dates/active revision on reserved audio-only lineage being replaced by
  processing dates, invented history or an assumed transcript during query/detail.
- A/B FTS capability or reindex logic being duplicated; titles/index becoming stale
  after a B edit; deletion appearing on later pages after a concurrent B operation.
- Shared-file or schema writes overlapping between fronts, and benchmark/deadline
  results measured while other verification is running.
- Opt-in partial hits absent from complete-only FTS, matched title tokens missing
  from literal routes, wildcard input interpreted as syntax, or partials changing
  after reindex. Require the real R41 hybrid/global-page contract.
- R42 barrier accepted while SQL still shows the meeting, DB loss restoring old
  receipts, or an old UI callback re-showing deletion; fail closed on bad registry.

### Task 1: SearchService, queries, pagination and fallback

**Files:** `src/memory/search.py` (new), `src/memory/models.py` (M2, add query/page
DTOs and `CitationTarget` type), `src/memory/repository.py` (M2, read helpers),
`src/memory/database.py` (shared; change only if the accepted contract needs it,
under root lease); tests in
`tests/test_memory_search.py`, `tests/test_memory_search_fallback.py`,
`tests/test_memory_search_benchmark.py` (marked slow, skipped by default).

**Interfaces:** `SearchService(connection_factory)` with
`list_meetings(query: MeetingQuery) -> MeetingPage`,
`search(query: SearchQuery) -> SearchPage`, `reindex() -> ReindexReport`,
`capabilities() -> SearchCapabilities`; pure helpers `normalize_query_text(str) -> str`
and `build_fts_expression(str) -> str`; typed errors `InvalidInput` (bad cursor or
limit) reusing M2 error types. `SearchHit` carries `meeting_id`, `revision_id`,
`segment_id`, `snippet`, `start_ms`, `end_ms`, `timing_source`, `partial`.
The planned search `reindex` facade delegates to the accepted B maintenance
operation; its exact adapter/result is a dependency to reconcile, not a new writer.
Use the existing accepted FTS probe before adding any capability implementation.

- [ ] Write failing tests for `build_fts_expression`: accents, case, quotes, two-letter
  token, operator-looking tokens, unbalanced quotes; assert the raw string never
  appears in the SQL text (capture through a tracing connection).
  Cover literal token matching across the same title/segment fields, with `%`, `_`
  and escape characters treated literally in both partial and missing-FTS routes.
- [ ] Implement normalization and expression building.
- [ ] Write failing tests for listing and search: empty query lists by date then id;
  filters before limit; cursor round trip across two pages with an insert in between;
  malformed cursor → `InvalidInput`; `deleting`, historical and partial exclusion;
  `include_partial` badge flag. Include unknown-date audio-only lineage and a
  reservation without a revision; retain unknown fields. Assert list/search exact
  R39 ordering across equal dates, NULL and non-NULL cursor boundaries and equal
  ranks; from inclusion/to exclusion and exclusion of NULL under either date
  bound; identical date rules in FTS/fallback; mismatched cursor query/filter/order
  and malformed NULL flags return `InvalidInput`. Cover inserts/deletes between
  pages without OFFSET, sentinels, duplicate rows or invented dates.
  Mixed complete/partial matches must cross the global LIMIT/rank/date-NULL keyset
  correctly; complete ranks retain BM25, partials are literal rank 0 and badged.
  Assert no historical/deleting/suppressed hits, no per-branch truncation and
  include_partial/backend cursor mismatch rejection. Literal snippets are actual
  text, at most 160 characters, with only measured/absent timing information.
- [ ] Implement queries with `bm25()` ordering, filters in SQL, opaque cursors and
  bounded `snippet()`.
  Add R41's read-only active-partial literal branch only when opted in; UNION ALL
  before global R39 cursor/LIMIT, with common filters/suppression applied first.
  Consume B5's accepted R42 guard through its private adapter before real read
  integration; one coherent suppression snapshot includes pending barriers whose
  SQL hide transaction failed. Never duplicate persistent FTS or registry parsing.
- [ ] Write failing tests for the FTS5-missing path and `reindex()` equivalence;
  reuse the accepted capability probe and implement literal fallback with `degraded`
  flag. Integrate/delegate reindex only after B task 5's contract is accepted, then
  run equivalence tests against its actual index maintenance.
  Compare complete-only and include_partial pages before/after B5 reindex; the
  persistent index remains complete-only and the partial read branch is unchanged.
  Simulate missing FTS for both literal branches/rank 0. Inject invalid/inaccessible
  R42 registry and assert typed recoverable fail-closed reads; rebuild from a
  durable pending/complete barrier must never expose a surviving receipt/FTS row.
- [ ] Generate the 1,000/100,000 synthetic fixture in a slow test; record p95 on the
  documented machine in the report; keep the test opt-in in CI.
- [ ] Run focused tests and the full offscreen suite; self-review; commit as
  "memory: search service, pagination and fallback".

### Task 2: Library tab, workers, filters, editing and empty states

**Files:** `src/ui/library.py` (new), `src/ui/main_window.py`, `src/ui/app.py`,
`src/ui/bridge.py`, `src/ui/contract.py` (extend `VisualController` with library
operations), `src/ui/fake_controller.py`, `src/ui/preview.py`; tests in
`tests/test_ui_library.py`, `tests/test_ui_lifecycle.py`.

**Interfaces:** `LibraryView(QWidget)` with signals `meeting_selected(meeting_id)`,
`segment_selected(CitationTarget)`; `VisualController` gains `list_meetings`,
`search`, `get_meeting`, `update_meeting`; `BackgroundJobs.submit` is reused with a
`request_id` and a `kind` per query type; the fake controller serves synthetic pages
so the preview runs without a database.
The real `update_meeting` adapter/title editor depends on accepted M2 B task 4;
root reserves any common UI/contract file against B's concurrent GUI integration.

- [ ] Write failing Qt tests: Biblioteca tab exists next to capture; typing five
  characters within 200 ms issues one query; an older delayed response is discarded;
  closing the detail drops its callback; both empty states and their actions.
  Queue an older result before a logically accepted B deletion, deliver it after
  the guard changes, and prove request invalidation prevents row/detail resurrection.
- [ ] Implement the tab, search field with debounce, filters, list model and detail
  pane fed only by DTOs from the controller.
- [ ] Write failing tests for title editing: stale `row_version` shows conflict and
  keeps the stored title; current version updates and refreshes the list.
- [ ] Implement title editing through accepted B task-4 `update_meeting`, preserving
  its error categories, version precondition and FTS maintenance contract.
- [ ] Write failing accessibility tests: tab order across search, filters, list,
  detail and editor; accessible names present; contrast check in both themes using
  the existing palette tokens.
- [ ] Extend `fake_controller.py` and `preview.py` so `here-gui-preview` shows a
  populated library with synthetic meetings.
- [ ] Run focused Qt tests and the full offscreen suite; self-review; independent
  agent review; commit as "ui: library tab with search, filters and editing".

## Closing the group

- Update `README.md` (library tab), `ARCHITECTURE.md` (search service and request
  flow), `docs/MILESTONE_3_PLAN.md` (criteria 1–10, 17–18 status), `ACCEPTANCE.md`
  (M3 responsiveness row with the benchmark run) and `WHATS_NEW.md`.
- Open one PR with the two task commits, self-review and independent agent review in
  the description. Root integrates any required accepted B changes, verifies the
  exact head and merges only after received/responded Copilot and green Windows CI
  plus independent review. No unmet title-edit/reindex dependency is marked complete.
  M3 B starts from the merged target and DTOs; milestone criteria remain unchanged.
- Root aligns the M3 milestone agenda with ruling 35 in the contract docs PR
  without changing acceptance criteria. After M2 B merges, M5 B may advance with
  its claims hook waiting for M4 B; other group dependencies remain in STATUS.
  This preparation creates no chat/worktree and authorizes no early implementation.
