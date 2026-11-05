# M3 search and library actions implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The author merges after Copilot review, CI and independent subagent review (rulings 16 and 17). Steps use checkbox
> syntax. **Status: not in execution (ruling 15).** Start only on an explicit user
> instruction, after M2 is merged into `main`.

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

## Review focus

- Operator-looking user input (`NEAR`, `*`, `^`, `-x`, `title:x`, unbalanced quotes).
- Cursor stability when a meeting is inserted or deleted between pages.
- Filters applied after `LIMIT` by mistake.
- A late response from a superseded request replacing the current result.
- Conflict handling on title edit with a stale `row_version`.
- FTS5-missing path leaving the library unusable.

### Task 1: SearchService, queries, pagination and fallback

**Files:** `src/memory/search.py` (new), `src/memory/models.py` (M2, add query/page
DTOs and `CitationTarget` type), `src/memory/repository.py` (M2, read helpers),
`src/memory/database.py` (M2, FTS5 capability probe); tests in
`tests/test_memory_search.py`, `tests/test_memory_search_fallback.py`,
`tests/test_memory_search_benchmark.py` (marked slow, skipped by default).

**Interfaces:** `SearchService(connection_factory)` with
`list_meetings(query: MeetingQuery) -> MeetingPage`,
`search(query: SearchQuery) -> SearchPage`, `reindex() -> ReindexReport`,
`capabilities() -> SearchCapabilities`; pure helpers `normalize_query_text(str) -> str`
and `build_fts_expression(str) -> str`; typed errors `InvalidInput` (bad cursor or
limit) reusing M2 error types. `SearchHit` carries `meeting_id`, `revision_id`,
`segment_id`, `snippet`, `start_ms`, `end_ms`, `timing_source`, `partial`.

- [ ] Write failing tests for `build_fts_expression`: accents, case, quotes, two-letter
  token, operator-looking tokens, unbalanced quotes; assert the raw string never
  appears in the SQL text (capture through a tracing connection).
- [ ] Implement normalization and expression building.
- [ ] Write failing tests for listing and search: empty query lists by date then id;
  filters before limit; cursor round trip across two pages with an insert in between;
  malformed cursor → `InvalidInput`; `deleting`, historical and partial exclusion;
  `include_partial` badge flag.
- [ ] Implement queries with `bm25()` ordering, filters in SQL, opaque cursors and
  bounded `snippet()`.
- [ ] Write failing tests for the FTS5-missing path and `reindex()` equivalence;
  implement the capability probe, literal fallback with `degraded` flag and reindex.
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

- [ ] Write failing Qt tests: Biblioteca tab exists next to capture; typing five
  characters within 200 ms issues one query; an older delayed response is discarded;
  closing the detail drops its callback; both empty states and their actions.
- [ ] Implement the tab, search field with debounce, filters, list model and detail
  pane fed only by DTOs from the controller.
- [ ] Write failing tests for title editing: stale `row_version` shows conflict and
  keeps the stored title; current version updates and refreshes the list.
- [ ] Implement title editing through `update_meeting`.
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
  the description, CI green; the author merges per rulings 16 and 17. Group B starts from the merged target
  and DTOs.
