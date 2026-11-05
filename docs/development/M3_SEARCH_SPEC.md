# M3 search and library actions specification

Status: written 2026-11-05, not in execution (ruling 15 in `DELIVERY.md`).
Implementation starts only on an explicit user instruction, after M2 is merged.
Governing design: `PRODUCT_DESIGN.md` §2, §3 and §5, a taken decision (ruling 3).
Spanish plan and acceptance criteria: `../MILESTONE_3_PLAN.md`.

## Intent

Let a person list, filter and search the M2 memory and reach the right meeting and
segment, from a Qt library view that stays responsive on a large library and from a
service that never exposes raw SQL or raw FTS syntax. This group delivers the
`SearchService`, the typed query and page DTOs, the library list and detail widgets,
and title editing. Playback, the citation target and the CLI are group B.

## Decisions

### Service placement and contracts

`src/memory/search.py` holds `SearchService`, built on the M2 repository and
connection factory; it imports neither Qt nor Typer. Contracts are the ones in
`PRODUCT_DESIGN.md` §3: `MeetingQuery` (optional title text, UTC from inclusive and
to exclusive, statuses, participant, `limit` 1–100, stable cursor), `SearchQuery`
(text plus the same filters, limit, cursor), `MeetingPage` and `SearchPage` with
`items` and `next_cursor`, `MeetingPatch` (title and user notes only). Dates are
stored ISO UTC and presented in local time by the UI. An empty query lists meetings
through `MeetingQuery`; it never issues an empty `MATCH`.

### FTS5 querying

The index exists from M2 over active complete revisions, tokenizer
`unicode61 remove_diacritics 2`, kept in sync inside the publication transaction and
purged on deletion. This group only reads it. User text is normalized (trim, collapse
whitespace, strip control characters) and turned into a safe FTS expression: each
token is double-quoted with internal quotes doubled, tokens are joined with implicit
AND, and no user character is interpreted as operator, prefix or column filter.
Ranking is `bm25()` then `started_at` descending then `meeting_id`. Filters are
applied in the SQL before `LIMIT`, never after. Cursors encode the last
`(rank, started_at, meeting_id, segment_id)` and are opaque strings; a malformed
cursor is `InvalidInput`.

Snippets come from FTS5 `snippet()` with plain-text markers and a bounded length;
the UI renders them as text, never as HTML. Results carry `meeting_id`,
`revision_id`, `segment_id`, snippet, `start_ms`/`end_ms` when present and
`timing_source`.

### Fallback and reindex

At open, the service probes FTS5 and records the capability. Without FTS5, listing
and filtering work unchanged; search uses a documented literal fallback (`LIKE` on
normalized segment text, bounded by the same filters and limit) and the page carries
a `degraded` flag the UI shows. `reindex()` rebuilds the derived index from active
complete revisions in one transaction per meeting batch and reports counts; without
FTS5 it reports the limitation instead of failing. A search result is identical
before and after `reindex()`.

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
callbacks. Title editing calls `MemoryService.update_meeting` with the displayed
`row_version`; a `Conflict` is shown and the row is refreshed, never overwritten.
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

No playback, no citation target resolution, no CLI, no participant editing beyond
what M2 already exposes, no provider call, no semantic search, no transcript text
editing, no changes to capture or overlay. The citation target DTO is declared here
only as a type in `src/memory/models.py` so group B and M4 share it.

## Technical references

- [SQLite FTS5](https://www.sqlite.org/fts5.html): `MATCH` syntax, `bm25()`,
  `snippet()`, tokenizer options and the need to quote user tokens.
- [Qt QThread and worker objects](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html):
  queued signals and the request-sequence pattern already used by `BackgroundJobs`.
