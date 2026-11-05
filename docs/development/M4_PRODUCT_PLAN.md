# M4 answers persistence, UI and CLI implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The user merges (ruling 4). Steps use checkbox
> syntax. **Status: not in execution (ruling 15).** Start only on an explicit user
> instruction after M4 group A is merged.

**Goal:** Durable, honestly invalidated answers and extractions, and the user-facing
"Preguntar" view, extraction view and CLI commands over the grounded core.
**Architecture:** three catalog tables with cascading FKs; a cache key over revision
hashes; Qt views on `BackgroundJobs` workers with request identities; CLI as a thin
adapter over the same service.
**Tech Stack:** Python 3.13, `sqlite3`, Pydantic, PySide6, Typer, pytest-qt; no new
dependency.
**Spec:** `docs/development/M4_PRODUCT_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_4_PLAN.md`.

## Global constraints

- Deleting a meeting leaves no text of it in answers, claims, FTS, caches or managed
  backups; this is tested with a canary phrase after restart.
- No raw provider payloads, evidence-bearing prompts, headers or secrets persisted or
  logged.
- Whole-library scope is never the default and never inferred from an empty selection.
- The provider notice is shown once per application session; no repeated dialogs.
- Synthesis is always labeled; citations show real titles, dates and timestamps only
  from the memory.
- The Qt thread never blocks; every service call is a `BackgroundJobs` job with an
  identity.
- `intelligence/` and `memory/` stay free of Qt and Typer. Touched files leave ruff
  clean and formatted; no new mypy errors in them (`CLAUDE.md` §6).

## Review focus

- Cascade on deletion reaching `claim_citations` and `claims` through `answers`.
- A result arriving after its meeting was deleted or its view closed.
- Cache key omitting a revision hash because of adjacent context segments.
- Historical marking when only one of several cited meetings was retried.
- Canary leakage into logs through `str(exc)` on provider errors.
- CLI scope parsing and exit codes.

### Task 1: Persistence, cache and invalidation

**Files:** `src/memory/database.py` (migration), `src/memory/models.py`,
`src/memory/repository.py`, `src/memory/service.py` (deletion extended),
`src/intelligence/store.py`, `src/intelligence/service.py`; tests in
`tests/test_intelligence_store.py`, `tests/test_memory_service.py`.

**Interfaces:** `AnswerStore.save(answer) -> AnswerRecord`, `find_cached(cache_key)`,
`mark_historical(revision_id)`, `discard_for_meeting(meeting_id)`; `cache_key(request,
revisions) -> str`; `MemoryService.delete_meeting` extended to remove citing answers
inside the same journaled operation; a deletion event the service exposes for
in-flight invalidation.

- [ ] Write failing migration tests: three tables, FKs, cascade from `meetings`
  through `answers` to `claims` and `claim_citations`; checks pass; M2 future-schema
  rejection intact.
- [ ] Write failing tests: cache key changes with any cited revision hash; same key
  returns the cached answer flagged cached; retry on one meeting marks only answers
  citing it historical.
- [ ] Write failing deletion tests: delete a cited meeting, restart, search a canary
  phrase in `answers`, `claims`, FTS, managed backups and caches, expect zero hits;
  delete during an in-flight request discards the result.
- [ ] Write failing redaction tests: canary key and canary prompt phrase absent from
  database, logs and artifacts after a mock-transport provider run, including provider
  error paths.
- [ ] Implement store, migration, cascade, cache key, historical marking and in-flight
  discard; sanitize provider exception messages before logging.
- [ ] Run focused tests and the full suite; self-review; commit as "intelligence:
  persist answers with honest invalidation".

### Task 2: "Preguntar" view, extraction view and CLI

**Files:** `src/ui/intelligence.py`, `src/ui/main_window.py`, `src/ui/contract.py`,
`src/ui/bridge.py`, `src/ui/preferences.py` (mode preference, notice shown flag),
`src/cli.py`; tests in `tests/test_ui_intelligence.py`, `tests/test_cli.py`.

**Interfaces:** `IntelligenceView` with scope control, question field, mode selector,
progress, cancellation, answer cards and evidence panel; `ExtractionView` from the
meeting detail; `ApplicationUiAdapter` gains `ask`, `extract`, `cancel_request` and
`open_citation(target)` delegating to M3's `LibraryView.open_target(target)` (the same
entry list, search results and citations use; it never starts playback); CLI commands `ask`
and `extract` with `--meeting`, `--all`, `--mode`, `--json` and exit codes
0/2/3/4/5.

- [ ] Write failing Qt tests: scope never defaults to whole library; provider notice
  once per session; cancellation ends cancelled; synthesis label visible; citation
  button opens detail with exact seek when a timestamp exists and "Sin marca de
  tiempo" otherwise; stale result after view close is dropped; provider error offers
  local mode; historical badge and explicit regenerate.
- [ ] Implement the views on `BackgroundJobs` with request identities and the
  deletion-event hook; wire navigation to the M3 `CitationTarget` opener.
- [ ] Write failing CLI tests: missing scope exits 2; `--mode provider` without key
  exits 5 with a hint; stable JSON for both commands; `--json` contains resolved
  citation targets and no secrets.
- [ ] Implement the CLI commands as thin adapters over `IntelligenceService`.
- [ ] Run focused Qt and CLI tests plus the full offscreen suite; self-review;
  independent agent review; commit as "intelligence: ask and extract views and CLI".

## Closing the group

- Update `README.md` (new commands, modes, what provider mode sends),
  `ARCHITECTURE.md` (answer persistence and invalidation), `docs/MILESTONE_4_PLAN.md`
  (criteria 10–18 status), `ACCEPTANCE.md` (M4 rows with the runs that close them)
  and `WHATS_NEW.md` with a short note on asking the memory with citations.
- Bump version to 0.5.0 (ruling 9) when all M4 criteria are verified.
- Open one PR with the two task commits, self-review and independent agent review in
  the description, CI green. The user merges. Then the agent writes
  `docs/MILESTONE_5_PLAN.md` and the first M5 SPEC/PLAN (ruling 14) if they do not
  exist yet.
