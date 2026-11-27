# M4 answers persistence, UI and CLI implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. Root owns leases, integration and merges.
> Steps use checkbox syntax. **Status: contract preparation only (ruling 20).**
> Bind to the reviewed A head with its PR open, land any required contract PR and
> await root dispatch. Rebase after A merges (CLAUDE §11).

**Goal:** Durable, honestly invalidated answers and extractions, and the user-facing
"Preguntar" view, extraction view and CLI commands over the grounded core.
**Architecture:** three catalog tables with FKs and an R51 whole-answer SQL purger;
a guarded cache over revision hashes; Qt views on `BackgroundJobs` workers with
request identities; CLI as a thin adapter over the same service.
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

## Handoff and file ownership

- M3 0.4.0 (`7239ba6`, PR #30) supplies R42/R50 suppression, the R51 deletion
  lifecycle and R52 composition, the Qt deletion ACK, and passive `Citation`
  navigation. Use the same injected memory service, roots and jobs. Root assigns
  leases on `src/ui/app.py` or `src/ui/library.py` if composition or ACK wiring
  needs edits there.
- At A handoff, verify the actual R79 snapshot/revalidation and `ask`/`extract`
  APIs, result/claim shapes (including local evidence), source and revision hash
  inventory, coverage, typed outcomes and cancellation/progress hooks. Confirm
  writer-side source revalidation and ownership of cache/history invalidation and
  deletion request matching. Resolve those interfaces before freezing T4/T5 tests.
- T4 owns schema, storage and cleanup. T5 consumes the T4 store, cache, guarded
  publication and invalidation interfaces. Root serializes the migration, shared
  `service.py` integration with A and the 0.5.0 bump. M5 B claims/answers remain a
  future consumer of accepted R51 cleanup and storage contracts.

## Review focus

- The R51 purger deletes whole affected answers before core relationships vanish.
  A failed cleanup rolls back together and R42 suppression hides pending records.
- A result arriving after its meeting was deleted or its view closed.
- Cache key omitting a revision hash because of adjacent context segments.
- Historical marking when only one of several cited meetings was retried.
- Canary leakage into logs through `str(exc)` on provider errors.
- CLI scope parsing and exit codes.

### Task 4: Persistence, cache and invalidation

**Files:** `src/memory/database.py` (migration), `src/memory/models.py`,
`src/memory/repository.py`, `src/memory/service.py` (existing R51 purger binding),
`src/intelligence/store.py`, `src/intelligence/service.py`; tests in
`tests/test_intelligence_store.py`, `tests/test_memory_service.py`.

**Interfaces:** `AnswerStore.save(answer) -> AnswerRecord`, `find_cached(cache_key)`,
`mark_historical(revision_id)`; `cache_key(request, revisions) -> str`; bind the
private SQL purger to the existing `MemoryService` cleanup and use the deletion
ACK/invalidation boundary for in-flight results. Confirm final store interfaces
against A before implementation.

- [ ] Write failing migration tests: three tables, FKs, answer-to-claim/citation
  cascades, integrity checks and future-schema rejection.
- [ ] Write failing tests: cache key changes with any cited revision hash; same key
  returns the cached answer flagged cached; retry on one meeting marks only answers
  citing it historical.
- [ ] Write failing deletion tests: delete a cited meeting, restart, search a canary
  phrase in `answers`, `claims`, FTS, managed backups and caches, expect zero hits;
  delete during an in-flight request discards the result. Cover rollback and pending
  replay, whole-answer removal across meetings, suppression before cache exposure,
  and a source change between validation and save.
- [ ] Write failing redaction tests: canary key and canary prompt phrase absent from
  database, logs and artifacts after a mock-transport provider run, including provider
  error paths.
- [ ] Implement store, migration, SQL purger, cache key, historical marking and
  in-flight discard. Sanitize provider exception messages before logging.
- [ ] Run focused tests and the full suite; self-review; commit as "intelligence:
  persist answers with honest invalidation".

### Task 5: "Preguntar" view, extraction view and CLI

**Files:** `src/ui/intelligence.py`, `src/ui/main_window.py`, `src/ui/contract.py`,
`src/ui/bridge.py`, `src/ui/preferences.py` (mode preference, notice shown flag),
`src/cli.py`; tests in `tests/test_ui_intelligence.py`, `tests/test_cli.py`.

**Interfaces:** `IntelligenceView` with scope control, question field, mode selector,
progress, cancellation, answer cards and evidence panel; `ExtractionView` from the
meeting detail; planned `ApplicationUiAdapter` operations `ask`, `extract` and
`cancel_request`; route navigation from a passive `Citation` to M3's
`LibraryView.open_target`, preserving user-initiated playback. CLI commands `ask`
and `extract` with `--meeting`, `--all`, `--mode`, `--json` and exit codes
0/2/3/4/5.

- [ ] Write failing Qt tests: scope never defaults to whole library; provider notice
  once per session; cancellation ends cancelled; synthesis label visible; citation
  button opens detail with exact seek when a timestamp exists and "Sin marca de
  tiempo" otherwise; stale result after view close is dropped; provider error offers
  local mode; historical badge and explicit regenerate.
- [ ] Implement the views on `BackgroundJobs` with request identities and the
  deletion ACK/invalidation path. Wire navigation to the M3 citation opener.
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
  the description. Root verifies Windows CI on the exact head and merges under
  CLAUDE §7. Ruling 64 omits Copilot for the pending PRs. Reconcile the existing M5
  plan and contracts against merged M4 (ruling 14), including the future M5 B claims
  and answers handoff.
