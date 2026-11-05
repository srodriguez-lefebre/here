# M4 answers persistence, UI and CLI specification

Status: written 2026-11-05, not in execution (ruling 15). Implementation starts only
on an explicit user instruction after M4 group A is merged. Governing design:
`PRODUCT_DESIGN.md` §3, §5, §6 and §8, a taken decision (ruling 3). Spanish plan and
acceptance criteria: `../MILESTONE_4_PLAN.md`.

## Intent

Keep every verified answer and extraction as durable, citable records that are
invalidated honestly when the underlying revisions change or a meeting is deleted,
and expose the grounded core to the user through a Qt "Preguntar" view, an
extraction view and two CLI commands. Nothing persisted can reproduce a deleted
meeting; nothing shown can hide which claims are synthesis.

## Decisions

### Persistence

A migration adds `answers`, `claims` and `claim_citations` to the M2 catalog.
`answers`: UUID, question, mode, provider, model, prompt and schema version, scope
(selected meeting ids or whole-library flag), coverage, status
(`completed`, `cancelled`, `failed`, `historical`), created_at, cache_key, usage
fields nullable. `claims`: UUID, answer FK, kind (`fact`, `synthesis`,
`uncertainty`, `decision`, `task`, `blocker`), text, owner_text and due_text nullable,
ordinal. `claim_citations`: claim FK, meeting FK, revision FK, segment FK, quote,
offsets nullable. Foreign keys cascade so that deleting a meeting removes every answer
that cites it, with its claims, in the same operation that M2 already journals in
`pending_operations`. No raw provider payloads, prompts with evidence text, headers or
secrets are persisted. Usage and cost fields are stored only when the provider returns
them; never a monetary estimate presented as a bill.

### Cache and invalidation

`cache_key` is a hash of question, mode, provider, model, prompt and schema version,
budget, scope and the `content_hash` of every revision used. A repeated question with
the same key returns the stored answer marked as cached. When a retry activates a new
revision for any cited meeting, answers citing the old revision become `historical`;
the UI shows them as such and offers explicit regeneration. Before showing or saving,
the service revalidates that no cited meeting is `deleting` or gone; if one is, the
answer is discarded entirely. A deletion notified while a request is in flight
discards that request's result and clears its text from the UI.

### Qt "Preguntar" view

`src/ui/intelligence.py` adds a view reachable from the M3 library: a scope control
showing the selected meetings or an explicit "toda la biblioteca" switch (never on by
default), the question field, the mode selector defaulting to local, a one-time per
application session notice of what provider mode sends, progress with cancellation,
and the result as answer cards. Each card shows the claim kind as a visible label,
the text, and citations rendered as buttons with real title, real date and timestamp
only when the segment has one. A citation button opens the M3 detail and player at
that segment through the shared `CitationTarget`; without a timestamp it opens the
text with "Sin marca de tiempo". Coverage reads "Se consultaron N fragmentos de M
reuniones" and marks truncation. Historical answers carry a badge and a regenerate
action. Errors from the provider show their state and a button to switch to local
mode; nothing is relabeled as AI output silently.

### Extraction view

From a meeting detail: "Extraer" produces decisions, tasks and blockers with the same
cards and citations, batched with progress and cancellation for the whole meeting;
local mode labels items "Candidato detectado".

### Threads and request identity

All service calls run in `BackgroundJobs` workers from `src/ui/bridge.py`; the Qt
thread never blocks. Each request has an identity; results for a superseded request
or a closed view are dropped. A deletion event invalidates pending requests that cite
the deleted meeting.

### CLI

`here ask "pregunta" [--meeting ID ...] [--all] [--mode local|provider] [--json]`
and `here extract ID [--mode local|provider] [--json]` under a new `ask`/`extract`
pair in `src/cli.py`, calling the same `IntelligenceService`. Omitting both
`--meeting` and `--all` is exit 2. Exit codes: 0 success, 2 invalid input, 3 not
found or conflict, 4 integrity or IO, 5 provider unavailable or verification failed.
JSON output contains full ids, mode, model, coverage, claims and citations with the
resolved `CitationTarget` fields; no secrets.

## Acceptance

- Migration adds the three tables with FKs; `foreign_key_check` and `integrity_check`
  pass; the M2 future-schema rejection still works.
- Deleting a meeting removes every answer citing it, its claims and citations; after
  restart no text from that meeting exists in `answers`, `claims`, FTS, managed
  backups or caches (canary phrase search).
- Deletion during an in-flight request discards the result; the UI shows no text from
  that meeting afterwards.
- Cache key changes when any cited revision's `content_hash` changes; a retry marks
  prior answers `historical`; the UI shows the badge and regenerates only on explicit
  action.
- A synthetic canary key and canary prompt phrase are absent from the database, logs
  and artifacts after a provider-mode run with mock transport.
- Qt tests: scope control never defaults to whole library; provider notice shown once
  per session; cancellation ends in `cancelled`; synthesis label visible; citation
  button opens detail and requests the exact seek when a timestamp exists and shows
  "Sin marca de tiempo" otherwise; provider error offers local mode.
- Stale result from a superseded request or a closed view is dropped.
- CLI: missing scope exits 2; `--mode provider` without key exits 5 with a
  configuration hint; JSON schema stable across both commands; exit codes as listed.
- Full Windows CI suite passes offscreen with a synthetic key and temporary roots.

## Boundaries

No embeddings, no automatic extraction after recording, no editing of claims, no
provider settings UI (M5), no export of answers (M5), no changes to the grounded core
beyond consuming it.

## Technical references

- [SQLite foreign key actions](https://www.sqlite.org/foreignkeys.html): cascade
  semantics used for answer removal.
- `src/ui/bridge.py` (`BackgroundJobs`) and `src/ui/contract.py`
  (`ApplicationUiAdapter`): the existing worker and adapter pattern the view follows.
