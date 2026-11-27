# M4 answers persistence, UI and CLI specification

Status: contract preparation on M3 0.4.0 (`7239ba6`, PR #30). Ruling 20 authorizes
M4. Root confirms the actual A handoff and lands the required contract PR before
dispatch. Under CLAUDE §11, B may start on the reviewed A branch with its PR open
and rebase after A merges. Governing design:
`PRODUCT_DESIGN.md` §3, §5, §6 and §8, a taken decision (ruling 3). Spanish plan and
acceptance criteria: `../MILESTONE_4_PLAN.md`.

## Intent

Keep every verified answer and extraction as durable, citable records that are
invalidated honestly when the underlying revisions change or a meeting is deleted,
and expose the grounded core to the user through a Qt "Preguntar" view, an
extraction view and two CLI commands. Nothing persisted can reproduce a deleted
meeting; nothing shown can hide which claims are synthesis.

## Decisions

### Inherited contracts and A handoff

M3 0.4.0 supplies the R42 durable suppression registry, R50 `catalog_read_view`,
phase-aware R51 deletion and the R52 same-service ownership boundary. M3's
`LibraryView.open_target` accepts a passive `Citation` and resolves it against
memory, including historical revisions. A missing or unvalidated WAV can leave
verified segment text available. Seek uses original timing and playback stays
user-initiated.

R79 defines A's planned `EvidenceSnapshot` and `revalidate_evidence`. Bind the
actual verified answer/extraction DTOs and exact evidence dependencies (including
context), historical status and cache metadata, cancellation/publication boundary,
and typed errors against A's reviewed head. Confirm how A transfers source identity
and revision hashes without passing raw prompts or unused transcript text to B's
store. A's active-revision revalidation and M3's historical navigation are separate
contracts.

### Persistence

A migration adds `answers`, `claims` and `claim_citations` to the M2 catalog.
`answers`: UUID, question, mode, provider, model, prompt and schema version, scope
(selected meeting ids or whole-library flag), coverage, status
(`completed`, `cancelled`, `failed`, `historical`), created_at, cache_key, usage
fields nullable. `claims`: UUID, answer FK, kind (`fact`, `synthesis`,
`uncertainty`, `decision`, `task`, `blocker`), text, owner_text and due_text nullable,
ordinal. `claim_citations`: claim FK, meeting FK, revision FK, segment FK, quote,
offsets nullable. Use FKs for integrity and dependent claim/citation cleanup.
Deleting a meeting removes the entire answer, including claims and citations for
other meetings, through R51's private SQL purger. Register it on the same composed
`MemoryService`. The purger identifies affected answer IDs before core relationships
are removed and deletes them in the existing final cleanup transaction on the
supplied repository connection. It runs no transaction control, other connection,
I/O or provider call. Failure rolls core and derived rows back together and keeps
the deletion pending and hidden for replay. Preserve the R52 roots and leases and
the existing cleanup of DB/WAL and managed backups before durable completion.

Persist only the intended answer record and its verifiable source associations.
Every retained transcript-derived part must be attributable to a guarded source for
invalidation and whole-answer purge. The A handoff must account for evidence used
in a result, including context that contributes to a claim or cache key. Keep raw
provider payloads, evidence-bearing prompts, headers and secrets out of storage and
logs. Store usage and cost only when supplied by the provider.

### Cache and invalidation

`cache_key` is a hash of question, mode, provider, model, prompt and schema version,
budget, scope and the `content_hash` of every revision used. A repeated question with
the same key returns the stored answer marked as cached. When a retry activates a new
revision for any cited meeting, answers citing the old revision become `historical`;
the UI shows them as such and offers explicit regeneration. Before saving and
publishing a fresh result, use A's active-source validation and guard the save
against a revision switch or deletion between validation and write. Bind the
writer-side validation with the memory owner. Use the existing serialized write
boundary and common suppression parser, without nesting R50 inside a write or
holding a view during a provider call.

Read cached and historical records under R50. Suppress the entire record if any
source meeting is pending/complete suppressed, deleting or absent, before cache
reuse or presentation. Historical display requires guarded, still-resolvable source
identity; it does not satisfy active-evidence publication or a fresh cache lookup.
An invalid guard fails closed. A deletion invalidates affected in-flight requests
and clears the corresponding displayed text; the UI callback does not own purge.

### Qt "Preguntar" view

`src/ui/intelligence.py` adds a view reachable from the M3 library: a scope control
showing the selected meetings or an explicit "toda la biblioteca" switch (never on by
default), the question field, the mode selector defaulting to local, a one-time per
application session notice of what provider mode sends, progress with cancellation,
and the result as answer cards. Each card shows the claim kind as a visible label,
the text, and citations rendered as buttons with real title, real date and timestamp
only when the segment has one. A citation button opens the M3 detail and player at
that segment by passing the passive `Citation` to M3; without a timestamp it opens the
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
or a closed view are dropped. Extend the existing deletion acknowledgement path for
affected view/request invalidation. Keep the worker-side bounded ACK and Qt-thread
ownership. After an accepted deletion, callbacks may not restore suppressed text.

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
