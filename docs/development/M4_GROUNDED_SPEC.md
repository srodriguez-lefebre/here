# M4 grounded core specification

Status: written 2026-11-05, not in execution (ruling 15). Implementation starts only
on an explicit user instruction after M3 closes. Governing design:
`PRODUCT_DESIGN.md` §3, §6, §8 and §9, a taken decision (ruling 3). Spanish plan and
acceptance criteria: `../MILESTONE_4_PLAN.md`.

## Intent

Answer questions over one or several meetings and extract decisions, tasks and
blockers, with every relevant statement backed by a citation that provably exists in
the context the system itself assembled. Two explicit modes: local evidence with no
network and no key, and a provider mode with structured output whose every citation is
mechanically verified before anything is shown. Transcript text is data, never
instructions. Missing evidence produces an honest abstention, not a guess.

## Decisions

### Package and dependencies

New package `src/intelligence/` with `models.py`, `retrieval.py`, `local.py`,
`providers.py`, `verifier.py`, `service.py`, registered in `pyproject.toml`. No Qt, no
Typer. The provider adapter reuses the OpenAI SDK already in the project and the
client factory in `src/transcription/client.py` (`build_client`,
`require_provider_key`); no new runtime dependency. Retrieval reads through
`MemoryService.search` and the segment repository of M2; it never opens SQLite on its
own.

### Requests and scope

`QuestionRequest` carries the question, an explicit scope (`meeting_ids` tuple, or
`whole_library=True` set on purpose), the M3 filters, `mode` (`local` or
`provider`) and a `Budget`. An empty `meeting_ids` with `whole_library=False` is
`InvalidInput`; it never silently widens to the whole library. `ExtractionRequest`
is the same for one meeting with `kinds` subset of `decision`, `task`, `blocker`.

### Retrieval and budgets

Lexical retrieval over the M2 FTS5 index: informative Spanish and English terms of
the question, joined with OR, scored with BM25, with adjacent segments as context.
Diversify per meeting: at most `Budget.per_meeting` (default 6) and
`Budget.total` (default 24) segments, plus a hard character budget for evidence
(default 24,000) and a hard output token budget (default 2,000). Only active complete
revisions by default. `Coverage` reports segments and meetings consulted and whether
any budget truncated; the caller shows it as "N fragmentos de M reuniones", never as
an exhaustive review. "Summarize this meeting" selects all segments of that meeting
and processes them in bounded batches with progress and cancellation.

### Local mode

`local.py` returns the retrieved passages as `Answer` items of kind `evidence`, each
with a `Citation`, plus `Extraction` candidates found by conservative multilingual
patterns (decided/acordamos/we agreed; tiene que/debe/will do; bloqueado/blocker/
waiting on). Labels are fixed: "Evidencia local" and "Candidato detectado". The
literal excerpt is preserved; owner and due text are copied verbatim or left null;
negations and hypotheticals are excluded by pattern, never inverted. No hit yields
the fixed abstention message. Local mode needs no network, no key and no provider.

### Provider mode and grounded request

`GroundedRequest` is assembled by the service, not by the UI: a fixed system
instruction, the question, and inert evidence blocks `E1`, `E2`, … each carrying
`meeting_id`, `revision_id`, `segment_id` and the original text, physically separated
from instructions. Nothing from transcript text is ever interpolated into the
instruction part. The adapter calls the SDK with a JSON schema for `ProviderResult`
(claims `{kind, text, evidence_ids, quotes, uncertainty}` with kind in
`fact|synthesis|uncertainty`; extractions `{kind, text, owner_text|null,
due_text|null, evidence_ids}` with kind in `decision|task|blocker`), `store=False`
where the endpoint supports it, a bounded timeout and bounded output. The model name
comes from `Settings.INTELLIGENCE_MODEL`, chosen at implementation time and recorded
as a dated ruling (ruling 7). The provider is called only from an explicit user
action in provider mode; a missing key or model is `ProviderUnavailable` and the
caller offers local mode. Cancellation is a `threading.Event` checked before each
call and before publishing a result.

### Verifier

`verifier.py` checks every claim and extraction against the exact evidence sent:
each `evidence_id` must be in the sent set; the referenced revision must still be the
snapshot used; each quote must be a substring of that segment's source text after
minimal normalization (whitespace, quotes, case); timestamps are never accepted from
the model and are resolved from the segment; field lengths and counts are bounded.
A rejection carries the reason per item. The service may retry once with the reasons
("bounded repair"); a second rejection is a visible failure that offers local mode.
Verification proves citation existence, not semantic truth; the result keeps claim
kinds so the UI can label synthesis.

### Results

`Answer` carries mode, model, coverage, verified claims and extractions, each with
`Citation` objects resolvable through `MemoryService.resolve_citation`. Visible
messages are redacted: no key, no meeting text by default. Typed errors as in
`PRODUCT_DESIGN.md` §3 plus `ProviderUnavailable` and `VerificationFailed`.

## Acceptance

- Three-meeting synthetic corpus: an answer cites at least two meetings when the
  corpus requires it, in local mode and in provider mode with a mock transport.
- No-hit question returns the abstention message with zero claims in both modes.
- Budgets hold: 50 meetings, 2,000 matching segments, at most 6 per meeting and 24
  total sent; `Coverage` matches the sent set and reports truncation.
- Empty selection without `whole_library` is `InvalidInput`; `whole_library` only
  when set explicitly.
- Verifier rejects, each as its own test: unknown `evidence_id`, empty quote, quote
  not a normalized substring, timestamp supplied by the model, oversized fields,
  revision changed between retrieval and verification. One repair attempt, then
  `VerificationFailed`.
- Prompt injection fixture (instructions to change mode, fetch a URL, run a command,
  ignore rules) has no effect on commands, network, schema or policy; the text is
  treated as evidence only.
- Local extraction fixture: negation not inverted, hypothesis not a decision, task
  without owner has null owner, "el viernes" kept verbatim.
- Provider mode never sends audio, file names or the whole library; the request
  payload in the mock transport is inspected for that.
- Missing key in provider mode is `ProviderUnavailable` before any network attempt;
  local mode works with the key unset.
- Cancellation before the provider call issues no request; cancellation after it
  publishes nothing.
- Opt-in real-provider smoke with synthetic text records model, date and reported
  usage in `ACCEPTANCE.md` and stores no key; it runs only when
  `HERE_PROVIDER_SMOKE_KEY` is set (ruling 19) and is skipped otherwise.
- Full Windows CI suite passes with `QT_QPA_PLATFORM=offscreen`, a synthetic key,
  temporary roots and no real `.env` or private recordings.

## Boundaries

No persistence of answers, no UI, no CLI (group B). No embeddings or vector store.
No automatic calls after recording. No global speaker identity. No normalization of
names or dates the text does not state. No change to M2 or M3 contracts beyond
reading them.

## Technical references

- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs):
  schema conformance is not semantic truth.
- [SQLite FTS5](https://www.sqlite.org/fts5.html): BM25 ranking and query syntax that
  must never receive raw user text as MATCH.
- `src/transcription/client.py`: existing client factory and the `chat.completions`
  call used by `cleanup_transcript`, the pattern to follow for the adapter.
