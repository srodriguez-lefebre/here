# M4 grounded core specification

Status: contract preparation on the merged M3 0.4.0 base. Ruling 20 authorizes M4
and ruling 79 sets the evidence-read contract. M4 A implementation starts after
this contract PR merges and root dispatches the work.
Governing design:
`PRODUCT_DESIGN.md` §3, §6, §8 and §9, a taken decision (ruling 3). Spanish plan and
acceptance criteria: `../MILESTONE_4_PLAN.md`.

## Intent

Answer questions over one or several meetings and extract decisions, tasks and
blockers, with every relevant statement backed by a citation that provably exists in
the context the system itself assembled. Two explicit modes: local evidence with no
network and no key, and a provider mode with structured output whose every citation is
mechanically verified before anything is shown. Transcript text is data, never
instructions. Missing evidence produces an honest abstention, not a guess.

## Inherited foundation and handoff boundary

Foundation: M2 A merge `53712c5beedafc2d40e2abf10c13362a1c3c9f78` and contract
merge `73d3cb384bd422de6899d8c7b761f3f3159ea3f1`. M3 A merged in PR #28 at
`a1a83629cdfab64dd249f877cdd443e7f7d96b08`. The M3 B handoff contracts merged
in PR #29 at `8d0b03d92524c78ca6983db0fc64e1048e7b2a6c`. M3 B closed 0.4.0 in
PR #30 at `7239ba64e03e7e3230b36b5d8d952cefbbe54e47` on 2026-11-27 at 14:21
Montevideo time, with reviewed head `d29f83a1c7f69caa3261bb7ddab579a75d4bc4b9`.
M4 inherits the search, read guard and citation APIs from that merge.

## Decisions

### Package and dependencies

New package `src/intelligence/` with `models.py`, `retrieval.py`, `local.py`,
`providers.py`, `verifier.py`, `service.py`, registered in `pyproject.toml`. No Qt, no
Typer. The provider adapter reuses the OpenAI SDK already in the project and the
client factory in `src/transcription/client.py` (`build_client`,
`require_provider_key`); no new runtime dependency. Retrieval consumes the memory-owned
query/read boundary: current M3 defines `SearchService(connection_factory)` in
`memory/search.py`, with `search(SearchQuery) -> SearchPage | MeetingPage`.
The factory returns a fresh LibraryConnection per operation and SearchService closes
it. Intelligence never opens SQLite or implements its own query/index/registry reader.
There is no assumed `MemoryService.search` facade. The exact scoped retrieval and
adjacent-segment snapshot handoff below must be accepted before implementing it.

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

### Query, scope and coherent evidence handoff

R39/R41/R45 remain memory-owned: opaque cursor/filter/backend binding, actual null
dates, complete-only persistent FTS, opt-in read-only partials and literal/degraded
fallback. Never fabricate dates, rank precision or another indexer. Search generation
covers the actual full rank corpus, including nonmatching/filtered/suppressed rows
still indexed. Canonical `memory.service.InvalidInput.reason="corpus_changed"`
rejects stale ranked continuation; evidence assembled across incompatible snapshots
cannot be combined. Ordinary listing keeps its original date/ID insertion behavior.

R47 normalized empty search returns MeetingPage with listing filters/limit/cursor
intact, not invented SearchHits or segments. That page is not evidence and cannot
widen the explicit M4 scope. Removing stopwords must not turn an empty retrieval
expression into an implicit whole-library request.

Merged M3 A exposes text and listing filters. Its FTS builder quotes each input
token and joins tokens with spaces, and its literal fallback combines term predicates
with AND. Merged M3 B's `SearchQuery.meeting_id: str | None = None` accepts
one canonical UUID, applies it before keyset and LIMIT, and binds it to cursor scope.
FTS ranking still uses global BM25. Empty search carries the same scope into listing.
R75/R76 use private Windows SQLite encoders for FTS and ACTIVE generation, preserving
generation bytes, public v2 cursors and canonical fallback within R50. Retrieval
consumes the public search contracts and has no native-loader dependency.
M4 A adds the memory-owned `SearchService.evidence_snapshot(query: EvidenceQuery)`
returning `EvidenceSnapshot`. `EvidenceQuery` carries an explicit `match` or `summary`
purpose, normalized literal terms, mutually exclusive `meeting_ids` or explicit
`whole_library`, inherited date/status/participant filters and `include_partial`, and
the `per_meeting`, `total` and `evidence_chars` limits from `Budget`. Apply scope
and suppression to candidate eligibility before ranked rounds and limits.

`match` with no terms yields no evidence and preserves scope. When FTS is available,
complete revisions use global BM25 and MATCH built from escaped literal OR terms.
Eligible opt-in partials use escaped literal OR, carry the partial flag and receive
neutral rank 0, as in M3. When FTS is unavailable, all eligible matches use escaped
literal OR and neutral rank 0. Preserve the date/ID tie-breakers and ranked rounds
for these routes, and surface degraded coverage in fallback. Source evidence is exact
transcript text. The memory boundary constructs both query forms from bound terms.

Use one R50 view for matching and context. Deduplicate segments. Within a meeting,
take ranked matches in order. Consume the first match of each meeting before the
second and sort each round by M3 order (rank, date descending with NULL last,
meeting ID, segment ID). Rank and return only IDs, ordinals, metadata and length
information in the candidate phase. Bound consideration in SQL before returning
rows to Python to `candidate_limit = 4 * Budget.total` overall and per meeting.
Use a metadata-only sentinel row to detect eligible candidates beyond the window.
Give matches priority, then consider at most two neighbors per selected match, its
immediate predecessor and successor by ordinal in the same revision. Deduplicate
and apply the same limits and bounded reads in reproducible order.

Fetch complete text only for selected candidates that pass the remaining-character
preflight. Account for embedded NUL in the length calculation, since SQLite's text
`length()` stops there. A byte-length preflight for such values may bound the read
to at most four UTF-8 bytes per remaining character, followed by an exact Unicode
character count. Recheck the exact text length before charging the budget. Omit
segments that cannot fit, mark truncation, and continue within the window if another
candidate fits. A sentinel marks truncation and contributes no evidence or coverage.
Compute coverage from the actual set included. The snapshot carries exact segment text,
meeting/revision/segment identity, revision content hash and citation timing metadata.
It carries no paths. Preserve public M3 search and cursor contracts and the schema.

`summary` requires exactly one selected meeting and no terms. Traverse its revision
in bounded ordinal batches. Bind continuation to the same `revision_id` and the last
considered ordinal, and keep source IDs on the blocks. The service reports progress
and checks cancellation between batches, surfaces truncation and revalidates the
evidence before publication. A revision change cannot continue an earlier batch.

R42/R50 suppression must guard all source/context reads and revalidation, before
exposure and keyset/LIMIT. The approved `catalog_read_view(library)` contract has
caller-owned fresh connection/close, B-owned lock-before-BEGIN and one transaction
through generation/fetch. Its immutable snapshot IDs/fingerprint include registry
UUIDs and exact normalized-origin/fingerprint aliases resolved in that DB view.
Pending and complete suppress; invalid/inaccessible state fails closed. The memory
boundary owns this integration, not an intelligence-side parser or nested transaction.
Search and citation resolution both use this guard in M3 0.4.0. Task 1 adds
`SearchService.revalidate_evidence(snapshot: EvidenceSnapshot) -> bool`. It opens a
separate coherent R50 view and checks each block against the current eligible meeting,
active revision and revision hash, segment identity, exact content and citation timing
metadata. Absence or staleness returns `False`; a guard failure prevents publication.
Revalidation accesses no media. Release the retrieval view before any provider call
and revalidate before publication. The resolver's existing historical navigation is
independent of this check. Both new methods still await implementation and acceptance.

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

R46 supplies the passive `Citation(meeting_id, revision_id, segment_id, quote=None,
start_char=None, end_char=None)`. Validate source membership and any supplied quote
or Unicode character span; offsets must address original source text, not bytes or
positions invented in normalized snippets. Preserve the verifier's existing minimal
normalization rule without broadening it. When it accepts a normalized match,
populate the passive citation with the exact original substring and original
character coordinates. Merged M3 B resolves membership and validates
literal quotes or paired, bounded Unicode offsets inside one `catalog_read_view`.
It derives navigation metadata and validates exact revision audio through managed
artifact role and path guards. Only a target resolved by memory can grant
navigation authority. Null dates/times stay absent, measured zero stays zero, and
request-scoped speakers retain their scope (R46/R48).

### Results

`Answer` carries mode, model, coverage, verified claims and extractions, each with
`Citation` objects resolvable through `MemoryService.resolve_citation`. Visible
messages are redacted: no key, no meeting text by default. Typed errors as in
`PRODUCT_DESIGN.md` §3 plus `ProviderUnavailable` and `VerificationFailed`.
Merged M3 B exposes `MemoryService.resolve_citation(Citation) -> CitationTarget`.
It returns `status="missing"` without a playable path when audio is unvalidated.
Audio authority requires the revision's single ordinal-zero WAV link, an
available asset owned by the meeting, validated artifact roles, and matching
file size and digest. Verified segment text may still be present. A missing
revision can yield a missing target for a bare reference when the segment is
also absent; a quote or span supplied in that case is invalid. The resolver also
accepts historical revisions. Use the verified source and revision state,
independently of playback status, to revalidate evidence before publication. The
playback resolver grants audio authority only for navigation and must never supply
audio or paths to the provider.

Future M4 B persistence consumes R51's private trusted `_derived_row_purgers` only
as ordered SQL on B's same final cleanup transaction before core relationships are
removed; no own transaction/commit/connection/I/O/provider. Failure rolls derived/core
rows back together into pending replay. It is not a read resolver or an M4 A task.
R51/R52 keep accepted deletion pending/hidden after a barrier even if SQL rows vanish.
Composition reuses the injected service and explicit configured roots; no second
service, new .env reader or origin-derived root probing belongs to intelligence.

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
