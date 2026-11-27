# M4 grounded core implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. Root owns leases, integration and merges
> under rulings 20/35, with current-head Windows CI and Copilot omitted under
> ruling 64. Ruling 69 selects GPT-6.2 Sol for active agent work.
> Steps use checkbox syntax. **Status: preparation only, no M4 GO.** Ruling 20
> replaces ruling 15's pause; implementation still waits for M3 closure, accepted
> actual handoffs and the required contract-document PR before root dispatch.

**Goal:** Questions and extractions over selected meetings, in a local mode with no
network and a provider mode whose every citation is mechanically verified.
**Architecture:** `intelligence/` service over the M2 catalog and FTS index; lexical
retrieval with budgets; a grounded request with inert evidence blocks; a verifier
between the provider and anything the user sees.
**Tech Stack:** Python 3.13, `sqlite3` FTS5 through M2 repositories, Pydantic for
schemas, the existing OpenAI SDK, pytest with mock transport; no new dependency.
**Spec:** `docs/development/M4_GROUNDED_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_4_PLAN.md`.

## Global constraints

- Local mode works with `OPENAI_API_KEY` unset; provider mode is entered only by an
  explicit user choice and fails typed before any network attempt when the key or
  model is missing.
- Transcript text is inert data: never interpolated into instructions, never a source
  of tools, links, code or policy.
- Memory resolves paths, titles, dates and timestamps used for navigation. The
  verifier rejects model-supplied timestamps and derives timing from the segment.
- Budgets are hard: 6 per meeting, 24 total, 24,000 evidence characters, 2,000 output
  tokens by default; truncation is reported, never hidden.
- `intelligence/` does not import Qt or Typer and does not open SQLite directly.
- Tests use synthetic corpora, temporary roots and a mock transport; the one real
  provider smoke reads `HERE_PROVIDER_SMOKE_KEY` only (ruling 19) and stores no key.
- Register `here.intelligence` in `pyproject.toml`. Touched files leave ruff clean and
  formatted; no new mypy errors in them (baseline not clean, `CLAUDE.md` §6).

## Actual contracts and pending handoffs

Foundation is merged M2 A `53712c5beedafc2d40e2abf10c13362a1c3c9f78` and contracts
`73d3cb384bd422de6899d8c7b761f3f3159ea3f1`. M3 A search merged in PR #28 at
`a1a83629cdfab64dd249f877cdd443e7f7d96b08` and the M3 B handoff contracts merged
in PR #29 at `8d0b03d92524c78ca6983db0fc64e1048e7b2a6c`. The provisional B
integration inspected here is `063d974f66ac42da85a82474cfc900a1335e54f3`, with the
34 transferred commits on that handoff base. B acceptance, merge and M3's
0.4.0 closure are pending. Root confirms the interfaces on the final integrated
base and merges the M4 contract-document PR before source work.

| Boundary | Current contract and unresolved handoff |
|---|---|
| Query owner | `memory.search.SearchService(connection_factory)`, `search(SearchQuery) -> SearchPage \| MeetingPage`; fresh LibraryConnection per operation, closed by the service. No presumed MemoryService.search. |
| Scoped evidence | Provisional B `SearchQuery.meeting_id` scopes one canonical UUID before keyset and LIMIT and binds it to the cursor. Empty search passes scope through to listing. Task 1 adds memory-owned `SearchService.evidence_snapshot(EvidenceQuery) -> EvidenceSnapshot` for selected-set literal OR, global BM25, ranked rounds, adjacent context and the budgets defined in the SPEC. Public M3 search, cursors and schema stay unchanged. |
| Read suppression | Search and citation resolution at `063d974` share R50 `catalog_read_view`, which owns lock-before-BEGIN and one transaction. The caller opens and closes the fresh connection. The snapshot carries resolved suppressed IDs and a fingerprint covering registry records and alias resolution. Registry roles and paths are preflighted under the lock and the boundary fails closed. The planned retrieval and `SearchService.revalidate_evidence(EvidenceSnapshot) -> bool` each use a separate coherent R50 view. |
| Native search generation | R75/R76 are private Windows SQLite encoders for FTS and ACTIVE generation. Consume public v2 cursors and preserve canonical fallback inside R50. Retrieval requires no native-loader binding. |
| Citation source | Candidate `MemoryService.resolve_citation(Citation) -> CitationTarget` validates membership, literal quotes, paired Unicode bounds, and exact revision/audio links with artifact role guards. Unvalidated audio yields `status="missing"` without a path, possibly with verified text. The resolver can return historical evidence; M4 still has to revalidate the selected active revision before publication. Confirm both bindings on the integrated M3 head. |
| Participants/root | R48 real IDs/roles/nullable assignments, request-scoped speakers; R40/R52 same injected service/configured root. No new .env reader or origin-derived root probing. |
| Future persistence | R51 private SQL purger on B's same final cleanup transaction; future M4 B only, never a citation resolver or an M4 A migration/task. |

Within the existing retrieval/verifier tests, preserve R39 NULL/date/keysets,
R41 complete/partial/fallback semantics and R45 whole-corpus generation. Reuse the
canonical InvalidInput.reason including corpus_changed; never concatenate stale
ranked pages. R47 empty search is MeetingPage, not evidence or implicit scope
expansion. The memory-owned evidence method applies selected-set eligibility and
suppression before keyset and LIMIT, with global BM25 and fair, repeatable ranked
rounds as defined in the SPEC. Exclude raw MATCH input, per-ID search fan-out, post-LIMIT
scope filtering and direct intelligence SQL. Do not hold a database
transaction/ownership lock while calling the provider. Actual guarded source
revalidation must precede publication; concurrent loss/change does not certify a
stale result. R42/R50 pending/complete suppression also applies to adjacent context,
with aliases resolved before exposure; invalid registry state fails closed.

Use the SPEC's metadata-only window (`4 * Budget.total` overall and per meeting,
plus a sentinel) before reading full text. Charge exact Unicode characters and
cover embedded NUL without an unbounded read. Restrict neighbor reads to the two
immediate ordinals of each selected match. Preserve R41 for `match`: global BM25
for indexed complete revisions, escaped literal OR and neutral rank for opted-in
partials and the degraded fallback, and literal transcript evidence for all routes.
For explicit `summary`, continue from the last considered ordinal on the same
revision with bounded batches, progress and cancellation between batches.

M3 B accepts `Citation` for desktop navigation and re-resolves identity through
the injected memory backend even when called with a `CitationTarget`. Navigation
uses the service passed to `create_default_controller(memory_service=...)`;
the controller's default processor receives the same instance. M3 B review,
playback evidence and seek verification remain integration gates.
In the candidate, a measured zero seek waits for a ready, seekable backend and
capture clears the pending seek. These playback behaviors provide no authority
for M4 text publication.

Keep quote offsets in original Unicode source coordinates even when the existing
minimal verifier normalization is used. Put the exact source substring in the
passive Citation. Revalidate the segment and revision even when playable audio is
unavailable. Keep audio and paths out of provider payloads. Do not normalize away
uncertain owners/dates, invent timestamps from targets, or promote a request speaker
label into a participant.
Future M4 B's derived-row purger is SQL-only on the supplied final transaction;
no commit/BEGIN/rollback/other connection/I/O/provider and no use from retrieval.
R51 accepted-barrier failures remain pending with canonical ID/reason/io, not fake
rejection after SQL disappearance. These are inherited boundaries, not added M4 A work.

## Prepared corpus and evidence boundaries

Root transfers the already prepared and independently inspected
`tests/grounded_fixtures.py` once from m2-preparation to the future independent RED
owner. Preserve that file and its provenance; do not regenerate it under a new name.
It provides four synthetic meetings/thirteen passages, literal evidence notes and
lifecycle descriptions, not product assertions or approved model answers.
The budget corpus is separately prepared at `a2ce1e1` in
`tests/fixtures/intelligence_corpus.py`. It yields 50 meetings and 2,000 OR
matches, with 400 in the selected set and 100 in the sparse-only subset. It
includes a shared nonmatching neighbor, long Unicode passages and a title-only
token. Root can transfer it once after the final base is bound. The RED owner
still ingests the data through accepted memory APIs and temporary roots, supplies
lifecycle and suppression cases, and writes the retrieval assertions and missing
English/mocked-provider/verifier cases. The pure corpus supplies inputs and counts;
ingestion, ranking, budget enforcement and coverage require tests on the bound head.

The prepared verifier payloads at `0c35b6b` in
`tests/fixtures/intelligence_provider_payloads.py` include bilingual E1/E2
quotes with original Unicode coordinates, isolated invalid responses and empty
claim/extraction arrays. Root transfers the fixture once to the RED owner with
its provenance. The mock claims use `false` provisionally for `uncertainty`.
The eventual schema owns that type and the exact size limits. The fixture supplies
mock response values and source anchors for verifier tests on the bound head.

All existing acceptance and complete checkbox blocks remain binding. Registering the
future package in pyproject is a later root-serialized task, not permission to edit
dependencies/version now. Choose and verify INTELLIGENCE_MODEL only at implementation
time, recording its dated ruling; no model is selected here. Provider traffic remains
explicit-action only, tests use mock transport, and only the opt-in synthetic smoke
uses HERE_PROVIDER_SMOKE_KEY without recording it. No key/Qt/SQLite access is added.

## Review focus

- Any path where question text or transcript text reaches the instruction part of a
  request.
- Verifier acceptance of an `evidence_id` not in the sent set, or of a quote that
  matches after more than minimal normalization.
- Revision changed between retrieval and verification.
- Budget accounting when adjacent context segments are added.
- Cancellation between retrieval, provider call and result publication.
- Request payload contents in provider mode: no audio, no file names, no whole library.

### Task 1: Models, citations and budgeted retrieval

**Files:** `src/intelligence/__init__.py`, `src/intelligence/models.py`,
`src/intelligence/retrieval.py`, `pyproject.toml`, `src/memory/models.py`
(`EvidenceQuery` and `EvidenceSnapshot`) and `src/memory/search.py`
(bounded retrieval and revalidation under R50); tests in
`tests/test_intelligence_models.py`, `tests/test_intelligence_retrieval.py`;
the existing `tests/grounded_fixtures.py` via root's one-time handoff, plus the
required budget corpus whose final test-helper location is bound before execution.

**Interfaces:** immutable DTOs `QuestionRequest`, `ExtractionRequest`, `Budget`,
`EvidenceBlock` (`id`, `meeting_id`, `revision_id`, `segment_id`, `text`,
`start_ms|None`, `end_ms|None`), `Coverage`, `Claim`, `Extraction`, `Answer`,
reusing M2 `Citation`; `Retriever.retrieve(request) -> RetrievalResult` with
evidence blocks and coverage, consuming `SearchService.evidence_snapshot` and
the revalidation boundary described above; canonical
`InvalidInput`, plus planned `ProviderUnavailable` and `VerificationFailed`.
These intelligence interfaces remain planned; no existing memory API is invented.

- [ ] Write failing tests: empty selection without `whole_library` raises
  `InvalidInput`; `whole_library` only when set; term extraction drops stopwords and
  short tokens; FTS query is built with bound parameters and escaped literal terms.
- [ ] Write failing budget tests on a 50-meeting, 2,000-hit corpus: at most 6 per
  meeting and 24 total, character budget respected with adjacent context counted,
  `Coverage` equals the sent set and flags truncation; only active complete revisions
  unless requested. Include mixed selected/unselected hits, literal OR terms and
  skewed meeting volumes. Verify selected-set eligibility before limits, global
  ranking, ranked rounds, stable ordering and context counted once. Cover a run of
  oversized candidates with the next fitting match inside and outside the window,
  sentinel truncation and exact lengths for text with embedded NUL. Assert partial
  and degraded literal OR routing, neutral ranks, canonical ties and source text.
  Check empty `match` abstention and explicit `summary` continuation, cancellation
  and progress across bounded batches, including a changed revision. Exercise
  `revalidate_evidence` against changed and absent sources and suppression failures.
- [ ] Implement models, retrieval, diversification and coverage; register the package.
- [ ] Run focused tests and the full offscreen suite; self-review; commit as
  "intelligence: models and budgeted retrieval".

### Task 2: Local evidence mode and conservative extraction

**Files:** `src/intelligence/local.py`, `src/intelligence/service.py`; tests in
`tests/test_intelligence_local.py`, `tests/test_intelligence_service.py`.

**Interfaces:** `IntelligenceService.ask(request, cancellation) -> Answer` and
`extract(request, cancellation) -> Extraction` dispatching on mode; `local.py`
exposes `evidence_answer(retrieval) -> Answer` and
`detect_candidates(blocks, kinds) -> list[Extraction]` with fixed labels
"Evidencia local" and "Candidato detectado" and the fixed abstention message.

- [ ] Write failing tests on the extraction fixture: negation ("no se aprobó") is not
  a decision; hypothesis is not a decision; task without owner has `owner_text=None`;
  "el viernes" kept verbatim; English and Spanish patterns both fire; duplicates by
  content and citation collapse while keeping all sources.
- [ ] Write failing tests: no-hit question yields the abstention message with zero
  claims; every local answer item carries a resolvable `Citation`; cancellation
  between retrieval batches stops with status cancelled.
- [ ] Implement local mode, candidate patterns and the service dispatch; provider
  mode raises `ProviderUnavailable` in this task.
- [ ] Run focused tests and the full suite; self-review; commit as "intelligence:
  local evidence mode".

### Task 3: Provider adapter, schema and verifier

**Files:** `src/intelligence/providers.py`, `src/intelligence/verifier.py`,
`src/intelligence/service.py`, `src/config/settings.py` (`INTELLIGENCE_MODEL`,
`INTELLIGENCE_MODE` default `local`); tests in `tests/test_intelligence_provider.py`,
`tests/test_intelligence_verifier.py`, `tests/test_intelligence_injection.py`,
opt-in `tests/test_intelligence_provider_smoke.py`.

**Interfaces:** `IntelligenceProvider` protocol with
`generate(request: GroundedRequest, cancellation) -> ProviderResult`; `OpenAIProvider`
built from `build_client()` with timeout and `store=False` where supported;
`ProviderResult` Pydantic schema; `verify(result, sent_blocks, snapshot) ->
Verified | Rejected(reasons)`; one bounded repair attempt then `VerificationFailed`.

- [ ] Write failing tests with a mock transport: the request separates instructions
  from evidence blocks; question and transcript text never appear in the instruction
  part; payload contains no audio, file names or unselected meetings; model name comes
  from settings; missing key is `ProviderUnavailable` before any call.
- [ ] Write failing verifier tests, one per rejection: unknown id, empty quote,
  non-substring quote, model-supplied timestamp, oversized field, changed revision;
  one repair then `VerificationFailed`; accepted result keeps claim kinds.
- [ ] Write the injection fixture test: transcript instructions to change mode, fetch
  a URL, run a command or ignore rules have no effect on commands, network, schema or
  policy.
- [ ] Implement the adapter, schema, verifier, repair loop and cancellation checks
  before each call and before publication.
- [ ] Add the opt-in real-provider smoke with synthetic text, gated by
  `HERE_PROVIDER_SMOKE_KEY` (ruling 19; skipped when unset), recording model, date and
  provider-reported usage to the evidence file without the key.
- [ ] Record the chosen `INTELLIGENCE_MODEL` as a dated ruling in `DELIVERY.md`
  (ruling 7).
- [ ] Run focused tests and the full suite; self-review; independent agent review;
  commit as "intelligence: provider adapter and citation verifier".

## Closing the group

- Update `ARCHITECTURE.md` (intelligence package, grounded request, verifier),
  `docs/MILESTONE_4_PLAN.md` (criteria 1–9, 12 status), `ACCEPTANCE.md` (M4 rows
  with the runs that close them, smoke model and date) and `WHATS_NEW.md` only if a
  user-facing capability ships (it does not until group B).
- Root coordinates the group PR with self-review and independent review, verifies
  current-head Windows CI and performs the guarded merge under rulings 20/35.
  For the pending PRs, omit Copilot under ruling 64. Reconcile the existing
  `M4_PRODUCT_PLAN.md` with accepted contracts before its dispatch, preserving the
  same-milestone open-PR overlap and later rebase allowed by CLAUDE §11.
