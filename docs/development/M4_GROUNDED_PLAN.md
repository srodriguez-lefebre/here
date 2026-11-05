# M4 grounded core implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The user merges (ruling 4). Steps use checkbox
> syntax. **Status: not in execution (ruling 15).** Start only on an explicit user
> instruction after M3 closes.

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
- The provider never supplies paths, titles, dates or timestamps used for navigation;
  the verifier drops them and the memory resolves them.
- Budgets are hard: 6 per meeting, 24 total, 24,000 evidence characters, 2,000 output
  tokens by default; truncation is reported, never hidden.
- `intelligence/` does not import Qt or Typer and does not open SQLite directly.
- Tests use synthetic corpora, temporary roots and a mock transport; the one real
  provider smoke is opt-in and stores no key.
- Register `here.intelligence` in `pyproject.toml`. Touched files leave ruff clean and
  formatted; no new mypy errors in them (baseline not clean, `CLAUDE.md` §6).

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
`src/intelligence/retrieval.py`, `pyproject.toml`; tests in
`tests/test_intelligence_models.py`, `tests/test_intelligence_retrieval.py`; a
synthetic corpus builder in `tests/fixtures/intelligence_corpus.py`.

**Interfaces:** immutable DTOs `QuestionRequest`, `ExtractionRequest`, `Budget`,
`EvidenceBlock` (`id`, `meeting_id`, `revision_id`, `segment_id`, `text`,
`start_ms|None`, `end_ms|None`), `Coverage`, `Claim`, `Extraction`, `Answer`,
reusing M2 `Citation`; `Retriever.retrieve(request) -> RetrievalResult` with
evidence blocks and coverage, built on `MemoryService.search` and the segment
repository; typed errors `InvalidInput`, `ProviderUnavailable`, `VerificationFailed`.

- [ ] Write failing tests: empty selection without `whole_library` raises
  `InvalidInput`; `whole_library` only when set; term extraction drops stopwords and
  short tokens; FTS query is built with bound parameters and escaped literal terms.
- [ ] Write failing budget tests on a 50-meeting, 2,000-hit corpus: at most 6 per
  meeting and 24 total, character budget respected with adjacent context counted,
  `Coverage` equals the sent set and flags truncation; only active complete revisions
  unless requested.
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
- [ ] Add the opt-in real-provider smoke with synthetic text, gated by an explicit
  environment flag, recording model and date to the evidence file without the key.
- [ ] Record the chosen `INTELLIGENCE_MODEL` as a dated ruling in `DELIVERY.md`
  (ruling 7).
- [ ] Run focused tests and the full suite; self-review; independent agent review;
  commit as "intelligence: provider adapter and citation verifier".

## Closing the group

- Update `ARCHITECTURE.md` (intelligence package, grounded request, verifier),
  `docs/MILESTONE_4_PLAN.md` (criteria 1–9, 12 status), `ACCEPTANCE.md` (M4 rows
  with the runs that close them, smoke model and date) and `WHATS_NEW.md` only if a
  user-facing capability ships (it does not until group B).
- Open one PR with the three task commits, self-review and independent agent review in
  the description, CI green. The user merges. Then group B starts from
  `M4_PRODUCT_PLAN.md`.
