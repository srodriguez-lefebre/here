# M3 playback, evidence navigation and CLI implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. Root owns leases/integration/merges after
> received/responded current-head Copilot, exact-head Windows CI and independent
> review (rulings 16/17/20/35). Steps use checkbox syntax.
> **Status: preparation only, not in execution.** Root may dispatch B on the
> frozen/reviewed M3 A branch with its PR open after accepted interface handoff
> and any necessary contract-docs PR merge; rebase B after A merges (CLAUDE §11).

**Goal:** Jump from any result or citation to the original segment and, with real
timing, to the audio at that millisecond; expose the same operations through
`here meetings` with stable JSON and exit codes.
**Architecture:** The planned `MemoryService.resolve_citation` turns a `Citation` into a
`CitationTarget` from the database; `src/ui/playback.py` wraps QMediaPlayer behind a
DTO-driven controller; the CLI `meetings` group calls the services and formats output.
**Tech Stack:** Python 3.13, PySide6 QtMultimedia, Typer, pytest, pytest-qt; no new
runtime dependency (QtMultimedia ships with PySide6; the frozen bundle must include
its plugins).
**Spec:** `docs/development/M3_LIBRARY_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_3_PLAN.md`.

## Global constraints

- Titles, dates, paths and times for navigation come only from the database through
  the resolver; never from free text or a search snippet.
- A segment without timing never seeks; no control implies precision that does not exist.
- Playback starts only from a user click; selection never plays.
- Controls are disabled while the snapshot reports preparing, recording or paused.
- The CLI implements no search or persistence; it calls the services.
- Tests use synthetic fixtures, generated WAVs and temporary roots; the real-marker
  WAV run is a recorded observation, not a CI test.
- Touched files leave ruff clean and formatted; no new mypy errors in them.

## Dependencies and exclusive scope before execution

Preparation foundation: M2 A merge
`53712c5beedafc2d40e2abf10c13362a1c3c9f78`; lifecycle/search contract PR #22 merge
`73d3cb384bd422de6899d8c7b761f3f3159ea3f1`. The latter accepts planning contracts,
not M3 A or M2 B implementation. Actual frozen/reviewed M3 A code/DTO handoff
and verified PR head remain **pending**. Under CLAUDE §11, root may dispatch M3 B
on that branch while M3 A's PR is open, once the actual interfaces are accepted
and any necessary contract-documentation PR is merged. M3 A's eventual merge SHA
remains **pending** for rebase and final integration/provenance; it is not a
prerequisite for preparation, the formal contract PR or this permitted root GO.
R35 leaves root sole owner of leases, integration and merges. Rebase B after A
merges; this preparation starts no implementation or new chat/front.

Use the accepted root M3_SEARCH/M2_LIFECYCLE documents at `73d3cb3`, not older
prepared drafts. Root supplies the actual accepted M3 A source and exact signatures
before the formal B maintenance PR and launch; future API names below are planned,
not proof that those methods/types exist in M2 A. Fakes do not close integration.

| Handoff / shared scope | Required owner or dependency |
|---|---|
| Query/page/Citation/CitationTarget/error definitions | Accepted M3 A DTOs, structured reasons and frozen/reviewed PR head; eventual merge SHA for rebase/final provenance, not initial GO. |
| Library/detail/request lifecycle | Accepted M3 A UI hooks/stale-callback handling; playback extends them under exclusive file leases. |
| Revision/audio evidence | Merged A schema2 links and receipt3/marker3 semantics, nullable dates and original timing. |
| Title/participant/assignment CLI actions | Real accepted M2 B4 APIs and version/error contract. |
| Deletion/reindex/R42 read guard | Real accepted B5 adapter and playback/delete handoff; no duplicate registry parser/index writer. |
| memory service/models/repository/database, CLI, application adapters | Root serializes writes with active M2 B/M3 A; no access to their owned worktrees. |
| Shared UI/tests, packaging, dependencies/version/global docs | Root grants exact file leases and serializes integration/metadata and milestone closure. |

Heavy tests/builds/mypy, benchmark/deadline suites and real seek observations require
root's exclusive host window. No runtime is authorized by this preparation.
Existing prepared playback/portability fixtures remain under their current owners;
transfer accepted fixtures through root without regenerating or reopening them here.

Carry these inherited requirements into the existing tasks below:

- R46's passive `Citation(meeting_id, revision_id, segment_id, quote=None,
  start_char=None, end_char=None)` is an identity/reference, not media authority.
  Bind the existing invalid-citation tests to source quote and Unicode character
  span/membership/bounds validation; do not use byte offsets or trust UI-provided
  target path, status or text. Resolve those values from guarded memory without
  inventing timestamps or replacing the selected revision's audio links.
- R50 approves `memory.service.catalog_read_view(library)`, with actual B5
  implementation/handoff still pending: caller opens/closes a fresh connection
  without an active transaction; B takes the lock before BEGIN and owns the single
  read transaction through generation/fetch. Consume SuppressionSnapshot's sorted
  `suppressed_meeting_ids` and `fingerprint`, including exact origin/fingerprint
  aliases resolved in that DB view, before exposure/keyset/LIMIT. Pending and
  complete suppress; invalid/inaccessible state fails closed with LibraryError.
  No second parser, nested lock/transaction or accepted deletion/reindex UI hook
  is implied by the approved contract.
- R47 empty normalized search returns `MeetingPage` with listing filters, limit
  and cursor intact; nonempty search returns `SearchPage`. CLI delegates and
  formats the real page type, never manufacturing a SearchHit or segment. Bind
  R45 error handling to canonical `memory.service.InvalidInput.reason` with
  `corpus_changed`, preserving existing exit 2 and the accepted JSON shape.
- Resolve exact meeting/revision/segment membership and that revision's ordered
  audio links/asset identity/hash/path. Reuse role/path guards before media access;
  missing/ambiguous evidence grants no guessed playable path. Do not choose latest
  or meeting-wide audio. Preserve receipt2/3 and marker1/2/3 compatibility.
- Keep absent capture dates/bounds unknown, measured zero valid and recording offsets
  applied once. No seek inferred from text, chunk order, pauses or processing dates.
  Nonretryable catalog warnings grant no media/provider/mutation authority.
- Consume accepted B5 R42 suppression before target/detail/path and CLI exposure,
  including old selections/receipts after catalog loss. Uncertain guards fail closed;
  actual ownership/path guards stay intact. Add no alternate suppression parser.
- Pass through R39 keysets and R41 complete/partial literal/fallback/degraded behavior
  from SearchService. R45 ranked generation mismatch remains structured
  InvalidInput/corpus_changed → existing CLI exit 2/accepted JSON error shape.
  Do not combine generations; Qt uses its existing fresh request/replacement path.
  Ordinary listing remains insert-stable under its date/ID keyset.

## R51/R52 bindings within the existing tasks

These approved contracts remain pending verified B5/native implementation and final
handoff signatures; reconcile them through the normal docs PR before M3 B source.
- Bind existing delete/error tests to phase-aware DeletionResult: after a valid
  durable barrier, including a prior attempt, ordinary failure is pending with
  canonical UUID, fixed reason and io/exit 4, never rejected/NotFound for absent SQL.
  Complete/exit 0 requires verified purge plus durable complete. Pre-acceptance typed
  errors retain their mappings; consume structured fields, not exception prose.
- The future before_delete adapter finishes Qt-thread stop/unload/load invalidation
  and waits for bounded ACK from the worker without blocking Qt or touching player
  off-thread. Invoke without SQL transaction/catalog locks the GUI may require.
  After ACK/failure reread phase, reacquire/revalidate identity, ownership and
  inventory; per R52/f09c0fc retain applicable native and canonical leases from
  before durable barrier through cleanup. Pre-ACK preflight is not acceptance,
  and ownership appearing during ACK must be caught before a new barrier.
  Choose the concrete timeout during adapter implementation, not in this draft.
- Never invoke private _derived_row_purgers as a resolver. Trusted SQL purgers run
  on the same final cleanup connection/transaction before relationships/core/FTS
  removal, with no own transaction/commit/I/O/provider; failure rolls back together.
- Future composition/CLI/retry use the same injected MemoryService and explicit
  configured/StartRequest root; register request output_dir before worker launch.
  CLI destructive operations use existing settings_operation/get_settings root.
  No second default service, core settings/.env reader, durable root registry or
  origin-derived probing. Bare destructive context fails closed with LibraryError/
  io4; a valid barrier remains pending/hidden, while reads/non-destructive repair
  continue. Validate declared roots/lock roles again after ACK. Actual B5/native
  cleanup ordering and lease lifetime remain a handoff dependency, not a UI rewrite.

R53 facade binding is now concrete at frozen/reviewed `d8ec92b`:
`SearchService(connection_factory, *, reindexer=None).reindex()` calls the same
injected MemoryService callback once, returns its identical R51 report and propagates
canonical errors unchanged. Constructor performs no maintenance; reindex owns no
factory/connection/snapshot/lock/probe/migration and does not retry or recount.
Missing callback fails only reindex with existing LibraryError, preserving reads.
Consume ApplicationUiAdapter's explicit query override and separate optional typed
maintenance capability; do not expand required read protocols or create another
backend, indexer, MemoryService or root. Existing B5/R52/UI and performance/group
handoff gates remain open; the facade checkpoint does not approve them.

## Review focus

- `setPosition` called before loaded or seekable, or applied to the wrong media
  after a fast meeting switch.
- A citation whose revision is historical or whose audio is missing.
- Playback enabled while a recording is active.
- Any path that triggers play without a click.
- Exit codes and JSON shape drifting between commands.
- Frozen bundle missing QtMultimedia plugins (coordinate with `packaging/windows.spec`).

### Task 3: Playback adapter and evidence navigation

**Files:** `src/ui/playback.py` (new), `src/ui/library.py` (group A),
`src/ui/main_window.py`, `src/ui/contract.py`, `src/ui/fake_controller.py`,
`src/memory/service.py` (M2, add `resolve_citation`), `src/memory/models.py`
(`Citation`, `CitationTarget` from group A), `packaging/windows.spec` (QtMultimedia
plugins); tests in `tests/test_memory_citation.py`, `tests/test_ui_playback.py`,
`tests/test_ui_library.py`, `tests/test_distribution_smoke.py` (plugin presence).

**Planned B interfaces (bind to actual accepted A DTOs before execution):**
`MemoryService.resolve_citation(citation: Citation) -> CitationTarget`;
`PlaybackController(parent)` with `load(target: CitationTarget)`, `play()`, `pause()`,
`seek_ms(int)`, `set_volume(float)`, `set_rate(float)`, `stop_and_unload()`,
signals `position_changed(int)`, `duration_changed(int)`, `state_changed(str)`,
`error(str)`; planned `VisualController.resolve_citation` and
`LibraryView.open_target(target)`. Root reconciles these intended hooks against
actual accepted UI/types; they are not methods verified in merged M2 A.

Interpret the existing resolver/navigation tests with exact revision audio links,
nullable timing/dates, R42 suppression and shared path guards. Fake media verifies
state/seek ordering only; the prepared real-marker fixture and recorded 250 ms
observation remain separate requirements.

- [ ] Write failing tests for `resolve_citation`: current, historical and missing
  targets; validated audio path under the managed root only; invalid citation →
  `InvalidInput`; offsets never include wall-clock pauses.
- [ ] Implement the resolver on the M2 repository.
- [ ] Write failing Qt tests with a fake `QMediaPlayer`: seek queued until loaded and
  seekable; meeting switch stops and unloads; missing file and codec error surface as
  text; controls disabled while recording/paused/preparing; no play without click.
- [ ] Implement `PlaybackController` and the player widget; wire it into the detail
  pane with "Reproducir desde mm:ss" and "Sin marca de tiempo".
- [ ] Write failing navigation tests: list item, search hit and synthetic citation open
  the same detail with the segment centered and neighbors visible; historical badge
  and link; missing-state rendering.
- [ ] Implement `open_target`, context rendering and badges.
- [ ] Add QtMultimedia plugins to `packaging/windows.spec` and a presence check in the
  distribution smoke test.
- [ ] Record the real-marker WAV seek observation (generated WAV with tones at known
  offsets, measured within 250 ms) in `ACCEPTANCE.md` with machine and WAV hash.
- [ ] Run focused tests and the full offscreen suite; self-review; commit as
  "ui: playback adapter and evidence navigation".

### Task 4: `here meetings` commands and JSON

**Files:** `src/cli.py` (new `meetings_app` Typer group), `src/entrypoints.py` (no
change expected; confirm), `src/ui/playback.py` (minimal standalone window for
`play`); tests in `tests/test_cli_meetings.py`, `tests/test_cli.py`.

**Interfaces:** Typer commands `list`, `show`, `search`, `rename`, `participant`,
`retry`, `play`, `import-session`, `reindex`, `delete` under `here meetings`; a shared
`emit(result, json: bool)` formatter; an `ExitCode` enum `0/2/3/4/5` mapped from the
typed service errors (`InvalidInput`→2, `NotFound`/`Conflict`→3,
`IntegrityFailure`/I/O→4, `ProviderUnavailable`→5). Actual imports/result/JSON shapes
come from the accepted A/B handoff, not new classes assumed present in M2 A.
The only provider route is the accepted explicit application retry adapter.

The existing pass-through/error tests consume R39/R41/R45 without another query
implementation: preserve opaque cursor/filter/backend/generation values, map
structured corpus_changed to InvalidInput/exit 2, keep ordinary listing insert
stability and retain unknown dates as null. Real title/participant/delete/reindex
integration waits for accepted B4/B5 services/guard; fakes do not satisfy it.

- [ ] Write failing tests for each command's success path with a temporary library:
  JSON shape, full IDs, ISO UTC dates, ordering and filters passed through to the
  services.
- [ ] Write failing tests for every error mapping, ambiguous abbreviation (exit 2 with
  candidates), `delete` without `--yes` (exit 2, nothing changed), and `play` without
  multimedia (explanatory error, target in JSON).
- [ ] Implement the command group, formatter and exit-code mapping; `play` reuses
  `PlaybackController` in a minimal window.
- [ ] Verify `here --help` lists the group and that no command constructs a provider
  client except `retry`.
- [ ] Run focused tests and the full offscreen suite; self-review; independent agent
  review; commit as "cli: meetings commands with stable JSON and exit codes".

## Closing the group

- Update `README.md` (library, player, `here meetings`), `ARCHITECTURE.md` (citation
  resolver and playback adapter), `docs/MILESTONE_3_PLAN.md` (criteria 11–18 status),
  `ACCEPTANCE.md` (M3 navigation row with the real-audio run) and `WHATS_NEW.md`.
- Bump version to 0.4.0 in `pyproject.toml` and the build/installer paths that embed
  it (`scripts/build_windows.ps1`, `packaging/installer.iss`, `windows-package.yml`).
  Root reconciles actual embedding paths at handoff (workflow is under
  `.github/workflows/`) and serializes the bump only after all M3 criteria and
  real dependencies pass; this preparation changes no version or dependency.
- Open one PR with the two task commits, self-review and independent agent review in
  the description. Root integrates accepted dependencies, receives/responds to
  Copilot on the current head, verifies exact-head Windows CI and performs the
  guarded merge under rulings 16/17/20. Real-audio certification remains required.
  Then reconcile the already-prepared `docs/MILESTONE_4_PLAN.md` and first M4
  SPEC/PLAN against actual merged M3 contracts under ruling 14; do not restart
  existing preparation or create another front without root dispatch.
