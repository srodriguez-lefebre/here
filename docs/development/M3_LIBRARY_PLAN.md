# M3 playback, evidence navigation and CLI implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. Root owns leases, integration and merges
> under CLAUDE §7 and rulings 16/20/35/64. Steps use checkbox syntax.
> **Status: preparation only, held before dispatch.** Root may dispatch B on the
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
- Playback starts only from a user click. Selection only opens the evidence.
- Controls are disabled while the snapshot reports preparing, recording or paused.
- The CLI implements no search or persistence; it calls the services.
- Tests use synthetic fixtures, generated WAVs and temporary roots; the real-marker
  WAV run is a recorded observation, not a CI test.
- Touched files leave ruff clean and formatted; no new mypy errors in them.

## Dependencies and exclusive scope before execution

M2 B merged in PR #25 at `b4b3d1beb9b05e33ae03ab8381e805aa6a23ffc8`, with
Windows CI `43461764790` and package workflow `43461764783` on reviewed head
`55d68e2`. The interface comparison pins integrated M3 A candidate
`31488693152067ae58eeacec08f0b01f47085fe4` and the reviewed deletion ACK
`d8fb75b37691cc9c4e57fa84bb545190b3b9304b`. This is a static comparison.
The earlier warm-search p95 of 879.48 ms fails the 500 ms gate. M3 A's PR is
unopened and its final reviewed head and gates remain pending. Root holds B until
that head is reviewed, the PR opens, the interfaces are accepted and any required
contract documentation is merged. Under CLAUDE §11, B can then start on A's
branch and rebase after A merges.

| Handoff / shared scope | Required owner or dependency |
|---|---|
| Query/page/Citation/CitationTarget/error definitions | Candidate DTOs and structured reasons in the SPEC. Confirm them on the final reviewed A PR head and record the merge SHA at rebase. |
| Library/detail/request lifecycle | Candidate generation and worker hooks. Playback extends the reviewed ACK in the existing UI lifecycle. |
| Revision/audio evidence | Merged schema 2 links and receipt 3/marker 3 semantics, nullable dates and original timing. |
| Title/participant/assignment CLI actions | Merged B4 version and error contracts. |
| Deletion/reindex/R42 read guard | Merged B5, candidate R53 facade and R52 ACK. Player release on deletion is Task 3 work. |
| Memory service/models/repository/search, CLI, UI and application adapters | Root serializes all shared writes with M3 A and M5 B. Work from separate worktrees and keep the M3 A candidate frozen during its verification. |
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
- Merged B5 supplies R50's `memory.service.catalog_read_view(library)`. The caller
  opens and closes a fresh connection without an active transaction. The guard
  takes the lock before BEGIN and owns the single
  read transaction through generation/fetch. Consume SuppressionSnapshot's sorted
  `suppressed_meeting_ids` and `fingerprint`, including exact origin/fingerprint
  aliases resolved in that DB view, before exposure/keyset/LIMIT. Pending and
  complete suppress; invalid/inaccessible state fails closed with LibraryError.
  Use the same parser and transaction boundary.
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
- Consume merged B5 R42 suppression before target/detail/path and CLI exposure,
  including old selections/receipts after catalog loss. Uncertain guards fail closed;
  actual ownership/path guards stay intact. Add no alternate suppression parser.
- Pass through R39 keysets and R41 complete/partial literal/fallback/degraded behavior
  from SearchService. R45 ranked generation mismatch remains structured
  InvalidInput/corpus_changed → existing CLI exit 2/accepted JSON error shape.
  Do not combine generations; Qt uses its existing fresh request/replacement path.
  Ordinary listing remains insert-stable under its date/ID keyset.

## R51/R52 bindings within the existing tasks

Use merged B5 and the reviewed ACK carried by the M3 A candidate.
- Bind existing delete/error tests to phase-aware DeletionResult: after a valid
  durable barrier, including a prior attempt, ordinary failure is pending with
  canonical UUID, fixed reason and io/exit 4, never rejected/NotFound for absent SQL.
  Complete/exit 0 requires verified purge plus durable complete. Pre-acceptance typed
  errors retain their mappings; consume structured fields, not exception prose.
- Extend the existing `_DeleteAcknowledgement` and
  `LibraryView._pause_for_delete(request)` to stop/unload and invalidate player
  loads/seeks on Qt before the tracked worker receives ACK. Keep the existing 1 s
  wait, settled-worker refresh and close cancellation. B5 invokes the callback
  without SQL transaction/catalog locks the GUI may require.
  After ACK/failure reread phase, reacquire/revalidate identity, ownership and
  inventory; per R52/f09c0fc retain applicable native and canonical leases from
  before durable barrier through cleanup. Pre-ACK preflight is not acceptance,
  and ownership appearing during ACK must be caught before a new barrier.
- Never invoke private _derived_row_purgers as a resolver. Trusted SQL purgers run
  on the same final cleanup connection/transaction before relationships/core/FTS
  removal, with no own transaction/commit/I/O/provider; failure rolls back together.
- Composition/CLI/retry use the same injected MemoryService and explicit
  configured/StartRequest root; register request output_dir before worker launch.
  CLI destructive operations use existing settings_operation/get_settings root.
  No second default service, core settings/.env reader, durable root registry or
  origin-derived probing. Bare destructive context fails closed with LibraryError/
  io4; a valid barrier remains pending/hidden, while reads/non-destructive repair
  continue. Merged B5 revalidates declared roots and lock roles after ACK.

The candidate carries the reviewed R53 facade:
`SearchService(connection_factory, *, reindexer=None).reindex()` calls the same
injected MemoryService callback once, returns its identical R51 report and propagates
canonical errors unchanged. Constructor performs no maintenance; reindex owns no
factory/connection/snapshot/lock/probe/migration and does not retry or recount.
Missing callback fails only reindex with existing LibraryError, preserving reads.
Consume ApplicationUiAdapter's explicit query override and separate optional typed
maintenance capability; do not expand required read protocols or create another
backend, indexer, MemoryService or root. Reindex delegates to merged B5 and does
not call the deletion ACK. Verify unchanged playback and identical report/error
propagation with the real delegate.

## Review focus

- `setPosition` called before loaded or seekable, or applied to the wrong media
  after a fast meeting switch.
- A citation whose revision is historical or whose audio is missing.
- Playback enabled while a recording is active.
- Any path that triggers play without a click.
- Tracked-worker deletion ACK returns only after Qt stop/unload/invalidation.
  A late load or seek cannot revive deleted media. Timeout or teardown remains
  phase-aware and replacement reads wait for worker settlement. Reindex leaves
  current playback intact.
- Exit codes and JSON shape drifting between commands.
- Frozen bundle missing QtMultimedia plugins (coordinate with `packaging/windows.spec`).

### Task 3: Playback adapter and evidence navigation

**Files:** `src/ui/playback.py` (new), `src/ui/library.py` (shared with A),
`src/ui/main_window.py`, `src/ui/contract.py`, `src/ui/fake_controller.py`,
`src/memory/service.py` (shared with M5 B, add `resolve_citation`),
`src/memory/models.py` (consume existing `Citation` and `CitationTarget`),
`packaging/windows.spec` (QtMultimedia
plugins); tests in `tests/test_memory_citation.py`, `tests/test_ui_playback.py`,
`tests/test_ui_library.py`, `tests/test_ui_delete_ack.py`,
`tests/test_distribution_smoke.py` (plugin presence). The ACK composition in
`src/ui/app.py` and ticket bridge in `src/ui/bridge.py` are existing interfaces.
Root grants their leases if wiring needs edits there.

**Planned B interfaces (bind to actual accepted A DTOs before execution):**
`MemoryService.resolve_citation(citation: Citation) -> CitationTarget`;
`PlaybackController(parent)` with `load(target: CitationTarget)`, `play()`, `pause()`,
`seek_ms(int)`, `set_volume(float)`, `set_rate(float)`, `stop_and_unload()`,
signals `position_changed(int)`, `duration_changed(int)`, `state_changed(str)`,
`error(str)`; planned `VisualController.resolve_citation` and
`LibraryView.open_target(target)`. Only the B resolver can supply playable audio.
The A segment signal provides reference IDs, and the A worker generations gate
the resolver response. A stale load or seek from before deletion must stay
invalidated after the worker ACK.

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
`play`), `src/memory/search.py` and `src/memory/models.py` (the `--meeting`
scope after the M3 A freeze and root's lease); tests in
`tests/test_cli_meetings.py`, `tests/test_cli.py` and
`tests/test_memory_search.py` for search scope. Application retry adapters in
`src/application/controller.py`, `src/application/processing.py` and
`src/application/recovery.py` are consumed as existing interfaces. Any needed
source change for the structured retry outcome requires root's explicit lease.

**Interfaces:** Typer commands `list`, `show`, `search`, `rename`, `participant`,
`retry`, `play`, `import-session`, `reindex`, `delete` under `here meetings`; a shared
`emit(result, json: bool)` formatter; an `ExitCode` enum `0/2/3/4/5` mapped from the
typed service errors (`InvalidInput`→2, `NotFound`/`Conflict`→3,
`IntegrityFailure`/`Busy`/I/O→4). Exit 5 is reserved for a structured provider
failure from explicit retry. Resolve that binding before implementation because
the candidate has no canonical `ProviderUnavailable`. Reuse the merged
`memory` service mappings and preserve its JSON shape. For `meetings`, carry
the canonical `InvalidInput.reason` when reporting search errors.

The existing pass-through/error tests consume R39/R41/R45 without another query
implementation: preserve opaque cursor/filter/backend/generation values, map
structured corpus_changed to InvalidInput/exit 2, keep ordinary listing insert
stability and retain unknown dates as null. Cover `--meeting` across all routes,
including an empty search, a page where other meetings would fill LIMIT and a
cursor reused with a different scope. Exercise real B4/B5 adapters. Retry tests
bind a guarded meeting ID to one eligible configured recovery candidate, exercise
absent, ambiguous or suppressed candidates and confirm lookup cannot replay an
unrelated session. Distinguish provider failure from catalog, I/O and ownership
outcomes at the application boundary.

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
- Open one PR against `main` with the two task commits, evidence, self-review and
  independent agent review in the description. Ruling 64 omits the Copilot step.
  Root verifies Windows CI on the final head, creates the merge commit with
  `--admin --match-head-commit` and records it in `DELIVERY.md` under CLAUDE §7.
  Real-audio certification remains required.
  Then reconcile the already-prepared `docs/MILESTONE_4_PLAN.md` and first M4
  SPEC/PLAN against actual merged M3 contracts under ruling 14; do not restart
  existing preparation or create another front without root dispatch.
