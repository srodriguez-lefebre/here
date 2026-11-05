# M3 playback, evidence navigation and CLI implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The user merges (ruling 4). Steps use checkbox
> syntax. **Status: not in execution (ruling 15).** Start only on an explicit user
> instruction, after M3 group A is merged into `main`.

**Goal:** Jump from any result or citation to the original segment and, with real
timing, to the audio at that millisecond; expose the same operations through
`here meetings` with stable JSON and exit codes.
**Architecture:** `MemoryService.resolve_citation` turns a `Citation` into a
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

**Interfaces:** `MemoryService.resolve_citation(citation: Citation) -> CitationTarget`;
`PlaybackController(parent)` with `load(target: CitationTarget)`, `play()`, `pause()`,
`seek_ms(int)`, `set_volume(float)`, `set_rate(float)`, `stop_and_unload()`,
signals `position_changed(int)`, `duration_changed(int)`, `state_changed(str)`,
`error(str)`; `VisualController.resolve_citation`; `LibraryView.open_target(target)`.

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
`IntegrityFailure`/I/O→4, `ProviderUnavailable`→5).

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
- Open one PR with the two task commits, self-review and independent agent review in
  the description, CI green. The user merges. Then, per ruling 14, write
  `docs/MILESTONE_4_PLAN.md` and the first M4 SPEC/PLAN if they do not exist yet.
