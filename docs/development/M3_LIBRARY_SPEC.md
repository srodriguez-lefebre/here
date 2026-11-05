# M3 playback, evidence navigation and CLI specification

Status: written 2026-11-05, not in execution (ruling 15 in `DELIVERY.md`).
Implementation starts only on an explicit user instruction, after M3 group A is
merged. Governing design: `PRODUCT_DESIGN.md` §3 and §5, a taken decision (ruling 3).
Spanish plan and acceptance criteria: `../MILESTONE_3_PLAN.md`.

## Intent

Take a person from a list item, a search hit or, later, a citation to the exact
original moment: the segment in context and, when the segment has real timing, the
audio at that millisecond. Add the `here meetings` commands so the CLI offers the
same operations with stable JSON. Playback must never pretend precision it does not
have, never recapture itself through loopback, and never start on its own.

## Decisions

### Common citation target

`CitationTarget` (declared in group A, `src/memory/models.py`) carries meeting title
and date, source text, visible speaker, optional `start_ms`/`end_ms`, an optional
validated local audio path and a state `current`, `historical` or `missing`.
`MemoryService.resolve_citation(Citation) -> CitationTarget` resolves it from the
database only: titles, dates, paths and times never come from free text. List,
search and M4 citations all produce a `Citation` and open the same detail through
the same resolver.

### Evidence navigation

Opening a target selects the segment, centers it in the detail pane and shows the
previous and next segments as context. With timing present the detail offers
"Reproducir desde mm:ss" built from `start_ms`; with `timing_source=absent` it shows
"Sin marca de tiempo" and no play-from-zero control. A `historical` target shows the
revision badge and a link to the current revision; a `missing` target shows what is
missing (audio file or revision) without a broken control.

### Playback adapter

`src/ui/playback.py` wraps `QMediaPlayer` plus `QAudioOutput` behind a small
`PlaybackController` the UI drives with DTOs. It accepts only the validated local
path from the target. It waits for `mediaStatus` loaded and `isSeekable()` before
`setPosition(start_ms)`; a seek requested earlier is queued once and applied when
seekable. Controls: play/pause, position slider, current and total time, volume,
speed from a fixed moderate set. Changing meeting stops and unloads the previous
media. Codec and missing-file errors are shown as text in the detail without
affecting list or search. While the application snapshot reports recording, paused
or preparing, playback controls are disabled with the reason visible, to avoid
loopback recapture. The controller exposes no autoplay entry point; group A's
contract and M4 may only select a target, never start playback.

### CLI

`src/cli.py` gains a `meetings` Typer group:

```text
here meetings list [--from ISO] [--to ISO] [--status ...] [--participant ID] [--json]
here meetings show ID [--json]
here meetings search "consulta" [--meeting ID] [--json]
here meetings rename ID "Título"
here meetings participant ID "Nombre" [--speaker KEY]
here meetings retry ID
here meetings play ID [--segment SEGMENT_ID]
here meetings import-session PATH
here meetings reindex
here meetings delete ID --yes
```

Every command calls `MemoryService` or `SearchService`; the CLI implements no search
or persistence of its own. Exit codes: 0 success, 2 invalid input, 3 not found or
conflict, 4 integrity or I/O failure, 5 provider (only `retry` can return it).
`--json` emits full IDs and ISO UTC dates; abbreviations are accepted only when the
resolution is unambiguous, otherwise exit 2 with the candidates. `delete` without
`--yes` exits 2 and prints what would be removed. `play` opens a minimal Qt window
with the same `PlaybackController`; when Qt multimedia is unavailable it exits with
an explanatory error and, with `--json`, prints the resolved target without claiming
playback.

### Real-audio verification

Seek accuracy is certified on a real WAV with audible markers at known offsets:
request a seek to each marker and measure the audible position within 250 ms. The
run, machine and WAV hash are recorded in `ACCEPTANCE.md`. A fake player proves the
contract, not the criterion.

## Acceptance

- From a list item, a search hit and a synthetic citation, the same resolver returns
  the same `CitationTarget` and the same detail opens with the segment centered and
  its neighbors visible.
- Segment with timing: "Reproducir desde mm:ss" appears and the controller requests
  `setPosition(start_ms)` only after loaded and seekable; segment without timing:
  "Sin marca de tiempo" and no playback-from-zero control.
- Historical target shows the badge and the link; missing target names what is
  missing and renders no broken control.
- Changing meeting stops and unloads the previous media; a missing file and a bad
  codec each show a text error while search and list keep working.
- With the snapshot in recording, paused or preparing, controls are disabled with the
  reason; after completion they are enabled again.
- No code path starts playback without a user click; a selection event from a
  citation does not play.
- Real WAV with markers: every seek lands within 250 ms, recorded in `ACCEPTANCE.md`.
- Each `here meetings` command returns the documented exit code for success, invalid
  input, not found, conflict, integrity failure and, for `retry`, provider failure;
  `--json` carries full IDs and ISO UTC dates; ambiguous abbreviations exit 2 with
  candidates; `delete` without `--yes` changes nothing.
- `play` without multimedia exits with an explanatory error and prints the target in
  JSON mode.
- Keyboard operation of the player (play/pause, seek step, volume) and accessible
  names on every control; contrast in both themes.
- Full Windows CI suite passes offscreen with a synthetic key and temporary roots;
  the real-audio run is a separate recorded observation.

## Boundaries

No questions, answers or extractions (M4); no export, import, settings or tray (M5);
no transcript text editing; no autoplay; no audio formats beyond the WAV M1 writes;
no provider call except the existing `retry` path through `MemoryService`.

## Technical references

- [QMediaPlayer for PySide6](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QMediaPlayer.html):
  media status, seekability and `setPosition`.
- [QAudioOutput](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QAudioOutput.html):
  volume and device output.
- [Typer](https://typer.tiangolo.com/): command groups and exit codes, as already used
  in `src/cli.py`.
