# here

`here` is a Python CLI for recording microphone and system audio, transcribing
the recording with OpenAI, and saving the result as a set of recoverable session
artifacts.

The project currently focuses on reliable transcription of long recordings.
Audio can be processed in chunks while recording continues, with an automatic
offline fallback if the live transcription pipeline fails.

## Requirements

- Windows for combined microphone and system-audio capture and diagnostics
- An OpenAI API key when transcription is requested; opening the desktop and
  local recovery does not require one
- Python 3.12.6 or later for source/wheel development; the Windows installer
  includes its own Python runtime

## Installation

For the Windows 11 x64 packaged application, run the generated
`here-0.2.0-windows-x64-setup.exe`. It installs for the current user under
`%LOCALAPPDATA%\Programs\here` and creates Start entries for here and its
uninstaller. Exit here before reinstalling or updating: the installer refuses
to replace a running application. Uninstalling preserves `%LOCALAPPDATA%\here`
and any configured data directory. It never enables automatic startup or recording.

The installer and portable onedir ZIP are build/CI artifacts; no public release
download or Authenticode application signature is claimed. See
[Windows build instructions](docs/development/WINDOWS_DISTRIBUTION.md).

For source development, from the repository root:

From the repository root:

```powershell
pip install -e .
```

This installs the current CLI commands:

```powershell
here
record
here-gui
```

`here-gui` starts the Qt interface with the real application controller. Its
visual layer can also be exercised without audio hardware or provider calls with:

```powershell
here-gui-preview
```

The preview uses synthetic state and level events and is deliberately separate
from the production entry point.

## Configuration

For an installed application, place configuration in
`%LOCALAPPDATA%\here\.env`, or select one file explicitly with `HERE_ENV_FILE`.
For source development, `.env.example` may be copied to a repository `.env`.
Provide an API key only when requesting transcription:

```env
OPENAI_API_KEY=sk-...
```

The example file also contains optional settings for output directories,
transcription models, and transcript cleanup.
`HERE_DATA_DIR` overrides the user data root; sessions default to its `sessions`
directory. Frozen builds do not read a development `.env`. `HERE_SETTINGS_FILE`
optionally selects an INI file for presentation preferences, including isolated
portable/test launches; ordinary launches retain the existing per-user preferences.

## Usage

```powershell
record
record alt
record mic
record mic alt
record os
record os alt
here trans path\to\audio.wav
here-gui
```

- `record` captures microphone and system audio together.
- `record mic` captures microphone audio only.
- `record os` captures system audio only.
- `alt` uses the configured alternative transcription model.
- `here trans` transcribes an existing audio file.
- `here-gui` opens the Windows application and its live recording overlay.

## Session Artifacts

Each recording creates a timestamped folder inside `transcriptions/`:

```text
transcriptions/YYYYMMDD_HHMMSS/
  audio.wav
  raw_transcript.txt
  transcript.txt
  transcript.md
  session.json
  chunks.json
  segments.json
```

If transcription fails, the audio and diagnostic metadata remain available so
the session can be retried without recording it again. Failed sessions also
include an `errors.json` file.

Successful live, offline, fallback, file transcription and retry paths preserve
`raw_transcript.txt` with the transcription text before optional cleanup.
`transcript.txt` and `transcript.md` contain the display text, which may be cleaned.
Keeping both lets you inspect the source text when cleanup changes the wording.

An unexpected Windows capture failure also preserves finalized audio and a failed
session for retry. If mixing fails, source WAV copies remain in that session.
Explicit recording cancellation deletes captured audio and creates no session.
Live handoff queues hold at most 256 audio blocks and eight transcription jobs;
exceeding either budget triggers offline transcription of the saved audio.

`segments.json` stores versioned provider evidence: original segment text,
nullable timing/speaker values, and request scope for chunk-local speaker labels.
Text-only responses retain an untimed segment with no inferred speaker.
The optional `timing_source` field distinguishes `provider`, `imported_explicit`
and `absent`; new provider evidence uses `absent` when no valid bound remains.
Contradictory bounds become missing, and a wall-clock pause never shifts a recorded
audio position. Historical segment documents without this field remain readable.
`segments.json` retains overlapping chunk evidence separately from the merged or cleaned
display transcript. `session.json.capture_sources` preserves original device names,
sample rates, channels, and frame counts after mixing.

Application-driven recordings can also include `events.json`, which records
pause/resume and processing lifecycle events without storing raw audio blocks.

The project produces transcripts and processing metadata. It does not currently
generate summaries or action items.

## Local catalog (M2 in progress)

The source candidate implements cataloging of recordings, file transcriptions
and explicit retries under the configured user data root. It keeps managed copies
and immutable transcript revisions; importing a saved session leaves the original
folder unchanged. Repeated import of the same origin is unchanged. Conflicting
content with the same meeting UUID is rejected; an explicit application retry
is intended to create the next revision while retaining earlier evidence.

These maintenance commands work without a provider key:

```powershell
$result = here memory ingest path\to\saved-session --json | ConvertFrom-Json
here memory show $result.meeting_id --json
here memory reconcile --json
```

`ingest` reports `status`, `meeting_id` and warnings. `show` returns the meeting,
audio assets, revisions and segments; `reconcile` reports local recovery warnings.
The intended exit codes are 0 for success, 2 for invalid input, 3 for a rejected conflict or
missing meeting, and 4 for an unavailable/corrupt catalog. `--json` keeps the result
machine-readable. Local catalog commands never transcribe or contact the provider.

The data root contains `library.sqlite3`, `meetings/<uuid>/revisions/`, managed
backups and staging material. Keep these together for recovery. Completed revision
receipts and later state snapshots allow reconstruction from managed evidence;
uncertain publications are retained with a warning. Unknown source files are not
imported merely because they share a session folder.

If only catalog publication fails, saved audio and text remain usable. The desktop
shows a separate library-pending warning, and later discovery retries local catalog
work without repeating transcription. A failed provider retry keeps the previous
complete revision active. Ordinary application retry still requires an explicit
action and the provider credential.

M2 is not released or complete. Candidate `6c422f0` passed 883 local tests with
nine explicit skips, but independent review requests seven corrections in audio
association, retry identity/state, ownership/recovery and result/error reporting.
Those contracts are not approved yet. Fix verification, re-review and current-head
hosted CI/Copilot gates are tracked in [STATUS.md](STATUS.md).
The packaged 0.2.0 delivery above retains its recorded M1 scope.

## Audio Diagnostics

Windows audio devices can be inspected without using an API key:

```powershell
here devices
here test mic
here test os
```

## Development

Run the test suite from the repository root:

```powershell
uv sync --dev
uv run pytest
```

Qt tests run without a display server:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest tests/test_live_logo_overlay.py tests/test_ui_lifecycle.py
```

The Windows GUI integrates directly through
`here.application.create_default_controller()`; it never launches the CLI as a
subprocess. See the application-contract section in
[ARCHITECTURE.md](ARCHITECTURE.md#application-contract).

See [ARCHITECTURE.md](ARCHITECTURE.md) for an overview of the internal flow and
main modules.
