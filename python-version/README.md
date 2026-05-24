# here

`here` is a CLI for recording microphone and/or system audio, transcribing the
recording with OpenAI, and saving the result as session artifacts.

The current Python version focuses on long recordings. It can process audio in
live chunks while recording continues, then falls back to offline transcription
if the live pipeline fails.

## Requirements

- Python 3.12.6 or later
- An OpenAI API key
- Windows for microphone + system-audio capture and audio diagnostics

## Installation

From `python-version/`:

```powershell
pip install -e .
```

This installs two entrypoints:

```powershell
here
record
```

## Configuration

Create `python-version/.env`:

```env
OPENAI_API_KEY=sk-...
TRANSCRIPTIONS_DIR=transcriptions
TRANSCRIPTION_MODEL=gpt-4o-transcribe-diarize
ALT_TRANSCRIPTION_MODEL=gpt-4o-transcribe
CLEANUP_MODEL=gpt-4.1-mini
CLEANUP_ENABLED=false
```

## Recording Commands

```powershell
record
record alt
record mic
record mic alt
record os
record os alt
here trans path\to\audio.wav
```

- `record` captures microphone + system audio together.
- `record mic` captures microphone audio only.
- `record os` captures system audio only.
- `alt` variants use `ALT_TRANSCRIPTION_MODEL`.
- `here trans {file}` transcribes an existing audio file without recording.

## Output Artifacts

Each successful recording creates a session folder:

```text
transcriptions/YYYYMMDD_HHMMSS/
  audio.wav
  transcript.txt
  transcript.md
  session.json
  chunks.json
```

If transcription fails after recording, `here` still creates a recoverable
session folder:

```text
transcriptions/YYYYMMDD_HHMMSS/
  audio.wav
  session.json
  chunks.json
  errors.json
```

`audio.wav` is the normalized/mixed audio that can be retried later.

### `transcript.txt`

The plain final transcript, written with UTF-8 BOM for broad editor
compatibility.

### `transcript.md`

A human-readable Markdown version with:

- session details
- audio sources
- processing flags
- transcript text

It does not generate summaries or action items.

### `session.json`

Pydantic-validated session metadata:

- schema version
- session id
- recording start/completion timestamps
- duration
- status: `pending`, `completed`, or `failed` (`pending` is reserved for transitional states)
- recoverable audio path
- source labels, device names, sample rates, channels, frames, and durations
- transcription and cleanup model names
- alternate model flag
- live pipeline and fallback flags
- generated artifact names

Temporary audio paths, raw audio, and secrets are not persisted.

### `errors.json`

Failed sessions include structured errors:

- stage
- error type
- message
- root cause type/message when the error was wrapped
- whether the error is retryable
- timestamp

### `chunks.json`

Pydantic-validated chunk metadata for debugging and future indexing.

Each chunk records:

- chunk index
- mode: `live` or `offline`
- start/end/duration seconds
- source count
- transcription start/finish timestamps
- status: `completed` or `failed`
- error text when a chunk fails

It intentionally does not include audio bytes, transcript text, API keys, or
global session metadata already present in `session.json`.

## Audio Diagnostics

Diagnostics are currently Windows-only and do not require an OpenAI API key.

```powershell
here devices
here test mic
here test os
```

- `here devices` shows the default microphone and WASAPI loopback devices.
- `here test mic` measures microphone signal for a few seconds.
- `here test os` measures system-audio loopback signal for a few seconds.

Optional duration:

```powershell
here test mic --duration 5
here test os --duration 5
```

The signal tests report peak, RMS, and whether signal was detected.

## Retrying Or Transcribing Existing Audio

Use `here trans` with any local audio file:

```powershell
here trans .\meeting.wav
```

If the file lives inside a failed session folder, such as:

```text
transcriptions/20260524_153000/audio.wav
```

then `here trans` updates that session in place. On success it writes
`transcript.txt` and `transcript.md`, marks `session.json` as `completed`, and
removes `errors.json`.

For an external audio file, `here trans` creates a new session folder and stores
a normalized `audio.wav` alongside the generated transcript artifacts.

## Development

Run the test suite from `python-version/`:

```powershell
python -m pytest
```
