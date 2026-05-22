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
```

- `record` captures microphone + system audio together.
- `record mic` captures microphone audio only.
- `record os` captures system audio only.
- `alt` variants use `ALT_TRANSCRIPTION_MODEL`.

## Output Artifacts

Each successful recording creates a session folder:

```text
transcriptions/YYYYMMDD_HHMMSS/
  transcript.txt
  transcript.md
  session.json
  chunks.json
```

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
- source labels, device names, sample rates, channels, frames, and durations
- transcription and cleanup model names
- alternate model flag
- live pipeline and fallback flags
- generated artifact names

Temporary audio paths, raw audio, and secrets are not persisted.

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

## Development

Run the test suite from `python-version/`:

```powershell
python -m pytest
```
