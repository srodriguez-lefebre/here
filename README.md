# here

`here` is a Python CLI for recording microphone and system audio, transcribing
the recording with OpenAI, and saving the result as a set of recoverable session
artifacts.

The project currently focuses on reliable transcription of long recordings.
Audio can be processed in chunks while recording continues, with an automatic
offline fallback if the live transcription pipeline fails.

## Requirements

- Python 3.12.6 or later
- An OpenAI API key
- Windows for combined microphone and system-audio capture and diagnostics

## Installation

From the repository root:

```powershell
pip install -e .
```

This installs two commands:

```powershell
here
record
```

## Configuration

Copy `.env.example` to `.env` and provide your API key:

```env
OPENAI_API_KEY=sk-...
```

The example file also contains optional settings for output directories,
transcription models, and transcript cleanup.

## Usage

```powershell
record
record alt
record mic
record mic alt
record os
record os alt
here trans path\to\audio.wav
```

- `record` captures microphone and system audio together.
- `record mic` captures microphone audio only.
- `record os` captures system audio only.
- `alt` uses the configured alternative transcription model.
- `here trans` transcribes an existing audio file.

## Session Artifacts

Each recording creates a timestamped folder inside `transcriptions/`:

```text
transcriptions/YYYYMMDD_HHMMSS/
  audio.wav
  transcript.txt
  transcript.md
  session.json
  chunks.json
```

If transcription fails, the audio and diagnostic metadata remain available so
the session can be retried without recording it again. Failed sessions also
include an `errors.json` file.

The project produces transcripts and processing metadata. It does not currently
generate summaries or action items.

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
python -m pytest
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for an overview of the internal flow and
main modules.
