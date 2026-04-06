# here MVP

`here` is a small CLI for recording audio and generating a text transcription with OpenAI.

This MVP is intentionally narrow in scope:

- `record` captures microphone + system audio together and saves a single transcript
- `record mic` captures microphone only
- `record os` captures system audio only
- long recordings are chunked automatically before transcription

The main goal is to validate a simple meeting capture workflow with the shortest possible path from audio to text.

## Requirements

- Python 3.12.6 or later
- An OpenAI API key
- Windows for combined `record` capture (`mic + os`)

## Installation

Install the project in your local environment so the CLI entrypoint is available:

```bash
pip install -e .
```

After installation, the commands below are available from the `MVP/` directory:

```bash
record
record mic
record os
```

## Configuration

Create `MVP/.env` with at least:

```env
OPENAI_API_KEY=sk-...
```

Optional settings:

```env
TRANSCRIPTIONS_DIR=transcriptions
PULSE_SERVER=
TRANSCRIPTION_MODEL=gpt-4o-transcribe-diarize
CLEANUP_MODEL=gpt-4.1-mini
CLEANUP_ENABLED=false
```

### Settings

- `OPENAI_API_KEY`: Required. OpenAI API key used for transcription.
- `TRANSCRIPTIONS_DIR`: Output directory for generated `.txt` files. Defaults to `MVP/transcriptions`.
- `PULSE_SERVER`: Optional PulseAudio server used for Linux/WSL microphone or system-audio capture.
- `TRANSCRIPTION_MODEL`: Transcription model. Defaults to `gpt-4o-transcribe-diarize`.
- `CLEANUP_MODEL`: Reserved for optional post-processing. Defaults to `gpt-4.1-mini`.
- `CLEANUP_ENABLED`: Enables cleanup after transcription. Defaults to `false`.

## Command Reference

### `record`

Records microphone + system audio, transcribes the mixed audio, and writes a timestamped `.txt` file.

```bash
record
```

Optional output directory:

```bash
record --output-dir custom_transcriptions
```

### `record mic`

Records microphone input only.

```bash
record mic
```

### `record os`

Records system audio only.

```bash
record os
```

## Output

Each run generates a UTF-8 with BOM (`utf-8-sig`) text file in the configured transcription directory:

```text
transcriptions/YYYYMMDD_HHMMSS.txt
```

Temporary WAV files are created during recording/transcription and deleted automatically after the transcript is written.

Long recordings are chunked automatically during processing so the final transcript can exceed a single-upload audio limit while still producing one final `.txt` output.

## Platform Notes

- `record` currently supports combined microphone + system audio capture on Windows only.
- `record os` on Windows uses WASAPI loopback.
- `record os` on Linux/WSL depends on PulseAudio monitor sources and local audio configuration.
- `record mic` works on both Windows and Linux/WSL, but the underlying audio backend differs by platform.

