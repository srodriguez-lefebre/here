# Python Version Architecture

## Overview

`python-version` is the active implementation of `here`. It records audio,
transcribes it, and writes a structured session artifact folder.

The main flow is:

1. CLI command starts a recording.
2. Recording backend captures microphone, system audio, or both.
3. Captured blocks are sent to the live transcription pipeline.
4. Live processing cuts chunks near silence, normalizes/mixes audio, and
   transcribes chunks in the background.
5. Once recording ends, output creates a recoverable session folder and
   materializes normalized/mixed `audio.wav`.
6. If live processing fails, the CLI falls back to offline session
   transcription using the recoverable audio.
7. Output writing creates completed or failed artifacts.

## Main Modules

- `cli.py`: command entrypoint and orchestration.
- `recording/`: platform-specific capture and Windows diagnostics.
- `live_processing.py`: live chunk generation, transcription workers, and
  live chunk metadata.
- `audio/`: mixing, chunk planning, silence boundary selection, and text merge.
- `transcription/`: OpenAI client, offline transcription service, and segment
  timeline merge.
- `output/`: Pydantic metadata models, Markdown rendering, and session artifact
  writing.

## Output Layer

The output layer owns persisted artifacts.

`session.json` stores global session metadata, including the recording window
and the concrete device names captured for each source when the backend exposes
them. `chunks.json` stores chunk-level metadata only, without audio, transcript
text, or duplicated global fields.

Failed sessions are still persisted. They include `audio.wav`, `session.json`,
`chunks.json`, and `errors.json`, so API/provider failures can be retried without
re-recording.

This split keeps artifacts useful for debugging now and ready for a later local
search/RAG layer.

## Existing Audio Transcription

`here trans {file}` builds the same recoverable session shape from an existing
audio file. When `{file}` is already `audio.wav` inside a failed session folder,
the command updates that session in place.

## Diagnostics

Windows diagnostics are intentionally separate from transcription. They inspect
audio devices and measure signal without needing OpenAI credentials.
