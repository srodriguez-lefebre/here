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
5. If live processing fails, the CLI falls back to offline session
   transcription.
6. Output writing creates `transcript.txt`, `transcript.md`, `session.json`,
   and `chunks.json`.

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

`session.json` stores global session metadata. `chunks.json` stores chunk-level
metadata only, without audio, transcript text, or duplicated global fields.

This split keeps artifacts useful for debugging now and ready for a later local
search/RAG layer.

## Diagnostics

Windows diagnostics are intentionally separate from transcription. They inspect
audio devices and measure signal without needing OpenAI credentials.
