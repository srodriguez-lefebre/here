# Architecture

## Overview

`here` records audio, transcribes it, and writes a structured session artifact
folder.

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
- `application/`: stable states, events, snapshots, telemetry, and commands shared
  by every interface.
- `ui/`: Qt Widgets presentation, the live-logo overlay, and the single adapter
  from visual actions to the application contract.

## Windows Presentation Layer

The Windows interface and live logo run in the same process as the application
controller. `ApplicationEventBridge` is the only asynchronous entry into Qt: it
marshals application callbacks onto the UI event loop, then distributes immutable
snapshots and the reduced combined audio level. Widgets never open an audio
device, parse logs, inspect session files, or invoke the CLI.

`ApplicationUiAdapter` is the narrow integration seam. It translates UI intent
into the stable application commands, including a combined-source `StartRequest`.
The main window and overlay depend on that seam, so controller implementation
changes do not leak into painting or interaction code. A fake implementation of
the real contract supports deterministic preview and headless tests.

The overlay is a transparent, frameless, always-on-top Qt tool window. Each new
active session positions it in the lower-right of the current screen's available
area with a 24-pixel margin; dragging changes only that instance and is never
saved. Its 104-pixel surface redraws at roughly 30 frames per second, independently
from capture callbacks. Audio updates are reduced to one normalized level and are
ignored outside the recording state.

Visual state decisions are intentionally explicit:

- recording uses a reactive perimeter in the configured accent color;
- pause is a static ring with a pause mark;
- preparation, stopping, and processing use an indeterminate spiral in that color;
- success and failure use fixed green-check and red-cross marks for three seconds;
- cancellation uses a neutral gray minus mark for 1.8 seconds, never a success or
  error symbol.

The accent preference is persisted through `QSettings`; overlay position is not.
Closing the main window while `snapshot.has_active_work` is true hides it while
the overlay keeps the process reachable. Closing while idle requests full process
exit. If work ends while the window is hidden, the process exits after the
terminal indicator finishes.

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
