# Architecture

## Overview

`here` records audio, transcribes it, and writes a structured session artifact
folder.

The main flow is:

1. An interface creates the application controller and starts a recording.
2. Recording backend captures microphone, system audio, or both.
3. Captured blocks are sent to the live transcription pipeline.
4. Live processing cuts chunks near silence, normalizes/mixes audio, and
   transcribes chunks in the background.
5. Once recording ends, output creates a recoverable session folder and
   materializes normalized/mixed `audio.wav`.
6. If live processing fails, the application layer falls back to offline session
   transcription using the recoverable audio.
7. Output writing creates completed or failed artifacts.

## Main Modules

- `application/`: interface-independent lifecycle, commands, events, processing,
  recovery, and the production composition root.
- `cli.py`: secondary input adapter. It delegates recording lifecycle to the
  application controller.
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

`events.json` is written when lifecycle events exist. Pause and resume events
contain both wall-clock time and the duration of material actually recorded.
`session.json.total_paused_seconds` records the accumulated wall-clock pause;
no silence is inserted into `audio.wav`.

This split keeps artifacts useful for debugging now and ready for a later local
search/RAG layer.

## Existing Audio Transcription

`here trans {file}` builds the same recoverable session shape from an existing
audio file. When `{file}` is already `audio.wav` inside a failed session folder,
the command updates that session in place.

## Diagnostics

Windows diagnostics are intentionally separate from transcription. They inspect
audio devices and measure signal without needing OpenAI credentials.

## Application contract

GUI and CLI code import `create_default_controller` from `here.application`.
The returned controller is asynchronous: `start()` returns after scheduling
preparation, state changes arrive through `subscribe()`, and capture or
processing never runs on the caller's UI thread.

The stable states are `idle`, `preparing`, `recording`, `paused`, `stopping`,
`processing`, `completed`, `failed`, and `cancelled`. Commands are `start`,
`pause`, `resume`, `stop`, `cancel`, and `retry`. Audio-level events expose only
bounded peak/RMS aggregates for the combined signal at no more than 15 updates
per second; raw audio never crosses the UI contract.

Cancellation semantics depend on the current state:

- cancelling while preparing, recording, or paused deletes raw temporary audio
  and produces no session;
- cancelling while stopping or processing preserves normalized `audio.wav`,
  writes a recoverable session with status `cancelled`, and permits `retry()`;
- an already-issued provider request may finish before the worker observes the
  cancellation flag, but no later retry or processing stage is started.

Only one job can be active in a controller. A fresh job can start from any
terminal state. Listener callbacks run on worker threads, so the Qt adapter must
bridge them to queued Qt signals before touching widgets.

## Programmatic Windows capture

`recording.start_recording()` opens selected or default Windows devices and
returns a controllable handle. Pausing drains device buffers without writing
frames, preventing both stale audio and synthetic silence on resume. Stopping
returns a `RecordingSession`; cancelling removes all temporary sources.

The first implementation remains a single process and supports Windows capture
only. Device and long-duration validation still require the real-hardware matrix
described in the milestone plan; unit tests use deterministic adapters and do
not claim hardware certification.

## Decisions made during implementation

- Application events are immutable values and listener failure is isolated so
  a UI subscriber cannot stop capture.
- Peak and RMS are combined using the maximum current source value. This favors
  an unmistakable recording indicator over a meter intended for mastering.
- Offline provider failures receive three total attempts with incremental waits
  of 0.5 and 1 second; the terminal failure is appended to recoverable metadata.
- Session metadata stays schema version 1 because all new fields are additive
  and have backward-compatible defaults. `events.json` has its own versioned
  envelope.
- Device identifiers are optional in `StartRequest`; absence means the current
  Windows default. Selection and remembered preferences remain interface and
  configuration concerns.
