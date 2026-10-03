# Capture durability specification

## Intent

Complete the reliability foundation required by milestone 1. A recoverable capture
failure must preserve useful audio; a slow live transcription worker must not exhaust
memory or block the capture writer; model-provided timing and speaker data must survive
serialization. Successful and destructive-cancellation behavior remain compatible.

## Decisions

Capture writes its primary WAV before offering the block to optional live processing.
Bound the capture handoff to a finite queue (256 blocks) and live chunk work to eight
jobs. When pressure exceeds either budget, disable the live path explicitly and fall
back to offline transcription from saved audio. No unbounded retry queue and no silent
text success after missing live chunks. Terminal shutdown cannot hang on a full queue.

A capture exception with material includes a typed partial RecordingSession and
original cause. Writers must be closed before recovering. Controller failures persist
audio and error metadata without trying to claim a completed transcript, expose a
recoverable session directory and permit retry. A true user cancellation still deletes
all raw data and leaves no session. Opening failure before any audio exists remains a
clear nonrecoverable error.

TranscriptionResult must carry the final structured segments, separate from cleaned
display text. Persist a versioned `segments.json` containing original text, nullable
start/end and nullable speaker. Do not infer global human identities from provider
chunk-local speaker labels. Preserve capture source/device provenance through mixing.
Old callers without segments remain supported and old session metadata stays readable.

## Acceptance

- Reader failure after frames were written yields recoverable audio and failed metadata.
- Retry can operate on the resulting audio without recording again.
- Explicit cancel after frames were written removes those frames.
- Both bounded queues fail live processing deterministically under pressure, primary
  capture continues, offline fallback receives complete primary audio.
- Stop, abort and cleanup terminate when either queue was full.
- Segments from live and offline transcription retain nullable timing/speaker values
  and are serialized independently of optional transcript cleanup.
- Original device names and raw-source properties remain in session provenance.
- Existing capture, processing, merge, persistence, CLI and UI tests continue to pass.

## Deferred to the next M1 tasks

Persistent in-progress journal/restart recovery, default device display and signal
diagnostic UI, per-user data paths, graceful explicit exit, Windows installation and
real-hardware long-duration acceptance. These are not declared completed by this task.
