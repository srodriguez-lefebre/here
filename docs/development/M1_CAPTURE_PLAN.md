# Capture durability implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use
> checkbox syntax. User explicitly delegated decisions and execution.

**Goal:** Preserve useful audio and structured transcript evidence under failure and load.
**Architecture:** Primary disk capture is authoritative; optional live work has finite
queues and falls back to offline. Typed capture failure carries partial material into
application persistence. Segment artifacts preserve provider facts.
**Tech Stack:** Python, NumPy, soundfile, PyAudioWPatch, pytest, Pydantic.
**Spec:** `docs/development/M1_CAPTURE_SPEC.md`.

## Global Constraints

- Existing destructive recording cancellation must remove audio and create no session.
- No Qt or Typer dependency in capture or application layers.
- Queue budgets: 256 capture blocks, eight pending transcription jobs.
- Missing timestamps and speakers remain missing; no fabricated identities.
- All original session artifacts remain readable; changes are additive.
- Do not access or publish real credentials or private recording contents.

## Review Focus

- Capture failure after one source has material, including writer finalization.
- Full queues during concurrent stop/abort: termination without capture starvation.
- Offline fallback after partial live success: no missing or duplicated final material.
- Cleaned transcript display text must not rewrite original segment evidence.
- Mixing must not erase source/device provenance in recoverable metadata.

### Task 1: Durable capture and transcript evidence

**Files:** `src/recording/models.py`, `src/recording/windows.py`,
`src/application/controller.py`, `src/application/processing.py`,
`src/live_processing.py`, `src/transcription/client.py`,
`src/transcription/service.py`, `src/output/metadata.py`,
`src/output/session_writer.py`, `src/cli.py`; focused tests in `tests/`.

**Interfaces:** Add a typed capture exception exposing a partial `RecordingSession`;
add backward-compatible segments to `TranscriptionResult`; writer accepts optional
structured transcript segments and capture-source provenance. Preserve existing
public start/pause/resume/stop/cancel/retry operations.

- [ ] Write failing tests for partial Windows-reader failure recovery and destructive
  cancellation using deterministic adapters, not real private audio.
- [ ] Run focused tests and record the expected red evidence.
- [ ] Implement partial-session exception and controller failed-session persistence;
  preserve original exception cause, metadata and retry availability.
- [ ] Write/run failing queue-pressure/stop/abort tests; implement finite queues,
  nonblocking handoff and explicit offline fallback after pressure.
- [ ] Write/run failing live/offline segment round-trip and provenance tests; carry
  final structured segments to a versioned artifact without inventing data.
- [ ] Run all focused tests and the full Qt-offscreen suite, record results, self-review.
- [ ] Commit in cohesive groups: failure/load handling, then transcript/provenance.
- [ ] Write report with red/green evidence to `.superpowers/m1-capture/report.md`.

The controller performs independent review, opens the PR, requests Copilot, handles
and replies to all observations, verifies the reviewed head, then merges.
