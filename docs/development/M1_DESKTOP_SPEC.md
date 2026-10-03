# Windows capture desktop and restart recovery

## Intent and fixed behavior

The existing production GUI must open without a credential and explain how to configure
one only when recording/transcription needs it. Recording uses the Windows defaults,
shows their actual names, offers a finite signal test, and remains responsive. Explicit
exit during active work stops and saves; ordinary window close keeps the current hide
behavior. Startup exposes local interrupted/failed/cancelled sessions for explicit retry.
No startup operation issues provider calls or begins recording by itself.

## Decisions

Settings use `%LOCALAPPDATA%/here` on Windows, with `HERE_DATA_DIR` override for portable
tests and user-managed local storage. `TRANSCRIPTIONS_DIR` remains an explicit override;
new default is `<data_dir>/sessions`. Configuration remains `.env` compatible: explicit
`HERE_ENV_FILE`, otherwise user data `.env`, and development project `.env` only as
backward-compatible fallback outside a frozen build. Secrets are optional `SecretStr`
at settings construction, required only by provider client creation. No credential
value enters logs, previews, QSettings or committed artifacts.

Resolve exactly one `.env`: explicit `HERE_ENV_FILE` wins, including when missing;
otherwise existing user-data `.env`, otherwise development `.env` only outside frozen
builds. Process environment values take precedence. Missing/empty/whitespace key fails
explicit start/retry before device opening, live/temp allocation or provider calls,
with a safe configuration-path message. Preserve existing recovery records. GUI,
diagnostics and local recovery never construct a provider client; editing the effective
file must be reflected by a later explicit start without restarting the application.

Introduce a capture journal in `<sessions>/.captures/<uuid>/` with version, UUID,
start/updated wall times, source descriptors/relative WAV names, pause events, recorded
frames and state. Every source writes into this owned directory rather than anonymous
system temp. While frames arrive, checkpoint at intervals no greater than one second;
closing/pausing/stopping force a checkpoint. Serialize audio write, header update,
flush and close on the owning writer thread. `SoundFile.flush()` alone does not update
the WAV header: use an explicit libsndfile header command or validated format/data-offset
recovery from a copy. Publish journal frame counts only after audio/header checkpoint
succeeds. Process-kill recovery does not imply power-loss guarantees. Journal writes
use same-directory temporary plus replace. Source paths
must be beneath this capture directory; journal content is data, not authority to read
arbitrary files. Preserve useful material if startup finds an interrupted journal.

Long scheduling or suspend gaps must not make catch-up silence uninterruptible. Check
stop, cancellation and pause intent between inserted blocks, checkpoint before closing,
and send only actually persisted frames to live processing in disk order. A bounded
control request must not drain an entire historical silence backlog before it acts.
Record the interruption/gap truthfully; a silent placeholder is not captured speech.

The journal lives until the final recoverable session metadata is persisted. Allocate
capture UUID and final destination mapping before normalization creates a destination;
reuse the same identity/path after restart. Persist that UUID as additive optional
`SessionMetadata.meeting_id`; keep the existing human-readable `session_id` unchanged.
M2 consumes the same UUID rather than allocating another identity for recovered audio.
Live provider work can run during capture,
protected by primary audio/journal. Before post-capture provider finalization or offline
processing, close/validate normalized audio and atomically publish pending metadata.
Stage audio and metadata under owned paths and replace in-directory, preserving the
last valid record on failure. Commit final metadata after every referenced artifact is
ready; remove journal only after commit. Deduplicate journal/session by UUID; completed
metadata wins. Startup scans only the configured sessions root and `.captures`, validates
versions and paths, and offers recoverable results. It never deletes unrecognized files
or converts malformed journals into a successful session. Conversion to a failed /
interrupted recoverable session is a local idempotent operation with original causes.
Destructive cancellation removes only its owned capture directory and no final session.

Validate journal sources, mapped destination, normalized audio and every session metadata
source reference against the expected owned directory/root before opening, copying,
normalizing, writing or deleting. Reject absolute/traversal paths and resolved
outside-root links/reparse points, and revalidate at retry rather than trusting an old
discovery result. A malformed entry preserves its files and cannot hide valid candidates.

The existing CLI file-transcription path also reads neighboring session metadata and
may reuse that directory. When reusing managed session state, preflight metadata,
chunks/errors, every advertised audio/output reference and destination before reads
or provider allocation. Preserve ordinary explicit file transcription: a user-supplied
original input outside a session remains legitimate; neighboring metadata is never
authority to follow a redirected managed entry. Both interfaces use the shared safe
artifact helpers rather than assuming writer validation protects earlier reads.

Consolidate the retained private CLI `_save_transcription` compatibility helper with
the shared SessionProcessor. Current recording commands already use the shared
controller; this helper is exercised by compatibility tests. Its normalization-
failure record must likewise preserve and reference locally retryable raw sources,
with geometry, original cause and safe owned paths. Leaving anonymous temporary WAVs
outside a failed session is not recoverable session evidence. Test an actual tiny
WAV, normalization failure, preserved local source and successful later retry.

`RecoveryService.discover() -> list[RecoveryCandidate]` returns session directory,
display ID, recorded duration, status, error summary and can_retry. It uses original
or normalized preserved audio. The minimal GUI provides a recovery selector and retry
action, refreshed at startup and after work ends; full history navigation remains H3.

The ready capture handle exposes immutable actual opened-source descriptors (label,
device name/index when available, sample rate/channels). Controller publishes these
with readiness. A subsequent enumeration never replaces actual opened names; default
StartRequest continues to omit IDs so changing Windows defaults is respected.

`AudioDiagnosticsService` provides default device information and microphone/loopback
signal test results via a Qt background worker. Both test and enumerate have bounded
duration/timeouts and truthful no-signal/error messages. Capture controls reject a
diagnostic running concurrently with recording. Names are refreshed at capture start
and record the actual opened source names, not stale preflight guesses. No manual
selection UI and no second capture stream for visual telemetry.

Diagnostics and capture share a serialized audio-resource reservation. Release it only
after streams close and the worker completes, including timeout cleanup. Requests have
identities; Qt receives queued signals on its event loop and ignores stale deliveries.
Retain worker ownership until completion, disconnect shutdown delivery and never destroy
a running QThread or force-stop a thread holding resources. Use bounded backend
open/read/cancellation (an isolated helper process can be terminated safely after its
deadline); a UI timer alone is not resource cleanup. Discovery/normalization also runs
off the GUI thread.

Add an explicit `Salir` action. While recording/paused it requests stop and waits for
terminal persistence; while preparing it requests stop as soon as recording is ready;
while stopping/processing it waits. It never routes through destructive cancel.
Exit after a failure still preserves its recovery record. Ordinary close semantics and
overlay restoration stay intact. Controller stop/save is idempotent and accepts
PREPARING through an atomic intent latch checked at handle publication. Opening
failure/timeout resolves exit intent with partial recovery where available. Qt never
waits/joins synchronously. Quit only after persistence or durable failure AND worker
completion acknowledgement, not an early terminal-state event. Repeated exit is harmless;
persistence failure leaves the journal for recovery.

## Acceptance

- GUI opens with no key; recording fails clearly and locally if key is absent.
- Settings paths are writable per user and frozen code never uses its install folder
  for data or automatically includes a development `.env`.
- A killed Windows synthetic subprocess with the production writer still open after a
  checkpoint handshake is discoverable/retryable; a fresh process reads expected
  mono/multichannel committed frames and sample values without finally/close.
- Kill around reservation, mapping, normalization, pending publication, metadata
  replacement and journal removal; repeated discovery/materialization preserves one
  identity/path. Completed metadata with leftover journal stays completed.
- Pending processing session survives a killed worker/provider and is offered again.
- Tampered journal/session/destination paths and replaced links between discovery/retry
  are rejected without external reads/writes/deletion; other valid entries still appear.
- Actual device names are displayed, signal tests work asynchronously, and no-signal
  is distinct from device error.
- Barrier-controlled exit before/simultaneous readiness, opening failure, timeout and
  delayed cleanup after terminal publication demonstrates no cancel, no Qt wait and
  persistence plus worker completion before quit.
- Stalled diagnostics, capture attempts during timeout cleanup and stale delivery after
  shutdown demonstrate no overlapping streams or running-thread destruction.
- A large deterministic clock jump followed by stop, cancel or pause during catch-up
  demonstrates bounded reader operations and matching persisted/live frame order.
- Exit while recording, paused, preparing and processing follows stop/save semantics;
  idle close, hidden work and terminal overlay behavior remain unchanged.
- Unit tests and Qt integration tests cover the new flow; real Windows smoke is logged
  separately from synthetic process-kill and hardware long-duration acceptance.

## Boundaries

No automatic retry, retention purge, manual device selector, background service,
resident idle default, or credential UI. Recovery candidates do not substitute for the
durable product memory of H2.

## Technical references

- [libsndfile header checkpoint commands](https://github.com/libsndfile/libsndfile/blob/master/docs/command.md).
- [SoundFile flush implementation](https://github.com/bastibe/python-soundfile/blob/master/soundfile.py).
