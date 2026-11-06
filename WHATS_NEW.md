# What's new

## M2 in progress — preserved evidence and local catalog

Successful recordings, file transcriptions and retries now keep the uncleaned
text in `raw_transcript.txt` alongside the display transcript and segment evidence.
This makes cleanup changes inspectable across live processing and offline fallback.
Text-only responses keep their words without inventing timestamps or speakers;
nullable timing provenance and recorded-audio offsets remain explicit.

The catalog implementation now links saved sessions to durable identities and
immutable transcript revisions, so an explicit retry can retain earlier evidence.
Local ingest, inspect and reconcile commands work without the provider. A separate
library-pending warning keeps saved audio/text available when catalog work needs
recovery; opening the app replays local persistence rather than transcription.

Scoped fixes now preserve revision-specific audio links, legacy identity and
actual ownership through capture/recovery. Publication tokens distinguish an
already committed replay from a fresh retry; WAV-only legacy recovery retains
unknown capture dates. Final publication now seals the exact local artifacts before
the final session metadata commit, so startup can recover a completed saved result
without repeating transcription. Unmatched final evidence remains preserved with
a warning. Invalid receipt envelopes no longer prevent valid neighbors rebuilding;
capture-failure cancellation is sealed before its first catalog publication while
retaining the original error and audio.

The catalog group M2 A is now merged in PR21. Ordinary precommit callback errors
restore earlier evidence, the controller/default processor share one catalog,
and blocked completed publications stay visible without offering another
transcription. Malformed pending markers and unavailable legacy ownership mappings
also leave a bounded nonretryable warning while retaining original data and strict
owner/path guards.

Independent reviews and all five Copilot observation threads are complete.
The final exact-head Windows suite passes980 tests/two wheel skips; the fresh
installed wheel separately passes all four distribution checks, including both
installed entry points. These are automatic synthetic checks, with no new
hardware/provider/clean-install certification. M2 remains unreleased pending
lifecycle/deletion and milestone acceptance; version0.2.0 stays current. Detailed
runs, source hashes and historical failures remain in ACCEPTANCE/DELIVERY.

## 0.2.0 — Windows M1 delivery

- Production Qt desktop and recording overlay share the application controller.
- Local settings, saved audio and owned recovery remain usable without a provider
  key; transcription requests require the key explicitly.
- Windows delivery includes a windowed GUI, console CLI/helper companion and
  per-user installer. Reinstall/uninstall preserves user data; active applications
  prevent replacement, and installation does not enable startup or recording.
- Locked build tools, actual frozen startup/helper checks, installer lifecycle,
  complete namespaced notices, native inventory and SHA-256 artifacts are provided.

The user excluded the two-hour capture/resource and long generated-audio/provider
acceptance runs. Existing short hardware/provider observations retain their actual
source/version limits. M2–M5 are not implemented by this delivery.
