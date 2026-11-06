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
unknown capture dates. Pending markers use version 2 and publication receipts
version 3; the DB stays at schema 2. The integrated suite passes 940 tests with
seven skips; whole-group review remains open on crash replay, malformed receipts
and capture-cancellation catalog state. Its three Important findings await fixes
and final documentation review. M2 is unreleased; version 0.2.0 stays current.

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
