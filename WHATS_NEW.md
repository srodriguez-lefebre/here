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

Integrated source `7bfd0cb` uses final pending markers version 3 with readers for
versions 1/2; receipts remain version 3, compatible with version 2, and the DB
stays at schema 2. The final group/recovery focus passes 26 tests; the earlier
940-pass/seven-skip full is historical evidence for `8087e25`. Full 3 at `7bfd0cb`
has eight failures/945 passes/seven skips from two capture-recovery test wrappers
rejecting the additive precommit callback; both wheel smokes pass. The narrow
fixture correction is integrated at `ef70f78`, preserving all crash assertions.
Its 13-case focus passes; full 4 passes **953 tests/seven skips in 358.94s** with
both wheel smokes, using the unchanged product wheel. Final scoped independent
review APPROVED source, fixture and docs `8845877`; GR1–GR3/GD1/GD2 are addressed.
Hosted gates remain pending. M2 is unreleased; version 0.2.0 stays current.

PR #21 Copilot corrections now restore earlier evidence after ordinary precommit
callback rejection, keep the controller/default processor on one catalog, and
show blocked completed publication through the existing recovery warning without
offering another transcription. Source `c3cdf08` passes eight new cases and the
325-case affected focus (one symlink skip). Full 5 passes 961/seven skips in
371.80s with both wheel smokes. Independent final scoped review remains pending;
the full4/approval above retain their previous-source scope.

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
