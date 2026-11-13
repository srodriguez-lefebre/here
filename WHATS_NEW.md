# What's new

## 0.3.0 — Local catalog and lifecycle

Give saved meetings a useful title and link
participants to the speaker labels in a transcript. You can correct an
attribution while keeping the original transcript and audio available for
reference.

Local reconciliation resumes interrupted catalog work. Confirmed deletion
removes the managed meeting and its derivatives while preserving the original
folder of an imported session. If cleanup needs another attempt, the result
reports that it is pending.

Recordings, file transcriptions and explicit retries preserve raw text, segment
timing provenance and revision-specific audio alongside the display transcript.
Startup resumes local publication when evidence matches, so recovering a saved
result needs no transcription request. Evidence is recorded in
[ACCEPTANCE.md](docs/development/ACCEPTANCE.md).

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
