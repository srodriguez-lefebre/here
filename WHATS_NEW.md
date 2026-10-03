# What's new

## Desktop configuration and error feedback

- Local configuration editor for provider credentials, models, cleanup and sessions.
- Persistent indicator size and transparency controls.
- Missing-key explanation before capture, visible error attention and bounded
  structural diagnostics even when a session has not been created.

These changes are reviewed in their own PR after the M1 delivery; merging them
requires a later user instruction. No new Copilot review is requested.

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
