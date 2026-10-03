# Product acceptance evidence

Every row is open until the named run and inspectable result are recorded. Automated
fakes prove software contracts, not physical hardware or provider accuracy. Test data
must be synthetic; credentials and private conversations never enter evidence files.

| Milestone | Acceptance requirement | Evidence required | Current state |
|---|---|---|---|
| M1 | Both default Windows sources captured at 44.1/48 kHz | Actual device/format and WAV frame/duration report | Short actual two-source capture passed; long run open |
| M1 | Two-hour meeting without unbounded growth or unexplained gaps | Frames/known markers, peak memory, disk, queue and final artifacts | Open |
| M1 | Pause/resume audio timeline, destructive cancel and saved exit | Qt and Windows synthetic marker runs | Short native pause/resume and paused save-exit passed; synthetic restart/exit barriers passed; long/installed checks open |
| M1 | Provider/timestamps and diarization | Synthetic real-provider run and segment output inspected | Single/multiple-source TTS smokes passed; long diarization open |
| M1 | Failure/crash/network/disconnection/disk-pressure recovery | Fault-injection plus bounded hardware run | PR12 faults, writer/publication process kills and helper-owner death passed; native recoverable saved failure passed; long/installed fault matrix open |
| M1 | Installed application opens, records and uninstalls | Fresh Windows installation and bundled-runtime smoke | Open |
| M2 | Identity/relations/revisions and lifecycle durable | SQLite constraints, restart, migration and crash boundary tests | Open |
| M2 | Deletion clears managed memory without resurrection | DB/FTS/files/answers/backup inspection after restart | Open |
| M3 | Listing/filter/search and original moment navigation | End-to-end Qt selection and seek plus real WAV marker playback | Open |
| M3 | Large library remains responsive | 1,000 meetings/100,000 segments benchmark and event-loop check | Open |
| M4 | Multi-meeting answers/extractions with valid evidence | Source-resolution and semantic synthetic corpus, provider smoke | Open |
| M4 | Uncertainty/injection/cancellation/deletion remain honest | No-evidence, invalid-citation, concurrent delete/redaction tests | Open |
| M5 | Offline first run, settings, tray/startup and safe exit | Qt, process concurrency and Windows settings tests | Open |
| M5 | Portable data including optional audio and citations | Export/import on a fresh root, hashes and hostile ZIP tests | Open |
| M5 | Install/update/uninstall preserve user memory | N to N+1 installation and separate uninstall test | Open |
| All | PR groups reviewed by Copilot and merged | PR URLs, reviewed heads, comment dispositions, CI and merge SHAs | First M1 capture group PR12 merged; remaining groups open |

## Runs

Sanitized, versioned observations from the real synthetic-provider runs are in
[`M1_PROVIDER_EVIDENCE.json`](M1_PROVIDER_EVIDENCE.json). Only authored fixture
content, source formats, provider segments and audio hashes are included; no local
session paths, credentials or private recordings are published.

- 2026-10-03: original main `2bdb6ef`, 163 tests passed (Python environment in original
  checkout), and fresh managed worktree baseline 163 passed (Python 3.14.7, 42.19 s).
- Windows default device enumeration: microphone `Microphone (3- Arctis Nova 7P)`,
  44100 Hz, two channels; loopback `Headphones (3- Arctis Nova 7P) [Loopback]`,
  48000 Hz, two channels. Enumeration only; no claim of captured content.
- 2026-10-03 01:44 America/Montevideo: configured real provider, model
  `gpt-4o-transcribe-diarize`, accepted a generated Windows TTS fixture (~11 seconds).
  Production SessionProcessor preserved audio, four correct source utterances, four
  provider timestamp ranges and speaker A with request/chunk scope in `segments.json`.
  Evidence is local ignored `.superpowers/provider-smoke/result.json` plus session
  `20261003_014354`. No private recording was uploaded; no credential was displayed or
  written to evidence. This verifies single-request provider integration, not speaker
  identity across requests or full long-duration acceptance.
- 2026-10-03 02:05 America/Montevideo: alternating Windows David/Zira voices in two
  synthetic stereo sources at 44,100/48,000 Hz (26.159 seconds) passed the actual
  mixing, normalization, provider and artifact path. Seven provider segments retain
  timestamps and alternating A/B speakers in request scope; source format/frame
  provenance and normalized audio are preserved. Local evidence:
  `.superpowers/provider-multisource/result.json`, session `20261003_020516`.
  A spurious quote in the first utterance is preserved as original provider output.
  The first returned start precedes its synthetic source offset by 300 ms; later
  utterance starts are within 40 ms. This measures provider estimates, not an exact
  acoustic alignment guarantee, physical capture or long-meeting certification.
- Capture correctness suite at review-fix commit `4860095`: 211 passed, including
  ten regressions for the two Copilot findings. Changed-file Ruff and diff checks pass.
  Hosted Windows/Ubuntu tests and wheel/source builds passed at CI correction head
  `545174e`, run `37099618688`.
- Hosted review-fix CI at `240858e`, run `37103530702`: 266 tests passed in Windows
  (8.35 s) and Ubuntu (7.71 s), including the locally skipped symlink test; wheel and
  source builds passed. This is evidence for that immutable head, not later commits.
- 2026-10-03 05:37 UTC, code `5ec4e92`: short actual default-device capture with an
  authored TTS fixture on loopback, pause/resume and stop. Wall duration 8.281 s,
  pause 2.000 s, stop latency 0.107 s. Microphone: 271,360 frames at 44,100 Hz,
  stereo (6.153 s); system loopback: 295,936 frames at 48,000 Hz, stereo (6.165 s).
  Persisted, returned and live frame counts matched for both sources; nonzero loopback
  signal observed. No provider call; probe-owned raw audio was removed. The original
  probe used raw integer samples for peak, so no calibrated amplitude/clipping claim
  is made. This verifies a short device/control path, not long capture, journal recovery
  or installed-runtime behavior.
- Corrective code `7ed996a`: one full local offscreen suite, 385 passed/6 skipped in
  10.62 s. New tests reproduce external-entry recovery routes, partial output writes
  and reversed timing across aliases/objects/typed serialization. Real hardlinks and
  junctions passed; six symlink cases require local Windows privileges. Hosted checks
  for the final PR head are recorded in the PR and evaluated before merge.

- First M1 capture group merged 2026-10-03 07:52:15 UTC: PR #12, exact reviewed
  head `3162a74`, merge `ca07aed`. Current-head CI `37107198007` passed 469 tests
  on Windows (14.24 s, no skips), 467 on Ubuntu (7.97 s, two Windows-only junction
  skips), and wheel/source builds on both. All six locally skipped symlink cases
  ran successfully on both hosts. Local final-code suite: 463 passed/6 skipped in
  12.98 s; independent review accepted the two latest corrections. Final Copilot
  review `5399598693` confirms the fixes and records the legacy private CLI helper
  limitation assigned to desktop Task 1; no full-M1 completion is claimed. Details:
  [`M1_CAPTURE_EVIDENCE.json`](M1_CAPTURE_EVIDENCE.json). Short physical observations
  retain their original code provenance in
  [`M1_NATIVE_EVIDENCE.json`](M1_NATIVE_EVIDENCE.json).

## Measurement decisions

Desktop source `76d317a` passed 558 tests with six existing local symlink-privilege
skips (53.72 s), separate Task 1/2 independent reviews and a bounded native Qt check.
The native check observed actual 44.1/48 kHz default-device names, asynchronous
three-second quiet microphone and authored-tone loopback diagnostics, credential-free
local recovery and a paused save-exit with intentionally injected local transcription
failure. Audio remained locally retryable until probe-owned cleanup; no provider call
was issued. The maximum Qt heartbeat gap was 49.6 ms. This is source-runtime evidence,
not frozen or long-provider acceptance; details and limits are in
[`M1_DESKTOP_EVIDENCE.json`](M1_DESKTOP_EVIDENCE.json).

Long meeting means at least 7,200 seconds of recording wall time excluding deliberate
pauses. Compare known playback markers and recorded durations; do not require exact
hardware-clock synchronization between two sources. Initial marker seek tolerance is
250 ms. Memory must remain bounded by queue and current-chunk budgets; metadata may
grow with the number of chunks. Set a concrete hardware memory budget after measuring
the bundled application's idle baseline, and report baseline plus delta.

Clean-install verification must use an independent Windows environment without the
developer virtual environment or repo on PATH. A bundled executable launched from a
temporary directory tests independence but alone does not prove a clean installation.
