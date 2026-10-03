# Product acceptance evidence

Every row is open until the named run and inspectable result are recorded. Automated
fakes prove software contracts, not physical hardware or provider accuracy. Test data
must be synthetic; credentials and private conversations never enter evidence files.

| Milestone | Acceptance requirement | Evidence required | Current state |
|---|---|---|---|
| M1 | Both default Windows sources captured at 44.1/48 kHz | Actual device/format and WAV frame/duration report | Short actual two-source capture passed; long run omitted by user |
| M1 | Two-hour meeting without unbounded growth or unexplained gaps | Frames/known markers, peak memory, disk, queue and final artifacts | Omitted by explicit user instruction; not run |
| M1 | Pause/resume audio timeline, destructive cancel and saved exit | Qt and Windows synthetic marker runs | Short native pause/resume and paused save-exit passed; synthetic restart/exit barriers passed; no repeated installed capture claim; long run omitted by user |
| M1 | Provider/timestamps and diarization | Synthetic real-provider run and segment output inspected | Single/multiple-source TTS smokes passed; long transcription/diarization omitted by explicit user instruction |
| M1 | Failure/crash/network/disconnection/disk-pressure recovery | Fault-injection plus bounded hardware run | PR12 faults, writer/publication process kills and helper-owner death passed; native recoverable saved failure passed; no repeated installed fault matrix claim; long run omitted by user |
| M1 | Installed application opens, records and uninstalls | Fresh Windows installation and bundled-runtime smoke | Local Windows 11 frozen GUI/CLI, Start launch, active-update refusal, reinstall/uninstall and data preservation passed; recording retains earlier source evidence; final source incorporation and clean hosted package CI pending |
| M2 | Identity/relations/revisions and lifecycle durable | SQLite constraints, restart, migration and crash boundary tests | Open |
| M2 | Deletion clears managed memory without resurrection | DB/FTS/files/answers/backup inspection after restart | Open |
| M3 | Listing/filter/search and original moment navigation | End-to-end Qt selection and seek plus real WAV marker playback | Open |
| M3 | Large library remains responsive | 1,000 meetings/100,000 segments benchmark and event-loop check | Open |
| M4 | Multi-meeting answers/extractions with valid evidence | Source-resolution and semantic synthetic corpus, provider smoke | Open |
| M4 | Uncertainty/injection/cancellation/deletion remain honest | No-evidence, invalid-citation, concurrent delete/redaction tests | Open |
| M5 | Offline first run, settings, tray/startup and safe exit | Qt, process concurrency and Windows settings tests | Open |
| M5 | Portable data including optional audio and citations | Export/import on a fresh root, hashes and hostile ZIP tests | Open |
| M5 | Install/update/uninstall preserve user memory | N to N+1 installation and separate uninstall test | Open |
| All | PR groups reviewed and merged | PR URLs, reviewed heads, comment dispositions, CI and merge SHAs | PR12 merged; remaining M1 integration/peer review/Windows CI open. User stopped new Copilot requests after desktop review 5402608228; distribution uses peer review and CI. M2–M5 outside this run |

## Current execution scope

The user's latest explicit instruction on 2026-10-03 omits both the two-hour
capture/pause/recovery/resource run and long generated-audio transcription with
timestamps/diarization. These cases are excluded from this delivery, not passed.
Complete the remaining Windows distribution, actual runtime/installation checks,
notices/checksums, reviewed PRs and evidence audit; then close M1 under this revised
scope and stop. Existing short hardware/fault/provider observations retain their
actual provenance. M2-M5 remain outside this run.

The historical matrix and runs below retain the original requirements and observations.
The long-run exclusions above override their earlier open status for this delivery.
No omitted requirement is represented as tested or passing.

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

Whole-branch corrective source `490b77c` passed 569 tests with six existing local
symlink-privilege skips (56.92 s) and scoped independent acceptance. A native Qt
authored-WAV probe confirmed recovery when the normalized file is already absent,
retained choices after missing-key failure, and an explicit retry after editing the
effective configuration in the same window. It rebuilt 240 frames while preserving
identity/selected filename/raw bytes; no physical capture or provider was used.


PR [#13](https://github.com/srodriguez-lefebre/here/pull/13) is open. Initial hosted
checks at `ef7010e` failed: Ubuntu exposed eager PortAudio initialization in pure
subprocess imports and a Windows-only path assumption in a test; Windows reported a
native Qt teardown exception whose cause remains unconfirmed. Commits `cc947e0` and
`98e19b1` defer unrelated Linux hardware loading, make platform tests explicit, and
address Copilot review `5400263899`/inline `4172753832`: retain the original child
capture error once instead of obscuring it with a generic parent error. Real child
failure/timeout/death regressions passed. The final committed-source suite on actual
Python 3.13.15/Qt 6.11.2 passed 581 tests with six existing local privilege skips
(56.67 s); scoped independent review accepted the corrections. Bounded Qt diagnostics
did not reproduce the native exception; verbose hosted diagnostics were added without
a speculative product patch. A new current-head Copilot review and hosted checks
still precede merge. Installer and long-duration M1 acceptance remain open.


A second hosted run at `62089a4` exposed a post-stop read-watchdog race and repeated
the native Qt teardown failure. Copilot review `5400371073` also identified the
watchdog issue in its body. Commits `0b29897` and `63030f0` use the dedicated closing
deadline after stop, deliver background results only after actual worker-thread exit,
and retain/drain test desktops before widget destruction. Three deterministic failing
regressions verify the closing cause and normal/closed job ownership; six corrected
cases pass. Final source `63030f0`, Python 3.13.15/Qt 6.11.2: 585 passed, six existing
local privilege skips (56.93 s), scoped independent acceptance. A native Windows 11
Qt authored-WAV recovery/retry/exit check also passed with actual cleanup, no provider
or hardware, and 41.8 ms maximum heartbeat gap. These contract corrections do not
prove the original access-violation cause; new-head hosted checks and Copilot review
remain merge gates. M1 installer and long-duration acceptance are still open.


Hosted CI at `bd90996` passed 591 Windows tests (50.64 s, no skips), 588 Ubuntu
tests (33.28 s, three Windows-only skips), and wheel/source builds on both. Copilot
review `5400431791` then identified a high-severity postcommit cleanup defect and a
medium empty-directory-variable defect; both were accepted. Commits `f141189` and
`e3b8917` stage obsolete files before metadata-last publication, restore on handled
precommit failure, retain failed-rollback evidence, and keep owned cleanup failures
from falsifying committed success; empty LOCALAPPDATA/XDG uses the home fallback.
Thirteen RED cases became fifteen GREEN cases. Final committed-source suite:
600 passed, six existing local privilege skips (57.67 s), with scoped independent
acceptance. New-head hosted checks and Copilot review remain merge gates. Retained
owned backups and handled-I/O rollback are explicit limits; no atomic process-kill
transaction or full-M1 completion is claimed.


Hosted checkpoint `9299d59` passed 606 Windows tests (56.09 s, no skips),
603 Ubuntu tests (33.85 s, three Windows-only skips), and wheel/source builds.
Copilot review `5400522452` then found two medium body issues. Both are fixed:
`7b5a124` scopes recovery identity to its validated directory and journal
reservation; `3100d3e` makes only completed capture-file cleanup best effort,
preserving completed artifacts and persisted events even when residual WAV/path
validation or deletion fails. Direct discard/cancellation, precommit/provider/
publication failures and actual live/native closure remain strict. Seventeen RED
failures became twenty-four GREEN cases at `3100d3e` (624 passed, six existing
local privilege skips, 63.45 s). Scoped review exposed a copied-session retry
association defect: `e6e93a0` associates only the exact journal reservation,
leaving foreign bytes intact and invalid journal/path guards strict. Two RED
public-flow failures became nine GREEN cases. Final committed-source suite:
633 passed, 6 existing local privilege skips
(64.53 s); scoped corrective review accepted.
Residual capture evidence can remain after cleanup faults. Current-head hosted
checks and Copilot are still required before merge; full M1 remains open.


Copilot body review `5400663671` identified loss of original child-device errors
in a running controller: persistence and the UI selected the IPC wrapper even
though the journal retained the original cause. The accepted correction at
`9b21f57` derives capture errors from current validated journal events and
uses the persisted capture metadata in the desktop, preserving type, message
and occurrence time. Restart recovery uses the same persistence selection;
no-journal/parent-only failures retain a valid fallback. Meaningful regressions:
7 RED failures became
18 GREEN cases. Final committed-source suite:
654 passed, 6 existing local privilege
skips (68.06 s), with scoped independent acceptance. Previous
`35aed69` hosted CI passed 639 Windows/636 Ubuntu tests and both builds; new-head
Copilot and hosted checks are still required. Raw recovery and actual worker
ownership remain intact; no full-M1 or installed-runtime completion is claimed.


Copilot body review `5400816114` identified retryable audio candidates blocked by
a malformed optional meeting ID being used as a capture-journal locator.
Correction `0ca9a3a` checks canonical UUID form before journal lookup; absent,
malformed or noncanonical optional IDs associate no journal. Selected audio and
metadata remain validated, with their original identity values preserved. Canonical
journal document/path/media checks and exact reservation ownership remain strict.
This M1 compatibility rule grants no journal authority and does not certify M2
canonical catalog identity. Regressions: 15 RED
failures became 32 GREEN cases. Final source
suite: 677 passed, 6 existing local
privilege skips (71.7 s); scoped independent review accepted.
New-head hosted checks and Copilot remain merge gates; full M1 remains open.


Copilot body review `5400937500` identified CLI failure exits before actual worker
completion and pause requests hiding stalled device readers. Correction `662baba`:
CLI failure and interrupt exits cancel only active work and independently await unfinished actual worker completion in finally, so cancellation races cannot skip the fence and FAILED is not destructively cancelled. Original failures retain exit 1; original KeyboardInterrupt retains exit 130 with ordinary cleanup failure logged, without claiming a failed wait completed or catching BaseException.
Each actual child reader advances one monotonic counter only after active IO/write or paused availability/drain operations complete. Existing bounded ticks carry those counters separately from audio/live frames; parent deadlines renew only for strictly increased counters of that source. Missing, stale or lower values preserve history; sender activity and parent pause cannot mask a stalled reader. The separate post-stop closing deadline is unchanged.
Reader health originates from each actual child reader; parent pause state and
sender activity cannot certify device progress. Terminal failure does not grant
destructive cancellation. Real-gated CLI regressions reproduced four premature exits and two interrupt cleanup failures; final CLI/controller coverage passed 53 cases. Nine intended real-child watchdog failures became 16 passing pause cases, including either stalled source, blocked paused drain/availability, stale/missing/rollback ticks, healthy pauses, resume and paused stop with actual child/worker closure.
Final source suite: 705 passed, 6
existing local privilege skips (114.77 s); scoped independent
review accepted. New-head hosted checks and Copilot remain required; full M1 is open.
Actual Windows 11 two-source smoke at this source passed: 44.1/48 kHz stereo, 4.007 s healthy pause with a 3 s reader deadline and unchanged primary frame counts, both sources resumed, then paused stop in 0.262 s. Actual helper/owned threads closed; provider calls were zero and owned raw audio was removed after closure. This is short native reader/IPC proof, separate from GUI, frozen and long acceptance.


Copilot review `5401125631` identified healthy inserted-silence catch-up being
classified as a stalled reader and prior normalized WAVs remaining orphaned after
managed raw-audio retries. Both findings were accepted. Final source `1fa3688`:
Each reader advances its progress counter after each successfully inserted silence write and live delivery. Per-block stop/pause checks, explicit scheduling-gap journal evidence, independent reader watchdogs and the distinct closing deadline remain in place; blocked writes still time out.
Only after final completed/failed metadata publication, cleanup considers the prior validated recoverable normalized filename (audio.wav or audio_<32 lowercase hex>.wav). It rereads committed metadata, verifies final status/identity/current audio, protects selected input, current output/recoverable references and prior/current capture sources, revalidates containment/link safety, and removes only that single unreferenced prior file. Unknown files/names remain. Expected cleanup faults warn without replacing committed success or the original provider failure. Managed normalization failure preserves all prior manifest/error/audio bytes; new-session failure publication is unchanged.
Three real-child healthy catch-up RED failures became five passing catch-up cases; reader guards passed 107 cases. Thirteen authored-PCM cleanup/normalization RED failures with two existing publication-retention guards led to a final 243-case CLI/output focus with six existing privilege skips; unknown/current/raw references and cleanup faults are covered.
Final committed-source suite on CPython 3.13.15 / Qt 6.11.2: 734
passed, 6 existing local privilege skips, 138.64 s.
Changed-file lint/format and diff checks passed; scoped independent review accepted.
New-head Windows-only CI and formal Copilot review remain required before merge.
The user explicitly omitted two-hour capture/resource and long generated-audio
transcription acceptance; neither is run or claimed as passing. Finish remaining
Windows distribution and revised-scope M1 closure, then stop.


Copilot review `5402529641` found message-less background exceptions becoming
empty error strings and being treated as successful diagnostics/retry. Source
`843279c` now preserves nonempty messages and reports the actual exception type
when its message is empty; `None` remains the success sentinel. Actual worker/QTimer
ownership and closed-job delivery semantics remain unchanged.
Three actual Qt empty-message RED failures became 22 passing background/desktop cases, including visible diagnostic errors and retained explicit retry selection. The test-only pause diagnostics separately passed real-child failure-reporting experiments and all 33 isolated-capture cases.
Final committed-source Windows suite: 740 passed,
6 existing local privilege skips in 140.88 s;
changed-file checks and scoped independent review passed.
Historical Windows CI `37150084536` at `d89a386` failed one pause-marker assertion
with 739 other cases passing. The exact case and 12 bounded real-child repetitions
passed locally; its original cause remains unconfirmed. Test-only commit `0576648`
adds failure diagnostics after actual owned reap without changing timeouts,
watchdogs or active journal access. Its isolated-capture focus passed 33 cases.
New-head Windows CI and formal Copilot review remain gates. User-omitted long
capture/resource and long transcription cases remain omitted, not passed.

Long meeting means at least 7,200 seconds of recording wall time excluding deliberate
pauses. Compare known playback markers and recorded durations; do not require exact
hardware-clock synchronization between two sources. Initial marker seek tolerance is
250 ms. Memory must remain bounded by queue and current-chunk budgets; metadata may
grow with the number of chunks. Set a concrete hardware memory budget after measuring
the bundled application's idle baseline, and report baseline plus delta.

Clean-install verification must use an independent Windows environment without the
developer virtual environment or repo on PATH. A bundled executable launched from a
temporary directory tests independence but alone does not prove a clean installation.
