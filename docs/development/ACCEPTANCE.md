# Product acceptance evidence

Every row is open until the named run and inspectable result are recorded. Automated
fakes prove software contracts, not physical hardware or provider accuracy. Test data
must be synthetic; credentials and private conversations never enter evidence files.

| Milestone | Acceptance requirement | Evidence required | Current state |
|---|---|---|---|
| M1 | Both default Windows sources captured at 44.1/48 kHz | Actual device/format and WAV frame/duration report | Devices enumerated; capture pending |
| M1 | Two-hour meeting without unbounded growth or unexplained gaps | Frames/known markers, peak memory, disk, queue and final artifacts | Open |
| M1 | Pause/resume audio timeline, destructive cancel and saved exit | Qt and Windows synthetic marker runs | Open |
| M1 | Provider/timestamps and diarization | Synthetic real-provider run and segment output inspected | Single/multiple-source TTS smokes passed; long diarization open |
| M1 | Failure/crash/network/disconnection/disk-pressure recovery | Fault-injection plus bounded hardware run | Open |
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
| All | PR groups reviewed by Copilot and merged | PR URLs, reviewed heads, comment dispositions, CI and merge SHAs | Open |

## Runs

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

## Measurement decisions

Long meeting means at least 7,200 seconds of recording wall time excluding deliberate
pauses. Compare known playback markers and recorded durations; do not require exact
hardware-clock synchronization between two sources. Initial marker seek tolerance is
250 ms. Memory must remain bounded by queue and current-chunk budgets; metadata may
grow with the number of chunks. Set a concrete hardware memory budget after measuring
the bundled application's idle baseline, and report baseline plus delta.

Clean-install verification must use an independent Windows environment without the
developer virtual environment or repo on PATH. A bundled executable launched from a
temporary directory tests independence but alone does not prove a clean installation.
