# M1 revised-scope delivery audit

The user requested finishing M1 on Windows and stopping. The two-hour
capture/pause/recovery/resource run and long generated-audio transcription with
timestamps/diarization were explicitly excluded on 2026-10-03. They are omitted,
not run or passed. Ubuntu is optional manual-only; M2–M5 remain outside this run.

| Criterion | Observed evidence and boundary |
| --- | --- |
| Shared capture/controller, primary WAV, short default-source capture | `M1_CAPTURE_EVIDENCE.json`, `M1_NATIVE_EVIDENCE.json`; prior short real Windows observations retain their original source/runtime. |
| Pause/resume, stop/save, cancel, crash/recovery and bounded fault contracts | `M1_DESKTOP_EVIDENCE.json` and the Windows automated suite; source software contracts and prior short native runs are separate from frozen/installed execution. |
| Provider timestamps/diarization | `M1_PROVIDER_EVIDENCE.json` contains prior authored short real-provider runs. Long generated-audio provider acceptance is omitted by user instruction. |
| Wheel layout, no-key/off-repo startup, Qt, fixed child IPC/reaping | Incorporated-source build `1740b2d` passed from a clean tracked tree: actual noneditable wheel, frozen GUI/CLI, visible desktop, job drain, fixed invalid-request rejection and waiting-child kill/reap. See `M1_DISTRIBUTION_EVIDENCE.json`. |
| Current-user installer, Start launch, active-app replacement refusal, reinstall/uninstall/data preservation | Incorporated-source installer on local Windows 11 build 26200: stable AppId passed isolated lifecycle, all installed payload hashes, visible Start launch, no automatic startup, active replacement rejection (exit 1), idle close, reinstall/uninstall with preserved synthetic data. Clean hosted checks remain pending. |
| Notices, payload and exact provenance | `THIRD_PARTY_NOTICES.md`, full component text trees, `bundle-inventory.json`, `build-provenance.json`, `compiler-provenance.json` and `SHA256SUMS.txt` accompany artifacts. Unexposed native versions remain explicitly unknown. |
| PR review, Copilot dispositions, CI, merge | Desktop PR13 merged as `7ed78af`; distribution incorporates it. Root records remaining distribution peer review, actual Windows source/package CI and guarded merge. User stopped Copilot requests after desktop review 5402608228; no new distribution request. Local runtime proof does not imply hosted results or a merged distribution. |

The package check never opens audio devices, starts recording, calls providers, or
reads existing sessions/preferences. Installed recording/fault behavior is not
inferred from a startup report: earlier hardware/provider observations and source
tests retain their own limits. No public release upload or application Authenticode
signature is claimed. Final M1 closure uses the user's revised scope and actual
root-recorded integration/review/CI evidence, then work stops.
