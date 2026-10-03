# Windows installable delivery implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development task-by-task.
> Product/build/review/merge decisions were explicitly delegated by the user.

**Goal:** Deliver and verify installable Windows runtime artifacts for M1.
**Architecture:** Locked non-editable wheel -> PyInstaller onedir -> per-user Inno
installer; clean-runner verification plus independent local capture acceptance.
**Tech Stack:** uv, PyInstaller, Inno Setup 6.7.0, PowerShell, GitHub Actions.
**Spec:** `docs/development/M1_DISTRIBUTION_SPEC.md`.

The user's latest explicit instruction on 2026-10-03 omits both the two-hour
capture/pause/recovery/resource run and long generated-audio transcription with
timestamps/diarization. These cases are excluded from this delivery, not passed.
Complete the remaining Windows distribution, actual runtime/installation checks,
notices/checksums, reviewed PRs and evidence audit; then close M1 under this revised
scope and stop. Existing short hardware/fault/provider observations retain their
actual provenance. M2-M5 remain outside this run.


Execution status: predecessor desktop/recovery PR #13 is merged. Task 1 starts on
`codex/m1-windows-distribution` from that verified main revision. Director ruling:
deliver build/installer/clean-runtime proof and the revised-scope closure audit in
the remaining M1 PR, preserving commits, independent review, dispositions of all
received Copilot findings, Windows CI and guarded merge. The user explicitly
omitted long capture/resource and long generated-audio transcription runs. No bundle
or installation completion is inferred from predecessor desktop evidence. Stop after M1.
The user also made review `5402608228` the last Copilot review; do not request another.

## Global Constraints

- Windows 11 x64; no system service or automatic recording/startup.
- Current-user install/uninstall never removes the data root or existing personal data.
- No credentials, personal transcripts or development scratch in bundle/installer.
- Verify runnable artifacts before claiming build/installation succeeds.
- Hardware, provider and clean installation have distinct acceptance evidence.

## Review Focus

- Explicit src-to-here package mapping must work without an editable finder.
- Windowed app has no stderr/stdout; logging/errors cannot crash normal launch.
- Missing required Qt plugins, WASAPI or soundfile DLL in frozen bundle.
- Update while recording must preserve current work and prevent unsafe replacement.
- Installer/script paths with spaces, existing data and nonadministrator user.

### Task 1: Build, installer and clean-runtime validation

**Files:** `packaging/windows.spec`, `packaging/gui_entry.py`,
`packaging/cli_entry.py`, `packaging/installer.iss`, `scripts/build_windows.ps1`,
`scripts/smoke_windows.ps1`, `.github/workflows/windows-package.yml`,
`pyproject.toml`, `uv.lock`, `src/ui/gui.py` and safe bootstrap as needed, tests/docs.

- [x] Add failing packaged-startup/import/path acceptance checks using actual wheel and
  isolated environment. Configuration-only scripts are verified by actual execution.
- [x] Add locked build group and build script with explicit compiler path/download
  verification, non-editable project layout and resource/notice inclusion.
- [x] Implement windowed and CLI entries, spec and current-user installer.
- [x] Build installer, launch off-repo with no key, and inspect report/bundle inventory.
- [x] Test isolated install/reinstall/uninstall with synthetic canaries and Start entries.
- [ ] Run Windows packaging CI, upload artifacts/checksums and record exact evidence.
- [ ] Update installation docs/architecture/What's New; make cohesive build/runtime and
  verification/documentation commits. Independent review and Windows source/package CI precede merge; no further Copilot request per explicit user instruction.

### Task 2: Revised-scope milestone closure audit

Included in the remaining distribution PR; no separate long-acceptance PR.

**Files:** `docs/development/M1_EVIDENCE.md`, `docs/development/ACCEPTANCE.md`,
authoritative plans and delivery ledger.

- [ ] Record exact source, frozen-runtime, installer and clean Windows CI evidence.
- [x] Reconcile each remaining M1 criterion with its actual observed proof and limits.
- [x] Mark the two-hour capture/resource and long generated-audio transcription cases
  omitted by explicit user instruction; do not execute replacement long-run tests.
- [ ] Record the reviewed merge and close M1 under the revised scope, then stop.
  Leave M2-M5 plans intact without implementing them.

Local incorporated-source delivery at clean build head `1740b2d` passed the complete
locked build command and actual stable-AppId installer lifecycle on Windows 11 26200.
Sanitized artifacts, hashes, runtime and limits are in `M1_DISTRIBUTION_EVIDENCE.json`.
Documentation and implementation commits are complete; independent review, actual
Windows source/package CI and guarded merge remain coordinator-owned gates.
