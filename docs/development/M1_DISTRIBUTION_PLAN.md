# Windows installable delivery implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development task-by-task.
> Product/build/review/merge decisions were explicitly delegated by the user.

**Goal:** Deliver and verify installable Windows runtime artifacts for M1.
**Architecture:** Locked non-editable wheel -> PyInstaller onedir -> per-user Inno
installer; clean-runner verification plus independent local capture acceptance.
**Tech Stack:** uv, PyInstaller, Inno Setup 6.7.0, PowerShell, GitHub Actions.
**Spec:** `docs/development/M1_DISTRIBUTION_SPEC.md`.

## Global Constraints

- Windows 11 x64; no system service or automatic recording/startup.
- Current-user install/uninstall never removes the data root or existing personal data.
- No credentials, personal transcripts or development scratch in bundle/installer.
- Verify runnable artifacts before claiming build/installation succeeds.
- Hardware, provider and clean installation have distinct acceptance evidence.

## Review Focus

- Explicit src-to-here package mapping must work without an editable finder.
- Windowed app has no stderr/stdout; logging/errors cannot crash normal launch.
- Missing Qt multimedia, WASAPI or soundfile DLL in frozen bundle.
- Update while recording must preserve current work and prevent unsafe replacement.
- Installer/script paths with spaces, existing data and nonadministrator user.

### Task 1: Build, installer and clean-runtime validation

**Files:** `packaging/windows.spec`, `packaging/gui_entry.py`,
`packaging/cli_entry.py`, `packaging/installer.iss`, `scripts/build_windows.ps1`,
`scripts/smoke_windows.ps1`, `.github/workflows/windows-package.yml`,
`pyproject.toml`, `uv.lock`, `src/ui/gui.py` and safe bootstrap as needed, tests/docs.

- [ ] Add failing packaged-startup/import/path acceptance checks using actual wheel and
  isolated environment. Configuration-only scripts are verified by actual execution.
- [ ] Add locked build group and build script with explicit compiler path/download
  verification, non-editable project layout and resource/notice inclusion.
- [ ] Implement windowed and CLI entries, spec and current-user installer.
- [ ] Build installer, launch off-repo with no key, and inspect report/bundle inventory.
- [ ] Test isolated install/reinstall/uninstall with synthetic canaries and Start entries.
- [ ] Run Windows packaging CI, upload artifacts/checksums and record exact evidence.
- [ ] Update installation docs/architecture/What's New; make cohesive build/runtime and
  verification/documentation commits. Independent review then Copilot/CI gates precede merge.

### Task 2: Capture acceptance and milestone closure audit

**Files:** `scripts/validate_capture.py`, `docs/development/M1_EVIDENCE.md`,
`docs/development/ACCEPTANCE.md`, authoritative plans and delivery ledger.

- [ ] Define executable synthetic marker/resource tests with a 7,200-second audio
  timeline; bounded duration-memory/queue tests must exercise the production pipeline.
- [ ] Run provider smoke on generated Windows TTS only, never upload private audio;
  inspect source segment timing, speaker availability and recovery artifacts.
- [ ] Run bounded Windows device/signal and pause/stop/cancel/recovery checks; document
  exact tested devices, duration and any unsupported conditions.
- [ ] Execute available long real-hardware and clean-installed scenarios from the
  acceptance matrix; record frame/marker/memory/disk results and explicit open cases.
- [ ] Compare every M1 finished criterion with authoritative evidence. Close only proven
  criteria; retain original scope and progress on later milestones if hardware evidence
  needs an external environment.
