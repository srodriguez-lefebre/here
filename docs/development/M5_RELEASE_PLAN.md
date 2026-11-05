# M5 release and acceptance implementation plan

> **For agentic workers:** read `CLAUDE.md` first. Task-by-task, red/green tests, self
> review, then independent agent review. The user merges (ruling 4). Steps use checkbox
> syntax. **Status: not in execution (ruling 15).** Start only on an explicit user
> instruction after M5 group B merges. Hardware rows need the clean machine and the
> credential the user provides on request (ruling 5); ask for them when a row needs
> them, with the protocol written, not before.

**Goal:** Version 1.0.0 packaged with multimedia and FTS5 verified, an About screen,
an honest upgrade path from 0.2.0, and the final Windows acceptance matrix.
**Architecture:** The existing PyInstaller onedir + Inno Setup pipeline parametrized
by version; smoke-check probes in `src/distribution.py`; evidence files per run under
`docs/development/`.
**Tech Stack:** PyInstaller 6.22.3 (locked), Inno Setup 6.7.0 (verified download),
PowerShell 7, `uv`, PySide6, pytest, pytest-qt.
**Spec:** `docs/development/M5_RELEASE_SPEC.md`.
**Plan (Spanish):** `docs/MILESTONE_5_PLAN.md`.

## Global constraints

- No version string hard-coded outside `pyproject.toml` after this group.
- The installer keeps refusing to replace a running application; never add
  `CloseApplications=yes`.
- Nothing claims Authenticode, a public release or an updater.
- Real installations and real data are never uninstalled or overwritten for testing;
  the smoke script's collision refusals stay.
- Evidence files contain hashes, versions, device names and timings only; no
  credentials, private recordings or user paths.
- Touched files leave ruff clean and formatted; no new mypy errors in them.

## Review focus

- Multimedia plugins actually loaded by the frozen player, not only present on disk.
- Version propagation to installer name, wheel glob, smoke script and About screen.
- Upgrade over a 0.2.0 library: migration backup exists and the old files are intact.
- Uninstall versus explicit data deletion: two different tests, two different results.
- The two-hour row stays "omitted"; no test or doc upgrades it to passed.

### Task 1: Build, installer, probes and About screen

**Files:** `packaging/windows.spec`, `packaging/installer.iss`,
`scripts/build_windows.ps1`, `scripts/smoke_windows.ps1`, `scripts/windows_notices.py`,
`.github/workflows/windows-package.yml`, `src/distribution.py`, `src/ui/about.py`
(new), `src/ui/main_window.py`, `pyproject.toml` (`version = "1.0.0"`,
`[project.urls]`), `docs/development/WINDOWS_DISTRIBUTION.md`; tests in
`tests/test_distribution_smoke.py`, `tests/test_ui_about.py`,
`tests/test_packaging_version.py`.

**Interfaces:** `smoke_check` report gains `checks.fts5_available` and
`checks.multimedia_loadable`; `read_project_version() -> str` shared by scripts via a
tiny Python helper; `AboutDialog` reads `build-provenance.json` and `[project.urls]`.

- [ ] Write failing tests: packaging scripts and installer take the version from
  `pyproject.toml` (grep-based test for `0.2.0` literals); smoke report has the two new
  probes and still reports no device, recording or provider activity.
- [ ] Parametrize version in `build_windows.ps1`, `installer.iss` and
  `smoke_windows.ps1`; add the probes to `src/distribution.py`.
- [ ] Re-enable the multimedia backend and plugins M3 uses in `windows.spec`; run the
  notices audit for any new binary; justify each in the inventory.
- [ ] Write failing Qt tests for `AboutDialog`: version, provenance fields, link
  shown when the URL is configured and responds, limitation text otherwise.
- [ ] Implement `AboutDialog` and the `[project.urls]` entry; bump version to 1.0.0.
- [ ] Build locally from a fresh locked environment; run `smoke_windows.ps1` on the
  local test AppId variant; record evidence under `M5_RELEASE_EVIDENCE.json`.
- [ ] Run focused tests and the full offscreen suite; self-review; commit as
  "release: 1.0.0 packaging with multimedia, probes and About screen".

### Task 2: Upgrade, acceptance matrix, accessibility and evidence

**Files:** `scripts/upgrade_windows.ps1` (new), `docs/development/ACCEPTANCE.md`,
`docs/development/M5_RELEASE_EVIDENCE.json` (new), `docs/MILESTONE_5_PLAN.md`,
`docs/development/WINDOWS_DISTRIBUTION.md`, `src/ui/settings.py` (data deletion
action), Qt accessibility tests across `tests/test_ui_*.py`.

**Interfaces:** `upgrade_windows.ps1 -OldInstaller PATH -NewInstaller PATH
-EvidenceDirectory PATH` installs N with a synthetic library, installs N+1, verifies
data, backup and startup, then uninstalls and checks preservation; a separate
`-DeleteData` run exercises the explicit deletion action.

- [ ] Obtain the real 0.2.0 installer from M1 CI artifacts; if expired, rebuild from
  `beeedac` and record the hash as a reconstruction in the evidence file.
- [ ] Write `upgrade_windows.ps1` reusing the smoke script's owned AppId, group and
  canary helpers; dry-run it with the local test variant.
- [ ] Write failing Qt accessibility tests for first run, settings, library, detail,
  ask and about: tab order, accessible names, visible focus, contrast against tokens.
- [ ] Implement the fixes those tests expose; implement the explicit data-deletion
  action with confirmation.
- [ ] Request the clean machine and credential from the user with the written
  protocol; run installation, Start launch, both-source recording, playback with seek,
  local and provider intelligence smoke with synthetic audio and text, export and
  import, upgrade, uninstall and data deletion; capture screenshots in light and dark.
- [ ] Record every run in `M5_RELEASE_EVIDENCE.json` and the matrix rows in
  `ACCEPTANCE.md`; write the two-hour row as "omitted by ruling 5".
- [ ] Run the full suite and the Windows package workflow at the release head;
  self-review; independent agent review; commit as "release: upgrade test, acceptance
  matrix and accessibility evidence".

## Closing the group

- Update `README.md` (1.0.0 installer name, upgrade and uninstall behavior, About),
  `ARCHITECTURE.md` (packaging probes), `docs/MILESTONE_5_PLAN.md` (criteria 14–20
  status and the milestone close), `ACCEPTANCE.md`, `WHATS_NEW.md` with the 1.0.0
  note, and `STATUS.md`.
- Open one PR with the two task commits, self-review and independent agent review in
  the description, CI green. The user merges and decides whether to publish a release;
  the agent never uploads a public release on its own.
