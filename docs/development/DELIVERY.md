# Autonomous product delivery

The user delegated product decisions, implementation, verification, GitHub pull requests,
Copilot review requests, replies to every review observation and merges on 2026-10-03.
This document is the recovery map for continuous execution; a milestone is not complete
merely because its code exists or its unit tests pass.

## Product authority and architecture

`docs/MASTER_PLAN.md` defines the five milestones. `docs/MILESTONE_1_PLAN.md` and
`docs/LIVE_LOGO_PLAN.md` govern capture and the overlay. Retain one Python process,
PySide6 Widgets, a shared application controller, Windows 11 support, explicit
cancellation semantics and local user-owned audio. Derived knowledge must always link
to its supporting meeting and original segment; missing timestamps or speakers stay
missing. No cloud synchronization, accounts, unsolicited transmission or automatic
audio deletion is introduced.

For structured memory use transactional SQLite with foreign keys and versioned
migrations, retaining portable recoverable session artifacts. Browsing and local search
must work without an API key. Remote intelligence must be an explicit user command,
with source validation and a labeled offline extractive path. The detailed specifications
and individual execution plans are added here before their implementation.

## Delivery sequence

1. M1 capture reliability: preserve partial capture on failures, bounded live queues,
   structured timestamps/speakers at the artifact boundary; then default devices,
   diagnostics, restart recovery, stop-save-exit, per-user paths; then Windows installer
   and an honest acceptance matrix. Split into independently reviewable PRs.
2. M2 structured memory: durable identities, transcript revisions, segments,
   participants, atomic catalog updates and recovery reconciliation; lifecycle and
   deletion integrity; migration of existing artifact sessions.
3. M3 navigation: GUI and CLI meeting list/detail/filter/search, contextual results
   and original-audio playback at recorded timestamps. Missing timing never implies
   a fabricated seek position.
4. M4 intelligence: retrieve across selected meetings, answer and extract decisions,
   tasks and blockers with validated evidence references, explicit uncertainty and
   preserved generation provenance.
5. M5 daily product: configuration, data control, export/import, reliable startup and
   logs, installer/uninstaller, explicit update checks and practical interoperability.

## Review and merge gates

- Each implementation task follows meaningful red/green tests and self-review.
- A fresh independent reviewer checks task requirements and code quality.
- Each PR has multiple cohesive commits where appropriate and is validated on its head.
- Push and create the PR, attach it to this chat, request `@copilot` with GitHub CLI.
- Wait for Copilot review of the current head; reply to every inline observation with
  its disposition, evidence and fix commit or reasoned rejection. Fix real defects.
- Re-request review after substantive fixes; merge only after successful checks and
  no unresolved material findings. Never treat review silence as approval.
- Use merge commits to preserve the requested commit groups. Update this ledger with
  PR, head, review and validation evidence before proceeding.

## Acceptance and evidence

Baseline: main `2bdb6ef`, 163 passing tests on the original workspace. Work occurs in
`C:/Users/savar/.codex/worktrees/five-milestones/here`, not the user's main checkout.
The local Windows default devices enumerate at 44.1 kHz microphone and 48 kHz loopback.
Enumeration does not certify captured signal, long meetings or a clean installation.

Define a long meeting as at least two hours. Automated virtual-duration tests measure
bounded resource behavior and completeness, but do not replace a two-hour hardware
run. Keep manual/hardware/provider/clean-install evidence explicit and leave acceptance
open wherever it has not actually been observed.

## Progress

- 2026-10-03: user delegation recorded; isolated managed worktree created.
- GitHub admin/push access and Copilot reviewer availability confirmed.
- Read-only capture audit and product architecture delegated to independent agents.
- Ruling: use the explicit autonomous delegation instead of repeated design approvals;
  preserve written specs and plans for review. Cost if wrong: reversible implementation
  rework, visible in committed decisions and PRs.
- Ruling: retain historical local docs ignored; version only the authoritative plans
  and new `docs/development/` evidence. Cost if wrong: older documents need explicit
  inclusion later; no private local scratch is accidentally published.
- Ruling: preserve supplied audio and facts; do not certify unobserved hardware tests.
  Cost if wrong: release acceptance takes longer, while code can continue advancing.
- M1 capture implementation: commits `c20ac3b` and `081ef94`; synthetic full suite
  reached 194 passing tests. Real synthetic TTS/provider smoke preserved four timed,
  scoped utterances. Wheel built and installed non-editably into isolated Python 3.13;
  GUI imports, Qt offscreen initialization and actual CLI help passed off-repository.
- Independent capture review reproduced two P1 defects: catch-up silence was omitted
  from the live timeline, and destructive cancellation during failure cleanup could
  still persist discarded audio. Both accepted for correction before PR creation.
- Ruling: serialize capture cancellation against the recovery-persistence cutover.
  Accepted capture cancellation deletes its partial sources; cancellation after the
  atomic processing cutover preserves audio and reports recoverable cancellation.
  Cost if wrong: recovery-state implementation rework; discarded private material
  must never survive an accepted destructive cancellation.
- Independent desktop-plan review adopted seven amendments covering actual WAV header
  checkpoints, atomic publication, preparation exit, diagnostic ownership, path
  validation, ready-time device provenance and credential precedence.
