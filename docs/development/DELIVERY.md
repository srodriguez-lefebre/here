# Autonomous product delivery

The user delegated product decisions, implementation, verification, GitHub pull requests,
Copilot review requests, replies to every review observation and merges on 2026-10-03.
This document is the recovery map for continuous execution; a milestone is not complete
merely because its code exists or its unit tests pass.

## Current authorized objective

On 2026-10-03 the user narrowed execution to **finish M1, then stop**. M2-M5 remain
future plans; their implementation is outside this run. The user also requested
Windows-only development and validation. Automatic CI and release gates use Windows;
the previous Ubuntu workflow is retained separately for explicit manual dispatch and
will not be run here. Earlier Ubuntu results below remain historical observations.
Prioritize the remaining desktop fixes, Windows distribution and necessary M1
acceptance; no later-milestone work or optional validation expands this scope.

The user's latest explicit instruction on 2026-10-03 omits both the two-hour
capture/pause/recovery/resource run and long generated-audio transcription with
timestamps/diarization. These cases are excluded from this delivery, not passed.
Complete the remaining Windows distribution, actual runtime/installation checks,
notices/checksums, reviewed PRs and evidence audit; then close M1 under this revised
scope and stop. Existing short hardware/fault/provider observations retain their
actual provenance. M2-M5 remain outside this run.

## Product authority and architecture

`docs/MASTER_PLAN.md` defines the five milestones. `docs/MILESTONE_1_PLAN.md` and
`docs/LIVE_LOGO_PLAN.md` govern capture and the overlay. Retain one application owner,
PySide6 Widgets, a shared application controller, Windows 11 support, explicit
cancellation semantics and local user-owned audio. Derived knowledge must always link
to its supporting meeting and original segment; missing timestamps or speakers stay
missing. No cloud synchronization, accounts, unsolicited transmission or automatic
audio deletion is introduced.

Ruling: allow a short-lived isolated hardware helper where the backend cannot safely
bound opening, reading or cancellation. The desktop spec permits that isolation;
the GUI, overlay and shared controller remain one application, with no persistent
service. A timeout cannot release audio resources until the helper actually closes.
This is a deliberate refinement of the original single-Python-process design, needed
to recover from a stalled driver without destroying a resource-owning thread. Cost
if wrong: bounded IPC complexity and runtime overhead, to verify in desktop/bundle tests.

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

Planned PR groups preserve task commits and independent task reviews: M1 capture,
desktop recovery and distribution/acceptance; M2 immutable catalog (tasks 1–3),
lifecycle/erasure (4–5) and shared persistence (6–7); M3 search/actions (1–2) and
library/CLI/evidence (3–6); M4 grounded core (1–5) and product flows/evidence (6–8);
M5 settings/runtime (1–2), portability/privacy (3–5) and updates/product acceptance
(6–8). These are delivery boundaries, not completed milestones or published releases.

The first group, M1 capture reliability, merged as [PR #12](https://github.com/srodriguez-lefebre/here/pull/12)
on 2026-10-03 07:52:15 UTC, merge `ca07aed8f6d5abf43c85e03ca0107d60096c3e03`.
The next branch is `codex/m1-desktop-recovery`; its two planned tasks implement the
restart-safe storage and desktop controls. All five milestones remain open.

Desktop Task 1 is independently approved at `d366711`: per-user configuration,
optional credentials, writer-owned WAV/journal checkpoints, stable capture UUID and
local recovery, shared CLI pending publication and interruptible catch-up. One
important independent finding was fixed before approval: file transcription now
publishes recoverable pending metadata before provider work. The final source-head
suite passed 524 tests with six existing Windows symlink-privilege skips (34.02 s);
new real junction/hardlink and subprocess-kill cases ran. No desktop Copilot review,
hardware-long, installer or whole-M1 completion is claimed here.

Desktop Task 2 is independently approved at `76d317a`: immutable actual opened devices,
asynchronous diagnostics/recovery, atomic preparation stop, completion-gated stop/save
exit and bounded hardware helpers. Actual Windows Job Object parent-death and stalled
backend/IPC-pressure regressions ran. Final source-head suite: 558 passed, six existing
symlink-privilege skips in 53.72 s; changed-file lint/format and diff checks passed.
The current synchronous provider transport remains cooperative and can wait through
its SDK timeout/retries; Qt retains ownership without blocking its event loop.

A bounded native Qt Windows check at that same source head passed without provider
calls: no-key startup, local interrupted discovery, key preflight preserving the journal,
actual default-device names, three-second microphone no-signal and authored-tone
loopback signal diagnostics, ordinary active close/hide and repeated Salir from pause.
An intentional local transcription failure left discoverable audio; exit followed
durable persistence and actual cleanup (0.372 s for this local fixture only).
Qt heartbeat maximum observed gap was 49.6 ms across 688 samples. Probe-owned raw
audio was removed after worker/reservation cleanup. Real frozen helper launch,
installation, long recording and real-provider long acceptance remain open.
Safe observations: [`M1_DESKTOP_EVIDENCE.json`](M1_DESKTOP_EVIDENCE.json).

Whole-branch review found two recovery integration defects: a missing normalized WAV
hid valid raw-backed sessions, and a missing-key retry removed the selector choices.
Commit `490b77c` fixes both and strengthens the preparation-stop test. Five failing
regressions became eleven passing; the final committed-head suite passed 569 tests
with six existing symlink-privilege skips (56.92 s). Scoped independent review accepts
the whole-branch source gate. A separate native Qt authored-WAV probe at `490b77c`
confirmed fresh raw-backed recovery, retained choices after missing-key error and one
explicit same-window retry after editing the effective env file, reconstructing 240
frames with UUID, human ID, selected filename and raw bytes preserved. No hardware
or provider was used for this correction probe. External Copilot and head CI gates
still precede merge; installer and full-M1 acceptance remain separate.

The legacy private CLI helper deferred in PR #12 review `5399598693` is consolidated
with SessionProcessor in desktop Task 1; its actual tiny-WAV normalization-failure,
preserved-local-source and later-retry regression passes. The original deferral below
records the historical first-PR disposition, rather than an outstanding source defect.


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
- Main ruleset `14750729` requires an approving/code-owner review and explicitly
  permits repository administrators to merge. The delegated owner account has ADMIN
  permission. Use that configured exception, without changing rules, only after the
  gates above pass and with `--match-head-commit`. Copilot's COMMENTED reviews and
  independent agent reviews are recorded as such; neither is a human APPROVED review.

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
- Release decision: planned M1 Windows delivery is `v0.2.0`; the complete five-milestone
  product targets `v0.3.0`. These are targets until built, reviewed and published.
  Existing official release is tag `MVP`, display name `v0.1.0`, published 2026-04-11,
  without runtime assets. Treat that non-version tag as a legacy release, not a
  trusted comparable installer update. Reuse the actual M1 artifact for M5 upgrade tests.
- Review corrections in `7602c18`: every catch-up silence block follows disk order into
  live capture; shared atomic cutover prevents accepted destructive cancellation from
  persisting audio, and late processing cancellation retains matching metadata.
  Six regression cases and affected suites passed (55); full committed-head suite:
  200 passed in 16.86 seconds. Independent re-review precedes PR creation.
- First M1 subset PR: https://github.com/srodriguez-lefebre/here/pull/12. Independent
  task and whole-branch reviews accepted the implementation. GitHub recorded the
  requested Copilot review; review `5399146085` inspected head `c3dc790` and identified
  two accepted defects (comments `4171819200`, `4171819228`). Commit `4860095` fixes
  successful-result cleanup after late cancellation and valid nested timestamp alias
  fallback. Ten cases reproduced nine failures before correction; 57 focused tests
  and the full suite of 211 passed. Both observations were replied to and resolved.
- Initial CI exposed an existing UTC string-comparison assumption and a missing EGL
  runtime on Ubuntu. Commit `545174e` adds deterministic UTC/non-UTC checks using aware
  datetime equality and installs libegl1. Windows and Ubuntu jobs both passed on that
  head (run `37099618688`). Later review-fix CI runs `37102433292` (`6f1f88d`) and
  `37103530702` (`240858e`) also passed. The latter ran 266 tests on each platform,
  including the symlink case omitted locally, and built wheel/source distributions.
- Further Copilot observations were accepted and replied to: oversized timestamp
  integers (`5399210060`) fixed in `6ac6a0f`; stale segment evidence (`5399274184`)
  fixed in `e48f9ee`; cleanup ordering tightened in `6f1f88d` and explained in the
  reply to review `5399321887`. Pair-publication failure (`5399348275`, inline
  `4171988549`) fixed in `240858e`, independently accepted, replied to and resolved.
- Review `5399404470` of `240858e`, received 2026-10-03 06:39:41 UTC, identified
  unsafe recovery output entries (inline `4172034351`) and reversed timing pairs.
  Commit `a9cb09a` rejects pre-existing redirects, hardlinks and nonregular managed
  entries before retry reads/provider calls and stages text, normalized WAV and source
  copies before replacement. Commit `7ed996a` keeps segment text/speaker but sets both
  contradictory time bounds to null before offsets, including directly typed segments.
  The stale-ledger observation is addressed by this event register.
- Corrective-code checkpoint `7ed996a`: full offscreen suite **385 passed, 6 skipped
  in 10.62 s**. All skips need local Windows symlink privileges; real hardlink and
  junction cases passed. Focused security suite: 129 passed/6 skipped; focused timing
  suite: 168 passed/1 skipped. Ruff, format and diff checks passed. This local result
  does not substitute for hosted CI of the final PR head.
- Final capture corrections: `ac46ff8` validates every referenced WAV geometry before
  retry normalization/provider work and derives provenance duration from the header;
  `3162a74` coerces direct typed bounds before order checks, preserving valid companion
  values and source text. Review `5399510672` findings were replied to and resolved;
  independent scoped review accepted requirements and quality. Final code suite:
  463 passed/6 local symlink-privilege skips in 12.98 s. Hosted CI `37107198007`
  at exact `3162a74` passed 469 Windows tests without skips and 467 Ubuntu tests with
  two Windows-only junction skips; both distributions built on both systems.
- Copilot review `5399598693` inspected exact `3162a74` and confirmed those two fixes.
  Its body also identifies a retained private CLI helper's normalization-failure
  record without local raw references. The complete source call graph shows only test
  callers of `_save_transcription`; current recording commands use the shared
  controller/SessionProcessor, whose failure persistence already owns local sources.
  Ruling: consolidate this pre-existing helper in desktop Task 1, with a real tiny-WAV
  failure-to-local-retry regression, alongside early CLI managed-session preflight;
  it does not block the delivered production capture subset. Cost if wrong: external
  callers of this private helper need their original temporary audio until that PR.
  The observation is accepted and publicly answered in comment `5966927016`, not
  declared fixed or silently discarded. M1 stays open until consolidation is verified.
- All six inline threads were replied to/resolved and every review-body observation
  dispositioned before merge. PR #12 merged using the configured administrator
  permission and exact-head guard at `3162a74`, preserving its commit history in
  `ca07aed`. This is a COMMENTED Copilot review plus independent agent reviews, not a
  human APPROVED review. Full immutable evidence is in
  [`M1_CAPTURE_EVIDENCE.json`](M1_CAPTURE_EVIDENCE.json); short native observations are
  in [`M1_NATIVE_EVIDENCE.json`](M1_NATIVE_EVIDENCE.json), with original code provenance.

Desktop Task 1 also closes the parked catch-up control and CLI early-read boundaries
and preserves capture UUID as additive `meeting_id` for M2. These amendments are in
the desktop spec/plan. Journals, optional credentials and desktop controls are now
implemented and independently reviewed on this branch, with their own observations;
they were not facts implied by the first capture merge.

This register records immutable observed checkpoints. The live
[PR review/check/merge record](https://github.com/srodriguez-lefebre/here/pull/12)
is the authority for subsequent review dispositions, final-head CI and merge status;
earlier passing runs never waive a later-head gate. No milestone is complete here:
the desktop branch still needs its whole-branch/Copilot/current-head CI merge gates;
installed distribution, two-hour acceptance and M2–M5 still need their planned evidence.
