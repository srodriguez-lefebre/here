# M2 lifecycle and deletion specification

Status: written 2026-11-05, not in execution (ruling 15). Second PR group of
milestone 2; depends on the merged contracts of `M2_CATALOG_SPEC.md` (group A).
Governing design: `PRODUCT_DESIGN.md` §4 "Estados, publicación y recuperación" and
"Borrado íntegro", a taken decision (ruling 3). Spanish plan and acceptance criteria:
`../MILESTONE_2_PLAN.md`.

## Intent

Make a meeting a thing the user can name, attribute and remove. Participants are
explicit, speaker labels are associated by hand and reversibly, edits are optimistic
and versioned, startup reconciliation finishes whatever a crash interrupted, and
deleting a meeting leaves nothing derived from it anywhere the application manages.

## Decisions

### Participants and speaker assignment

`participants` rows are created manually with a display name; a name never implies
biometric identity. `meeting_participants` links a participant to a meeting with an
optional role. `speaker_assignments` maps a `speaker_key` within its scope (meeting,
revision or provider request) to an optional participant. Assignment is explicit,
reversible and never inferred across requests: `speaker_0` from two chunks stays two
unassigned labels until the user links each one. Removing an assignment keeps the
segment's original `speaker_key`.

### Optimistic editing

`MeetingPatch` carries optional title and user notes only; no free SQL fields.
`update_meeting(meeting_id, patch, expected_version)` compares `meetings.row_version`
inside `BEGIN IMMEDIATE`, increments it on success and returns `Conflict` when the
stored version differs. The CLI and the GUI both pass the version they displayed.
Edits never touch transcript revisions or audio.

### Durable status transitions

Group A introduced `meetings.status`. This group owns the transitions that do not
come from a controller outcome: startup marks a `recording` or `processing` meeting
whose journal or pending operation is stale as `interrupted` with its available
material; `deleting` is entered by the deletion use case and left only by completion.
No transition issues a network call. `lifecycle_events` records every transition with
`occurred_at`, `recorded_duration_ms` and a validated payload without secrets.

### Reconciliation

`MemoryService.reconcile() -> RecoveryReport` runs at startup and on demand. It
walks `pending_operations` in state `prepared` and either completes or reverts each
one idempotently, using only paths validated below the managed root; it marks stale
`recording`/`processing` meetings `interrupted`; it verifies `audio_assets` presence
and hash and sets `available`/`missing`/`corrupt`; it never deletes unrecognized
files, never converts a malformed journal into a completed meeting, and never imports
anything whose origin fingerprint is in `deleted_origins`. The report lists what it
completed, reverted, marked and skipped, with reasons.

### Deletion

The GUI confirms once what will be removed; the CLI requires `--yes`. The service
rejects deletion of a meeting with capture, processing, import or export in progress
(`Busy`). Playback, if any, is stopped and its file handle released before the
operation starts (M3 wires the player; this group exposes the hook).

Sequence: transaction sets `status=deleting`, removes the meeting from FTS and from
every query by default, writes a `pending_operations` row of kind `delete` with the
list of owned relative paths, and commits. Then the file phase removes, in order,
owned transcript snapshots, segment documents, audio assets, temporary exports and
caches under `meetings/<uuid>/`, refusing any path that resolves outside the managed
root or through a link. Then a final transaction removes rows (revisions, segments,
assignments, events, assets, FTS remnants, M4 answers and claims that cite the meeting
once those tables exist) and the meeting itself, records the origin fingerprint in
`deleted_origins`, marks the operation complete and checkpoints the WAL where
possible. Managed backups that contain the meeting are removed or regenerated.

If Windows denies a file removal, the operation stays `prepared`, the meeting stays
`deleting` and hidden, the UI shows "deletion pending", and the next reconciliation
retries locally. Deletion never claims success it did not achieve. Deleting an
imported meeting never touches the external origin folder; the UI says so when the
meeting came from an import. The application does not promise forensic erasure on
SSDs or deletion of copies the user made elsewhere.

### FTS consistency

The FTS5 index created by group A is updated inside the same transactions that
activate a revision or delete a meeting. `reindex` rebuilds it from active complete
revisions only. Partial revisions are indexed only when explicitly selected (M3).

## Acceptance

- Creating a participant, linking it to a meeting, assigning a `speaker_key` in one
  request scope, and removing the assignment leaves the segment's original label
  intact and the participant row present.
- Two chunks with `speaker_0` each remain two independent unassigned labels; no
  automatic merge by equal label or similar name.
- `update_meeting` with a stale `expected_version` returns `Conflict` and changes
  nothing; with the current version it increments `row_version` and records a
  lifecycle event.
- Killing the process after the deletion transaction commits but before files are
  removed, then restarting, finishes the deletion: no rows, no FTS hits, no files
  under `meetings/<uuid>/`, origin fingerprint present in `deleted_origins`.
- Killing the process during the file phase with one file held open (simulated
  permission denial) leaves the meeting `deleting` and hidden, reports "deletion
  pending", and a later reconciliation completes it once the handle is released.
- A deletion attempt while the meeting is `processing` returns `Busy`.
- Deleting an imported meeting leaves the external origin folder byte-identical, and
  reconciliation does not re-import it because of `deleted_origins`.
- A managed backup taken before migration that contains the deleted meeting is removed
  or regenerated; inspection of every `.sqlite3` and `.bak` under the data root finds
  no row, FTS token or text of the deleted meeting.
- Startup with a stale `processing` meeting and no pending operation marks it
  `interrupted`, lists its available audio, and issues zero provider calls.
- Startup with a `prepared` publication operation from group A completes or reverts
  it idempotently on repeated runs.
- An asset whose file is missing or whose hash changed is marked `missing`/`corrupt`
  without deleting anything and without hiding the meeting's text.
- Paths in `pending_operations` that resolve outside the root or through a symlink or
  junction are refused and reported; sibling valid operations still run.
- Full Windows CI suite passes with `QT_QPA_PLATFORM=offscreen`, a synthetic
  `OPENAI_API_KEY`, temporary roots and no real `.env` or private recordings.

## Boundaries

No listing, search UI, playback, export, import, answers or settings UI. No automatic
retention or purge. No deletion of anything outside the managed root. No global
speaker identity. No provider call. The M4 `answers`/`claims` purge is a hook this
group defines and M4 fills; this group's tests use a synthetic table in its place.

## Technical references

- [SQLite FTS5 secure-delete](https://www.sqlite.org/fts5.html#the_secure_delete_configuration_option):
  distinct from the core `secure_delete` pragma; probe at runtime before relying on it.
- [SQLite WAL checkpoint](https://www.sqlite.org/wal.html#ckpt): required so deleted
  content does not linger in the `-wal` file.
- [Windows file deletion semantics](https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-deletefilew):
  a file opened without `FILE_SHARE_DELETE` cannot be removed; retry on reconciliation.
