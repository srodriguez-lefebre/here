# M5 portability specification

Status: preparation draft refreshed 2026-11-13; no implementation started or
approved by this draft. Ruling 20 supersedes the historical ruling 15 pause.
CLAUDE §11 and ruling 35 allow the core export/import work after **M2 B merges**,
subject to the normal contract-document PR and root's real B5 handoff and dispatch.
M5 A is not a blanket prerequisite for that core; claims/answers wait for M4 B
and UI/settings integration retains its actual dependencies. Governing design:
`PRODUCT_DESIGN.md` §7 "Exportación/importación portable", a taken decision
(ruling 3). Spanish plan and acceptance criteria: `../MILESTONE_5_PLAN.md`, criteria
10–12; its full milestone obligations remain unchanged.

## Intent

Let the user take selected meetings to another machine and bring them back without
losing identity, timing, participants, citations or audio integrity, and without ever
letting a package write outside the managed staging area or execute anything. The
package is the only portable form of the library; the raw SQLite file never leaves
the data root.

## Inherited contracts and pending handoffs

The accepted foundation is M2 A merge
`53712c5beedafc2d40e2abf10c13362a1c3c9f78` and contract-document merge
`73d3cb384bd422de6899d8c7b761f3f3159ea3f1`. These identify the foundation,
not an implemented portability service. M2 B merge and its final B5 API/evidence,
M4 B claims/answer adapters, M3 library/CLI and citation playback handoffs, and M5 A
settings integration remain **PENDING**. Proposed interfaces below remain plans.

- Preserve the actual schema-2 catalog's immutable revision identity and number,
  active revision, segment timing and exact `revision_audio_assets` relationships.
  Audio belongs to the referenced revision, not whichever asset is newest.
  Unknown dates and unavailable timing remain null/unknown; do not infer them from
  paths, filesystem timestamps or export time. The manifest timestamp is only the
  export timestamp. Preserve participant IDs, nullable assignments and roles
  (R46/R48); speaker labels alone are not global participant identities.
- Receipt 3 and marker 3 are managed durability/recovery contracts, not permission
  to replay arbitrary package files. Package metadata is validated data; import
  uses the accepted M2 publication/reconciliation path. No parallel ingestion,
  receipt rebuild, FTS rebuild or provider transcription path is introduced.
  Pending operations, source markers and suppression records are not portable
  authority. Existing malformed-marker/unknown-owner warnings and native ownership
  and path guards remain fail-closed; import does not clear or bypass them.
- R42 suppression applies before export exposure and before import publication,
  including pending deletions and exact origin/fingerprint aliases. A surviving
  receipt or archive does not restore a deleted identity. Invalid/inaccessible
  suppression fails closed, without inferring purge authority. Final publication
  and cleanup must use B's accepted operation ownership and inventory.
- R50 specifies `memory.service.catalog_read_view(library)`: the caller opens and
  closes a fresh LibraryConnection without an active transaction; B owns the
  existing lock and the single read transaction. Its immutable
  `SuppressionSnapshot.suppressed_meeting_ids` and `fingerprint` include resolved
  aliases in that same view. Export reuses the guard's read internals under the
  same lock, without a second parser, nested transaction or writer migration.
- R51's private `_derived_row_purgers: list[Callable[[MemoryRepository, str], None]]`
  runs trusted SQL on the same connection at the start of the final delete
  transaction, before removing core relations and FTS rows. A purger owns no
  transaction, additional connection, file I/O or provider work. Failure rolls back
  core and derived SQL together and leaves deletion pending for idempotent replay.
  This is a deletion hook. Snapshot, citation verification and publication of
  answers and claims require the actual M4 B handoff.
- Search effects use the existing M3 R45 corpus generation/invalidation contract;
  ordinary listing remains insertion-stable. No new cursor, index or p95 allowance.

The seams below were checked against reviewed M2 B source `ec57204`.
`get_meeting` and `show` each open their own read scope. `_operation_locked`
is the existing migration-capable writer body. The export read uses the R50
internals and closes its SQLite transaction before streaming or publishing.
Where native ownership applies, preserve B's explicit registered roots and
bounded native-before-catalog owner acquisition under the publication lock.
`recovery_publication` and `_publisher_operation` implement R62 for native
session markers, parent/reservation and receipt proof. Portable import extends
the existing journal and repository without borrowing that native retry scope.

Compare the normalized local import origin and manifest fingerprint directly
with durable R42 pairs before inserting `ingest_sources`, even if the current
catalog has no row resolving that pair to an alias. Keep the UUID check as well.
Recognize the registered `import-*` role in startup staging cleanup and R54
deletion staging recovery after catalog loss. Extend the existing managed
inventory and receipt rebuild for the local group role while preserving ordinary
receipt 2/3, legacy audio-only lineage and trusted retry semantics. The exact
app-created export temporary is verified separately before R42 completes and
does not add filesystem work to the R51 SQL hook.

After M2 B's merge, the independent core comprises archive inspection/name and
size validation, manifest/hash handling, streamed export/extraction, and meeting,
revision, segment, participant and audio round-trip through M2's publication path.
It does not require settings UI or intelligence to implement those portions.
Claims/answers and their final serialization, verification and publication adapters
remain blocked on M4 B. An answer-free synthetic core fixture proves only that
subset; empty placeholders or omitted unfinished adapters cannot satisfy the
answer/citation acceptance below. Final searching/seeking, Qt/CLI controls and
settings-root integration require the real M3/M5 A handoffs. No temporary product
flow or duplicate adapter is authorized to bypass a missing dependency.

## Decisions

### Package format

A `.here.zip` archive (ZIP, deflate or stored) with:

- `manifest.json`: `format_version`, exporter version, UTC timestamp, list of meeting
  ids, for every file its relative path, size and SHA-256, `audio_included` flag, and
  the ids of participants and answers included.
- `meetings/<meeting-uuid>/meeting.json` (metadata, status, lifecycle events, legacy
  id), `revisions/<n>/revision.json` and `segments.json` for every revision,
  `audio/<asset-uuid>.wav` only when audio is included.
- `participants.json` and `speaker_assignments.json` for the included meetings.
- `answers/<answer-uuid>.json` with claims and citations only when every meeting the
  answer cites is in the package; otherwise the answer is omitted and listed under
  `omitted` in the manifest.

Never included: the SQLite database, WAL or backups, absolute paths, `.env`,
credentials, logs, settings, pending operations or staging content.

Construct the manifest and entity documents from explicit field allowlists.
Portable content includes user-authored metadata and transcript evidence, entity IDs,
revision numbers, active and historical links, lifecycle events with their defined
payload fields, participant roles and assignments, and audio metadata and digests.
Represent revision audio links by their meeting, revision, asset and ordinal with
the validated source basename as a label. Resolve package audio through the asset
UUID and its package-relative entry, independent of that label.
Keep local revision origins, managed storage paths, warnings, local row versions,
operation markers, receipts and suppression evidence out of the archive. Keep
hashes and sizes for all listed payloads and validate the projection on import.

### Export

`PortabilityService.export(ids, destination, include_audio, progress, cancellation)
-> ExportResult` uses the composed `MemoryService` and its configured root.
The memory-owned context `_portable_export_view(ids: tuple[str, ...]) ->
ContextManager[_PortableExportView]` validates canonical IDs, opens a fresh
connection and acquires the catalog lock. It reads all selected meetings and R50
suppression evidence in one read transaction. Suppressed or missing meetings are
unavailable through the existing typed errors. The detached view holds meeting
versions, the suppression fingerprint, all portable rows and relations, and
root-relative audio references held locally for streaming. It exposes neither
a connection nor a repository. Close the read transaction and connection. Under
the same catalog lock, use B5's bounded acquisition of existing per-meeting OS
owner leases and applicable native leases in stable order. Release the catalog
lock before returning the detached view. Keep the owner leases until export exits.
A live export excludes deletion of those meetings.

With audio, check space for the sum of asset sizes plus manifest and refuse early.
Use B5's managed path and role checks before opening each asset. Stream and verify
its recorded size and digest without holding a SQLite transaction. Missing or
invalid bytes prevent publication of an audio-included package.

Within the export view, `_open_export_destination(destination: Path) ->
ContextManager[_OwnedExportTemporary]` exclusively creates a temporary sibling of
the requested destination. Before writing the first package byte, durably register
the locally allocated operation ID, selected meeting IDs, exact transient sibling
and strong file and parent identities in the configured library. Keep only the
metadata needed to recover that operation. Bind the open output, rename and cleanup
to this scoped ownership proof. The Windows binding for checking file and parent
identity at use remains an implementation check. A name, path, operation
ID or successful check separated from the mutation is insufficient on its own.

Write to the owned output and finish and close the ZIP stream before publication.
The view's `publish(temporary: _OwnedExportTemporary, *,
cancelled: Callable[[], bool]) -> None` is a one-shot call within its context.
It reuses the existing catalog lock and R50 read internals with a fresh connection.
Recheck the suppression fingerprint, each meeting's `row_version`, and
the full ordered detached state used by the export, including participant,
assignment, event and audio relationships and managed references. A mismatch
raises `Conflict`. Close the read transaction and perform only the final atomic
rename of the proven sibling while retaining that same lock and the owner leases.
Keep the public R50 lifetime unchanged and use shared internals to avoid recursive
locks or BEGIN. Check cancellation immediately before the rename. Retire the local
registration after verified publication or cleanup of the owned temporary. If the
process dies after rename but before retiring the registration, verify the parent
and absence of that sibling without opening or removing the published archive.

On cancellation, failure or startup after a process kill, the scoped operation
cleans up only its proven transient sibling. Startup and deletion coordinate with
the existing catalog lock and meeting leases before retrying cleanup. Live exports
remain busy. If a registration, file or parent cannot be verified, retain it,
report a bounded recovery warning and leave cleanup pending. Deletion can complete
only after its required temporary cleanup is verified, with existing R42/R51
pre-acceptance and pending behavior. This proof is specific to the app-created
transient sibling. The existing managed-root purge still handles its own inventory,
and neither mechanism follows an external source path or claims authority over a
published ZIP, the destination directory or unrelated files. Keep registration
discoverable by startup if SQLite has to be rebuilt.

Text or Markdown export of one meeting from detail reuses `src/output/markdown.py`
and includes transcript and metadata only.

### Import

`PortabilityService.import_archive(source, progress, cancellation) -> ImportReport`:

1. Inspect the central directory without extracting: refuse more than 10,000 entries,
   more than 20 GiB declared uncompressed or a manifest `format_version` newer than
   supported. For each entry refuse an uncompressed size greater than 100 times
   its compressed size. A nonempty entry with zero compressed bytes is rejected.
   An empty entry does not exceed the ratio limit.
2. Validate every entry name: reject absolute paths, `..` components, UNC prefixes,
   drive prefixes, alternate data streams (`:`), Windows device names, names that
   normalize (NFC, case, trailing dots and spaces) to a duplicate, symlink entries,
   and anything outside the manifest.
3. Extract into `<data_dir>/.staging/import-<uuid>/` with bounded streaming,
   verifying each file's size and SHA-256 against the manifest as it is written.
4. Validate relations in staging: every revision belongs to a listed meeting, every
   citation resolves to an included revision and segment, every audio asset matches
   a meeting.
5. Publish through M2's recoverable operations (`pending_operations`, rename into
   `meetings/<uuid>/`, one transaction per meeting). The private
   `MemoryService._publish_portable_import(batch: _ValidatedPortableImport, *,
   cancelled: Callable[[], bool])
   -> tuple[_PortableMeetingOutcome, ...]` accepts detached, fully validated
   meeting bundles, a locally registered staging handle and the cancellation
   predicate. Preflight all meetings against the destination under the existing
   catalog lock before any meeting is published. Check UUID ownership, R42
   suppression, path and asset roles, and the complete portable projection.
   A shared participant UUID may be reused only for an exact display-name match.
   Equal names with different
   UUIDs retain their distinct identities. Cross-entity collisions and any
   differing content for an existing meeting UUID yield `conflict`, with no
   meeting published from that batch. An exact match yields `unchanged`, with
   no changes to local rows, files or provenance. Compare all portable fields
   and digests, excluding archive- and installation-local metadata. A no-audio
   package compares the same no-audio projection of the existing meeting and
   leaves existing audio intact, with local byte availability excluded from
   that comparison. Unsafe or suppressed input yields `rejected`. A batch
   stopped by a conflict reports `rejected` for otherwise insertable meetings.
   Each newly committed UUID yields `inserted`.
6. Remove the staging directory on success or failure; a crash mid-publication is
   reconciled by M2 at next startup.

Before extracting any entry, open the memory-owned
`_portable_import_stage(inventory: _PortableImportInventory) ->
ContextManager[_PortableImportStage]`. Extend M2's private pre-journal
registration to bind `.staging/import-<uuid>` and its exact inventory to a
locally allocated operation ID. Under the existing lock, create the directory
exclusively and flush the local registration before releasing that lock or
extracting payloads. Keep its real OS operation lease in staging through
publication and ordinary cleanup. Only validated format-1 roles and
application-generated working paths enter the local inventory. This phase
writes only inside `.staging/`, including its registration and lease, and
reuses the existing catalog lock on the configured library.
Validate every child against that inventory and the existing path/ownership
guards before cleanup, including partial files in registered roles left by an
interrupted extraction. Unknown or ambiguous material is retained with a recovery
warning. Archive metadata never authorizes deletion, and a live import lease
prevents startup reconciliation from removing its staging. Remove owned payload
files before the local proof. Release and unlink the stage lease under the
catalog lock after payload cleanup, so Windows can remove the lock file without
an ownership gap. An interruption before registration can leave only the empty
stage and recognized registration or lease metadata.

Build each meeting's complete managed tree inside import staging before the
locked publication phase. Derive canonical local audio paths from asset UUIDs
under their first referencing revision, using the existing `audio_<uuidhex>.wav`
role and explicit links for reuse across revisions. Validate source labels with
the shared path policy and keep them as labels. Hold the same catalog lock and
applicable meeting owner leases from whole-batch preflight through the last
commit to keep the preflight valid. Use the existing publisher and reconciler to
register each new meeting's prepared operation in `pending_operations` before
atomically renaming the staged tree to its absent `meetings/<uuid>` destination.
The entity transaction inserts all revisions and exact IDs, numbers, active
pointer, timestamps and nullable values, segments, lifecycle events, participants,
assignments, assets, audio links, source evidence and active FTS through the
existing repository rules. It performs no file I/O.
Finish with local publication receipts and lifecycle state derived from committed
rows. Before the first member receipt can be observed, durably register the
import group and its exact member revision IDs against the meeting and local
operation. Every imported receipt is written in a group-bound role and identifies
its group from first publication. Use the portable role and binding to reject a
missing or malformed group reference without falling back to a legacy receipt or
trusted retry. Ordinary M2 receipts and retries keep their existing semantics.

The reconciler rolls back a prepared meeting when the local journal and catalog
prove its entity transaction did not commit. It completes durable evidence when
they prove the full commit. Emit local completion proof in the existing lifecycle
state only after verifying the full committed entity set and the exact registered
revision set with all matching local receipts and artifact identities. Keep group
registration and completion proof in managed local artifacts that survive catalog
loss. When restoring an absent catalog meeting, the existing rebuild accepts the
group as a unit after verifying that proof and every member. It retains a partial,
mixed, or unproven group for recovery without restoring a subset. Preserve that
requirement even when the meeting has later ordinary revisions. Extend the existing
delete inventory to cover the same registered managed roles.
Check cancellation before each meeting publication. A later kill or cancellation
preserves earlier committed meetings and leaves remaining meetings absent. Startup
reconciles meeting operations before removing their package staging and clears the
transient stage registration after verified cleanup.

Record the SHA-256 of the validated manifest bytes as the package fingerprint in
`ingest_sources` as one meeting-level source per inserted meeting. Its origin
is the normalized absolute local managed meeting directory, providing a per-meeting
key for the same package. Use that local origin on the imported revisions and derive
their managed storage paths from validated local IDs.
Treat package fingerprints as local import provenance and use the existing UUID
and exact normalized origin/fingerprint pairs for suppression. Never accept a
package-provided origin, tombstone, receipt or state as local authority.

For `audio_included=false`, preserve asset IDs, sizes, hashes, recorded metadata
and exact revision links, using the existing `missing` state and a locally derived
managed path. No WAV is created and playback remains unavailable. Textual
evidence and its timing remain unchanged. With audio included, require the
declared WAV bytes and their verified sizes and digests. Apply the same complete
projection to playback-relevant availability so a no-audio copy cannot count as
unchanged for a package with audio bytes.

Nothing in a package is executed, opened as code, or followed as a path outside
staging. Imported audio becomes a managed copy; deleting the meeting later deletes the
copy, not the package.

### UI and CLI

Library view gains Export selected (with audio checkbox) and Import package, both on
background workers with progress and cancel. CLI: `here meetings export ID... --out
PATH [--no-audio]` and `here meetings import PATH [--json]`, exit codes 0/2/3/4 as in
`PRODUCT_DESIGN.md` §5.

## Acceptance

- Export of three synthetic meetings with and without audio yields a manifest whose
  hashes match the files; the archive appears only after completion; cancellation
  leaves no partial file.
- Export refuses when free space is insufficient and when a meeting changed between
  snapshot and publish.
- Import into a fresh root reproduces meeting ids, legacy ids, revisions and their
  numbers, segments with times and speakers, participants, assignments, answers with
  citations and audio hashes; searching and seeking the imported meeting works.
- Reimport of the identical package reports `unchanged` for every meeting; a package
  with one modified revision reports `conflict` and leaves the library untouched.
- A hostile corpus (absolute path, `..`, UNC, drive prefix, ADS, `CON`, NFC and case
  duplicates, symlink entry, future manifest, 1:100,000 ratio entry, 10,001 entries,
  a file missing from the manifest, a hash mismatch) is rejected before publication,
  with nothing written outside `.staging/` and the staging directory removed.
- Process kill after extraction and before publication leaves the library unchanged
  after M2 reconciliation; kill after one meeting's commit leaves that meeting
  complete and the rest absent.
- Answers citing a meeting outside the package are omitted and listed; answers citing
  only included meetings are imported with verified citations.
- Full Windows CI suite passes with temporary roots and synthetic packages only.

## Boundaries

No cloud transfer, no sharing links, no merging of two libraries beyond import, no
encryption of packages, no import of formats other than `.here.zip` and the M1
session folders already handled by M2 ingestion.

## Technical references

- [zipfile](https://docs.python.org/3/library/zipfile.html): central directory
  inspection, `ZipInfo.file_size` and `compress_size` for ratio checks.
- [Windows naming conventions](https://learn.microsoft.com/windows/win32/fileio/naming-a-file):
  reserved device names, trailing dots and spaces, alternate data streams.
