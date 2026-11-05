# M5 portability specification

Status: written 2026-11-05, not in execution (ruling 15). Implementation starts only on
an explicit user instruction after group A of M5 merges. Governing design:
`PRODUCT_DESIGN.md` §7 "Exportación/importación portable", a taken decision
(ruling 3). Spanish plan and acceptance criteria: `../MILESTONE_5_PLAN.md`, criteria
10–12.

## Intent

Let the user take selected meetings to another machine and bring them back without
losing identity, timing, participants, citations or audio integrity, and without ever
letting a package write outside the managed staging area or execute anything. The
package is the only portable form of the library; the raw SQLite file never leaves
the data root.

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

### Export

`PortabilityService.export(ids, destination, include_audio, progress, cancellation)
-> ExportResult` takes a consistent snapshot: it reads metadata and revision
references in one read transaction, pins `row_version` per meeting, and before
publishing verifies the versions are unchanged; a concurrent change aborts with
`Conflict`. With audio it checks free space for the sum of asset sizes plus manifest
and refuses early. Output is written to a temporary file in the destination directory
and renamed into place on completion; cancellation removes the temporary file. Audio
is streamed, never loaded whole. A text or Markdown export of one meeting from the
detail view reuses the existing writer in `src/output/markdown.py` and includes
transcript and metadata only.

### Import

`PortabilityService.import_archive(source, progress, cancellation) -> ImportReport`:

1. Inspect the central directory without extracting: refuse more than 10,000 entries,
   more than 20 GiB declared uncompressed, a compressed-to-uncompressed ratio above
   1:100 on any entry, or a manifest `format_version` newer than supported.
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
   `meetings/<uuid>/`, one transaction per meeting). An identical meeting (same UUID
   and content hashes) is a no-op reported as `unchanged`; the same UUID with
   different content is reported as `conflict` and nothing is written; a new UUID is
   `inserted`. `ingest_sources` records the package fingerprint.
6. Remove the staging directory on success or failure; a crash mid-publication is
   reconciled by M2 at next startup.

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
