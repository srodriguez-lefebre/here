"""Local immutable file publication with a recoverable SQLite operation journal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import UUID, uuid4

from here.memory.artifacts import (
    SessionArtifacts,
    file_hash,
    managed_path,
    normalized_origin,
    read_artifacts,
    read_meeting_snapshot,
    read_meeting_state,
    read_pending_intent,
    read_retry_artifacts,
    validate_artifact_roles,
    write_durable_json,
)
from here.memory.database import (
    Busy,
    LibraryConnection,
    LibraryError,
    migrate,
    open_library,
    storage_error,
)
from here.memory.models import (
    AudioAsset,
    DeletionResult,
    IngestSource,
    Meeting,
    MeetingDetail,
    MeetingParticipant,
    MeetingPatch,
    MeetingStatus,
    Participant,
    PendingOperation,
    ReindexReport,
    RevisionAudioAsset,
    Segment,
    SpeakerAssignment,
    SuppressionSnapshot,
    TranscriptRevision,
)
from here.memory.repository import MemoryRepository
from here.output.ownership import FileLease, LeaseBusy
from here.output.paths import session_artifact_path, validate_session_directory


@dataclass(frozen=True)
class FtsUnavailable:
    code: str = "fts_unavailable"
    message: str = "FTS5 unavailable; the library remains usable without its search index"


@dataclass(frozen=True)
class IngestResult:
    status: str
    meeting_id: str | None = None
    warnings: tuple[object, ...] = ()
    error_category: str | None = None


@dataclass(frozen=True)
class RecoveryOutcome:
    identity: str
    reason: str


@dataclass(frozen=True)
class RecoveryReport:
    warnings: tuple[object, ...] = ()
    completed: tuple[RecoveryOutcome, ...] = ()
    reverted: tuple[RecoveryOutcome, ...] = ()
    marked: tuple[RecoveryOutcome, ...] = ()
    skipped: tuple[RecoveryOutcome, ...] = ()


class Conflict(LibraryError):
    """The displayed version is stale; reload before editing again."""


class NotFound(LibraryError):
    """The requested catalog identity does not exist."""


class InvalidInput(LibraryError):
    """The request does not satisfy the explicit lifecycle input contract."""

    def __init__(self, message: str, *, reason: str = "invalid_input") -> None:
        super().__init__(message)
        self.reason = reason


class OwnershipBusy(Busy):
    """A live job owns the session; no provider or compatibility mutation may start."""


def _is_uuid(value: object) -> bool:
    try:
        return isinstance(value, str) and str(UUID(value)) == value
    except ValueError:
        return False


def _receipt_envelope(value: object) -> dict[str, Any]:
    """Validate untrusted containers before mapping methods or referenced reads."""
    if not isinstance(value, dict):
        raise ValueError("Managed publication receipt must be an object")
    for name in ("meeting", "revision", "hashes"):
        if not isinstance(value.get(name), dict):
            raise ValueError(f"Managed publication receipt {name} must be an object")
    for name in ("audio_assets", "segments", "revision_audio_assets", "ingest_sources"):
        entries = value.get(name)
        if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
            raise ValueError(f"Managed publication receipt {name} must be an array of objects")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in value["hashes"].items()
    ):
        raise ValueError("Managed publication receipt has invalid artifact hashes")
    return value


@dataclass(frozen=True)
class _Publication:
    publication_id: str
    revision_id: str
    revision_of: str | None
    origin_path: str
    origin_fingerprint: str
    source_kind: str = "session"

    def __post_init__(self) -> None:
        if (
            not _is_uuid(self.publication_id)
            or not _is_uuid(self.revision_id)
            or self.revision_of is not None
            and not _is_uuid(self.revision_of)
            or not isinstance(self.origin_path, str)
            or not self.origin_path
            or not isinstance(self.origin_fingerprint, str)
            or len(self.origin_fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in self.origin_fingerprint)
            or self.source_kind not in {"session", "legacy_audio_only"}
        ):
            raise ValueError("Invalid publication provenance")

    @classmethod
    def read(cls, value: object) -> _Publication:
        if not isinstance(value, dict):
            raise ValueError("Invalid publication provenance envelope")
        return cls(**value)


def _utc_timestamp(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    measured = datetime.fromisoformat(value) if isinstance(value, str) else value
    if measured.tzinfo is None or measured.utcoffset() is None:
        return None  # the original manifest preserves an unspecified legacy timezone
    return measured.astimezone(timezone.utc).isoformat()


def fts_available(connection: sqlite3.Connection) -> bool:
    """Probe the running SQLite, rather than assuming a compile-time option."""
    try:
        connection.execute("CREATE VIRTUAL TABLE temp.here_fts_probe USING fts5(text)")
        connection.execute("DROP TABLE temp.here_fts_probe")
        return True
    except sqlite3.Error as error:
        if (
            getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_ERROR
            and str(error) == "no such module: fts5"
        ):
            return False
        raise storage_error(error) from error


class _FileLock(FileLease):
    def __init__(self, path: Path, *, timeout: float = 0.25, create: bool = True) -> None:
        try:
            super().__init__(path, timeout=timeout, create=create)
        except LeaseBusy as exc:
            raise Busy(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class _DeletionRecord:
    meeting_id: str
    origins: tuple[tuple[str, str], ...]
    phase: str

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "meeting_id": self.meeting_id,
            "origins": [
                {"origin_path": path, "origin_fingerprint": fingerprint}
                for path, fingerprint in self.origins
            ],
            "phase": self.phase,
        }


class _CatalogRepository(MemoryRepository):
    """Operation-local registry evidence, loaded while holding the publication lease."""

    def __init__(self, library: LibraryConnection, records: tuple[_DeletionRecord, ...]) -> None:
        super().__init__(library)
        self._deletion_records = records

    def _remember_deletion(self, record: _DeletionRecord) -> None:
        records = {item.meeting_id: item for item in self._deletion_records}
        records[record.meeting_id] = record
        self._deletion_records = tuple(records[identity] for identity in sorted(records))


def _catalog_records(repo: MemoryRepository) -> tuple[_DeletionRecord, ...]:
    """Never fall back to filesystem I/O from a transactional helper."""
    if not isinstance(repo, _CatalogRepository):
        raise LibraryError("Catalog suppression evidence was not loaded for this operation")
    return repo._deletion_records


def _origin_pair(value: object) -> tuple[str, str]:
    """Validate stored origin metadata lexically; never inspect an external path."""
    if not isinstance(value, dict) or set(value) != {"origin_path", "origin_fingerprint"}:
        raise ValueError("Invalid deletion origin envelope")
    path, fingerprint = value["origin_path"], value["origin_fingerprint"]
    if (
        not isinstance(path, str)
        or not os.path.isabs(path)
        or os.path.normcase(os.path.normpath(path)) != path
        or any(ord(character) < 32 for character in path)
        or not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise ValueError("Invalid deletion origin pair")
    return path, fingerprint


def _deletion_registry(data_dir: Path) -> tuple[tuple[_DeletionRecord, ...], tuple[str, ...]]:
    """The sole registry parser, shared by locked reads, writers and recovery."""
    try:
        directory = managed_path(data_dir, ".deletions", directory=True)
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except FileNotFoundError:
            return (), ()
        records = []
        stages = []
        # Validate EVERY role/path before opening even the first final record.
        finals = []
        for entry in entries:
            stage = re.fullmatch(r"\.([0-9a-f-]{36})\.json\.[A-Za-z0-9_-]{8}\.stage", entry.name)
            if stage is not None and _is_uuid(stage[1]):
                session_artifact_path(directory, entry.name)
                stages.append(entry.name)
            elif entry.suffix == ".json" and _is_uuid(entry.stem):
                finals.append(session_artifact_path(directory, entry.name))
            else:
                raise ValueError("Unrecognized deletion registry entry retained")
        for path in finals:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or set(value) != {"schema_version", "meeting_id", "origins", "phase"}
                or type(value["schema_version"]) is not int
                or value["schema_version"] != 1
                or value["meeting_id"] != path.stem
                or value["phase"] not in ("pending", "complete")
                or not isinstance(value["origins"], list)
            ):
                raise ValueError("Invalid deletion registry record")
            origins = tuple(sorted(_origin_pair(origin) for origin in value["origins"]))
            if len(set(origins)) != len(origins):
                raise ValueError("Duplicate deletion origin pair")
            records.append(_DeletionRecord(path.stem, origins, value["phase"]))
        return tuple(records), tuple(stages)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise LibraryError("Deletion registry unavailable; local recovery is required") from exc


def _suppressed_meeting_ids(
    library: LibraryConnection, records: tuple[_DeletionRecord, ...]
) -> tuple[str, ...]:
    """Resolve aliases from the current SQL snapshot without I/O or hashing."""
    ids = {record.meeting_id for record in records}
    pairs = {pair for record in records for pair in record.origins}
    if pairs:
        for row in library.execute(
            "SELECT meeting_id,origin_path,origin_fingerprint FROM ingest_sources"
        ):
            if (row[1], row[2]) in pairs:
                if not _is_uuid(row[0]):
                    raise LibraryError("Invalid suppressed catalog identity")
                ids.add(row[0])
    return tuple(sorted(ids))


def _suppression_snapshot(
    library: LibraryConnection, records: tuple[_DeletionRecord, ...]
) -> SuppressionSnapshot:
    resolved = _suppressed_meeting_ids(library, records)
    digest = hashlib.sha256()
    for value in ("here-suppression-v1", [record.document() for record in records], resolved):
        encoded = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return SuppressionSnapshot(resolved, digest.hexdigest())


@contextmanager
def catalog_read_view(library: LibraryConnection) -> Iterator[SuppressionSnapshot]:
    """Own only the bounded publication lease and a fresh caller's read transaction."""
    if library.connection.in_transaction:
        raise LibraryError("Catalog read view requires a fresh connection without a transaction")
    lock: _FileLock | None = None
    owned_transaction = False
    try:
        lock = _FileLock(managed_path(library.data_dir, ".memory.lock"))
        library.execute("BEGIN")
        owned_transaction = True
        records, _ = _deletion_registry(library.data_dir)
        yield _suppression_snapshot(library, records)
    except sqlite3.Error as exc:
        raise storage_error(exc) from exc
    except (OSError, ValueError) as exc:
        raise LibraryError("Catalog read view unavailable") from exc
    finally:
        try:
            if owned_transaction:
                library.connection.rollback()
        finally:
            if lock is not None:
                lock.close()


def _remove_tree(root: Path, relative: str, names: set[str] | None = None) -> None:
    """Remove registered protocol files; unknown material defers cleanup."""
    directory = managed_path(root, relative, directory=True)
    if not directory.exists():
        return
    entries = list(directory.iterdir())
    unprepared = names is None and directory.name != "states"
    if names is None and directory.name == "states":
        names = set()
        for entry in entries:
            try:
                read_meeting_state(directory, entry.name, directory.parent.name)
            except (OSError, ValueError, TypeError, KeyError):
                continue
            names.add(entry.name)
    elif names is None:
        names = set()
        intent = session_artifact_path(directory, ".publication.json")
        if intent.exists():
            plan = json.loads(intent.read_text(encoding="utf-8"))
            if (
                not isinstance(plan, dict)
                or plan.get("schema_version") != 1
                or plan.get("operation_id") != directory.name
                or not isinstance(plan.get("hashes"), dict)
                or any(
                    not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                    for name, digest in plan["hashes"].items()
                )
            ):
                raise ValueError("Invalid staging ownership manifest")
            names.update(plan["hashes"])
    if directory.name != "states":
        validate_artifact_roles(
            directory,
            (entry.name for entry in entries if entry.name != ".publication.json"),
            require_audio_declaration=unprepared,
        )
        names = names | {".publication.json"}
    # A UUID parent or familiar suffix alone never grants deletion authority.
    if any(entry.name not in names for entry in entries):
        raise ValueError(f"Unrecognized material retained; cleanup deferred: {relative}")
    for entry in entries:
        session_artifact_path(directory, entry.name)
    for entry in entries:
        session_artifact_path(directory, entry.name).unlink()
    directory.rmdir()


class MemoryService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).absolute()
        self._owners: dict[str, _FileLock] = {}
        self.before_delete: list[Callable[[str], None]] = []
        self._derived_row_purgers: list[Callable[[MemoryRepository, str], None]] = []
        self._publication_context: ContextVar[tuple[str, str] | None] = ContextVar(
            "here_memory_publication", default=None
        )
        if not (self.data_dir / "library.sqlite3").exists():
            try:
                with self._operation():
                    pass
            except (OSError, ValueError, LibraryError):
                # Eager initialization is convenient, but a recording must be
                # allowed to start even when its independent catalog is unavailable.
                # The actual operation reports the typed failure to its caller.
                pass

    @contextmanager
    def _operation(self) -> Iterator[MemoryRepository]:
        validate_session_directory(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        lock = _FileLock(self.data_dir / ".memory.lock")
        library: LibraryConnection | None = None
        try:
            records, _ = _deletion_registry(self.data_dir)
            library = open_library(self.data_dir)
            migrate(library)
            yield _CatalogRepository(library, records)
        finally:
            if library is not None:
                library.close()
            lock.close()

    def _fts(self, repo: MemoryRepository) -> bool:
        if not fts_available(repo.connection):
            return False
        repo.library.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5("
            "meeting_id UNINDEXED, revision_id UNINDEXED, segment_id UNINDEXED, text, title, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        return True

    def reindex(self) -> ReindexReport:
        """Replace the one derived index, preserving partial and historical evidence."""
        with self._operation() as repo:
            records, _ = _deletion_registry(self.data_dir)
            suppression = _suppression_snapshot(repo.library, records)
            with repo.transaction():
                if not self._fts(repo):
                    return ReindexReport(warnings=(FtsUnavailable(),))
                repo.library.execute("DELETE FROM segments_fts")
                # Visibility precedes insertion, and no partial branch writes FTS.
                suppressed = set(suppression.suppressed_meeting_ids)
                rows = repo.library.execute(
                    "SELECT m.id,m.title,r.id AS revision_id,s.id AS segment_id,s.text "
                    "FROM meetings m JOIN transcript_revisions r ON r.id=m.active_revision_id "
                    "AND r.meeting_id=m.id JOIN segments s ON s.revision_id=r.id "
                    "WHERE r.status='complete' AND m.status!='deleting' ORDER BY m.id,s.ordinal"
                ).fetchall()
                meetings = set()
                count = 0
                for row in rows:
                    if row["id"] in suppressed:
                        continue
                    repo.library.execute(
                        "INSERT INTO segments_fts VALUES (?,?,?,?,?)",
                        (
                            row["id"],
                            row["revision_id"],
                            row["segment_id"],
                            row["text"],
                            row["title"],
                        ),
                    )
                    meetings.add(row["id"])
                    count += 1
            return ReindexReport(len(meetings), count)

    def _delete_inventory(self, repo: MemoryRepository, identity: str) -> dict[str, Any]:
        """Read only A-registered publication/state roles; unknown children stay unowned."""
        root = managed_path(self.data_dir, f"meetings/{identity}", directory=True)
        files: dict[str, str] = {}
        directories = {f"meetings/{identity}"}
        revisions = managed_path(self.data_dir, f"meetings/{identity}/revisions", directory=True)
        if revisions.exists():
            directories.add(revisions.relative_to(self.data_dir).as_posix())
            for directory in sorted(revisions.iterdir()):
                if not _is_uuid(directory.name):
                    continue
                validate_session_directory(directory)
                registration = session_artifact_path(directory, ".publication.json")
                receipt = session_artifact_path(directory, "catalog.json")
                if not registration.exists():
                    # No suffix/UUID-only authority, even for a familiar filename.
                    continue
                plan = json.loads(registration.read_text(encoding="utf-8"))
                if (
                    not isinstance(plan, dict)
                    or set(plan) != {"schema_version", "operation_id", "hashes"}
                    or type(plan["schema_version"]) is not int
                    or plan["schema_version"] != 1
                    or not _is_uuid(plan["operation_id"])
                    or not isinstance(plan["hashes"], dict)
                    or any(
                        not isinstance(name, str)
                        or not isinstance(digest, str)
                        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                        for name, digest in plan["hashes"].items()
                    )
                ):
                    raise ValueError("Invalid registered publication inventory")
                validate_artifact_roles(directory, plan["hashes"])
                operation = repo.get_pending_operation(plan["operation_id"])
                if operation is not None:
                    payload = json.loads(operation.payload_json)
                    if (
                        operation.kind != "publish_revision"
                        or operation.meeting_id != identity
                        or operation.destination_path
                        != directory.relative_to(self.data_dir).as_posix()
                        or payload.get("revision_id") != directory.name
                        or payload.get("hashes") != plan["hashes"]
                    ):
                        raise ValueError("Publication inventory disagrees with registered journal")
                if receipt.exists():
                    snapshot = _receipt_envelope(json.loads(receipt.read_text(encoding="utf-8")))
                    if (
                        snapshot.get("schema_version") not in {2, 3}
                        or snapshot["meeting"].get("id") != identity
                        or snapshot["revision"].get("meeting_id") != identity
                        or snapshot["revision"].get("id") != directory.name
                        or snapshot["hashes"] != plan["hashes"]
                    ):
                        raise ValueError("Publication receipt disagrees with cleanup inventory")
                    if snapshot["schema_version"] == 3:
                        proof = _Publication.read(snapshot.get("publication"))
                        if (
                            snapshot.get("operation_id") != plan["operation_id"]
                            or proof.revision_id != directory.name
                        ):
                            raise ValueError("Publication cleanup provenance mismatch")
                    assets = {
                        asset["id"]: AudioAsset(**asset) for asset in snapshot["audio_assets"]
                    }
                    for asset in assets.values():
                        path = managed_path(self.data_dir, asset.relative_path)
                        if asset.meeting_id != identity or not path.is_relative_to(root):
                            raise ValueError("Cleanup audio must belong to its meeting")
                        if path.suffix.lower() != ".wav":
                            raise ValueError("Cleanup audio must have a WAV role")
                        validate_artifact_roles(path.parent, (path.name,))
                    for ordinal, values in enumerate(snapshot["revision_audio_assets"]):
                        link = RevisionAudioAsset(**values)
                        if (
                            link.meeting_id != identity
                            or link.revision_id != directory.name
                            or link.asset_id not in assets
                            or link.ordinal != ordinal
                        ):
                            raise ValueError("Cleanup audio relationship mismatch")
                        validate_artifact_roles(directory, (link.source_file,))
                    files[receipt.relative_to(self.data_dir).as_posix()] = file_hash(receipt)
                directories.add(directory.relative_to(self.data_dir).as_posix())
                files[registration.relative_to(self.data_dir).as_posix()] = file_hash(registration)
                for name, digest in plan["hashes"].items():
                    path = session_artifact_path(directory, name)
                    if path.exists() and file_hash(path) != digest:
                        raise ValueError("Registered cleanup artifact hash mismatch")
                    files[path.relative_to(self.data_dir).as_posix()] = digest
        states = managed_path(self.data_dir, f"meetings/{identity}/states", directory=True)
        if states.exists():
            directories.add(states.relative_to(self.data_dir).as_posix())
            for path in sorted(states.iterdir()):
                read_meeting_snapshot(states, path.name, identity)
                files[path.relative_to(self.data_dir).as_posix()] = file_hash(path)
        return {"schema_version": 1, "files": files, "directories": sorted(directories)}

    def _validate_delete_operation(
        self, operation: PendingOperation, record: _DeletionRecord
    ) -> dict[str, Any]:
        if (
            not _is_uuid(operation.id)
            or operation.kind != "delete"
            or operation.meeting_id != record.meeting_id
            or operation.staging_path is not None
            or operation.destination_path != f"meetings/{record.meeting_id}"
        ):
            raise ValueError("Invalid pending delete identity or path")
        value = json.loads(operation.payload_json)
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "files", "directories", "origins"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or not isinstance(value["files"], dict)
            or not isinstance(value["directories"], list)
            or not isinstance(value["origins"], list)
            or tuple(sorted(_origin_pair(pair) for pair in value["origins"])) != record.origins
        ):
            raise ValueError("Invalid pending delete inventory envelope")
        root = managed_path(self.data_dir, f"meetings/{record.meeting_id}", directory=True)
        for relative, digest in value["files"].items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise ValueError("Invalid delete file inventory")
            path = managed_path(self.data_dir, relative)
            if not path.is_relative_to(root) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("Delete inventory escapes its meeting")
            parts = path.relative_to(root).parts
            if len(parts) == 3 and parts[0] == "revisions" and _is_uuid(parts[1]):
                if path.name not in {"catalog.json", ".publication.json"}:
                    validate_artifact_roles(path.parent, (path.name,))
            elif len(parts) == 2 and parts[0] == "states":
                if path.exists():
                    read_meeting_snapshot(path.parent, path.name, record.meeting_id)
                elif re.fullmatch(r"[1-9][0-9]*-[0-9a-f-]{36}\.json", path.name) is None:
                    raise ValueError("Invalid deleted state role")
            else:
                raise ValueError("Unregistered delete inventory role")
            if path.exists() and file_hash(path) != digest:
                raise ValueError("Delete inventory file changed")
        for relative in value["directories"]:
            if not isinstance(relative, str):
                raise ValueError("Invalid delete directory inventory")
            path = managed_path(self.data_dir, relative, directory=True)
            parts = path.relative_to(root).parts if path.is_relative_to(root) else ("unsafe",)
            if parts not in ((), ("states",), ("revisions",)) and not (
                len(parts) == 2 and parts[0] == "revisions" and _is_uuid(parts[1])
            ):
                raise ValueError("Unknown delete directory role")
        return value

    def _delete_owner(
        self, repo: MemoryRepository, identity: str, leases: ExitStack, *, accepted: bool = False
    ) -> None:
        path = managed_path(self.data_dir, f".owners/{identity}.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        lease = _FileLock(path, timeout=0)
        leases.callback(lease.close)
        meeting = repo.get_meeting(identity)
        if (
            not accepted
            and meeting is not None
            and meeting.status in {MeetingStatus.RECORDING, MeetingStatus.PROCESSING}
        ):
            raise Busy("Capture or processing must finish before deletion")

    def _delete_journal(self, repo: MemoryRepository, record: _DeletionRecord) -> PendingOperation:
        rows = repo.library.execute(
            "SELECT * FROM pending_operations WHERE kind='delete' AND meeting_id=?",
            (record.meeting_id,),
        ).fetchall()
        if len(rows) > 1:
            raise ValueError("Ambiguous delete operation")
        if rows:
            operation = PendingOperation(**dict(rows[0]))
            self._validate_delete_operation(operation, record)
            return operation
        inventory = self._delete_inventory(repo, record.meeting_id)
        inventory["origins"] = record.document()["origins"]
        return PendingOperation(
            str(uuid4()),
            "delete",
            "prepared",
            record.meeting_id,
            destination_path=f"meetings/{record.meeting_id}",
            payload_json=json.dumps(inventory),
        )

    def _prepare_delete(
        self, repo: MemoryRepository, record: _DeletionRecord, operation: PendingOperation
    ) -> tuple[object, ...]:
        warnings: tuple[object, ...] = ()
        with repo.transaction():
            if repo.get_pending_operation(operation.id) is None:
                repo.insert_pending_operation(operation)
            meeting = repo.get_meeting(record.meeting_id)
            if meeting is not None and meeting.status != MeetingStatus.DELETING:
                occurred_at = datetime.now(timezone.utc).isoformat()
                repo.library.execute(
                    "UPDATE meetings SET status='deleting',row_version=row_version+1,updated_at=? "
                    "WHERE id=?",
                    (occurred_at, record.meeting_id),
                )
                repo.record_event(
                    record.meeting_id,
                    "status_changed",
                    occurred_at,
                    {"from": meeting.status, "to": "deleting"},
                )
            if self._fts(repo):
                # Optional FTS capability is distinct from core secure_delete.
                try:
                    repo.connection.execute(
                        "INSERT INTO segments_fts(segments_fts,rank) VALUES('secure-delete',1)"
                    )
                except sqlite3.OperationalError:
                    pass  # Final compaction is required on either capability path.
                repo.library.execute(
                    "DELETE FROM segments_fts WHERE meeting_id=?", (record.meeting_id,)
                )
            else:
                warnings = (FtsUnavailable(),)
            for _, fingerprint in record.origins:
                repo.library.execute(
                    "INSERT OR IGNORE INTO deleted_origins VALUES (?,?)",
                    (fingerprint, datetime.now(timezone.utc).isoformat()),
                )
        return warnings

    def _purge_delete_files(self, operation: PendingOperation, record: _DeletionRecord) -> None:
        inventory = self._validate_delete_operation(operation, record)

        # Keep A's proof until all registered content has been removed, enabling
        # restart with SQL loss during the file phase. Unknown material is retained.
        def purge_order(relative: str) -> tuple[int, bool, str]:
            path = Path(relative)
            if path.name in {"catalog.json", ".publication.json"}:
                role = 4
            elif "states" in path.parts:
                role = 3
            elif path.suffix.lower() == ".wav":
                role = 2
            elif path.suffix.lower() in {".txt", ".md"}:
                role = 0
            else:
                role = 1
            return role, path.name == ".publication.json", relative

        files = sorted(inventory["files"], key=purge_order)
        for relative in files:
            path = managed_path(self.data_dir, relative)
            path.unlink(missing_ok=True)
        for relative in sorted(inventory["directories"], key=lambda name: (-name.count("/"), name)):
            path = managed_path(self.data_dir, relative, directory=True)
            if path.exists():
                path.rmdir()  # Never recursively remove unregistered children.

    def _finish_delete(
        self, repo: MemoryRepository, record: _DeletionRecord, operation: PendingOperation
    ) -> DeletionResult:
        warnings = self._prepare_delete(repo, record, operation)
        self._purge_delete_files(operation, record)
        repo.library.execute("PRAGMA secure_delete=ON")
        with repo.transaction():
            for purge in tuple(self._derived_row_purgers):
                purge(repo, record.meeting_id)
            exists = repo.library.execute(
                "SELECT 1 FROM sqlite_master WHERE name='segments_fts' AND type='table'"
            ).fetchone()
            if exists:
                repo.library.execute(
                    "DELETE FROM segments_fts WHERE meeting_id=?", (record.meeting_id,)
                )
                repo.library.execute("INSERT INTO segments_fts(segments_fts) VALUES('rebuild')")
            repo.purge_meeting(record.meeting_id)
        # VACUUM also removes old FTS token/free-page residue on runtimes without
        # FTS secure-delete. Checkpoints are bounded and busy is never completion.
        checkpoint = repo.library.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is not None and checkpoint[0] != 0:
            raise Busy("Deletion WAL cleanup is busy")
        repo.library.execute("VACUUM")
        checkpoint = repo.library.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is not None and checkpoint[0] != 0:
            raise Busy("Deletion WAL cleanup is busy")
        for backup in repo.library.managed_backup_paths():
            backup.unlink()
        # All required physical work precedes completion, including retained
        # per-meeting owner cleanup (the shared publication lock is never removed).
        if repo.get_meeting(record.meeting_id) is not None:
            raise LibraryError("Deletion core cleanup remains pending")
        with repo.transaction():
            if operation.phase == "prepared":
                repo.transition_pending_operation(operation.id, "complete")
        complete = replace(record, phase="complete")
        directory = managed_path(self.data_dir, ".deletions", directory=True)
        write_durable_json(
            session_artifact_path(directory, f"{record.meeting_id}.json"), complete.document()
        )
        if isinstance(repo, _CatalogRepository):
            repo._remember_deletion(complete)
        return DeletionResult("completed", record.meeting_id, warnings=warnings)

    @staticmethod
    def _delete_failure(identity: str, error: Exception, *, accepted: bool) -> DeletionResult:
        reason = "busy" if isinstance(error, Busy) else "purge_failed" if accepted else "io"
        return DeletionResult(
            "pending" if accepted else "rejected",
            identity,
            reason,
            ("Deletion pending local recovery" if accepted else "Deletion could not start",),
            "io",
        )

    def delete_meeting(self, meeting_id: str) -> DeletionResult:
        self._identity(meeting_id)
        # An unlocked observation can prove prior acceptance for phase-aware
        # Busy results. It never grants mutation; all work revalidates under lock.
        records, _ = _deletion_registry(self.data_dir)
        accepted = any(record.meeting_id == meeting_id for record in records)
        hook_error: Exception | None = None
        try:
            with self._operation() as repo, ExitStack() as leases:
                records, _ = _deletion_registry(self.data_dir)
                accepted = any(record.meeting_id == meeting_id for record in records)
                if not accepted:
                    self._meeting(repo, meeting_id)
                    self._delete_owner(repo, meeting_id, leases)
                    self._delete_inventory(repo, meeting_id)
            for release in tuple(self.before_delete):
                try:
                    release(meeting_id)
                except Exception as exc:
                    hook_error = exc
                    break
            with self._operation() as repo, ExitStack() as leases:
                records, _ = _deletion_registry(self.data_dir)
                record = next(
                    (record for record in records if record.meeting_id == meeting_id), None
                )
                accepted = record is not None
                if hook_error is not None:
                    return DeletionResult(
                        "pending" if accepted else "rejected",
                        meeting_id,
                        "hook_failed",
                        ("Deletion release acknowledgement failed",),
                        "io",
                    )
                if not accepted:
                    self._meeting(repo, meeting_id)
                self._delete_owner(repo, meeting_id, leases, accepted=accepted)
                if record is None:
                    inventory = self._delete_inventory(repo, meeting_id)
                    origins = tuple(
                        sorted(
                            {
                                (row[0], row[1])
                                for row in repo.library.execute(
                                    "SELECT origin_path,origin_fingerprint FROM ingest_sources "
                                    "WHERE meeting_id=?",
                                    (meeting_id,),
                                )
                            }
                        )
                    )
                    record = _DeletionRecord(meeting_id, origins, "pending")
                    # Validate every pair before persisting the minimal barrier.
                    for origin in record.document()["origins"]:
                        _origin_pair(origin)
                    inventory["origins"] = record.document()["origins"]
                    operation = PendingOperation(
                        str(uuid4()),
                        "delete",
                        "prepared",
                        meeting_id,
                        destination_path=f"meetings/{meeting_id}",
                        payload_json=json.dumps(inventory),
                    )
                    self._validate_delete_operation(operation, record)
                    directory = managed_path(self.data_dir, ".deletions", directory=True)
                    directory.mkdir(exist_ok=True)
                    write_durable_json(
                        session_artifact_path(directory, f"{meeting_id}.json"), record.document()
                    )
                    if isinstance(repo, _CatalogRepository):
                        repo._remember_deletion(record)
                    accepted = True
                else:
                    operation = self._delete_journal(repo, record)
                return self._finish_delete(repo, record, operation)
        except (InvalidInput, NotFound, Conflict, Busy) as exc:
            # Atomic replace may have succeeded before its caller observed failure.
            records, _ = _deletion_registry(self.data_dir)
            accepted = accepted or any(record.meeting_id == meeting_id for record in records)
            if not accepted:
                raise
            return self._delete_failure(meeting_id, exc, accepted=True)
        except Exception as exc:
            records, _ = _deletion_registry(self.data_dir)
            accepted = accepted or any(record.meeting_id == meeting_id for record in records)
            return self._delete_failure(meeting_id, exc, accepted=accepted)

    def active_revision(self, meeting_id: str | None) -> str | None:
        if not meeting_id:
            return None
        library = open_library(self.data_dir)
        try:
            with catalog_read_view(library) as suppression:
                if meeting_id in suppression.suppressed_meeting_ids:
                    return None
                repo = MemoryRepository(library)
                meeting = repo.get_meeting(meeting_id)
                return meeting.active_revision_id if meeting else None
        finally:
            library.close()

    def retry_parent(
        self, session_dir: Path, meeting_id: str | None
    ) -> tuple[str, str | None] | None:
        """Resolve trusted retry lineage without modifying or hashing a legacy origin."""
        library = open_library(self.data_dir)
        try:
            with catalog_read_view(library) as suppression:
                repo = MemoryRepository(library)
                if meeting_id is not None and _is_uuid(meeting_id):
                    if meeting_id in suppression.suppressed_meeting_ids:
                        raise NotFound("Meeting has been deleted")
                    meeting = repo.get_meeting(meeting_id)
                    return (meeting.id, meeting.active_revision_id) if meeting else None
                rows = library.execute(
                    "SELECT DISTINCT m.id,m.active_revision_id FROM ingest_sources i "
                    "JOIN meetings m ON m.id=i.meeting_id WHERE i.origin_path=?",
                    (normalized_origin(session_dir),),
                ).fetchall()
                visible = [row for row in rows if row[0] not in suppression.suppressed_meeting_ids]
                if rows and not visible:
                    raise NotFound("Origin has been deleted")
                rows = visible
                if len(rows) > 1:
                    raise ValueError("Legacy origin conflict: more than one catalog identity")
                return (rows[0][0], rows[0][1]) if rows else None
        finally:
            library.close()

    def ingest_session(self, session_dir: Path, *, revision_of: str | None = None) -> IngestResult:
        return self._ingest_session(session_dir, revision_of=revision_of)

    @contextmanager
    def _publication(self, publication_id: str, source_kind: str) -> Iterator[None]:
        """Bind the private application intent without changing ordinary import API."""
        if not _is_uuid(publication_id):
            raise ValueError("Invalid publication UUID")
        token = self._publication_context.set((publication_id, source_kind))
        try:
            yield
        finally:
            self._publication_context.reset(token)

    def _replay_publication(
        self,
        repo: MemoryRepository,
        source: SessionArtifacts,
        publication_id: str,
        revision_of: str | None,
        source_kind: str,
    ) -> IngestResult | None:
        for row in repo.library.execute(
            "SELECT * FROM pending_operations WHERE kind='publish_revision' AND phase!='reverted'"
        ):
            operation = PendingOperation(**dict(row))
            try:
                payload = json.loads(operation.payload_json)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            value = payload.get("publication")
            if not isinstance(value, dict) or value.get("publication_id") != publication_id:
                continue
            proof = _Publication.read(value)
            revision = repo.get_revision(proof.revision_id)
            if (
                revision is not None
                and revision.meeting_id == operation.meeting_id
                and payload.get("revision_id") == revision.id
                and proof.revision_of == revision_of
                and proof.origin_path == source.origin_path == revision.origin
                and proof.origin_fingerprint == source.fingerprint
                and proof.source_kind == source_kind
                and revision.content_hash == source.content_hash
            ):
                return IngestResult("unchanged", revision.meeting_id)
            return IngestResult(
                "rejected",
                operation.meeting_id,
                ("Publication conflict: committed provenance not proven",),
                "conflict",
            )
        return None

    @staticmethod
    def _known_source_kind(repo: MemoryRepository, identity: str) -> str:
        for row in repo.library.execute(
            "SELECT kind,payload_json FROM pending_operations WHERE meeting_id=?", (identity,)
        ):
            try:
                payload = json.loads(row["payload_json"])
                if not isinstance(payload, dict):
                    continue
                if row["kind"] == "reserve_audio_only":
                    if payload.get("source_kind") == "legacy_audio_only":
                        return "legacy_audio_only"
                elif row["kind"] == "publish_revision" and "publication" in payload:
                    proof = _Publication.read(payload["publication"])
                    if proof.source_kind == "legacy_audio_only" and repo.get_revision(
                        proof.revision_id
                    ):
                        return proof.source_kind
            except (ValueError, TypeError):
                continue
        return "session"

    def _source_kind(self, identity: str) -> str:
        library = open_library(self.data_dir)
        try:
            return self._known_source_kind(MemoryRepository(library), identity)
        finally:
            library.close()

    def _reserve_audio_only(self, directory: Path, duration_ms: int) -> tuple[str, str | None]:
        """Reserve actual WAV evidence and a lease, without inventing a transcript."""
        audio = session_artifact_path(directory, "audio.wav")
        origin = normalized_origin(directory)
        fingerprint = hashlib.sha256(
            json.dumps(((audio.name, file_hash(audio)),)).encode()
        ).hexdigest()
        identity: str | None = None
        acquired = False
        try:
            with self._operation() as repo:
                records, _ = _deletion_registry(self.data_dir)
                if (origin, fingerprint) in {pair for record in records for pair in record.origins}:
                    raise NotFound("Origin has been deleted")
                rows = repo.library.execute(
                    "SELECT DISTINCT meeting_id FROM ingest_sources WHERE origin_path=?", (origin,)
                ).fetchall()
                if len(rows) > 1:
                    raise ValueError("Legacy origin conflict: more than one catalog identity")
                identity = rows[0][0] if rows else str(uuid4())
                if identity in _suppression_snapshot(repo.library, records).suppressed_meeting_ids:
                    raise NotFound("Meeting has been deleted")
                if not isinstance(identity, str) or not _is_uuid(identity):
                    raise ValueError("Invalid reserved audio-only identity")
                if identity in self._owners:
                    raise OwnershipBusy("A job already owns this audio-only origin")
                self._acquire_owner(identity)
                acquired = True
                with repo.transaction():
                    meeting = repo.get_meeting(identity)
                    if meeting is None:
                        repo.insert_meeting(
                            Meeting(
                                id=identity,
                                title=directory.name,
                                status=MeetingStatus.PROCESSING,
                                duration_ms=duration_ms,
                            )
                        )
                        repo.record_event(
                            identity,
                            "status_changed",
                            datetime.now(timezone.utc).isoformat(),
                            {"from": None, "to": "processing"},
                        )
                    else:
                        self._change_status(repo, identity, MeetingStatus.PROCESSING)
                    if repo.get_ingest_source(origin, fingerprint) is None:
                        repo.insert_ingest_source(
                            IngestSource(str(uuid4()), origin, fingerprint, identity, None)
                        )
                    repo.insert_pending_operation(
                        PendingOperation(
                            str(uuid4()),
                            "reserve_audio_only",
                            "complete",
                            identity,
                            payload_json=json.dumps(
                                {
                                    "source_kind": "legacy_audio_only",
                                    "origin_path": origin,
                                    "origin_fingerprint": fingerprint,
                                }
                            ),
                        )
                    )
                self._write_state(repo, identity)
                return identity, meeting.active_revision_id if meeting else None
        except BaseException:
            if acquired and identity is not None:
                self.finish_capture(identity)
            raise

    def prepare_retry(self, session_dir: Path, source_id: str | None) -> tuple[str, str]:
        """Resolve a canonical parent before an explicit producer mutates the v1."""
        parent = self.retry_parent(session_dir, source_id)
        if parent and parent[1] is not None:
            return parent[0], parent[1]
        if parent and self._source_kind(parent[0]) == "legacy_audio_only":
            marker = session_artifact_path(session_dir, "memory_pending.json")
            pending = read_pending_intent(session_dir, source_id) if marker.exists() else None
            with self._publication(
                (pending.publication_id if pending else None) or str(uuid4()), "legacy_audio_only"
            ):
                result = self._ingest_session(session_dir, trusted_legacy=True)
        else:
            result = self._ingest_session(session_dir, trusted_legacy=True)
        if result.status == "rejected" or result.meeting_id is None:
            raise ValueError(f"Legacy identity could not be resolved: {result.warnings}")
        active = self.active_revision(result.meeting_id)
        if active is None:
            raise LibraryError("Legacy retry has no canonical active parent")
        return result.meeting_id, active

    def ingest_retry(self, session_dir: Path, *, revision_of: str) -> IngestResult:
        """Publish explicit trusted retry evidence under its expected active parent."""
        return self._ingest_session(session_dir, revision_of=revision_of, trusted_legacy=True)

    def _ingest_session(
        self, session_dir: Path, *, revision_of: str | None = None, trusted_legacy: bool = False
    ) -> IngestResult:
        identity: str | None = None
        operation: PendingOperation | None = None
        staging_rel: str | None = None
        committed = False
        source_read = False
        warnings: tuple[object, ...] = ()
        active: TranscriptRevision | None = None
        # Opening the catalog is a typed storage boundary; CLI must distinguish a
        # broken database from a rejected source or publication conflict.
        with self._operation() as repo:
            records, _ = _deletion_registry(self.data_dir)
            suppression = _suppression_snapshot(repo.library, records)
            try:
                source = (
                    read_retry_artifacts(session_dir)
                    if trusted_legacy
                    else read_artifacts(session_dir)
                )
                source_read = True
                if source.metadata.meeting_id in suppression.suppressed_meeting_ids or (
                    source.origin_path,
                    source.fingerprint,
                ) in {pair for record in records for pair in record.origins}:
                    return IngestResult(
                        "rejected",
                        source.metadata.meeting_id,
                        ("Origin or identity has been deleted",),
                        "conflict",
                    )
                publication_context = self._publication_context.get()
                if publication_context is not None:
                    replay = self._replay_publication(
                        repo, source, publication_context[0], revision_of, publication_context[1]
                    )
                    if replay is not None:
                        return replay
                previous_origin = repo.get_ingest_source(source.origin_path, source.fingerprint)
                source_identity = source.metadata.meeting_id
                try:
                    canonical = (
                        source_identity is not None
                        and str(UUID(source_identity)) == source_identity
                    )
                except ValueError:
                    canonical = False
                identity = (source_identity if canonical else None) or (
                    previous_origin.meeting_id if previous_origin else str(uuid4())
                )
                if (
                    not canonical
                    and publication_context is not None
                    and publication_context[1] == "legacy_audio_only"
                ):
                    rows = repo.library.execute(
                        "SELECT DISTINCT meeting_id FROM ingest_sources WHERE origin_path=?",
                        (source.origin_path,),
                    ).fetchall()
                    if len(rows) != 1:
                        raise ValueError("Audio-only publication requires its reserved identity")
                    identity = rows[0][0]
                if not canonical and revision_of is not None:
                    parent = repo.get_revision(revision_of)
                    if parent is None:
                        return IngestResult(
                            "rejected",
                            warnings=("Revision conflict: missing parent",),
                            error_category="conflict",
                        )
                    identity = parent.meeting_id
                if identity in suppression.suppressed_meeting_ids:
                    return IngestResult(
                        "rejected", identity, ("Origin or identity has been deleted",), "conflict"
                    )
                if previous_origin and revision_of is None:
                    return IngestResult("unchanged", previous_origin.meeting_id)
                previous = repo.get_meeting(identity)
                active = (
                    repo.get_revision(previous.active_revision_id)
                    if (previous and previous.active_revision_id)
                    else None
                )
                if active and active.content_hash == source.content_hash and revision_of is None:
                    with repo.transaction():
                        repo.insert_ingest_source(
                            IngestSource(
                                str(uuid4()),
                                source.origin_path,
                                source.fingerprint,
                                identity,
                                active.id,
                            )
                        )
                    return IngestResult("unchanged", identity)
                if active and revision_of != active.id:
                    return IngestResult(
                        "rejected", identity, ("Identity conflict: content differs",), "conflict"
                    )
                if revision_of is not None and (not active or revision_of != active.id):
                    return IngestResult(
                        "rejected", identity, ("Revision conflict: stale retry",), "conflict"
                    )
                revision_id = str(uuid4())
                number = repo.library.execute(
                    "SELECT COALESCE(MAX(revision_number),0)+1 FROM transcript_revisions "
                    "WHERE meeting_id=?",
                    (identity,),
                ).fetchone()[0]
                operation_id = str(uuid4())
                staging_rel = f".staging/{operation_id}"
                destination_rel = f"meetings/{identity}/revisions/{revision_id}"
                # Validate destination components before copying or publishing.
                destination = managed_path(self.data_dir, destination_rel, directory=True)
                staging = managed_path(self.data_dir, staging_rel, directory=True)
                staging.mkdir(parents=True)
                existing_audio = repo.library.execute(
                    "SELECT * FROM audio_assets WHERE meeting_id=? ORDER BY id", (identity,)
                ).fetchall()
                known_assets = {row["sha256"]: AudioAsset(**dict(row)) for row in existing_audio}
                assets = []
                links = []
                audio_names = {item["name"] for item in source.audio}
                for ordinal, audio in enumerate(source.audio):
                    values = dict(audio)
                    name = values.pop("name")
                    asset = known_assets.get(audio["sha256"])
                    if asset is None:
                        asset = AudioAsset(
                            id=str(uuid4()),
                            meeting_id=identity,
                            relative_path=f"{destination_rel}/{name}",
                            **values,
                        )
                        assets.append(asset)
                        known_assets[asset.sha256] = asset
                    links.append(RevisionAudioAsset(identity, revision_id, asset.id, ordinal, name))
                new_audio_names = {Path(a.relative_path).name for a in assets}
                plan = {
                    name: digest
                    for name, digest in source.files
                    if name not in audio_names or name in new_audio_names
                }
                write_durable_json(
                    session_artifact_path(staging, ".publication.json"),
                    {
                        "schema_version": 1,
                        "operation_id": operation_id,
                        "hashes": plan,
                    },
                )
                hashes = {}
                for name, digest in sorted(
                    plan.items(), key=lambda item: item[0] != "session.json"
                ):
                    # Preserve raw source copies as well; only the canonical audio
                    # is deduplicated across immutable transcript revisions.
                    src = session_artifact_path(source.directory, name)
                    dst = session_artifact_path(staging, name)
                    shutil.copy2(src, dst)
                    with dst.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    if file_hash(dst) != digest:
                        raise ValueError("Source changed while staging")
                    hashes[name] = digest
                metadata = source.metadata
                status = "processing" if metadata.status == "pending" else metadata.status
                audio_only = self._known_source_kind(repo, identity) == "legacy_audio_only"
                meeting = Meeting(
                    id=identity,
                    legacy_session_id=metadata.session_id,
                    title=source.directory.name,
                    started_at=None if audio_only else _utc_timestamp(metadata.started_at),
                    completed_at=None if audio_only else _utc_timestamp(metadata.completed_at),
                    duration_ms=round(metadata.duration_seconds * 1000),
                    paused_ms=round(metadata.total_paused_seconds * 1000),
                    status=MeetingStatus(status),
                )
                revision = TranscriptRevision(
                    id=revision_id,
                    meeting_id=identity,
                    revision_number=number,
                    raw_text=source.raw_text,
                    presentation_text=source.presentation_text,
                    provider="openai",
                    model=metadata.transcription_model,
                    origin=source.origin_path,
                    content_hash=source.content_hash,
                    status="complete" if status == "completed" else "partial",
                )
                segments = [
                    Segment(id=str(uuid4()), revision_id=revision_id, **item)
                    for item in source.segments
                ]
                payload = {"revision_id": revision_id, "hashes": hashes}
                if publication_context is not None:
                    payload["publication"] = asdict(
                        _Publication(
                            publication_context[0],
                            revision_id,
                            revision_of,
                            source.origin_path,
                            source.fingerprint,
                            publication_context[1],
                        )
                    )
                operation = PendingOperation(
                    operation_id,
                    "publish_revision",
                    "prepared",
                    identity,
                    staging_rel,
                    destination_rel,
                    json.dumps(payload),
                )
                with repo.transaction():
                    repo.insert_pending_operation(operation)
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(destination)
                warnings = (repo.library.storage_warning,) if repo.library.storage_warning else ()
                if audio_only:
                    warnings += ("Capture dates are unknown for this legacy audio-only source",)
                elif meeting.started_at is None or meeting.completed_at is None:
                    warnings += ("Legacy date has no timezone; catalog UTC date left unspecified",)
                with repo.transaction():
                    if previous is None:
                        repo.insert_meeting(meeting)
                    else:
                        repo.library.execute(
                            "UPDATE meetings SET status=?, completed_at=?, duration_ms=?, "
                            "paused_ms=?, legacy_session_id=COALESCE(legacy_session_id,?), "
                            "started_at=COALESCE(started_at,?), "
                            "title=CASE WHEN title='' THEN ? ELSE title END WHERE id=?",
                            (
                                status,
                                meeting.completed_at,
                                meeting.duration_ms,
                                meeting.paused_ms,
                                meeting.legacy_session_id,
                                meeting.started_at,
                                meeting.title,
                                identity,
                            ),
                        )
                    for asset in assets:
                        repo.insert_audio_asset(asset)
                    repo.insert_revision(revision)
                    for link in links:
                        repo.insert_revision_audio_asset(link)
                    repo.insert_segments(segments)
                    if previous_origin is None:
                        repo.insert_ingest_source(
                            IngestSource(
                                str(uuid4()),
                                source.origin_path,
                                source.fingerprint,
                                identity,
                                revision_id,
                            )
                        )
                    if self._fts(repo):
                        indexed_meeting = repo.get_meeting(identity)
                        assert indexed_meeting is not None
                        repo.library.execute(
                            "DELETE FROM segments_fts WHERE meeting_id=?", (identity,)
                        )
                        if revision.status == "complete":
                            for segment in segments:
                                repo.library.execute(
                                    "INSERT INTO segments_fts VALUES (?,?,?,?,?)",
                                    (
                                        identity,
                                        revision_id,
                                        segment.id,
                                        segment.text,
                                        indexed_meeting.title,
                                    ),
                                )
                    else:
                        warnings += (FtsUnavailable(),)
                    repo.activate_revision(identity, revision_id)
                    if previous is None or previous.status != meeting.status:
                        repo.record_event(
                            identity,
                            "status_changed",
                            datetime.now(timezone.utc).isoformat(),
                            {"from": previous.status if previous else None, "to": meeting.status},
                        )
                committed = True
                self._write_receipt(repo, operation)
                self._write_state(repo, identity)
                with repo.transaction():
                    repo.transition_pending_operation(operation_id, "complete")
                return IngestResult("revised" if active else "inserted", identity, warnings)
            except Exception as exc:
                if committed:
                    # The entity transaction is authoritative. Leave its prepared
                    # operation for replay if receipt/state/completion fails.
                    return IngestResult(
                        "revised" if active else "inserted", identity, warnings + (str(exc),)
                    )
                # An ordinary failed publication reverts, unlike an abrupt process
                # death. Never let reconciliation activate a failed explicit retry.
                if operation is not None:
                    try:
                        self._reconcile_operation(repo, operation)
                    except Exception:
                        pass
                elif staging_rel is not None:
                    try:
                        _remove_tree(self.data_dir, staging_rel)
                    except (OSError, ValueError):
                        pass
                if isinstance(exc, LibraryError):
                    raise
                category = (
                    "invalid_input"
                    if isinstance(exc, ValueError)
                    or (not source_read and isinstance(exc, FileNotFoundError))
                    else "io"
                )
                return IngestResult("rejected", identity, (str(exc),), category)

    def _reconcile_operation(self, repo: MemoryRepository, operation: PendingOperation) -> None:
        records, _ = _deletion_registry(self.data_dir)
        if operation.kind == "delete":
            record = next(
                (item for item in records if item.meeting_id == operation.meeting_id), None
            )
            if record is None:
                raise ValueError("Delete journal has no durable suppression authority")
            self._validate_delete_operation(operation, record)
            self._finish_delete(repo, record, operation)
            return
        if (
            operation.meeting_id
            in _suppression_snapshot(repo.library, records).suppressed_meeting_ids
        ):
            # A publication/state operation cannot recreate deleted evidence.
            return
        if operation.kind == "persist_state":
            identity = operation.meeting_id
            if (
                not _is_uuid(operation.id)
                or not isinstance(identity, str)
                or not _is_uuid(identity)
                or operation.staging_path is not None
                or operation.destination_path is not None
                or operation.payload_json != "{}"
                or repo.get_meeting(identity) is None
            ):
                raise ValueError("Invalid pending lifecycle state operation")
            self._write_state(repo, identity)
            with repo.transaction():
                repo.transition_pending_operation(operation.id, "complete")
            return
        if operation.kind != "publish_revision":
            raise ValueError("Unknown pending operation")
        if not operation.meeting_id or str(UUID(operation.meeting_id)) != operation.meeting_id:
            raise ValueError("Invalid pending meeting identity")
        staging_rel = operation.staging_path or ""
        destination_rel = operation.destination_path or ""
        staging = managed_path(self.data_dir, staging_rel, directory=True)
        destination = managed_path(self.data_dir, destination_rel, directory=True)
        payload = json.loads(operation.payload_json)
        revision_id = str(UUID(payload["revision_id"]))
        if staging_rel != f".staging/{operation.id}" or destination_rel != (
            f"meetings/{operation.meeting_id}/revisions/{revision_id}"
        ):
            raise ValueError("Pending operation paths disagree with identity")
        validate_artifact_roles(destination if destination.exists() else staging, payload["hashes"])
        committed = repo.get_revision(revision_id) is not None
        if committed:
            if not destination.is_dir():
                raise ValueError("Committed revision files are missing")
            for name, digest in payload["hashes"].items():
                if file_hash(session_artifact_path(destination, name)) != digest:
                    raise ValueError("Published revision hash mismatch")
            self._write_receipt(repo, operation)
            self._write_state(repo, operation.meeting_id)
        else:
            _remove_tree(self.data_dir, destination_rel, set(payload["hashes"]))
        if staging.exists():
            _remove_tree(self.data_dir, staging_rel, set(payload["hashes"]))
        if not committed:
            # Remove only empty operation-created ancestors, preserving siblings.
            for directory in (destination.parent, destination.parent.parent):
                validate_session_directory(directory)
                try:
                    directory.rmdir()
                except OSError:
                    pass
        with repo.transaction():
            repo.transition_pending_operation(operation.id, "complete" if committed else "reverted")

    def _write_state(self, repo: MemoryRepository, identity: str) -> None:
        meeting = repo.get_meeting(identity)
        if meeting is None:
            return
        directory = managed_path(self.data_dir, f"meetings/{identity}/states", directory=True)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "schema_version": 2,
            "meeting": asdict(meeting),
            "source_kind": self._known_source_kind(repo, identity),
            "participants": [
                dict(row)
                for row in repo.library.execute(
                    "SELECT p.* FROM participants p JOIN meeting_participants l "
                    "ON p.id=l.participant_id WHERE l.meeting_id=? ORDER BY p.id",
                    (identity,),
                )
            ],
        }
        for table in (
            "meeting_participants",
            "speaker_assignments",
            "lifecycle_events",
            "audio_assets",
            "ingest_sources",
        ):
            snapshot[table] = [
                dict(row)
                for row in repo.library.execute(
                    f"SELECT * FROM {table} WHERE meeting_id=? ORDER BY 1", (identity,)
                )
            ]
        write_durable_json(
            session_artifact_path(directory, f"{meeting.row_version}-{uuid4()}.json"), snapshot
        )

    def _write_receipt(self, repo: MemoryRepository, operation: PendingOperation) -> None:
        """A receipt is written only after the entity transaction has committed."""
        directory = managed_path(self.data_dir, operation.destination_path or "", directory=True)
        receipt = session_artifact_path(directory, "catalog.json")
        if receipt.exists():
            return  # historical evidence is never rewritten
        payload = json.loads(operation.payload_json)
        revision = repo.get_revision(payload["revision_id"])
        if operation.meeting_id is None:
            raise ValueError("A publication receipt requires a meeting identity")
        meeting = repo.get_meeting(operation.meeting_id)
        if revision is None or meeting is None or revision.meeting_id != meeting.id:
            raise ValueError("A publication receipt requires committed catalog entities")
        snapshot = {
            "schema_version": 3 if "publication" in payload else 2,
            "meeting": asdict(meeting),
            "revision": asdict(revision),
            "segments": [asdict(segment) for segment in repo.get_segments(revision.id)],
            "audio_assets": [
                dict(row)
                for row in repo.library.execute(
                    "SELECT a.* FROM audio_assets a JOIN revision_audio_assets l "
                    "ON l.asset_id=a.id AND l.meeting_id=a.meeting_id "
                    "WHERE l.revision_id=? ORDER BY l.ordinal",
                    (revision.id,),
                )
            ],
            "revision_audio_assets": [
                asdict(link) for link in repo.get_revision_audio_assets(revision.id)
            ],
            "ingest_sources": [
                dict(row)
                for row in repo.library.execute(
                    "SELECT * FROM ingest_sources WHERE revision_id=? OR "
                    "(meeting_id=? AND revision_id IS NULL AND ?=1)",
                    (revision.id, meeting.id, revision.revision_number),
                )
            ],
            "hashes": payload["hashes"],
        }
        if "publication" in payload:
            proof = _Publication.read(payload["publication"])
            if proof.revision_id != revision.id or proof.origin_path != revision.origin:
                raise ValueError("Publication receipt provenance mismatch")
            snapshot["publication"] = asdict(proof)
            snapshot["operation_id"] = operation.id
        write_durable_json(receipt, snapshot)

    def _restore_lifecycle_snapshot(self, repo: MemoryRepository, snapshot: dict[str, Any]) -> None:
        """Restore supplements only after immutable revisions establish their evidence."""
        identity = snapshot["meeting"]["id"]
        for values in snapshot["audio_assets"]:
            asset = repo.get_audio_asset(values["id"])
            if asset is None or any(
                value != asdict(asset)[name] for name, value in values.items() if name != "status"
            ):
                raise ValueError("Lifecycle state cannot authorize an unreceipted audio asset")
            # Status comes from the actual managed bytes, never a historical health claim.
        for values in snapshot["ingest_sources"]:
            source = IngestSource(**values)
            if source.revision_id is not None:
                revision = repo.get_revision(source.revision_id)
                if revision is None or revision.meeting_id != identity:
                    raise ValueError("Lifecycle origin references an unreceipted revision")
            existing_source = repo.get_ingest_source(source.origin_path, source.origin_fingerprint)
            if existing_source is None:
                repo.insert_ingest_source(source)
            elif existing_source != source:
                raise ValueError("Lifecycle origin conflicts with immutable publication evidence")
        for values in snapshot["participants"]:
            participant = Participant(**values)
            existing_participant = repo.get_participant(participant.id)
            if existing_participant is None:
                repo.insert_participant(participant)
            elif existing_participant != participant:
                raise ValueError("Lifecycle participant identity conflicts with another meeting")
        for values in snapshot["meeting_participants"]:
            repo.library.execute(
                "INSERT INTO meeting_participants VALUES (?,?,?)",
                (identity, values["participant_id"], values["role"]),
            )
        for values in snapshot["speaker_assignments"]:
            self._speaker_scope(repo, identity, values["revision_id"], values["speaker_key"])
            repo.library.execute(
                "INSERT INTO speaker_assignments VALUES (?,?,?,?)",
                (identity, values["revision_id"], values["speaker_key"], values["participant_id"]),
            )
        for values in snapshot["lifecycle_events"]:
            repo.library.execute(
                "INSERT INTO lifecycle_events VALUES (?,?,?,?,?,?)",
                (
                    values["id"],
                    identity,
                    values["kind"],
                    values["occurred_at"],
                    values["recorded_duration_ms"],
                    values["payload"],
                ),
            )
        if (
            snapshot["source_kind"] == "legacy_audio_only"
            and self._known_source_kind(repo, identity) != "legacy_audio_only"
        ):
            reservations = [
                item for item in snapshot["ingest_sources"] if item["revision_id"] is None
            ]
            if not reservations:
                raise ValueError("Audio-only state requires its measured reservation origin")
            origin = reservations[0]
            repo.insert_pending_operation(
                PendingOperation(
                    str(uuid4()),
                    "reserve_audio_only",
                    "complete",
                    identity,
                    payload_json=json.dumps(
                        {
                            "source_kind": "legacy_audio_only",
                            "origin_path": origin["origin_path"],
                            "origin_fingerprint": origin["origin_fingerprint"],
                        }
                    ),
                )
            )

    def _rebuild(
        self,
        repo: MemoryRepository,
        warnings: list[object],
        *,
        can_restore: Callable[[str], bool] | None = None,
    ) -> RecoveryReport:
        completed: list[RecoveryOutcome] = []
        skipped: list[RecoveryOutcome] = []
        marked: list[RecoveryOutcome] = []
        records, _ = _deletion_registry(self.data_dir)
        suppression = _suppression_snapshot(repo.library, records)
        suppressed_origins = {pair for record in records for pair in record.origins}
        meetings_root = managed_path(self.data_dir, "meetings", directory=True)
        if not meetings_root.exists():
            return RecoveryReport()
        for directory in meetings_root.iterdir():
            try:
                identity = str(UUID(directory.name))
                if identity != directory.name:
                    raise ValueError("Invalid managed meeting UUID")
                if identity in suppression.suppressed_meeting_ids:
                    continue
                validate_session_directory(directory)
                # Existing catalog rows stay authoritative. Only absent meetings
                # are restored; uncertain unreceipted publications are not adopted.
                if repo.get_meeting(identity) is not None:
                    continue
                if can_restore is not None and not can_restore(identity):
                    continue
                revisions_path = managed_path(
                    self.data_dir, f"meetings/{identity}/revisions", directory=True
                )
                snapshots = []
                for revision_dir in revisions_path.iterdir() if revisions_path.exists() else ():
                    revision_id = str(UUID(revision_dir.name))
                    validate_session_directory(revision_dir)
                    receipt = session_artifact_path(revision_dir, "catalog.json")
                    if not receipt.exists():
                        warnings.append(f"Uncertain publication retained: {identity}/{revision_id}")
                        continue
                    snapshot = _receipt_envelope(json.loads(receipt.read_text(encoding="utf-8")))
                    if any(
                        (origin.get("origin_path"), origin.get("origin_fingerprint"))
                        in suppressed_origins
                        for origin in snapshot["ingest_sources"]
                    ):
                        raise ValueError(
                            "Suppressed origin receipt retained for owned cleanup only"
                        )
                    if (
                        snapshot.get("schema_version") not in {2, 3}
                        or snapshot["meeting"]["id"] != identity
                        or snapshot["revision"]["meeting_id"] != identity
                        or snapshot["revision"]["id"] != revision_id
                    ):
                        raise ValueError("Invalid managed publication receipt")
                    if snapshot["schema_version"] == 3:
                        proof = _Publication.read(snapshot.get("publication"))
                        if (
                            not _is_uuid(snapshot.get("operation_id"))
                            or proof.revision_id != revision_id
                            or proof.origin_path != snapshot["revision"]["origin"]
                        ):
                            raise ValueError("Managed publication provenance mismatch")
                    validate_artifact_roles(revision_dir, snapshot["hashes"])
                    local_audio = {
                        Path(asset["relative_path"]).name: asset["sha256"]
                        for asset in snapshot["audio_assets"]
                        if Path(asset["relative_path"]).parent
                        == revision_dir.relative_to(self.data_dir)
                    }
                    for name, digest in snapshot["hashes"].items():
                        if name in local_audio and local_audio[name] == digest:
                            # Audio health is measured below; readable text survives its loss.
                            continue
                        if file_hash(session_artifact_path(revision_dir, name)) != digest:
                            raise ValueError("Managed revision hash mismatch during rebuild")
                    for asset in snapshot["audio_assets"]:
                        path = managed_path(self.data_dir, asset["relative_path"])
                        if asset["meeting_id"] != identity or not path.is_relative_to(directory):
                            raise ValueError("Managed audio identity mismatch")
                        if path.suffix.lower() != ".wav":
                            raise ValueError("Managed audio asset must have a WAV role")
                        validate_artifact_roles(path.parent, (path.name,))
                        try:
                            size = path.stat().st_size
                        except FileNotFoundError:
                            asset["status"] = "missing"
                        else:
                            valid = (
                                size == asset["size_bytes"] and file_hash(path) == asset["sha256"]
                            )
                            asset["status"] = "available" if valid else "corrupt"
                    for segment in snapshot["segments"]:
                        if segment["revision_id"] != revision_id:
                            raise ValueError("Managed segment revision mismatch")
                    assets_by_id = {asset["id"]: asset for asset in snapshot["audio_assets"]}
                    for ordinal, link in enumerate(snapshot["revision_audio_assets"]):
                        if (
                            link["meeting_id"] != identity
                            or link["revision_id"] != revision_id
                            or link["asset_id"] not in assets_by_id
                            or link["ordinal"] != ordinal
                        ):
                            raise ValueError("Managed revision audio relationship mismatch")
                        validate_artifact_roles(revision_dir, (link["source_file"],))
                    for origin in snapshot["ingest_sources"]:
                        if origin["meeting_id"] != identity or origin["revision_id"] not in {
                            None,
                            revision_id,
                        }:
                            raise ValueError("Managed source identity mismatch")
                    snapshots.append(snapshot)
                snapshots.sort(key=lambda item: item["revision"]["revision_number"])
                meeting_values = snapshots[-1]["meeting"] if snapshots else None
                lifecycle_snapshot = None
                states = managed_path(self.data_dir, f"meetings/{identity}/states", directory=True)
                if states.exists():
                    for entry in states.iterdir():
                        try:
                            envelope = read_meeting_snapshot(states, entry.name, identity)
                            state = envelope["meeting"]
                        except (OSError, ValueError, TypeError, KeyError) as exc:
                            warnings.append(str(exc))
                            continue
                        if any(
                            (origin.get("origin_path"), origin.get("origin_fingerprint"))
                            in suppressed_origins
                            for origin in envelope.get("ingest_sources", [])
                        ):
                            raise ValueError("Suppressed lifecycle origin cannot restore a meeting")
                        if (
                            meeting_values is None
                            or state["row_version"] >= meeting_values["row_version"]
                        ):
                            meeting_values = state
                            lifecycle_snapshot = (
                                envelope if envelope["schema_version"] == 2 else None
                            )
                if meeting_values is None:
                    continue
                meeting = Meeting(**meeting_values)
                revision_ids = {snapshot["revision"]["id"] for snapshot in snapshots}
                if (
                    meeting.active_revision_id is not None
                    and meeting.active_revision_id not in revision_ids
                ):
                    raise ValueError("Lifecycle state cannot authorize an unreceipted revision")
                with repo.transaction():
                    repo.insert_meeting(replace(meeting, active_revision_id=None))
                    inserted_assets = set()
                    for snapshot in snapshots:
                        revision = TranscriptRevision(**snapshot["revision"])
                        repo.insert_revision(revision)
                        repo.insert_segments([Segment(**item) for item in snapshot["segments"]])
                        for values in snapshot["audio_assets"]:
                            if values["id"] not in inserted_assets:
                                repo.insert_audio_asset(AudioAsset(**values))
                                inserted_assets.add(values["id"])
                        for values in snapshot["revision_audio_assets"]:
                            repo.insert_revision_audio_asset(RevisionAudioAsset(**values))
                        for values in snapshot["ingest_sources"]:
                            repo.insert_ingest_source(IngestSource(**values))
                        if snapshot["schema_version"] == 3:
                            proof = _Publication.read(snapshot["publication"])
                            if proof.revision_of is not None:
                                parent = repo.get_revision(proof.revision_of)
                                if parent is None or parent.meeting_id != identity:
                                    raise ValueError("Managed publication parent mismatch")
                            operation_id = snapshot["operation_id"]
                            repo.insert_pending_operation(
                                PendingOperation(
                                    operation_id,
                                    "publish_revision",
                                    "complete",
                                    identity,
                                    f".staging/{operation_id}",
                                    f"meetings/{identity}/revisions/{revision.id}",
                                    json.dumps(
                                        {
                                            "revision_id": revision.id,
                                            "hashes": snapshot["hashes"],
                                            "publication": asdict(proof),
                                        }
                                    ),
                                )
                            )
                    if lifecycle_snapshot is not None:
                        self._restore_lifecycle_snapshot(repo, lifecycle_snapshot)
                    if meeting.active_revision_id:
                        repo.activate_revision(identity, meeting.active_revision_id)
                        repo.library.execute(
                            "UPDATE meetings SET row_version=? WHERE id=?",
                            (meeting.row_version, identity),
                        )
                    if self._fts(repo) and meeting.active_revision_id:
                        active = repo.get_revision(meeting.active_revision_id)
                        if active and active.status == "complete":
                            for segment in repo.get_segments(active.id):
                                repo.library.execute(
                                    "INSERT INTO segments_fts VALUES (?,?,?,?,?)",
                                    (identity, active.id, segment.id, segment.text, meeting.title),
                                )
                completed.append(
                    RecoveryOutcome(identity, "Catalog restored from managed committed evidence")
                )
                for row in repo.library.execute(
                    "SELECT id,status FROM audio_assets WHERE meeting_id=? AND status!='available'",
                    (identity,),
                ):
                    marked.append(RecoveryOutcome(row["id"], f"Audio asset marked {row['status']}"))
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                warnings.append(f"Managed meeting {directory.name} retained: {exc}")
                skipped.append(
                    RecoveryOutcome(
                        directory.name, "Managed evidence retained; rebuild unavailable"
                    )
                )
        return RecoveryReport(
            completed=tuple(completed), marked=tuple(marked), skipped=tuple(skipped)
        )

    def reconcile(self) -> RecoveryReport:
        warnings: list[object] = []
        completed: list[RecoveryOutcome] = []
        reverted: list[RecoveryOutcome] = []
        marked: list[RecoveryOutcome] = []
        skipped: list[RecoveryOutcome] = []
        with self._operation() as repo, ExitStack() as leases:
            records, stages = _deletion_registry(self.data_dir)
            if stages:
                warnings.append("Uncommitted deletion metadata stages retained")
            # Restore suppression/origins BEFORE the one existing receipt rebuild.
            # A record never grants arbitrary-file authority; A proof still does.
            for record in records:
                try:
                    with repo.transaction():
                        for _, fingerprint in record.origins:
                            repo.library.execute(
                                "INSERT OR IGNORE INTO deleted_origins VALUES (?,?)",
                                (fingerprint, datetime.now(timezone.utc).isoformat()),
                            )
                    with ExitStack() as deletion_leases:
                        self._delete_owner(repo, record.meeting_id, deletion_leases, accepted=True)
                        operation = self._delete_journal(repo, record)
                        result = self._finish_delete(repo, record, operation)
                    if result.status == "completed":
                        completed.append(RecoveryOutcome(record.meeting_id, "Deletion completed"))
                except Exception:
                    warnings.append("Deletion remains pending local recovery")
                    skipped.append(RecoveryOutcome(record.meeting_id, "Deletion cleanup pending"))
            suppressed = set(_suppression_snapshot(repo.library, records).suppressed_meeting_ids)
            if repo.library.storage_warning:
                warnings.append(repo.library.storage_warning)
            rows = repo.library.execute(
                "SELECT * FROM pending_operations WHERE phase='prepared'"
            ).fetchall()
            protected = {row["staging_path"] for row in rows}
            active_rows = repo.library.execute(
                "SELECT id FROM meetings WHERE status IN ('recording','processing')"
            ).fetchall()
            identities = {row[0] for row in active_rows} | {row["meeting_id"] for row in rows}
            live = set()
            checked = set()

            def protect(identity: str | None) -> None:
                if identity is None:
                    return
                if identity in checked:
                    return
                checked.add(identity)
                try:
                    if str(UUID(identity)) != identity:
                        raise ValueError("Invalid owner identity")
                    owner_path = managed_path(self.data_dir, f".owners/{identity}.lock")
                    owner_path.parent.mkdir(parents=True, exist_ok=True)
                    probe = _FileLock(owner_path, timeout=0)
                    leases.callback(probe.close)
                except Busy:
                    live.add(identity)
                    skipped.append(RecoveryOutcome(identity, "Live catalog owner; work preserved"))
                except (OSError, ValueError) as exc:
                    warnings.append(str(exc))
                    # An invalid owner path is not proof the job has stopped.
                    live.add(identity)
                    skipped.append(
                        RecoveryOutcome(identity, "Ownership unavailable; work preserved")
                    )

            for identity in identities:
                protect(identity)
            for row in rows:
                if row["meeting_id"] in live:
                    continue
                if row["kind"] == "delete":
                    # Already handled above with the same retained ownership path.
                    continue
                if row["meeting_id"] in suppressed:
                    continue
                try:
                    self._reconcile_operation(repo, PendingOperation(**dict(row)))
                    after = repo.get_pending_operation(row["id"])
                    if after is not None and after.phase == "complete":
                        completed.append(
                            RecoveryOutcome(after.id, "Committed local operation completed")
                        )
                    elif after is not None and after.phase == "reverted":
                        reverted.append(
                            RecoveryOutcome(after.id, "Uncommitted local operation reverted")
                        )
                except Exception as exc:
                    warnings.append(str(exc))
                    skipped.append(
                        RecoveryOutcome(row["id"], "Operation retained; local recovery required")
                    )

            def can_restore(identity: str) -> bool:
                protect(identity)
                return identity not in live

            rebuilt = self._rebuild(repo, warnings, can_restore=can_restore)
            completed.extend(rebuilt.completed)
            marked.extend(rebuilt.marked)
            skipped.extend(rebuilt.skipped)
            active_rows = repo.library.execute(
                "SELECT id FROM meetings WHERE status IN ('recording','processing')"
            ).fetchall()
            for row in active_rows:
                protect(row[0])
            staging = managed_path(self.data_dir, ".staging", directory=True)
            if staging.exists():
                for entry in staging.iterdir():
                    relative = entry.relative_to(self.data_dir).as_posix()
                    if relative in protected:
                        continue
                    try:
                        UUID(entry.name)  # only owned UUID staging trees
                        _remove_tree(self.data_dir, relative)
                    except (OSError, ValueError) as exc:
                        warnings.append(str(exc))
            state_operations: list[PendingOperation] = []
            with repo.transaction():
                for row in active_rows:
                    if row[0] not in live and row[0] not in suppressed:
                        self._change_status(repo, row[0], MeetingStatus.INTERRUPTED)
                        state_operations.append(self._state_operation(repo, row[0]))
                        marked.append(RecoveryOutcome(row[0], "Unowned work marked interrupted"))
            for operation in state_operations:
                try:
                    self._reconcile_operation(repo, operation)
                    completed.append(
                        RecoveryOutcome(operation.id, "Committed state snapshot completed")
                    )
                except (OSError, ValueError, LibraryError) as exc:
                    warnings.append(str(exc))
                    skipped.append(
                        RecoveryOutcome(
                            operation.id, "Committed state snapshot pending local recovery"
                        )
                    )
            for row in repo.library.execute("SELECT * FROM audio_assets").fetchall():
                if row["meeting_id"] in live or row["meeting_id"] in suppressed:
                    continue
                try:
                    path = managed_path(self.data_dir, row["relative_path"])
                    if not path.is_relative_to(self.data_dir / "meetings" / row["meeting_id"]):
                        raise ValueError("Managed audio identity mismatch")
                    if path.suffix.lower() != ".wav":
                        raise ValueError("Managed audio asset must have a WAV role")
                    validate_artifact_roles(path.parent, (path.name,))
                    try:
                        size = path.stat().st_size
                    except FileNotFoundError:
                        health = "missing"
                    else:
                        valid = size == row["size_bytes"] and file_hash(path) == row["sha256"]
                        health = "available" if valid else "corrupt"
                    if health != row["status"]:
                        with repo.transaction():
                            repo.library.execute(
                                "UPDATE audio_assets SET status=? WHERE id=?", (health, row["id"])
                            )
                        marked.append(RecoveryOutcome(row["id"], f"Audio asset marked {health}"))
                except (OSError, ValueError) as exc:
                    warnings.append(str(exc))
                    skipped.append(
                        RecoveryOutcome(row["id"], "Unsafe or inaccessible asset retained")
                    )
        return RecoveryReport(
            tuple(warnings), tuple(completed), tuple(reverted), tuple(marked), tuple(skipped)
        )

    def begin_capture(self, meeting_id: str, *, started_at: str | None = None) -> None:
        if str(UUID(meeting_id)) != meeting_id:
            raise ValueError("Invalid capture identity")
        if meeting_id not in self._owners:
            self._acquire_owner(meeting_id)
        with self._operation() as repo:
            records, _ = _deletion_registry(self.data_dir)
            if meeting_id in _suppression_snapshot(repo.library, records).suppressed_meeting_ids:
                raise NotFound("Meeting has been deleted")
            with repo.transaction():
                if repo.get_meeting(meeting_id) is None:
                    repo.insert_meeting(
                        Meeting(
                            id=meeting_id,
                            started_at=_utc_timestamp(started_at),
                            status=MeetingStatus.RECORDING,
                        )
                    )
                    repo.record_event(
                        meeting_id,
                        "status_changed",
                        datetime.now(timezone.utc).isoformat(),
                        {"from": None, "to": "recording"},
                    )
            self._write_state(repo, meeting_id)

    def set_status(self, meeting_id: str, status: str) -> None:
        status = MeetingStatus(status)
        with self._operation() as repo:
            operation = None
            with repo.transaction():
                self._change_status(repo, meeting_id, status)
                if repo.get_meeting(meeting_id) is not None:
                    operation = self._state_operation(repo, meeting_id)
            if operation is not None:
                self._reconcile_operation(repo, operation)

    def begin_processing(self, meeting_id: str) -> None:
        """Own the local processing lifetime, preserving any prior active evidence."""
        if str(UUID(meeting_id)) != meeting_id:
            raise ValueError("Invalid processing identity")
        if meeting_id in self._owners:
            raise OwnershipBusy("A processing job already owns this meeting")
        self._acquire_owner(meeting_id)
        with self._operation() as repo:
            records, _ = _deletion_registry(self.data_dir)
            if meeting_id in _suppression_snapshot(repo.library, records).suppressed_meeting_ids:
                raise NotFound("Meeting has been deleted")
            with repo.transaction():
                if repo.get_meeting(meeting_id) is None:
                    repo.insert_meeting(Meeting(id=meeting_id, status=MeetingStatus.PROCESSING))
                    repo.record_event(
                        meeting_id,
                        "status_changed",
                        datetime.now(timezone.utc).isoformat(),
                        {"from": None, "to": "processing"},
                    )
                else:
                    self._change_status(repo, meeting_id, MeetingStatus.PROCESSING)
            self._write_state(repo, meeting_id)

    def _acquire_owner(self, meeting_id: str) -> None:
        try:
            records, _ = _deletion_registry(self.data_dir)
            if any(record.meeting_id == meeting_id for record in records):
                raise NotFound("Meeting has been deleted")
            directory = managed_path(self.data_dir, ".owners", directory=True)
            directory.mkdir(parents=True, exist_ok=True)
            self._owners[meeting_id] = _FileLock(directory / f"{meeting_id}.lock")
        except (Busy, OSError) as exc:
            # Unlike a failed SQLite state update, this failure proves no lease.
            raise OwnershipBusy(str(exc)) from exc

    @staticmethod
    def check_owner(data_dir: Path, meeting_id: str | None) -> None:
        """Probe an actual cross-process lease without opening or writing SQLite."""
        if meeting_id is None:
            return
        try:
            if str(UUID(meeting_id)) != meeting_id:
                return
        except ValueError:
            return
        path = managed_path(data_dir, f".owners/{meeting_id}.lock")
        if path.exists():
            try:
                probe = _FileLock(path, timeout=0, create=False)
            except Busy as exc:
                raise OwnershipBusy(str(exc)) from exc
            probe.close()

    def finish_attempt(
        self, meeting_id: str | None, status: str, *, revision_of: str
    ) -> IngestResult:
        """A failed/cancelled retry changes state, never the completed evidence."""
        if status not in {"failed", "cancelled"}:
            raise ValueError("Only unsuccessful retry outcomes use state-only publication")
        with self._operation() as repo:
            with repo.transaction():
                try:
                    canonical = meeting_id is not None and str(UUID(meeting_id)) == meeting_id
                except ValueError:
                    canonical = False
                if meeting_id is None or not canonical:
                    parent = repo.get_revision(revision_of)
                    if parent is None:
                        return IngestResult(
                            "rejected", warnings=("Revision conflict: missing parent",)
                        )
                    meeting_id = parent.meeting_id
                meeting = repo.get_meeting(meeting_id)
                if meeting is None or meeting.active_revision_id != revision_of:
                    return IngestResult(
                        "rejected", meeting_id, ("Revision conflict: stale outcome",)
                    )
                self._change_status(repo, meeting_id, MeetingStatus(status))
                operation = self._state_operation(repo, meeting_id)
            warnings: tuple[object, ...] = ()
            try:
                self._reconcile_operation(repo, operation)
            except Exception:
                # Only housekeeping can fail after the status/event transaction commits.
                warnings = ("Status saved; local snapshot recovery pending",)
        return IngestResult("unchanged", meeting_id, warnings)

    def finish_capture(self, meeting_id: str, *, discard: bool = False) -> None:
        try:
            if discard:
                with self._operation() as repo:
                    with repo.transaction():
                        repo.library.execute(
                            "DELETE FROM meetings WHERE id=? AND status='recording' "
                            "AND active_revision_id IS NULL",
                            (meeting_id,),
                        )
                    if repo.get_meeting(meeting_id) is None:
                        directory = managed_path(
                            self.data_dir, f"meetings/{meeting_id}", directory=True
                        )
                        _remove_tree(self.data_dir, f"meetings/{meeting_id}/states")
                        try:
                            directory.rmdir()
                        except OSError:
                            pass
        finally:
            owner = self._owners.pop(meeting_id, None)
            if owner:
                owner.close()

    @staticmethod
    def _change_status(repo: MemoryRepository, meeting_id: str, status: MeetingStatus) -> None:
        records = _catalog_records(repo)
        if meeting_id in _suppressed_meeting_ids(repo.library, records):
            raise NotFound("Meeting has been deleted")
        meeting = repo.get_meeting(meeting_id)
        if meeting is None or meeting.status == status:
            return
        now = datetime.now(timezone.utc).isoformat()
        repo.library.execute(
            "UPDATE meetings SET status=?,row_version=row_version+1,updated_at=? WHERE id=?",
            (status, now, meeting_id),
        )
        repo.record_event(meeting_id, "status_changed", now, {"from": meeting.status, "to": status})

    @staticmethod
    def _identity(identity: str) -> None:
        if not _is_uuid(identity):
            raise InvalidInput("Use a canonical UUID")

    @staticmethod
    def _display_name(display_name: str) -> str:
        if not isinstance(display_name, str) or not display_name.strip():
            raise InvalidInput("Participant name must be a nonempty string")
        return display_name.strip()

    @staticmethod
    def _meeting(repo: MemoryRepository, meeting_id: str) -> Meeting:
        records = _catalog_records(repo)
        if meeting_id in _suppressed_meeting_ids(repo.library, records):
            raise NotFound("Meeting does not exist")
        meeting = repo.get_meeting(meeting_id)
        if meeting is None:
            raise NotFound("Meeting does not exist")
        if meeting.status == MeetingStatus.DELETING:
            raise Conflict("Meeting deletion is pending")
        return meeting

    @staticmethod
    def _detail(
        repo: MemoryRepository, meeting_id: str, warnings: tuple[object, ...] = ()
    ) -> MeetingDetail:
        meeting = MemoryService._meeting(repo, meeting_id)
        revisions = tuple(
            TranscriptRevision(**dict(row))
            for row in repo.library.execute(
                "SELECT * FROM transcript_revisions WHERE meeting_id=? ORDER BY revision_number",
                (meeting_id,),
            )
        )
        return MeetingDetail(
            meeting,
            tuple(
                AudioAsset(**dict(row))
                for row in repo.library.execute(
                    "SELECT * FROM audio_assets WHERE meeting_id=? ORDER BY id", (meeting_id,)
                )
            ),
            revisions,
            tuple(segment for revision in revisions for segment in repo.get_segments(revision.id)),
            tuple(
                link
                for revision in revisions
                for link in repo.get_revision_audio_assets(revision.id)
            ),
            warnings,
            tuple(
                Participant(**dict(row))
                for row in repo.library.execute(
                    "SELECT p.* FROM participants p JOIN meeting_participants l "
                    "ON p.id=l.participant_id WHERE l.meeting_id=? ORDER BY p.id",
                    (meeting_id,),
                )
            ),
            tuple(
                MeetingParticipant(**dict(row))
                for row in repo.library.execute(
                    "SELECT * FROM meeting_participants WHERE meeting_id=? ORDER BY participant_id",
                    (meeting_id,),
                )
            ),
            tuple(
                SpeakerAssignment(**dict(row))
                for row in repo.library.execute(
                    "SELECT * FROM speaker_assignments WHERE meeting_id=? "
                    "ORDER BY revision_id,speaker_key",
                    (meeting_id,),
                )
            ),
        )

    @contextmanager
    def _read_operation(self) -> Iterator[MemoryRepository]:
        """Materialize detached values from one explicit SQLite read snapshot."""
        library = open_library(self.data_dir)
        try:
            with catalog_read_view(library):
                records, _ = _deletion_registry(self.data_dir)
                yield _CatalogRepository(library, records)
        finally:
            library.close()

    def get_meeting(self, meeting_id: str) -> MeetingDetail:
        self._identity(meeting_id)
        with self._read_operation() as repo:
            records = _catalog_records(repo)
            if meeting_id in _suppression_snapshot(repo.library, records).suppressed_meeting_ids:
                raise NotFound("Meeting does not exist")
            warnings = (repo.library.storage_warning,) if repo.library.storage_warning else ()
            return self._detail(repo, meeting_id, warnings)

    def list_participants(self, meeting_id: str | None = None) -> tuple[Participant, ...]:
        if meeting_id is not None:
            self._identity(meeting_id)
        with self._read_operation() as repo:
            if meeting_id is None:
                rows = repo.library.execute("SELECT * FROM participants ORDER BY id")
            else:
                self._meeting(repo, meeting_id)
                rows = repo.library.execute(
                    "SELECT p.* FROM participants p JOIN meeting_participants l "
                    "ON p.id=l.participant_id WHERE l.meeting_id=? ORDER BY p.id",
                    (meeting_id,),
                )
            return tuple(Participant(**dict(row)) for row in rows)

    @staticmethod
    def _state_operation(repo: MemoryRepository, meeting_id: str) -> PendingOperation:
        operation = PendingOperation(str(uuid4()), "persist_state", "prepared", meeting_id)
        repo.insert_pending_operation(operation)
        return operation

    def _persist_edit(
        self,
        repo: MemoryRepository,
        meeting_id: str,
        operation: PendingOperation,
        warnings: tuple[object, ...] = (),
    ) -> MeetingDetail:
        if repo.library.storage_warning:
            warnings += (repo.library.storage_warning,)
        try:
            self._reconcile_operation(repo, operation)
        except Exception:
            # SQL has committed. The prepared operation retries local housekeeping.
            warnings += ("Committed lifecycle edit; state snapshot pending local recovery",)
        return self._detail(repo, meeting_id, warnings)

    def add_participant(self, display_name: str) -> Participant:
        participant = Participant(str(uuid4()), self._display_name(display_name))
        with self._operation() as repo:
            with repo.transaction():
                repo.insert_participant(participant)
        return participant

    def link_participant(
        self, meeting_id: str, participant_id: str, role: str | None = None
    ) -> MeetingDetail:
        self._identity(meeting_id)
        self._identity(participant_id)
        if role is not None and not isinstance(role, str):
            raise InvalidInput("Participant role must be a string or omitted")
        with self._operation() as repo:
            with repo.transaction():
                self._meeting(repo, meeting_id)
                if repo.get_participant(participant_id) is None:
                    raise NotFound("Participant does not exist")
                repo.library.execute(
                    "INSERT INTO meeting_participants VALUES (?,?,?) ON CONFLICT "
                    "(meeting_id,participant_id) DO UPDATE SET role=excluded.role",
                    (meeting_id, participant_id, role),
                )
                now = datetime.now(timezone.utc).isoformat()
                repo.library.execute(
                    "UPDATE meetings SET row_version=row_version+1,updated_at=? WHERE id=?",
                    (now, meeting_id),
                )
                repo.record_event(meeting_id, "participant_linked", now, {})
                operation = self._state_operation(repo, meeting_id)
            return self._persist_edit(repo, meeting_id, operation)

    @staticmethod
    def _speaker_scope(
        repo: MemoryRepository, meeting_id: str, scope: str | None, speaker_key: str
    ) -> None:
        if (
            not isinstance(scope, str)
            or not _is_uuid(scope)
            or not isinstance(speaker_key, str)
            or not speaker_key
        ):
            raise InvalidInput("Use the revision and its original request-scoped speaker key")
        revision = repo.get_revision(scope)
        if revision is None or revision.meeting_id != meeting_id:
            raise InvalidInput("Speaker scope must be a revision of this meeting")
        if not repo.library.execute(
            "SELECT 1 FROM segments WHERE revision_id=? AND speaker_key=? LIMIT 1",
            (scope, speaker_key),
        ).fetchone():
            raise InvalidInput("Speaker key does not exist in this request scope")

    def assign_speaker(
        self, meeting_id: str, speaker_key: str, participant_id: str | None, scope: str
    ) -> MeetingDetail:
        self._identity(meeting_id)
        if participant_id is not None:
            self._identity(participant_id)
        with self._operation() as repo:
            with repo.transaction():
                self._meeting(repo, meeting_id)
                self._speaker_scope(repo, meeting_id, scope, speaker_key)
                if participant_id is not None:
                    if repo.get_participant(participant_id) is None:
                        raise NotFound("Participant does not exist")
                    if not repo.library.execute(
                        "SELECT 1 FROM meeting_participants WHERE meeting_id=? "
                        "AND participant_id=?",
                        (meeting_id, participant_id),
                    ).fetchone():
                        raise InvalidInput("Link the participant to the meeting before assignment")
                    repo.library.execute(
                        "INSERT INTO speaker_assignments VALUES (?,?,?,?) ON CONFLICT "
                        "(meeting_id,revision_id,speaker_key) DO UPDATE SET "
                        "participant_id=excluded.participant_id",
                        (meeting_id, scope, speaker_key, participant_id),
                    )
                else:
                    repo.library.execute(
                        "DELETE FROM speaker_assignments WHERE meeting_id=? AND revision_id=? "
                        "AND speaker_key=?",
                        (meeting_id, scope, speaker_key),
                    )
                now = datetime.now(timezone.utc).isoformat()
                repo.library.execute(
                    "UPDATE meetings SET row_version=row_version+1,updated_at=? WHERE id=?",
                    (now, meeting_id),
                )
                repo.record_event(meeting_id, "speaker_assigned", now, {})
                operation = self._state_operation(repo, meeting_id)
            return self._persist_edit(repo, meeting_id, operation)

    def add_meeting_participant(
        self, meeting_id: str, display_name: str, *, speaker_key: str | None = None
    ) -> MeetingDetail:
        """Create, link and optionally assign atomically for the CLI command."""
        self._identity(meeting_id)
        participant = Participant(str(uuid4()), self._display_name(display_name))
        with self._operation() as repo:
            with repo.transaction():
                meeting = self._meeting(repo, meeting_id)
                if speaker_key is not None:
                    self._speaker_scope(repo, meeting_id, meeting.active_revision_id, speaker_key)
                repo.insert_participant(participant)
                repo.library.execute(
                    "INSERT INTO meeting_participants VALUES (?,?,NULL)",
                    (meeting_id, participant.id),
                )
                if speaker_key is not None:
                    repo.library.execute(
                        "INSERT INTO speaker_assignments VALUES (?,?,?,?) ON CONFLICT "
                        "(meeting_id,revision_id,speaker_key) DO UPDATE SET "
                        "participant_id=excluded.participant_id",
                        (meeting_id, meeting.active_revision_id, speaker_key, participant.id),
                    )
                now = datetime.now(timezone.utc).isoformat()
                repo.library.execute(
                    "UPDATE meetings SET row_version=row_version+1,updated_at=? WHERE id=?",
                    (now, meeting_id),
                )
                repo.record_event(meeting_id, "participant_linked", now, {})
                operation = self._state_operation(repo, meeting_id)
            return self._persist_edit(repo, meeting_id, operation)

    def update_meeting(
        self, meeting_id: str, patch: MeetingPatch, expected_version: int
    ) -> MeetingDetail:
        self._identity(meeting_id)
        if not isinstance(patch, MeetingPatch) or any(
            value is not None and not isinstance(value, str) for value in (patch.title, patch.notes)
        ):
            raise InvalidInput("Meeting patch accepts only string title and notes")
        if type(expected_version) is not int or expected_version < 1:
            raise InvalidInput("Expected version must be a positive integer")
        with self._operation() as repo:
            with repo.transaction():
                meeting = self._meeting(repo, meeting_id)
                if meeting.row_version != expected_version:
                    raise Conflict("Meeting changed; reload its current version")
                now = datetime.now(timezone.utc).isoformat()
                title = meeting.title if patch.title is None else patch.title
                notes = meeting.notes if patch.notes is None else patch.notes
                repo.library.execute(
                    "UPDATE meetings SET title=?,notes=?,row_version=row_version+1,updated_at=? "
                    "WHERE id=? AND row_version=?",
                    (title, notes, now, meeting_id, expected_version),
                )
                has_fts = self._fts(repo)
                if has_fts:
                    repo.library.execute(
                        "UPDATE segments_fts SET title=? WHERE meeting_id=?", (title, meeting_id)
                    )
                repo.record_event(
                    meeting_id,
                    "metadata_updated",
                    now,
                    {
                        "fields": [
                            name for name in ("title", "notes") if getattr(patch, name) is not None
                        ]
                    },
                )
                operation = self._state_operation(repo, meeting_id)
            return self._persist_edit(
                repo, meeting_id, operation, () if has_fts else (FtsUnavailable(),)
            )

    def show(self, meeting_id: str) -> dict[str, object] | None:
        with self._read_operation() as repo:
            records = _catalog_records(repo)
            if meeting_id in _suppression_snapshot(repo.library, records).suppressed_meeting_ids:
                return None
            meeting = repo.get_meeting(meeting_id)
            if meeting is None:
                return None
            result: dict[str, object] = {"meeting": asdict(meeting)}
            for table, key in (
                ("audio_assets", "audio_assets"),
                ("transcript_revisions", "revisions"),
            ):
                result[key] = [
                    dict(row)
                    for row in repo.library.execute(
                        f"SELECT * FROM {table} WHERE meeting_id=?", (meeting_id,)
                    )
                ]
            result["segments"] = [
                dict(row)
                for row in repo.library.execute(
                    "SELECT s.* FROM segments s JOIN transcript_revisions r ON r.id=s.revision_id "
                    "WHERE r.meeting_id=? ORDER BY r.revision_number,s.ordinal",
                    (meeting_id,),
                )
            ]
            result["revision_audio_assets"] = [
                dict(row)
                for row in repo.library.execute(
                    "SELECT l.* FROM revision_audio_assets l JOIN transcript_revisions r "
                    "ON r.id=l.revision_id WHERE l.meeting_id=? "
                    "ORDER BY r.revision_number,l.ordinal",
                    (meeting_id,),
                )
            ]
            result["participants"] = [
                dict(row)
                for row in repo.library.execute(
                    "SELECT p.* FROM participants p JOIN meeting_participants l "
                    "ON p.id=l.participant_id WHERE l.meeting_id=? ORDER BY p.id",
                    (meeting_id,),
                )
            ]
            for table in ("meeting_participants", "speaker_assignments", "lifecycle_events"):
                result[table] = [
                    dict(row)
                    for row in repo.library.execute(
                        f"SELECT * FROM {table} WHERE meeting_id=? ORDER BY 1", (meeting_id,)
                    )
                ]
            return result
