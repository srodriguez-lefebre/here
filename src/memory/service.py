"""Local immutable file publication with a recoverable SQLite operation journal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from contextlib import ExitStack, closing, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Iterable, Iterator
from uuid import UUID, uuid4

from here.memory.artifacts import (
    PORTABLE_GROUP_NAME,
    PORTABLE_RECEIPT_NAME,
    PendingIntent,
    PortableCompletion,
    PortableCompletionMember,
    PortableGroup,
    PortableReceipt,
    SessionArtifacts,
    file_hash,
    managed_path,
    normalized_origin,
    portable_completion_name,
    portable_stage_io_path,
    read_artifacts,
    read_meeting_snapshot,
    read_meeting_state,
    read_pending_intent,
    read_portable_completion,
    read_portable_group,
    read_portable_receipt,
    read_retry_artifacts,
    validate_artifact_roles,
    validate_portable_group_members,
    write_durable_json,
)
from here.memory.database import (
    Busy,
    IntegrityFailure,
    LibraryConnection,
    LibraryError,
    migrate,
    open_library,
    storage_error,
)
from here.memory.models import (
    AudioAsset,
    Citation,
    CitationTarget,
    DeletionResult,
    IngestSource,
    Meeting,
    MeetingDetail,
    MeetingPatch,
    MeetingStatus,
    Participant,
    PendingOperation,
    ReindexReport,
    RevisionAudioAsset,
    Segment,
    SuppressionSnapshot,
    TranscriptRevision,
)
from here.memory.portable_package import validate_entry_name
from here.memory.repository import MemoryRepository, _PortableExportRows
from here.output.export_binding import (
    ExportBindingError,
    ExportIdentity,
    WindowsExportSibling,
    open_export_sibling,
    recover_export_sibling,
    validate_export_identity,
)
from here.output.metadata import SessionMetadata
from here.output.ownership import FileLease, LeaseBusy
from here.output.paths import UnsafeSessionPath, session_artifact_path, validate_session_directory
from here.output.session_writer import read_session_metadata

if TYPE_CHECKING:
    from here.memory.portable_import import (
        ImportReport,
        _PortableImportedMeeting,
        _PortableImportInventory,
        _PortableMeetingOutcome,
        _ValidatedPortableImport,
    )


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


class _DeleteScopeUnavailable(LibraryError):
    def __init__(self) -> None:
        super().__init__("Private staging retained; pending deletion scope cannot be verified")


class _ExportCleanupPending(LibraryError):
    def __init__(self) -> None:
        super().__init__("Export temporary cleanup pending local recovery")


class _ImportCleanupPending(LibraryError):
    def __init__(self) -> None:
        super().__init__("Portable import staging cleanup pending local recovery")


class _ImportCommitUncertain(LibraryError):
    """The entity transaction body finished but its commit was not confirmed."""


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


_EXPORT_SIBLING_NAME = re.compile(r"\.here-export-([0-9a-f]{32})\.partial")
_EXPORT_REGISTRY_FIELDS = frozenset(
    {"schema_version", "operation_id", "meeting_ids", "sibling", "parent_identity", "file_identity"}
)


@dataclass(frozen=True, slots=True)
class _ExportRegistration:
    operation_id: str
    meeting_ids: tuple[str, ...]
    sibling: Path
    parent_identity: ExportIdentity
    file_identity: ExportIdentity

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "operation_id": self.operation_id,
            "meeting_ids": self.meeting_ids,
            "sibling": str(self.sibling),
            "parent_identity": self.parent_identity,
            "file_identity": self.file_identity,
        }

    @classmethod
    def read(cls, value: object, operation_id: str) -> _ExportRegistration:
        if not isinstance(value, dict) or value.keys() != _EXPORT_REGISTRY_FIELDS:
            raise ValueError("Invalid export registration fields")
        ids = value["meeting_ids"]
        sibling = value["sibling"]
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or not _is_uuid(operation_id)
            or value["operation_id"] != operation_id
            or not isinstance(ids, (list, tuple))
            or not ids
            or not all(_is_uuid(identity) for identity in ids)
            or tuple(ids) != tuple(sorted(set(ids)))
            or not isinstance(sibling, str)
        ):
            raise ValueError("Invalid export registration values")
        path = Path(sibling)
        match = _EXPORT_SIBLING_NAME.fullmatch(path.name)
        if (
            match is None
            or str(UUID(hex=match[1])) != operation_id
            or str(path) != sibling
            or not path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("Invalid export temporary role")
        parent = validate_export_identity(value["parent_identity"])
        file = validate_export_identity(value["file_identity"])
        if parent[:2] != file[:2] or parent[2] == file[2]:
            raise ValueError("Export temporary and parent identities disagree")
        return cls(operation_id, tuple(ids), path, parent, file)


def _export_registrations(data_dir: Path) -> Iterator[tuple[Path, _ExportRegistration]]:
    directory = managed_path(data_dir, ".exports", directory=True)
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except FileNotFoundError:
        return
    for entry in entries:
        path = session_artifact_path(directory, entry.name)
        if path.suffix == ".json" and _is_uuid(path.stem):
            yield (
                path,
                _ExportRegistration.read(json.loads(path.read_text(encoding="utf-8")), path.stem),
            )
            continue
        stage = re.fullmatch(r"\.([0-9a-f-]{36})\.json\.[A-Za-z0-9_-]{8}\.stage", path.name)
        if stage is None or not _is_uuid(stage[1]):
            raise ValueError("Unrecognized export registry entry retained")
        # Interrupted registry writes were never sufficient to start streaming.


_IMPORT_REGISTRATION = ".import.json"
_IMPORT_LEASE = ".import.lock"
_IMPORT_WORK = ".import-work.json"
_IMPORT_STAGE_NAME = re.compile(r"import-([0-9a-f-]{36})")


@dataclass(frozen=True, slots=True)
class _ImportRegistration:
    operation_id: str
    meeting_ids: tuple[str, ...]
    files: tuple[tuple[str, int, str], ...]

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "operation_id": self.operation_id,
            "meeting_ids": self.meeting_ids,
            "files": self.files,
        }

    @classmethod
    def read(cls, value: object, operation_id: str) -> _ImportRegistration:
        if (
            not isinstance(value, dict)
            or value.keys() != {"schema_version", "operation_id", "meeting_ids", "files"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or not _is_uuid(operation_id)
            or value["operation_id"] != operation_id
        ):
            raise ValueError("Invalid portable import stage registration")
        meeting_ids = value["meeting_ids"]
        files = value["files"]
        if (
            not isinstance(meeting_ids, (list, tuple))
            or not meeting_ids
            or not all(_is_uuid(identity) for identity in meeting_ids)
            or tuple(meeting_ids) != tuple(sorted(set(meeting_ids)))
            or not isinstance(files, (list, tuple))
            or not 1 <= len(files) < 10_000
        ):
            raise ValueError("Invalid portable import stage inventory")
        validated: dict[str, tuple[str, int, str]] = {}
        identities = frozenset(meeting_ids)
        total = 0
        for entry in files:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise ValueError("Invalid portable import stage file")
            name, size, digest = entry
            if (
                not isinstance(name, str)
                or type(size) is not int
                or not 0 <= size <= 20 * 1024**3
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("Invalid portable import stage file metadata")
            path = validate_entry_name(name)
            if not _valid_import_payload_role(path.parts, identities):
                raise ValueError("Invalid portable import stage role")
            if name in validated:
                raise ValueError("Duplicate portable import stage role")
            validated[name] = (name, size, digest)
            total += size
            if total > 20 * 1024**3:
                raise ValueError("Portable import stage inventory exceeds its byte limit")
        return cls(operation_id, tuple(meeting_ids), tuple(validated.values()))


def _valid_import_payload_role(parts: tuple[str, ...], meeting_ids: frozenset[str]) -> bool:
    match parts:
        case ("participants.json" | "speaker_assignments.json",):
            return True
        case ("meetings", identity, "meeting.json"):
            return identity in meeting_ids
        case ("meetings", identity, "revisions", number, "revision.json" | "segments.json"):
            return identity in meeting_ids and re.fullmatch(r"[1-9][0-9]{0,18}", number) is not None
        case ("meetings", identity, "audio", name):
            return identity in meeting_ids and name.endswith(".wav") and _is_uuid(name[:-4])
        case _:
            return False  # Answer payloads are gated on the M4 B handoff.


@dataclass(slots=True)
class _PortableImportStage:
    operation_id: str
    directory: Path
    _registration: _ImportRegistration
    retain_for_recovery: bool = False


@dataclass(frozen=True, slots=True)
class _ImportMeetingPlan:
    group: PortableGroup
    manifest_sha256: str
    hashes: dict[str, dict[str, str]]
    stage_root: Path | None = None

    def evidence_stage_name(self, role: str) -> str:
        return f".{self.group.operation_id}-{role}.stage"

    def document(self) -> dict[str, object]:
        return {
            "group": self.group.document(),
            "manifest_sha256": self.manifest_sha256,
            "hashes": self.hashes,
        }

    @classmethod
    def read(cls, value: object) -> _ImportMeetingPlan:
        if not isinstance(value, dict) or value.keys() != {"group", "manifest_sha256", "hashes"}:
            raise ValueError("Invalid portable meeting plan")
        group = value["group"]
        manifest = value["manifest_sha256"]
        hashes = value["hashes"]
        if (
            not isinstance(group, dict)
            or group.keys() != {"schema_version", "operation_id", "meeting_id", "revision_ids"}
            or type(group["schema_version"]) is not int
            or group["schema_version"] != 1
            or not _is_uuid(group["operation_id"])
            or not _is_uuid(group["meeting_id"])
            or not isinstance(group["revision_ids"], (tuple, list))
            or not all(_is_uuid(identity) for identity in group["revision_ids"])
            or len(set(group["revision_ids"])) != len(group["revision_ids"])
            or not isinstance(manifest, str)
            or re.fullmatch(r"[0-9a-f]{64}", manifest) is None
            or not isinstance(hashes, dict)
            or hashes.keys() != set(group["revision_ids"])
        ):
            raise ValueError("Invalid portable meeting ownership proof")
        for roles in hashes.values():
            if not isinstance(roles, dict) or any(
                not isinstance(name, str)
                or re.fullmatch(r"audio_[0-9a-f]{32}\.wav", name) is None
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for name, digest in roles.items()
            ):
                raise ValueError("Invalid portable meeting artifact inventory")
        return cls(
            PortableGroup(group["operation_id"], group["meeting_id"], tuple(group["revision_ids"])),
            manifest,
            hashes,
        )


def _read_import_work(
    stage_root: Path, registration: _ImportRegistration
) -> dict[str, _ImportMeetingPlan]:
    proof = session_artifact_path(stage_root, _IMPORT_WORK)
    try:
        with proof.open("rb") as stream:
            data = stream.read(8 * 1024**2 + 1)
    except FileNotFoundError:
        return {}
    if len(data) > 8 * 1024**2:
        raise ValueError("Portable import work inventory is too large")
    value = json.loads(data)
    if (
        not isinstance(value, dict)
        or value.keys() != {"schema_version", "operation_id", "plans"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["operation_id"] != registration.operation_id
        or not isinstance(value["plans"], list)
        or len(value["plans"]) > len(registration.meeting_ids)
    ):
        raise ValueError("Invalid portable import work inventory")
    plans = [_ImportMeetingPlan.read(entry) for entry in value["plans"]]
    result = {plan.group.meeting_id: plan for plan in plans}
    if (
        len(result) != len(plans)
        or len({plan.group.operation_id for plan in plans}) != len(plans)
        or result.keys() - set(registration.meeting_ids)
    ):
        raise ValueError("Portable import work does not match selected meetings")
    return result


def _write_import_intent(directory: Path, name: str, document: object) -> None:
    """Stage a locally assigned protocol name recoverable after process death."""
    directory = portable_stage_io_path(directory)
    path = session_artifact_path(directory, name)
    temporary = session_artifact_path(directory, f"{name}.stage")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _import_tree_entries(
    root: Path, plan: _ImportMeetingPlan, *, published: bool
) -> tuple[list[Path], list[Path]]:
    """Validate all existing entries before removing any incomplete owned tree."""
    root = portable_stage_io_path(root)
    validate_session_directory(root)
    files: list[Path] = []
    directories: list[Path] = []
    if not root.exists():
        return files, directories
    registration = session_artifact_path(root, PORTABLE_GROUP_NAME)
    if registration.exists():
        group = read_portable_group(root, plan.group.meeting_id)
        if group.document() != plan.group.document():
            raise ValueError("Portable meeting group changed during cleanup")
        files.append(registration)
    elif published:
        raise ValueError("Published portable meeting has no owned group proof")
    revisions = root / "revisions"
    allowed = {PORTABLE_GROUP_NAME, "revisions"}
    if not published:
        staged = session_artifact_path(root, f"{PORTABLE_GROUP_NAME}.stage")
        if staged.exists():
            files.append(staged)
        allowed.add(staged.name)
    if any(entry.name not in allowed for entry in root.iterdir()):
        raise ValueError("Unknown portable meeting work retained")
    validate_session_directory(revisions)
    if revisions.exists():
        for revision in revisions.iterdir():
            hashes = plan.hashes.get(revision.name)
            if hashes is None:
                raise ValueError("Unknown portable revision work retained")
            validate_session_directory(revision)
            for entry in revision.iterdir():
                if entry.name not in hashes:
                    raise ValueError("Unknown portable revision material retained")
                file = session_artifact_path(revision, entry.name)
                if published and file_hash(file) != hashes[entry.name]:
                    raise ValueError("Portable revision material changed during cleanup")
                files.append(file)
            directories.append(revision)
        directories.append(revisions)
    directories.append(root)
    return files, directories


def _stage_payload_tree(
    root: Path, files: Iterable[tuple[str, int, str]]
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Validate every child before returning any removal authority."""
    root = portable_stage_io_path(root)
    inventory = {Path(name): size for name, size, _ in files}
    directories = {parent for file in inventory for parent in file.parents}
    present_files: list[Path] = []
    present_directories: list[Path] = []

    def inspect(directory: Path) -> None:
        validate_session_directory(directory)
        for entry in directory.iterdir():
            relative = entry.relative_to(root)
            if relative in directories:
                inspect(entry)
            elif relative in inventory:
                path = session_artifact_path(directory, entry.name)
                if path.stat().st_size > inventory[relative]:
                    raise ValueError("Portable import stage payload exceeded its registered size")
                present_files.append(path)
            else:
                raise ValueError("Unknown portable import stage material retained")
        present_directories.append(directory)

    if root.exists():
        inspect(root)
    return tuple(present_files), tuple(present_directories)


def _read_import_registration(stage_root: Path) -> _ImportRegistration:
    match = _IMPORT_STAGE_NAME.fullmatch(stage_root.name)
    if match is None or not _is_uuid(match[1]):
        raise ValueError("Invalid portable import stage role")
    path = session_artifact_path(stage_root, _IMPORT_REGISTRATION)
    with path.open("rb") as stream:
        data = stream.read(8 * 1024**2 + 1)
    if len(data) > 8 * 1024**2:
        raise ValueError("Portable import stage registration is too large")
    return _ImportRegistration.read(json.loads(data), match[1])


def _purge_import_stage_payload(stage_root: Path, registration: _ImportRegistration) -> None:
    """Prove every file before removing only the registered stage payload."""
    allowed = {
        _IMPORT_REGISTRATION,
        _IMPORT_LEASE,
        _IMPORT_WORK,
        f"{_IMPORT_WORK}.stage",
        "payload",
        "work",
    }
    for name in (_IMPORT_REGISTRATION, _IMPORT_LEASE, _IMPORT_WORK):
        session_artifact_path(stage_root, name)
    plans = _read_import_work(stage_root, registration)
    evidence_roles = {
        plan.evidence_stage_name(role)
        for plan in plans.values()
        for role in ("state", "completion", *plan.group.revision_ids)
    }
    evidence_files = []
    for entry in stage_root.iterdir():
        if entry.name in evidence_roles:
            evidence_files.append(session_artifact_path(stage_root, entry.name))
        elif entry.name not in allowed:
            raise ValueError("Unknown portable import staging material retained")
    interrupted_work = session_artifact_path(stage_root, f"{_IMPORT_WORK}.stage")
    files, directories = _stage_payload_tree(stage_root / "payload", registration.files)
    work = stage_root / "work"
    work_files: list[Path] = []
    work_directories: list[Path] = []
    validate_session_directory(work)
    if work.exists():
        for entry in work.iterdir():
            if entry.name not in plans:
                raise ValueError("Unknown portable import meeting work retained")
            member_files, member_directories = _import_tree_entries(
                entry, plans[entry.name], published=False
            )
            work_files.extend(member_files)
            work_directories.extend(member_directories)
        work_directories.append(work)
    for file in (*files, *work_files, *evidence_files):
        session_artifact_path(file.parent, file.name).unlink()
    for directory in (*directories, *work_directories):
        validate_session_directory(directory)
        directory.rmdir()
    interrupted_work.unlink(missing_ok=True)


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


@dataclass
class _RecoveryPublication:
    repo: MemoryRepository
    directory: Path
    declared_identity: str | None
    intent: PendingIntent | None
    active: bool = True


def _sealed_origin_fingerprint(
    directory: Path, metadata: SessionMetadata, pending: PendingIntent
) -> str | None:
    """Recognize a matching tombstone from the seal without reading source bytes."""
    if pending.hashes is None:
        return None
    names = {"session.json", *metadata.output_files}
    names.update(item.audio_file for item in metadata.capture_sources if item.audio_file)
    if metadata.recoverable_audio:
        names.add(metadata.recoverable_audio)
    # Match the existing ingest fingerprint's file set, including its v1 fallbacks.
    names.update(("raw_transcript.txt", "transcript.txt", "segments.json"))
    if not metadata.recoverable_audio or (
        pending.revision_of is not None
        and not session_artifact_path(directory, metadata.recoverable_audio).exists()
    ):
        names.add("audio.wav")
    present = {name for name in names if session_artifact_path(directory, name).is_file()}
    if present != {name for name, _ in pending.hashes}:
        return None
    return hashlib.sha256(json.dumps(pending.hashes).encode()).hexdigest()


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
def _catalog_snapshot_locked(library: LibraryConnection) -> Iterator[SuppressionSnapshot]:
    """Read one snapshot while the caller owns the catalog publication lease."""
    if library.connection.in_transaction:
        raise LibraryError("Catalog read view requires a fresh connection without a transaction")
    owned_transaction = False
    try:
        try:
            library.execute("BEGIN")
            owned_transaction = True
            records, _ = _deletion_registry(library.data_dir)
            snapshot = _suppression_snapshot(library, records)
        except sqlite3.Error as exc:
            raise storage_error(exc) from exc
        except (OSError, ValueError) as exc:
            raise LibraryError("Catalog read view unavailable") from exc
        try:
            yield snapshot
        except sqlite3.Error as exc:
            raise storage_error(exc) from exc
    finally:
        if owned_transaction:
            library.connection.rollback()


def _catalog_publication_lock(data_dir: Path) -> _FileLock:
    try:
        return _FileLock(managed_path(data_dir, ".memory.lock"))
    except (OSError, ValueError) as exc:
        raise LibraryError("Catalog read view unavailable") from exc


@dataclass(slots=True)
class _PortableExportView:
    _memory: MemoryService
    _ids: tuple[str, ...]
    _suppression: SuppressionSnapshot
    rows: _PortableExportRows
    warnings: tuple[str, ...] = ()
    _active: bool = True
    _temporary: WindowsExportSibling | None = None
    _publish_attempted: bool = False
    _published: bool = False

    @contextmanager
    def open_audio(self, asset: AudioAsset) -> Iterator[BinaryIO]:
        if not self._active or not any(asset in item.audio_assets for item in self.rows.meetings):
            raise InvalidInput("Audio is outside the selected export snapshot")
        try:
            path = managed_path(self._memory.data_dir, asset.relative_path)
            revision_parent = self._memory.data_dir / "meetings" / asset.meeting_id / "revisions"
            if (
                asset.status != "available"
                or path.parent.parent != revision_parent
                or not _is_uuid(path.parent.name)
                or path.suffix.lower() != ".wav"
            ):
                raise ValueError("Invalid managed export audio role")
            imported = self._memory._portable_receipt_for_asset(path, asset.meeting_id)
            hashes = (
                imported.hashes
                if imported is not None
                else _delete_publication_plan(path.parent)["hashes"]
            )
            if hashes.get(path.name) != asset.sha256:
                raise ValueError("Managed audio disagrees with its publication proof")
            validate_artifact_roles(path.parent, (path.name,))
            stream = path.open("rb")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise IntegrityFailure("Selected audio is unavailable or invalid") from exc
        with stream:
            yield stream

    @contextmanager
    def _open_export_destination(self, destination: Path) -> Iterator[WindowsExportSibling]:
        if not self._active or self._temporary is not None or self._publish_attempted:
            raise InvalidInput("Export destination scope is unavailable")
        if not destination.name.casefold().endswith(".here.zip"):
            raise InvalidInput("Export destination must name a .here.zip archive")
        destination = destination.absolute()
        # Application-managed directories recognize a fixed set of owned roles.
        # A user package must not become unknown material in those directories.
        if destination.parent.is_relative_to(self._memory.data_dir):
            raise InvalidInput("Choose a package destination outside the managed library")
        registration: Path | None = None
        cleanup_verified = False
        try:
            with ExitStack() as native_scope:
                try:
                    temporary = native_scope.enter_context(
                        open_export_sibling(destination, excluded_root=self._memory.data_dir)
                    )
                    self._temporary = temporary
                    registration = self._memory._register_export(self._ids, temporary)
                    yield temporary
                finally:
                    if self._temporary is not None and not self._published:
                        try:
                            self._temporary.remove()
                        except ExportBindingError as exc:
                            raise _ExportCleanupPending() from exc
                        cleanup_verified = True
                    self._temporary = None
        except ExportBindingError as exc:
            if not self._published:
                raise LibraryError("Could not open or publish the owned export temporary") from exc
            self.warnings += (str(_ExportCleanupPending()),)
        finally:
            if (
                registration is not None
                and (self._published or cleanup_verified)
                and not self.warnings
            ):
                try:
                    self._memory._retire_export(registration)
                except _ExportCleanupPending as exc:
                    if not self._published:
                        raise
                    self.warnings += (str(exc),)

    def publish(self, temporary: WindowsExportSibling, *, cancelled: Callable[[], bool]) -> None:
        if not self._active or temporary is not self._temporary or self._publish_attempted:
            raise InvalidInput("Export publication requires its active temporary")
        self._publish_attempted = True

        def before_rename() -> None:
            if cancelled():
                from here.memory.portability import ExportCancelled

                raise ExportCancelled("Export cancelled")

        try:
            with closing(_catalog_publication_lock(self._memory.data_dir)):
                try:
                    suppression, rows = self._memory._portable_rows_locked(self._ids)
                except NotFound as exc:
                    raise Conflict(
                        "Catalog changed during export; select the meetings again"
                    ) from exc
                if self._suppression != suppression or self.rows != rows:
                    raise Conflict("Catalog changed during export; select the meetings again")
                temporary.replace_destination(before_rename=before_rename)
                self._published = True
        except OSError:
            if not self._published:
                raise
            self.warnings += (str(_ExportCleanupPending()),)


@contextmanager
def catalog_read_view(library: LibraryConnection) -> Iterator[SuppressionSnapshot]:
    """Own only the bounded publication lease and a fresh caller's read transaction."""
    if library.connection.in_transaction:
        raise LibraryError("Catalog read view requires a fresh connection without a transaction")
    lock = _catalog_publication_lock(library.data_dir)
    try:
        with _catalog_snapshot_locked(library) as snapshot:
            yield snapshot
    finally:
        lock.close()


def _legacy_retry_parent(
    library: LibraryConnection, directory: Path, suppressed: tuple[str, ...]
) -> tuple[str, str | None] | None:
    """Resolve existing lineage without inspecting source material or inventing an ID."""
    rows = library.execute(
        "SELECT DISTINCT m.id,m.active_revision_id FROM ingest_sources i "
        "JOIN meetings m ON m.id=i.meeting_id WHERE i.origin_path=?",
        (normalized_origin(directory),),
    ).fetchall()
    visible = [row for row in rows if row[0] not in suppressed]
    if rows and not visible:
        raise NotFound("Origin has been deleted")
    if len(visible) > 1:
        raise ValueError("Legacy origin conflict: more than one catalog identity")
    return (visible[0][0], visible[0][1]) if visible else None


@contextmanager
def _recovery_read_view_locked(
    data_dir: Path,
    directory: Path,
    declared_identity: str | None,
    records: tuple[_DeletionRecord, ...],
) -> Iterator[str | None]:
    """Already under the sole publication lock; own only any necessary SQL view."""
    known = _is_uuid(declared_identity)
    direct = tuple(record.meeting_id for record in records)
    if known and declared_identity in direct:
        raise NotFound("Meeting has been deleted")
    pairs = any(record.origins for record in records)
    library: LibraryConnection | None = None
    owned_transaction = False
    try:
        try:
            if pairs or not known:
                database = managed_path(data_dir, "library.sqlite3")
                if database.exists():
                    library = open_library(data_dir)
                    library.execute("BEGIN")
                    owned_transaction = True
                elif pairs:
                    raise LibraryError("Recovery alias verification requires an available catalog")
        except sqlite3.Error as exc:
            raise storage_error(exc) from exc
        except UnsafeSessionPath:
            raise
        except (OSError, ValueError) as exc:
            raise LibraryError("Recovery SQL read view unavailable") from exc

        identity = declared_identity
        if library is not None:
            suppressed = _suppressed_meeting_ids(library, records)
            if known:
                if identity in suppressed:
                    raise NotFound("Meeting has been deleted")
            else:
                parent = _legacy_retry_parent(library, directory, suppressed)
                if parent is not None:
                    identity = parent[0]
        validate_session_directory(directory)
        yield identity
    finally:
        if library is not None:
            try:
                if owned_transaction:
                    library.connection.rollback()
            finally:
                library.close()


@contextmanager
def recovery_read_view(
    data_dir: Path, directory: Path, declared_identity: str | None
) -> Iterator[str | None]:
    """Hold a real recovery visibility scope without constructing or initializing SQL."""
    lock: _FileLock | None = None
    try:
        try:
            root = Path(data_dir).absolute()
            validate_session_directory(root)
            root.mkdir(parents=True, exist_ok=True)
            lock = _FileLock(managed_path(root, ".memory.lock"))
            records, _ = _deletion_registry(root)
            if declared_identity is not None and not isinstance(declared_identity, str):
                raise ValueError("Recovery identity must be text or absent")
            directory = Path(directory).absolute()
        except sqlite3.Error as exc:
            raise storage_error(exc) from exc
        except UnsafeSessionPath:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise LibraryError("Recovery read view unavailable") from exc
        with _recovery_read_view_locked(root, directory, declared_identity, records) as identity:
            yield identity
    finally:
        if lock is not None:
            lock.close()


def _journal_path_key(root: Path, relative: str | None) -> str | None:
    """Block lexical aliases; this key never grants cleanup authority."""
    if relative is None:
        return None
    if (
        not isinstance(relative, str)
        or not relative
        or any(ord(character) < 32 for character in relative)
    ):
        raise LibraryError("Invalid staging journal reference; material retained")
    spelling = relative.replace("\\", "/")
    windows = PureWindowsPath(spelling)
    if (windows.drive and not windows.is_absolute()) or spelling.startswith(("//?/", "//./")):
        raise LibraryError("Ambiguous staging journal reference; material retained")
    return os.path.normcase(os.path.normpath(os.path.join(root, spelling)))


def _stage_references(
    root: Path, rows: Iterable[sqlite3.Row], relative: str, operation_id: str
) -> list[sqlite3.Row]:
    key = _journal_path_key(root, relative)
    return [
        row
        for row in rows
        if row["id"] == operation_id
        or key
        in (
            _journal_path_key(root, row["staging_path"]),
            _journal_path_key(root, row["destination_path"]),
        )
    ]


def _validate_delete_roles(directory: Path, names: Iterable[str]) -> None:
    """Existing WAVs still need role proof; already removed WAVs need no unlink."""
    remaining = []
    for name in names:
        path = session_artifact_path(directory, name)
        if path.suffix.lower() != ".wav" or path.exists():
            remaining.append(name)
    validate_artifact_roles(directory, remaining)


def _delete_publication_plan(directory: Path) -> dict[str, Any]:
    registration = session_artifact_path(directory, ".publication.json")
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
    return plan


def _delete_file_order(relative: str) -> tuple[int, bool, str]:
    path = Path(relative)
    if path.name == PORTABLE_GROUP_NAME:
        role = 5
    elif path.name in {"catalog.json", ".publication.json", "session.json", PORTABLE_RECEIPT_NAME}:
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


def _remove_tree(
    root: Path,
    relative: str,
    names: set[str] | None = None,
    *,
    order: Callable[[str], tuple[int, bool, str]] | None = None,
) -> None:
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
    if order is not None:
        sort_order = order
        entries.sort(key=lambda entry: sort_order(entry.name))
    for entry in entries:
        session_artifact_path(directory, entry.name).unlink()
    directory.rmdir()


class MemoryService:
    def __init__(self, data_dir: Path, *, sessions_root: Path | None = None) -> None:
        self.data_dir = Path(data_dir).absolute()
        self._sessions_roots: dict[str, Path] = {}
        if sessions_root is not None:
            self._register_sessions_root(sessions_root)
        self._owners: dict[str, _FileLock] = {}
        self.before_delete: list[Callable[[str], None]] = []
        self._derived_row_purgers: list[Callable[[MemoryRepository, str], None]] = []
        self._publication_context: ContextVar[tuple[str, str] | None] = ContextVar(
            "here_memory_publication", default=None
        )
        self._recovery_publication_context: ContextVar[_RecoveryPublication | None] = ContextVar(
            "here_memory_recovery_publication", default=None
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

    def _register_sessions_root(self, root: Path) -> None:
        """Retain explicit composition authority lexically, without resolving or probing it."""
        directory = Path(os.path.abspath(root))
        self._sessions_roots.setdefault(os.path.normcase(str(directory)), directory)

    def _retain_native_owners(self, identity: str, leases: ExitStack) -> None:
        """Probe only declared native roots; retain existing applicable leases."""
        if not self._sessions_roots:
            raise LibraryError("Explicit native ownership context is required for cleanup")
        if not _is_uuid(identity):
            raise LibraryError("Cleanup requires a canonical meeting identity")
        try:
            for key in sorted(self._sessions_roots):
                root = self._sessions_roots[key]
                # Shared validators reject redirects, junctions, non-directories,
                # unsafe owner roles and multiple links before any file is opened.
                path = session_artifact_path(root / ".capture_owners", f"{identity}.lock")
                try:
                    owner = _FileLock(path, timeout=0, create=False)
                except FileNotFoundError:
                    continue  # Never create a declared root/owner to discover activity.
                leases.callback(owner.close)
        except (OSError, ValueError) as exc:
            raise LibraryError("Native ownership context could not be verified") from exc

    def _retain_catalog_owner(self, identity: str, leases: ExitStack) -> None:
        path = managed_path(self.data_dir, f".owners/{identity}.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        leases.enter_context(closing(_FileLock(path, timeout=0)))

    def _portable_rows_locked(
        self, ids: tuple[str, ...]
    ) -> tuple[SuppressionSnapshot, _PortableExportRows]:
        with (
            closing(open_library(self.data_dir)) as library,
            _catalog_snapshot_locked(library) as suppression,
        ):
            if set(ids).intersection(suppression.suppressed_meeting_ids):
                raise NotFound("Selected meeting does not exist")
            rows = MemoryRepository(library).portable_export_rows(ids)
            if tuple(item.meeting.id for item in rows.meetings) != ids or any(
                item.meeting.status == MeetingStatus.DELETING for item in rows.meetings
            ):
                raise NotFound("Selected meeting does not exist")
            return suppression, rows

    @contextmanager
    def _portable_export_view(self, ids: tuple[str, ...]) -> Iterator[_PortableExportView]:
        if not ids:
            raise InvalidInput("Select at least one meeting for export")
        for identity in ids:
            self._identity(identity)
        if len(set(ids)) != len(ids):
            raise InvalidInput("Select each meeting once for export")
        ids = tuple(sorted(ids))
        owners = ExitStack()
        view: _PortableExportView | None = None
        try:
            with closing(_catalog_publication_lock(self.data_dir)):
                suppression, rows = self._portable_rows_locked(ids)
                if any(
                    item.meeting.status in {MeetingStatus.RECORDING, MeetingStatus.PROCESSING}
                    for item in rows.meetings
                ):
                    raise Busy("Capture or processing must finish before export")
                for identity in ids:
                    self._retain_native_owners(identity, owners)
                    self._retain_catalog_owner(identity, owners)
            view = _PortableExportView(self, ids, suppression, rows)
            yield view
        finally:
            if view is not None:
                view._active = False
            try:
                owners.close()
            except OSError:
                if view is None or not view._published:
                    raise
                warning = str(_ExportCleanupPending())
                if warning not in view.warnings:
                    view.warnings += (warning,)

    def _register_export(self, ids: tuple[str, ...], temporary: WindowsExportSibling) -> Path:
        match = _EXPORT_SIBLING_NAME.fullmatch(temporary.path.name)
        if match is None:
            raise InvalidInput("Export sibling has an invalid operation role")
        operation_id = str(UUID(hex=match[1]))
        entry = _ExportRegistration(
            operation_id, ids, temporary.path, temporary.parent_identity, temporary.file_identity
        )
        try:
            with closing(_catalog_publication_lock(self.data_dir)):
                directory = managed_path(self.data_dir, ".exports", directory=True)
                directory.mkdir(exist_ok=True)
                path = session_artifact_path(directory, f"{operation_id}.json")
                if path.exists():
                    raise Conflict("Export registration already exists")
                write_durable_json(path, entry.document())
                return path
        except (OSError, ValueError) as exc:
            raise LibraryError("Export temporary registration could not be committed") from exc

    def _retire_export(self, registration: Path) -> None:
        try:
            with closing(_catalog_publication_lock(self.data_dir)):
                session_artifact_path(registration.parent, registration.name).unlink()
        except (OSError, ValueError, LibraryError) as exc:
            raise _ExportCleanupPending() from exc

    def _recover_exports_locked(self, *, deleting: str | None = None) -> bool:
        """Recover a stale sibling only after claiming its selected meeting leases."""
        pending = False
        try:
            registrations = tuple(_export_registrations(self.data_dir))
            for path, entry in registrations:
                if deleting is not None and deleting not in entry.meeting_ids:
                    continue
                try:
                    with ExitStack() as owners:
                        for identity in entry.meeting_ids:
                            if identity != deleting:
                                self._retain_native_owners(identity, owners)
                                self._retain_catalog_owner(identity, owners)
                        recover_export_sibling(
                            entry.sibling,
                            entry.parent_identity,
                            entry.file_identity,
                            excluded_root=self.data_dir,
                        )
                        session_artifact_path(path.parent, path.name).unlink()
                except Busy:
                    pending |= deleting is not None
                except (OSError, ValueError, LibraryError):
                    pending = True
        except (OSError, ValueError, LibraryError):
            pending = True
        if pending and deleting is not None:
            raise _ExportCleanupPending()
        return pending

    @contextmanager
    def _portable_import_stage(
        self, inventory: _PortableImportInventory
    ) -> Iterator[_PortableImportStage]:
        operation_id = str(uuid4())
        registration = _ImportRegistration(
            operation_id,
            tuple(sorted(inventory.meeting_ids)),
            tuple((file.path, file.size, file.sha256) for file in inventory.files),
        )
        _ImportRegistration.read(registration.document(), operation_id)
        relative = f".staging/import-{operation_id}"
        stage_root = managed_path(portable_stage_io_path(self.data_dir), relative, directory=True)
        stage = _PortableImportStage(
            operation_id, self.data_dir / relative / "payload", registration
        )
        owned = False
        with ExitStack() as ownership:
            try:
                with closing(_catalog_publication_lock(self.data_dir)):
                    stage_root.parent.mkdir(exist_ok=True)
                    stage_root.mkdir()
                    owned = True
                    ownership.enter_context(
                        closing(_FileLock(stage_root / _IMPORT_LEASE, timeout=0))
                    )
                    _write_import_intent(stage_root, _IMPORT_REGISTRATION, registration.document())
                    (stage_root / "payload").mkdir()
                yield stage
            finally:
                if owned and not stage.retain_for_recovery:
                    try:
                        with closing(_catalog_publication_lock(self.data_dir)):
                            self._remove_import_stage_locked(stage_root, ownership)
                    except (OSError, ValueError, LibraryError) as exc:
                        raise _ImportCleanupPending() from exc

    @staticmethod
    def _remove_import_stage_locked(stage_root: Path, ownership: ExitStack) -> None:
        stage_root = portable_stage_io_path(stage_root)
        if not session_artifact_path(stage_root, _IMPORT_REGISTRATION).exists():
            # No producer may write payloads before a durable registration.
            if any(
                entry.name not in {_IMPORT_LEASE, f"{_IMPORT_REGISTRATION}.stage"}
                for entry in stage_root.iterdir()
            ):
                raise ValueError("Unregistered portable import staging material retained")
            session_artifact_path(stage_root, f"{_IMPORT_REGISTRATION}.stage").unlink(
                missing_ok=True
            )
        else:
            _purge_import_stage_payload(stage_root, _read_import_registration(stage_root))
        ownership.close()
        session_artifact_path(stage_root, _IMPORT_LEASE).unlink(missing_ok=True)
        session_artifact_path(stage_root, _IMPORT_WORK).unlink(missing_ok=True)
        session_artifact_path(stage_root, _IMPORT_REGISTRATION).unlink(missing_ok=True)
        stage_root.rmdir()

    def _recover_import_stage_locked(
        self,
        repo: MemoryRepository,
        stage_root: Path,
        *,
        deleting: str | None = None,
    ) -> None:
        relative = stage_root.relative_to(self.data_dir).as_posix()
        stage_root = managed_path(portable_stage_io_path(self.data_dir), relative, directory=True)
        pending_deletes = {
            record.meeting_id
            for record in _catalog_records(repo)
            if record.phase == "pending" and record.meeting_id != deleting
        }
        if (deleting is not None or pending_deletes) and session_artifact_path(
            stage_root, _IMPORT_REGISTRATION
        ).exists():
            registration = _read_import_registration(stage_root)
            if deleting is not None and deleting not in registration.meeting_ids:
                return
            selected_pending = pending_deletes.intersection(registration.meeting_ids)
            if selected_pending:
                plans = _read_import_work(stage_root, registration)
                for identity in selected_pending.intersection(plans):
                    root = managed_path(self.data_dir, f"meetings/{identity}", directory=True)
                    if not session_artifact_path(root, PORTABLE_GROUP_NAME).exists():
                        continue
                    group = read_portable_group(root, identity)
                    if group.operation_id != plans[identity].group.operation_id:
                        continue
                    if group.document() != plans[identity].group.document():
                        raise ValueError("Pending portable deletion disagrees with registered work")
                    if deleting is not None:
                        raise Busy("Shared portable staging is required by a pending deletion")
                    return
            if deleting is not None:
                pending = repo.library.execute(
                    "SELECT meeting_id, staging_path FROM pending_operations "
                    "WHERE kind='publish_portable' AND phase='prepared'"
                )
                if any(
                    row["meeting_id"] != deleting
                    and isinstance(row["staging_path"], str)
                    and row["staging_path"].startswith(relative + "/")
                    for row in pending
                ):
                    raise Busy("Shared portable staging is awaiting another meeting's recovery")
        with ExitStack() as owners:
            try:
                owners.enter_context(
                    closing(_FileLock(stage_root / _IMPORT_LEASE, timeout=0, create=False))
                )
            except FileNotFoundError:
                pass  # A failed registration or partially retired stage has no owner.
            except Busy:
                if deleting is not None:
                    raise
                return
            self._remove_import_stage_locked(stage_root, owners)

    def _portable_import_outcome(
        self,
        repo: MemoryRepository,
        bundle: _PortableImportedMeeting,
        batch: _ValidatedPortableImport,
    ) -> _PortableMeetingOutcome | None:
        from here.memory.portable_import import _PortableMeetingOutcome, matches_portable_meeting

        identity = bundle.detail.meeting.id
        destination = managed_path(self.data_dir, f"meetings/{identity}", directory=True)
        origin = normalized_origin(destination)
        records = _catalog_records(repo)
        suppressed = _suppressed_meeting_ids(repo.library, records)
        if identity in suppressed or any(
            (origin, batch.manifest_sha256) in record.origins for record in records
        ):
            return _PortableMeetingOutcome(identity, "rejected", "suppressed")
        self._require_portable_commit_sealed(repo, identity)
        existing = repo.portable_export_rows((identity,)).meetings
        if existing:
            if not matches_portable_meeting(bundle, existing[0], include_audio=batch.include_audio):
                return _PortableMeetingOutcome(identity, "conflict", "different_meeting")
            if batch.include_audio and any(
                not self._portable_existing_audio(asset) for asset in existing[0].audio_assets
            ):
                return _PortableMeetingOutcome(identity, "conflict", "audio_unavailable")
            return _PortableMeetingOutcome(identity, "unchanged")
        if destination.exists() or repo.get_ingest_source(origin, batch.manifest_sha256):
            return _PortableMeetingOutcome(identity, "conflict", "destination_exists")
        detail = bundle.detail
        for participant in detail.participants:
            present = repo.get_participant(participant.id)
            if present is not None and present != participant:
                return _PortableMeetingOutcome(identity, "conflict", "participant_exists")
        for table, ids in (
            ("transcript_revisions", (item.id for item in detail.revisions)),
            ("segments", (item.id for item in detail.segments)),
            ("audio_assets", (item.id for item in detail.audio_assets)),
            ("lifecycle_events", (item.id for item in bundle.lifecycle_events)),
        ):
            if any(
                repo.library.execute(f"SELECT 1 FROM {table} WHERE id=?", (key,)).fetchone()
                for key in ids
            ):
                return _PortableMeetingOutcome(identity, "conflict", "catalog_identity_exists")
        return None

    def _portable_existing_audio(self, asset: AudioAsset) -> bool:
        try:
            path = managed_path(self.data_dir, asset.relative_path)
            if (
                asset.status != "available"
                or path.parent.parent != self.data_dir / "meetings" / asset.meeting_id / "revisions"
                or not _is_uuid(path.parent.name)
                or path.suffix.lower() != ".wav"
            ):
                return False
            validate_artifact_roles(path.parent, (path.name,))
            return path.stat().st_size == asset.size_bytes and file_hash(path) == asset.sha256
        except (OSError, ValueError):
            return False

    def _portable_receipt_for_asset(self, path: Path, identity: str) -> PortableReceipt | None:
        root = self.data_dir / "meetings" / identity
        registration = session_artifact_path(root, PORTABLE_GROUP_NAME)
        if not registration.exists():
            return None
        group = read_portable_group(root, identity)
        if path.parent.name not in group.revision_ids:
            return None
        return read_portable_receipt(path.parent, group)

    def _prepare_portable_import_work(
        self, batch: _ValidatedPortableImport, *, cancelled: Callable[[], bool]
    ) -> dict[str, _ImportMeetingPlan]:
        from here.memory.portability import ImportCancelled

        stage = batch.stage
        payload = portable_stage_io_path(stage.directory)
        stage_root = payload.parent
        plans: dict[str, _ImportMeetingPlan] = {}
        for bundle in batch.meetings:
            meeting = bundle.detail.meeting
            revisions = tuple(item.id for item in bundle.detail.revisions)
            group = PortableGroup(str(uuid4()), meeting.id, revisions)
            hashes: dict[str, dict[str, str]] = {revision: {} for revision in revisions}
            for asset, _ in bundle.audio_payloads:
                path = Path(asset.relative_path)
                hashes[path.parent.name][path.name] = asset.sha256
            plan = _ImportMeetingPlan(group, batch.manifest_sha256, hashes)
            plans[meeting.id] = _ImportMeetingPlan.read(plan.document())
        with closing(_catalog_publication_lock(self.data_dir)):
            if _read_import_registration(stage_root) != stage._registration:
                raise ValueError("Portable import scope changed before publication preparation")
            _write_import_intent(
                stage_root,
                _IMPORT_WORK,
                {
                    "schema_version": 1,
                    "operation_id": stage.operation_id,
                    "plans": [plan.document() for plan in plans.values()],
                },
            )
            (stage_root / "work").mkdir()
        for bundle in batch.meetings:
            if cancelled():
                raise ImportCancelled("Portable import cancelled")
            identity = bundle.detail.meeting.id
            root = stage_root / "work" / identity
            root.mkdir()
            plan = plans[identity]
            for revision_id in plan.group.revision_ids:
                (root / "revisions" / revision_id).mkdir(parents=True)
            _write_import_intent(root, PORTABLE_GROUP_NAME, plan.group.document())
            for asset, relative in bundle.audio_payloads:
                path = Path(asset.relative_path)
                target = session_artifact_path(root / "revisions" / path.parent.name, path.name)
                source = managed_path(payload, relative)
                with source.open("rb") as incoming, target.open("xb") as output:
                    remaining = asset.size_bytes
                    digest = hashlib.sha256()
                    while True:
                        if cancelled():
                            raise ImportCancelled("Portable import cancelled")
                        block = incoming.read(min(1024 * 1024, remaining + 1))
                        if not block:
                            break
                        remaining -= len(block)
                        if remaining < 0:
                            raise IntegrityFailure(
                                "Prepared import audio exceeded its registered size"
                            )
                        digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if remaining or digest.hexdigest() != asset.sha256:
                    raise IntegrityFailure(
                        "Prepared import audio does not match its registered proof"
                    )
        return plans

    def _publish_portable_import(
        self, batch: _ValidatedPortableImport, *, cancelled: Callable[[], bool]
    ) -> ImportReport:
        from here.memory.portability import ImportCancelled
        from here.memory.portable_import import ImportReport, _PortableMeetingOutcome

        if cancelled():
            raise ImportCancelled("Portable import cancelled")
        plans = self._prepare_portable_import_work(batch, cancelled=cancelled)
        with self._operation() as repo, ExitStack() as owners:
            outcomes: dict[str, _PortableMeetingOutcome] = {}
            pending_recovery_ids: list[str] = []
            for identity in sorted(plans):
                self._retain_native_owners(identity, owners)
                self._retain_catalog_owner(identity, owners)
            with _catalog_snapshot_locked(repo.library):
                for bundle in batch.meetings:
                    outcome = self._portable_import_outcome(repo, bundle, batch)
                    if outcome is not None:
                        outcomes[outcome.meeting_id] = outcome
            blocked = any(
                outcome.status in {"conflict", "rejected"} for outcome in outcomes.values()
            )
            for bundle in batch.meetings:
                identity = bundle.detail.meeting.id
                if identity in outcomes:
                    continue
                if blocked or cancelled():
                    if not blocked and not any(
                        item.status == "inserted" for item in outcomes.values()
                    ):
                        raise ImportCancelled("Portable import cancelled")
                    outcomes[identity] = _PortableMeetingOutcome(
                        identity, "rejected", "batch_stopped"
                    )
                    continue
                try:
                    warning = self._publish_portable_meeting(
                        repo, batch.stage, bundle, plans[identity]
                    )
                except _ImportCommitUncertain:
                    pending_recovery_ids.append(identity)
                    blocked = True
                except (OSError, ValueError, LibraryError, sqlite3.Error):
                    outcomes[identity] = _PortableMeetingOutcome(identity, "rejected", "local_io")
                    blocked = True
                else:
                    outcomes[identity] = _PortableMeetingOutcome(identity, "inserted", warning)
                    blocked = bool(warning)
            return ImportReport(
                tuple(outcomes[identity] for identity in plans if identity in outcomes),
                pending_recovery_ids=tuple(pending_recovery_ids),
            )

    def _publish_portable_meeting(
        self,
        repo: MemoryRepository,
        stage: _PortableImportStage,
        bundle: _PortableImportedMeeting,
        plan: _ImportMeetingPlan,
    ) -> str:
        identity = plan.group.meeting_id
        source_rel = (
            (stage.directory.parent / "work" / identity).relative_to(self.data_dir).as_posix()
        )
        destination_rel = f"meetings/{identity}"
        source = managed_path(portable_stage_io_path(self.data_dir), source_rel, directory=True)
        destination = managed_path(self.data_dir, destination_rel, directory=True)
        destination.parent.mkdir(exist_ok=True)
        operation = PendingOperation(
            plan.group.operation_id,
            "publish_portable",
            "prepared",
            identity,
            source_rel,
            destination_rel,
            json.dumps(plan.document()),
        )
        origin = normalized_origin(destination)
        detail = bundle.detail
        local_bundle = replace(
            bundle,
            detail=replace(
                detail, revisions=tuple(replace(item, origin=origin) for item in detail.revisions)
            ),
        )
        provenance = IngestSource(str(uuid4()), origin, plan.manifest_sha256, identity)
        with repo.transaction():
            repo.insert_pending_operation(operation)
            # Keep the recovery evidence even if the prepared commit has no acknowledgement.
            stage.retain_for_recovery = True
        source.rename(destination)
        entity_body_complete = False
        try:
            with repo.transaction():
                repo.insert_portable_meeting(local_bundle, provenance)
                self._index_active_revision(repo, detail.meeting)
                entity_body_complete = True
        except (OSError, ValueError, LibraryError, sqlite3.Error):
            if entity_body_complete:
                raise _ImportCommitUncertain(
                    "Portable meeting commit could not be confirmed"
                ) from None
            raise
        try:
            registered = replace(plan, stage_root=portable_stage_io_path(stage.directory.parent))
            self._finish_portable_receipts(repo, destination, registered)
            with repo.transaction():
                repo.transition_pending_operation(operation.id, "complete")
        except (OSError, ValueError, LibraryError, sqlite3.Error):
            return "local_recovery_pending"
        stage.retain_for_recovery = False
        return ""

    def _finish_portable_receipts(
        self, repo: MemoryRepository, destination: Path, plan: _ImportMeetingPlan
    ) -> None:
        identity = plan.group.meeting_id
        group = read_portable_group(destination, identity)
        if group.document() != plan.group.document() or group.group_sha256 is None:
            raise ValueError("Portable publication group disagrees with committed journal")
        meeting = repo.get_meeting(identity)
        if meeting is None:
            raise ValueError("Portable publication requires a committed meeting")
        revisions = tuple(
            row[0]
            for row in repo.library.execute(
                "SELECT id FROM transcript_revisions WHERE meeting_id=? ORDER BY revision_number",
                (identity,),
            )
        )
        states = destination / "states"
        completion_path = session_artifact_path(
            states, portable_completion_name(group.operation_id)
        )
        receipts = []
        if completion_path.exists():
            completion = read_portable_completion(states, group)
            for revision_id in group.revision_ids:
                receipt = read_portable_receipt(destination / "revisions" / revision_id, group)
                receipts.append(receipt)
            validate_portable_group_members(group, completion, receipts)
            return
        stage_root = plan.stage_root
        if stage_root is None:
            raise ValueError("Portable publication requires registered evidence staging")
        if revisions != group.revision_ids:
            raise ValueError("Portable publication does not match committed revision history")
        origin = normalized_origin(destination)
        provenance = repo.get_ingest_source(origin, plan.manifest_sha256)
        if (
            provenance is None
            or provenance.meeting_id != identity
            or provenance.revision_id is not None
        ):
            raise ValueError("Portable publication does not match committed source evidence")
        for index, revision_id in enumerate(group.revision_ids):
            revision = repo.get_revision(revision_id)
            if revision is None or revision.origin != origin:
                raise ValueError("Portable publication revision has no committed origin")
            revision_dir = destination / "revisions" / revision_id
            for name, digest in plan.hashes[revision_id].items():
                if file_hash(session_artifact_path(revision_dir, name)) != digest:
                    raise ValueError("Portable publication WAV changed before receipt")
            receipt_path = session_artifact_path(revision_dir, PORTABLE_RECEIPT_NAME)
            snapshot = self._revision_snapshot(repo, meeting, revision, plan.hashes[revision_id])
            snapshot.update(
                schema_version=4,
                operation_id=group.operation_id,
                meeting_id=identity,
                revision_id=revision_id,
                ingest_sources=[asdict(provenance)] if index == 0 else [],
            )
            if not receipt_path.exists():
                write_durable_json(
                    receipt_path,
                    snapshot,
                    staged=stage_root / plan.evidence_stage_name(revision_id),
                )
            receipt = read_portable_receipt(revision_dir, group)
            if receipt.snapshot != snapshot:
                raise ValueError("Portable publication receipt disagrees with committed evidence")
            receipts.append(receipt)
        state_path = self._write_state(repo, identity, portable_plan=plan)
        if state_path is None:
            raise ValueError("Portable publication is missing committed lifecycle state")
        completion = PortableCompletion(
            group.operation_id,
            identity,
            tuple(
                PortableCompletionMember(
                    receipt.revision_id, receipt.receipt_sha256, receipt.hashes
                )
                for receipt in receipts
            ),
            group.group_sha256,
            state_path.name,
            file_hash(state_path),
            read_meeting_snapshot(state_path.parent, state_path.name, identity),
        )
        validate_portable_group_members(group, completion, receipts)
        write_durable_json(
            completion_path,
            completion.document(),
            staged=stage_root / plan.evidence_stage_name("completion"),
        )

    def _portable_operation_plan(self, operation: PendingOperation) -> _ImportMeetingPlan:
        plan = _ImportMeetingPlan.read(json.loads(operation.payload_json))
        identity = plan.group.meeting_id
        source = (operation.staging_path or "").split("/")
        match = _IMPORT_STAGE_NAME.fullmatch(source[1]) if len(source) == 4 else None
        if (
            operation.kind != "publish_portable"
            or operation.id != plan.group.operation_id
            or operation.meeting_id != identity
            or operation.destination_path != f"meetings/{identity}"
            or match is None
            or not _is_uuid(match[1])
            or source[0] != ".staging"
            or source[2:] != ["work", identity]
        ):
            raise ValueError("Portable publication journal disagrees with managed roles")
        return plan

    def _reconcile_portable_meeting(
        self, repo: MemoryRepository, operation: PendingOperation, *, owners_held: bool
    ) -> None:
        plan = self._portable_operation_plan(operation)
        identity = plan.group.meeting_id
        destination = managed_path(self.data_dir, operation.destination_path or "", directory=True)
        source = managed_path(
            portable_stage_io_path(self.data_dir), operation.staging_path or "", directory=True
        )
        stage_root = source.parent.parent
        with ExitStack() as owners:
            if not owners_held:
                self._delete_owner(repo, identity, owners, accepted=True)
            if stage_root.exists():
                owners.enter_context(
                    closing(_FileLock(stage_root / _IMPORT_LEASE, timeout=0, create=False))
                )
                registration = _read_import_registration(stage_root)
                work = _read_import_work(stage_root, registration)
                if work.get(identity) != plan:
                    raise ValueError("Portable publication disagrees with registered work")
                plan = replace(plan, stage_root=stage_root)
            if repo.get_meeting(identity) is not None:
                self._finish_portable_receipts(repo, destination, plan)
                phase = "complete"
            else:
                if source.exists() and destination.exists():
                    raise ValueError("Portable publication has ambiguous owned copies")
                if destination.exists():
                    if plan.stage_root is None or not source.parent.is_dir():
                        raise ValueError("Portable rollback requires its registered work slot")
                    _import_tree_entries(destination, plan, published=True)
                    # Return the proven tree to its private cleanup scope before
                    # discarding the SQL journal's prepared phase.
                    portable_stage_io_path(destination).rename(source)
                phase = "reverted"
            with repo.transaction():
                repo.transition_pending_operation(operation.id, phase)

    @staticmethod
    def _require_portable_commit_sealed(repo: MemoryRepository, identity: str) -> None:
        pending = repo.library.execute(
            "SELECT 1 FROM pending_operations WHERE meeting_id=? "
            "AND kind='publish_portable' AND phase='prepared' LIMIT 1",
            (identity,),
        ).fetchone()
        if pending is not None:
            raise Busy("Portable meeting is awaiting local recovery before editing")

    @contextmanager
    def _operation(
        self, *, recovery_warnings: list[object] | None = None
    ) -> Iterator[MemoryRepository]:
        validate_session_directory(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        lock = _FileLock(self.data_dir / ".memory.lock")
        try:
            if recovery_warnings is not None and self._recover_exports_locked():
                recovery_warnings.append(str(_ExportCleanupPending()))
            with self._operation_locked() as repo:
                yield repo
        finally:
            lock.close()

    @contextmanager
    def _operation_locked(self) -> Iterator[MemoryRepository]:
        records, _ = _deletion_registry(self.data_dir)
        library = open_library(self.data_dir)
        try:
            migrate(library)
            yield _CatalogRepository(library, records)
        finally:
            library.close()

    def _fts(self, repo: MemoryRepository) -> bool:
        if not fts_available(repo.connection):
            return False
        repo.library.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5("
            "meeting_id UNINDEXED, revision_id UNINDEXED, segment_id UNINDEXED, text, title, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        return True

    def _index_active_revision(self, repo: MemoryRepository, meeting: Meeting) -> None:
        if not self._fts(repo) or meeting.active_revision_id is None:
            return
        active = repo.get_revision(meeting.active_revision_id)
        if active is not None and active.status == "complete":
            for segment in repo.get_segments(active.id):
                repo.library.execute(
                    "INSERT INTO segments_fts VALUES (?,?,?,?,?)",
                    (meeting.id, active.id, segment.id, segment.text, meeting.title),
                )

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
        portable_group = None
        group_path = session_artifact_path(root, PORTABLE_GROUP_NAME)
        if group_path.exists():
            portable_group = read_portable_group(root, identity)
            files[group_path.relative_to(self.data_dir).as_posix()] = file_hash(group_path)
        pending_plan = (
            self._pending_portable_cleanup_plan(repo, portable_group)
            if portable_group is not None
            else None
        )
        revisions = managed_path(self.data_dir, f"meetings/{identity}/revisions", directory=True)
        if revisions.exists():
            directories.add(revisions.relative_to(self.data_dir).as_posix())
            for directory in sorted(revisions.iterdir()):
                if not _is_uuid(directory.name):
                    continue
                validate_session_directory(directory)
                if portable_group is not None and directory.name in portable_group.revision_ids:
                    portable_path = session_artifact_path(directory, PORTABLE_RECEIPT_NAME)
                    if portable_path.exists():
                        proof = read_portable_receipt(directory, portable_group)
                        files[portable_path.relative_to(self.data_dir).as_posix()] = (
                            proof.receipt_sha256
                        )
                        hashes = proof.hashes
                        if (
                            pending_plan is not None
                            and hashes != pending_plan.hashes[directory.name]
                        ):
                            raise ValueError(
                                "Portable cleanup receipt disagrees with the work plan"
                            )
                    elif pending_plan is not None:
                        hashes = pending_plan.hashes[directory.name]
                    elif not any(directory.iterdir()):
                        # The still-present group proves this exact empty member
                        # after deletion removed its receipt but before rmdir.
                        hashes = {}
                    else:
                        raise ValueError("Portable revision lacks owned cleanup proof")
                    directories.add(directory.relative_to(self.data_dir).as_posix())
                    for name, digest in hashes.items():
                        path = session_artifact_path(directory, name)
                        if path.exists() and file_hash(path) != digest:
                            raise ValueError("Portable cleanup artifact hash mismatch")
                        files[path.relative_to(self.data_dir).as_posix()] = digest
                    continue
                registration = session_artifact_path(directory, ".publication.json")
                receipt = session_artifact_path(directory, "catalog.json")
                if not registration.exists():
                    # No suffix/UUID-only authority, even for a familiar filename.
                    continue
                plan = _delete_publication_plan(directory)
                _validate_delete_roles(directory, plan["hashes"])
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
                        _validate_delete_roles(path.parent, (path.name,))
                    for ordinal, values in enumerate(snapshot["revision_audio_assets"]):
                        link = RevisionAudioAsset(**values)
                        if (
                            link.meeting_id != identity
                            or link.revision_id != directory.name
                            or link.asset_id not in assets
                            or link.ordinal != ordinal
                        ):
                            raise ValueError("Cleanup audio relationship mismatch")
                        _validate_delete_roles(directory, (link.source_file,))
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
                if portable_group is not None and path.name == portable_completion_name(
                    portable_group.operation_id
                ):
                    completion = read_portable_completion(states, portable_group)
                    receipts = tuple(
                        read_portable_receipt(revisions / revision, portable_group)
                        for revision in portable_group.revision_ids
                    )
                    validate_portable_group_members(portable_group, completion, receipts)
                    files[path.relative_to(self.data_dir).as_posix()] = file_hash(path)
                    continue
                read_meeting_snapshot(states, path.name, identity)
                files[path.relative_to(self.data_dir).as_posix()] = file_hash(path)
        return {"schema_version": 1, "files": files, "directories": sorted(directories)}

    def _pending_portable_cleanup_plan(
        self, repo: MemoryRepository, group: PortableGroup
    ) -> _ImportMeetingPlan | None:
        operation = repo.get_pending_operation(group.operation_id)
        if operation is not None:
            plan = self._portable_operation_plan(operation)
            if plan.group.document() != group.document():
                raise ValueError("Portable delete group disagrees with the catalog journal")
            return plan if operation.phase == "prepared" else None
        staging = managed_path(portable_stage_io_path(self.data_dir), ".staging", directory=True)
        if not staging.exists():
            return None
        found = None
        for entry in staging.iterdir():
            if not _IMPORT_STAGE_NAME.fullmatch(entry.name):
                continue
            registration_path = session_artifact_path(entry, _IMPORT_REGISTRATION)
            if not registration_path.exists():
                continue
            registration = _read_import_registration(entry)
            if group.meeting_id not in registration.meeting_ids:
                continue
            candidate = _read_import_work(entry, registration).get(group.meeting_id)
            if candidate is None or candidate.group.operation_id != group.operation_id:
                continue
            if found is not None or candidate.group.document() != group.document():
                raise ValueError("Portable delete has ambiguous local group ownership")
            found = candidate
        return found

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
            if parts == (PORTABLE_GROUP_NAME,):
                if path.exists():
                    read_portable_group(root, record.meeting_id)
            elif len(parts) == 3 and parts[0] == "revisions" and _is_uuid(parts[1]):
                if path.name == PORTABLE_RECEIPT_NAME and path.exists():
                    group = read_portable_group(root, record.meeting_id)
                    read_portable_receipt(path.parent, group)
                elif path.name not in {"catalog.json", ".publication.json", PORTABLE_RECEIPT_NAME}:
                    _validate_delete_roles(path.parent, (path.name,))
            elif len(parts) == 2 and parts[0] == "states":
                match = re.fullmatch(r"portable-group-([0-9a-f-]{36})\.json", path.name)
                if match is not None and _is_uuid(match[1]):
                    if path.exists():
                        group = read_portable_group(root, record.meeting_id)
                        if group.operation_id != match[1]:
                            raise ValueError("Portable completion does not match the delete group")
                elif path.exists():
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
        self._retain_native_owners(identity, leases)
        self._retain_catalog_owner(identity, leases)
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
        root_relative = f"meetings/{record.meeting_id}"
        group_relative = f"{root_relative}/{PORTABLE_GROUP_NAME}"
        group_owned = group_relative in inventory["files"]
        completion_prefix = f"{root_relative}/states/portable-group-"
        # Drop the completion seal before removing the lifecycle state it hashes.
        files = sorted(
            inventory["files"],
            key=lambda name: (
                (-1, False, name)
                if group_owned and name.startswith(completion_prefix)
                else _delete_file_order(name)
            ),
        )
        for relative in files:
            if relative != group_relative:
                managed_path(self.data_dir, relative).unlink(missing_ok=True)
        for relative in sorted(inventory["directories"], key=lambda name: (-name.count("/"), name)):
            path = managed_path(self.data_dir, relative, directory=True)
            if path.exists():
                if group_owned and relative == root_relative:
                    if any(entry.name != PORTABLE_GROUP_NAME for entry in path.iterdir()):
                        raise ValueError("Unregistered portable meeting material retained")
                    managed_path(self.data_dir, group_relative).unlink(missing_ok=True)
                path.rmdir()  # Never recursively remove unregistered children.

    def _delete_stage_journal(self, row: sqlite3.Row, relative: str) -> dict[str, Any]:
        operation = PendingOperation(**dict(row))
        if (
            operation.kind != "publish_revision"
            or operation.phase not in {"prepared", "complete", "reverted"}
            or not _is_uuid(operation.id)
            or not _is_uuid(operation.meeting_id)
            or relative != f".staging/{operation.id}"
            or operation.staging_path != relative
        ):
            raise ValueError("Inconsistent publication stage retained")
        payload = json.loads(operation.payload_json)
        if (
            not isinstance(payload, dict)
            or not _is_uuid(payload.get("revision_id"))
            or not isinstance(payload.get("hashes"), dict)
            or any(
                not isinstance(name, str)
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for name, digest in payload["hashes"].items()
            )
            or operation.destination_path
            != f"meetings/{operation.meeting_id}/revisions/{payload['revision_id']}"
        ):
            raise ValueError("Publication stage destination disagrees with identity")
        if "publication" in payload:
            proof = _Publication.read(payload["publication"])
            if proof.revision_id != payload["revision_id"]:
                raise ValueError("Publication stage provenance mismatch")
        directory = managed_path(self.data_dir, relative, directory=True)
        if directory.exists() and any(directory.iterdir()):
            plan = _delete_publication_plan(directory)
            if plan["operation_id"] != operation.id or payload["hashes"] != plan["hashes"]:
                raise ValueError("Publication stage disagrees with registered journal")
            _validate_delete_roles(directory, plan["hashes"])
        return payload

    def _recover_delete_staging(self, repo: MemoryRepository, record: _DeletionRecord) -> None:
        # Observe current authority, never infer a historical database-loss event.
        # An existing meeting row retains its normal SQL scope; verified complete
        # tombstones do not regress because unrelated private material appears.
        if record.phase != "pending" or repo.get_meeting(record.meeting_id) is not None:
            return
        try:
            staging = managed_path(self.data_dir, ".staging", directory=True)
            if not staging.exists():
                return
            rows = repo.library.execute("SELECT * FROM pending_operations").fetchall()
            for directory in sorted(staging.iterdir()):
                if _IMPORT_STAGE_NAME.fullmatch(directory.name):
                    continue  # Portable registrations are processed separately for every phase.
                if not _is_uuid(directory.name):
                    raise ValueError("Unidentified private staging retained")
                relative = directory.relative_to(self.data_dir).as_posix()
                references = _stage_references(self.data_dir, rows, relative, directory.name)
                if references:
                    if len(references) != 1:
                        raise ValueError("Ambiguous publication stage references retained")
                    # A validated SQL relation proves this stage's actual scope.
                    # Foreign stages stay untouched; own stages use R52 below.
                    self._delete_stage_journal(references[0], relative)
                    continue
                directory = managed_path(self.data_dir, relative, directory=True)
                if any(directory.iterdir()):
                    plan = _delete_publication_plan(directory)
                    if plan["operation_id"] != directory.name:
                        raise ValueError("Private staging plan identity mismatch")
                # R54 discards only genuinely unreferenced private copies, under
                # this same publication lock and the existing ALL-child guard.
                _remove_tree(self.data_dir, relative, order=_delete_file_order)
        except (OSError, ValueError, LibraryError, KeyError, TypeError) as exc:
            raise _DeleteScopeUnavailable() from exc

    def _purge_delete_stages(self, repo: MemoryRepository, identity: str) -> None:
        """Consume only actual meeting journals while their cleanup authority survives."""
        rows = repo.library.execute("SELECT * FROM pending_operations").fetchall()
        for row in rows:
            if row["meeting_id"] != identity or row["staging_path"] is None:
                continue
            if row["kind"] == "publish_portable":
                self._portable_operation_plan(PendingOperation(**dict(row)))
                continue  # The portable stage registration above owns its work tree.
            relative = row["staging_path"]
            references = _stage_references(self.data_dir, rows, relative, row["id"])
            if len(references) != 1:
                raise ValueError("Ambiguous publication stage references retained")
            payload = self._delete_stage_journal(row, relative)
            directory = managed_path(self.data_dir, relative, directory=True)
            if not directory.exists():
                continue
            # Existing A all-child preflight; retain W1 proof ordering on partial
            # unlink and keep the SQL journal until the final core purge succeeds.
            _remove_tree(self.data_dir, relative, set(payload["hashes"]), order=_delete_file_order)

    def _finish_delete(
        self, repo: MemoryRepository, record: _DeletionRecord, operation: PendingOperation
    ) -> DeletionResult:
        warnings = self._prepare_delete(repo, record, operation)
        self._recover_exports_locked(deleting=record.meeting_id)
        self._recover_delete_staging(repo, record)
        self._purge_delete_stages(repo, record.meeting_id)
        self._purge_delete_files(operation, record)
        staging = managed_path(self.data_dir, ".staging", directory=True)
        if staging.exists():
            try:
                for entry in staging.iterdir():
                    if _IMPORT_STAGE_NAME.fullmatch(entry.name):
                        self._recover_import_stage_locked(repo, entry, deleting=record.meeting_id)
            except (OSError, ValueError, LibraryError) as exc:
                raise _ImportCleanupPending() from exc
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
        warnings: tuple[str, ...] = (
            "Deletion pending local recovery" if accepted else "Deletion could not start",
        )
        if isinstance(
            error, (_DeleteScopeUnavailable, _ExportCleanupPending, _ImportCleanupPending)
        ):
            warnings += (str(error),)
        return DeletionResult(
            "pending" if accepted else "rejected",
            identity,
            reason,
            warnings,
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
        except LibraryError as exc:
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

    @contextmanager
    def recovery_read_view(
        self, directory: Path, declared_identity: str | None
    ) -> Iterator[str | None]:
        """Delegate to the same real module reader, preserving this explicit data root."""
        with recovery_read_view(self.data_dir, directory, declared_identity) as identity:
            yield identity

    @contextmanager
    def recovery_publication(
        self, directory: Path, declared_identity: str | None, *, expected_identity: str | None
    ) -> Iterator[None]:
        """Revalidate the handoff and keep the publication lock through marker cleanup."""
        if self._recovery_publication_context.get() is not None:
            raise Conflict("Recovery publication is already scoped")
        with ExitStack() as operation:
            try:
                directory = Path(directory).absolute()
                validate_session_directory(self.data_dir)
                self.data_dir.mkdir(parents=True, exist_ok=True)
                lock = _FileLock(managed_path(self.data_dir, ".memory.lock"))
                operation.callback(lock.close)
                repo = operation.enter_context(self._operation_locked())
            except sqlite3.Error as exc:
                raise storage_error(exc) from exc
            except UnsafeSessionPath:
                raise
            except (OSError, ValueError) as exc:
                raise LibraryError("Recovery publication catalog unavailable") from exc

            try:
                metadata = read_session_metadata(directory)
            except UnsafeSessionPath:
                raise
            except (OSError, ValueError) as exc:
                raise UnsafeSessionPath("Recovery metadata unavailable") from exc
            if metadata is None or metadata.meeting_id != declared_identity:
                raise UnsafeSessionPath("Recovery identity changed")

            try:
                pending = read_pending_intent(directory, declared_identity)
            except FileNotFoundError:
                pending = None
            except OSError as exc:
                raise LibraryError("Recovery publication marker unavailable") from exc

            intent = pending if metadata.status != "pending" else None
            try:
                self._validate_recovery_publication(
                    repo, directory, metadata, intent, expected_identity
                )
            except UnsafeSessionPath:
                raise
            except (OSError, ValueError, TypeError) as exc:
                raise LibraryError("Recovery publication authority unavailable") from exc
            scope = _RecoveryPublication(repo, directory, declared_identity, intent)
            token = self._recovery_publication_context.set(scope)
            try:
                yield
            finally:
                scope.active = False
                self._recovery_publication_context.reset(token)

    def _validate_recovery_publication(
        self,
        repo: MemoryRepository,
        directory: Path,
        metadata: SessionMetadata,
        intent: PendingIntent | None,
        expected_identity: str | None,
    ) -> None:
        records = _catalog_records(repo)
        suppressed = _suppressed_meeting_ids(repo.library, records)
        identity = metadata.meeting_id
        canonical = _is_uuid(identity)
        if identity in suppressed:
            raise NotFound("Meeting has been deleted")
        if not canonical:
            lineage = _legacy_retry_parent(repo.library, directory, suppressed)
            if lineage is not None:
                identity = lineage[0]
        if identity != expected_identity:
            raise OwnershipBusy("Recovery ownership changed; rediscover the session")
        if intent is None:
            return

        origin = normalized_origin(directory)
        deleted = {
            digest for record in records for path, digest in record.origins if path == origin
        }
        fingerprint = _sealed_origin_fingerprint(directory, metadata, intent)
        if fingerprint in deleted:
            raise NotFound("Origin has been deleted")

        committed_replay = self._is_committed_recovery_replay(
            repo, origin, intent, suppressed, expected_identity, fingerprint
        )
        if deleted and (fingerprint is None or not committed_replay):
            # A seal alone cannot prove that the current material differs.
            raise Conflict("Recovery origin cannot be verified; pending intent retained")

        audio_only = intent.source_kind == "legacy_audio_only"
        if audio_only and intent.revision_of is None and not committed_replay:
            if not self._has_audio_reservation(repo, origin, identity):
                raise Conflict("Audio-only publication requires its reserved identity")
        if intent.revision_of is not None:
            parent = repo.get_revision(intent.revision_of)
            if parent is None:
                raise Conflict("Revision conflict: missing parent")
            if parent.meeting_id in suppressed:
                raise NotFound("Meeting has been deleted")
            if parent.meeting_id != expected_identity:
                raise OwnershipBusy("Recovery retry target changed; rediscover the session")
            meeting = repo.get_meeting(parent.meeting_id)
            if meeting is None or meeting.status == MeetingStatus.DELETING:
                raise Conflict("Revision conflict: unavailable parent meeting")
            if audio_only and not (
                self._is_audio_only_parent(repo, parent)
                or self._has_audio_reservation(repo, origin, parent.meeting_id)
            ):
                raise Conflict("Revision conflict: missing audio-only parent provenance")
            if intent.intent == "status" and metadata.status not in {"failed", "cancelled"}:
                raise Conflict("Only unsuccessful retry outcomes use state-only publication")
            if meeting.active_revision_id != parent.id and not committed_replay:
                raise Conflict("Revision conflict: stale retry")

    @staticmethod
    def _has_audio_reservation(repo: MemoryRepository, origin: str, identity: str | None) -> bool:
        if not _is_uuid(identity):
            return False
        rows = repo.library.execute(
            "SELECT p.payload_json,i.origin_fingerprint FROM pending_operations p "
            "JOIN ingest_sources i ON i.meeting_id=p.meeting_id "
            "WHERE p.kind='reserve_audio_only' AND p.phase='complete' "
            "AND p.meeting_id=? AND i.origin_path=? AND i.revision_id IS NULL",
            (identity, origin),
        )
        for row in rows:
            try:
                payload = json.loads(row[0])
            except (ValueError, TypeError):
                continue
            if payload == {
                "source_kind": "legacy_audio_only",
                "origin_path": origin,
                "origin_fingerprint": row[1],
            }:
                return True
        return False

    @contextmanager
    def _publisher_operation(
        self,
        *,
        revision_of: str | None,
        directory: Path | None = None,
        declared_identity: str | None = None,
    ) -> Iterator[MemoryRepository]:
        scope = self._recovery_publication_context.get()
        if scope is None:
            with self._operation() as repo:
                yield repo
            return
        intent = scope.intent
        if (
            not scope.active
            or scope.repo.library.data_dir != self.data_dir
            or intent is None
            or intent.revision_of != revision_of
        ):
            raise Conflict("Recovery publisher intent changed; rediscover the session")
        if directory is None:
            if intent.intent != "status" or declared_identity != scope.declared_identity:
                raise Conflict("Recovery publisher target changed; rediscover the session")
        else:
            publication = self._publication_context.get()
            if (
                intent.intent != "ingest"
                or directory.absolute() != scope.directory
                or publication is None
                or publication[1] != intent.source_kind
                or (intent.publication_id is not None and publication[0] != intent.publication_id)
            ):
                raise Conflict("Recovery publisher target changed; rediscover the session")
        yield scope.repo

    def _publication_records(self, repo: MemoryRepository) -> tuple[_DeletionRecord, ...]:
        scope = self._recovery_publication_context.get()
        if scope is not None and scope.active and scope.repo is repo:
            return _catalog_records(repo)
        records, _ = _deletion_registry(self.data_dir)
        return records

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
                return _legacy_retry_parent(
                    library, session_dir, suppression.suppressed_meeting_ids
                )
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

    @staticmethod
    def _publication_candidates(
        repo: MemoryRepository, *, meeting_id: str | None = None
    ) -> Iterator[tuple[PendingOperation, dict[str, Any], dict[str, Any]]]:
        sql = "SELECT * FROM pending_operations WHERE kind='publish_revision' AND phase!='reverted'"
        parameters: tuple[object, ...] = ()
        if meeting_id is not None:
            sql += " AND meeting_id=?"
            parameters = (meeting_id,)
        for row in repo.library.execute(sql, parameters):
            operation = PendingOperation(**dict(row))
            try:
                payload = json.loads(operation.payload_json)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            value = payload.get("publication")
            if isinstance(value, dict):
                yield operation, payload, value

    @staticmethod
    def _publication_proof(
        repo: MemoryRepository, operation: PendingOperation, payload: dict[str, Any], value: object
    ) -> tuple[PendingOperation, _Publication, TranscriptRevision | None]:
        proof = _Publication.read(value)
        revision = repo.get_revision(proof.revision_id)
        if (
            revision is None
            or revision.meeting_id != operation.meeting_id
            or payload.get("revision_id") != revision.id
            or proof.origin_path != revision.origin
        ):
            revision = None
        return operation, proof, revision

    def _is_audio_only_parent(self, repo: MemoryRepository, parent: TranscriptRevision) -> bool:
        candidates = self._publication_candidates(repo, meeting_id=parent.meeting_id)
        for operation, payload, value in candidates:
            if (
                value.get("revision_id") != parent.id
                or value.get("source_kind") != "legacy_audio_only"
            ):
                continue
            try:
                _, _, revision = self._publication_proof(repo, operation, payload, value)
            except (ValueError, TypeError):
                continue
            if revision is not None:
                return True
        return False

    def _is_committed_recovery_replay(
        self,
        repo: MemoryRepository,
        origin: str,
        intent: PendingIntent,
        suppressed: tuple[str, ...],
        expected_identity: str | None,
        fingerprint: str | None,
    ) -> bool:
        if intent.intent != "ingest" or intent.publication_id is None:
            return False
        matches = [
            candidate
            for candidate in self._publication_candidates(repo)
            if candidate[2].get("publication_id") == intent.publication_id
        ]
        if not matches:
            return False
        if len(matches) != 1:
            raise Conflict("Publication conflict: ambiguous publication token")
        _, proof, revision = self._publication_proof(repo, *matches[0])
        if revision is None or (
            proof.origin_path != origin
            or proof.revision_of != intent.revision_of
            or proof.source_kind != intent.source_kind
        ):
            raise Conflict("Publication conflict: committed provenance not proven")
        if revision.meeting_id in suppressed:
            raise NotFound("Meeting has been deleted")
        if revision.meeting_id != expected_identity:
            raise OwnershipBusy("Recovery publication target changed; rediscover the session")
        meeting = repo.get_meeting(revision.meeting_id)
        if meeting is None or meeting.status == MeetingStatus.DELETING:
            raise Conflict("Publication conflict: committed meeting unavailable")
        if intent.hashes is not None and proof.origin_fingerprint != fingerprint:
            raise Conflict("Publication conflict: pending seal disagrees with committed provenance")
        # The publisher still validates the current evidence and full receipt.
        return True

    def _replay_publication(
        self,
        repo: MemoryRepository,
        source: SessionArtifacts,
        publication_id: str,
        revision_of: str | None,
        source_kind: str,
    ) -> IngestResult | None:
        for operation, payload, value in self._publication_candidates(repo):
            if value.get("publication_id") != publication_id:
                continue
            _, proof, revision = self._publication_proof(repo, operation, payload, value)
            if (
                revision is not None
                and proof.revision_of == revision_of
                and proof.origin_path == source.origin_path
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
        fresh_path: Path | None = None
        committed = False
        source_read = False
        warnings: tuple[object, ...] = ()
        active: TranscriptRevision | None = None
        # Opening the catalog is a typed storage boundary; CLI must distinguish a
        # broken database from a rejected source or publication conflict.
        with self._publisher_operation(directory=session_dir, revision_of=revision_of) as repo:
            scoped = self._recovery_publication_context.get() is not None
            records = self._publication_records(repo)
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
                self._require_portable_commit_sealed(repo, identity)
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
                if scoped and destination.exists():
                    raise Conflict("Recovery publication destination already exists")
                staging = managed_path(self.data_dir, staging_rel, directory=True)
                staging.mkdir(parents=True)
                fresh_path = staging
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
                if scoped and destination.exists():
                    raise Conflict("Recovery publication destination already exists")
                with repo.transaction():
                    repo.insert_pending_operation(operation)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if scoped and destination.exists():
                    raise Conflict("Recovery publication destination already exists")
                staging.rename(destination)
                fresh_path = destination
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
                        if scoped and fresh_path is not None:
                            self._revert_fresh_publication(repo, operation, fresh_path)
                        elif not scoped:
                            self._reconcile_operation(repo, operation)
                    except Exception:
                        pass
                elif staging_rel is not None and (not scoped or fresh_path is not None):
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

    def _revert_fresh_publication(
        self, repo: MemoryRepository, operation: PendingOperation, fresh_path: Path
    ) -> None:
        """Compensate only the directory created by this still-locked writer."""
        if (
            operation.kind != "publish_revision"
            or operation.phase != "prepared"
            or not _is_uuid(operation.id)
            or not _is_uuid(operation.meeting_id)
        ):
            raise Conflict("Fresh publication journal changed; operation retained")
        payload = json.loads(operation.payload_json)
        revision = payload["revision_id"]
        staging_rel = f".staging/{operation.id}"
        destination_rel = f"meetings/{operation.meeting_id}/revisions/{revision}"
        if (
            not _is_uuid(revision)
            or repo.get_revision(revision) is not None
            or operation.staging_path != staging_rel
            or operation.destination_path != destination_rel
        ):
            raise Conflict("Fresh publication authority changed; operation retained")

        recorded = repo.get_pending_operation(operation.id)
        if recorded is not None and recorded != operation:
            raise Conflict("Fresh publication journal changed; operation retained")
        before_rename = fresh_path == self.data_dir / staging_rel
        if before_rename:
            # Disable destination authority before validating or removing the
            # staging we own. A failed rename may have hit a foreign destination.
            if recorded is not None:
                with repo.transaction():
                    repo.transition_pending_operation(operation.id, "reverted")
            relative = staging_rel
        elif (
            recorded is not None
            and fresh_path == self.data_dir / destination_rel
            and not managed_path(self.data_dir, staging_rel, directory=True).exists()
        ):
            relative = destination_rel
        else:
            raise Conflict("Fresh publication paths changed; operation retained")
        fresh_path = managed_path(self.data_dir, relative, directory=True)
        plan = _delete_publication_plan(fresh_path)
        hashes = payload["hashes"]
        if plan["operation_id"] != operation.id or plan["hashes"] != hashes:
            raise Conflict("Fresh publication inventory changed; operation retained")
        validate_artifact_roles(fresh_path, hashes)
        if any(
            file_hash(session_artifact_path(fresh_path, name)) != digest
            for name, digest in hashes.items()
        ):
            raise Conflict("Fresh publication files changed; operation retained")
        _remove_tree(self.data_dir, relative, set(hashes), order=_delete_file_order)
        if not before_rename:
            with repo.transaction():
                repo.transition_pending_operation(operation.id, "reverted")

    def _reconcile_operation(
        self, repo: MemoryRepository, operation: PendingOperation, *, owners_held: bool = False
    ) -> None:
        records = self._publication_records(repo)
        if operation.kind == "delete":
            record = next(
                (item for item in records if item.meeting_id == operation.meeting_id), None
            )
            if record is None:
                raise ValueError("Delete journal has no durable suppression authority")
            self._validate_delete_operation(operation, record)
            with ExitStack() as owners:
                if not owners_held:
                    self._delete_owner(repo, record.meeting_id, owners, accepted=True)
                self._finish_delete(repo, record, operation)
            return
        if (
            operation.meeting_id
            in _suppression_snapshot(repo.library, records).suppressed_meeting_ids
        ):
            # A publication/state operation cannot recreate deleted evidence.
            return
        if operation.kind == "publish_portable":
            self._reconcile_portable_meeting(repo, operation, owners_held=owners_held)
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
        with ExitStack() as owners:
            if (not committed or staging.exists()) and not owners_held:
                self._delete_owner(repo, operation.meeting_id, owners, accepted=True)
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
                repo.transition_pending_operation(
                    operation.id, "complete" if committed else "reverted"
                )

    def _write_state(
        self,
        repo: MemoryRepository,
        identity: str,
        *,
        portable_plan: _ImportMeetingPlan | None = None,
    ) -> Path | None:
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
        if portable_plan is None:
            path = session_artifact_path(directory, f"{meeting.row_version}-{uuid4()}.json")
            write_durable_json(path, snapshot)
        else:
            stage_root = portable_plan.stage_root
            if stage_root is None:
                raise ValueError("Portable state requires registered evidence staging")
            path = session_artifact_path(
                directory, f"{meeting.row_version}-{portable_plan.group.operation_id}.json"
            )
            if path.exists():
                if read_meeting_snapshot(directory, path.name, identity) != snapshot:
                    raise ValueError("Portable lifecycle state disagrees with committed evidence")
            else:
                write_durable_json(
                    path, snapshot, staged=stage_root / portable_plan.evidence_stage_name("state")
                )
        return path

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
        snapshot = self._revision_snapshot(repo, meeting, revision, payload["hashes"])
        snapshot["schema_version"] = 3 if "publication" in payload else 2
        if "publication" in payload:
            proof = _Publication.read(payload["publication"])
            if proof.revision_id != revision.id or proof.origin_path != revision.origin:
                raise ValueError("Publication receipt provenance mismatch")
            snapshot["publication"] = asdict(proof)
            snapshot["operation_id"] = operation.id
        write_durable_json(receipt, snapshot)

    @staticmethod
    def _revision_snapshot(
        repo: MemoryRepository,
        meeting: Meeting,
        revision: TranscriptRevision,
        hashes: dict[str, str],
    ) -> dict[str, Any]:
        return {
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
            "hashes": hashes,
        }

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
                states = managed_path(self.data_dir, f"meetings/{identity}/states", directory=True)
                group = None
                completion = None
                portable_receipts: dict[str, PortableReceipt] = {}
                if session_artifact_path(directory, PORTABLE_GROUP_NAME).exists():
                    group = read_portable_group(directory, identity)
                    completion = read_portable_completion(states, group)
                    portable_receipts = {
                        key: read_portable_receipt(revisions_path / key, group)
                        for key in group.revision_ids
                    }
                    validate_portable_group_members(group, completion, portable_receipts.values())
                snapshots = []
                for revision_dir in revisions_path.iterdir() if revisions_path.exists() else ():
                    revision_id = str(UUID(revision_dir.name))
                    validate_session_directory(revision_dir)
                    receipt = session_artifact_path(revision_dir, "catalog.json")
                    portable_path = session_artifact_path(revision_dir, PORTABLE_RECEIPT_NAME)
                    portable_receipt = portable_receipts.get(revision_id)
                    if portable_receipt is not None:
                        if receipt.exists():
                            raise ValueError("Portable revision has mixed receipt roles")
                        snapshot = portable_receipt.snapshot
                    elif portable_path.exists():
                        raise ValueError("Portable receipt lacks its complete local group")
                    elif not receipt.exists():
                        warnings.append(f"Uncertain publication retained: {identity}/{revision_id}")
                        continue
                    else:
                        snapshot = _receipt_envelope(
                            json.loads(receipt.read_text(encoding="utf-8"))
                        )
                    if any(
                        (origin.get("origin_path"), origin.get("origin_fingerprint"))
                        in suppressed_origins
                        for origin in snapshot["ingest_sources"]
                    ):
                        raise ValueError(
                            "Suppressed origin receipt retained for owned cleanup only"
                        )
                    if (
                        snapshot.get("schema_version") not in ({4} if portable_receipt else {2, 3})
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
                        if portable_receipt is None:
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
                lifecycle_snapshot = completion.state if completion is not None else None
                if lifecycle_snapshot is not None and meeting_values is None:
                    meeting_values = lifecycle_snapshot["meeting"]
                if states.exists():
                    for entry in states.iterdir():
                        if group is not None and entry.name == portable_completion_name(
                            group.operation_id
                        ):
                            continue
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
                    self._index_active_revision(repo, meeting)
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
        with self._operation(recovery_warnings=warnings) as repo, ExitStack() as leases:
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
                except Exception as exc:
                    warnings.append("Deletion remains pending local recovery")
                    if isinstance(
                        exc, (_DeleteScopeUnavailable, _ExportCleanupPending, _ImportCleanupPending)
                    ):
                        warnings.append(str(exc))
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
            native_checked = set()

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
                    if self._sessions_roots:
                        self._retain_native_owners(identity, leases)
                        native_checked.add(identity)
                except Busy:
                    live.add(identity)
                    skipped.append(RecoveryOutcome(identity, "Live local owner; work preserved"))
                except (OSError, ValueError, LibraryError) as exc:
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
                    skipped.append(
                        RecoveryOutcome(row["id"], "Local ownership prevents operation recovery")
                    )
                    continue
                if row["kind"] == "delete":
                    # Already handled above with the same retained ownership path.
                    continue
                if row["meeting_id"] in suppressed:
                    continue
                try:
                    self._reconcile_operation(
                        repo,
                        PendingOperation(**dict(row)),
                        owners_held=row["meeting_id"] in native_checked,
                    )
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
                journal_rows = repo.library.execute(
                    "SELECT id,kind,phase,meeting_id,staging_path,destination_path "
                    "FROM pending_operations"
                ).fetchall()
                for entry in staging.iterdir():
                    relative = entry.relative_to(self.data_dir).as_posix()
                    if relative in protected:
                        continue
                    try:
                        if _IMPORT_STAGE_NAME.fullmatch(entry.name):
                            if any(
                                row["kind"] == "publish_portable"
                                and row["phase"] == "prepared"
                                and isinstance(row["staging_path"], str)
                                and row["staging_path"].startswith(relative + "/")
                                for row in journal_rows
                            ):
                                continue
                            self._recover_import_stage_locked(repo, entry)
                            continue
                        if not _is_uuid(entry.name):
                            raise LibraryError(
                                "Invalid staging operation identity; material retained"
                            )
                        # R54 preserves A's private pre-journal cleanup authority.
                        # Check ALL journal kinds/references first: an inconsistent
                        # reference must not turn identified work into an orphan.
                        rows = _stage_references(self.data_dir, journal_rows, relative, entry.name)
                        if not rows:
                            # Same publication lock, existing A plan/role/path and
                            # all-children preflight; no meeting/native authority,
                            # invented re-hash or committed-data access is needed.
                            _remove_tree(self.data_dir, relative, order=_delete_file_order)
                            continue
                        if (
                            len(rows) != 1
                            or rows[0]["kind"] != "publish_revision"
                            or rows[0]["id"] != entry.name
                            or rows[0]["staging_path"] != relative
                            or not _is_uuid(rows[0]["meeting_id"])
                        ):
                            raise LibraryError("Inconsistent staging journal; material retained")
                        identity = rows[0]["meeting_id"]
                        protect(identity)
                        if identity in live or identity not in native_checked:
                            raise LibraryError(
                                "Staging cleanup ownership unavailable; material retained"
                            )
                        _remove_tree(self.data_dir, relative, order=_delete_file_order)
                    except (OSError, ValueError, LibraryError) as exc:
                        warnings.append(str(exc))
                        skipped.append(RecoveryOutcome(entry.name, "Staging cleanup deferred"))
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
        acquired = meeting_id not in self._owners
        if acquired:
            self._acquire_owner(meeting_id)
        with self._operation() as repo:
            records, _ = _deletion_registry(self.data_dir)
            if meeting_id in _suppression_snapshot(repo.library, records).suppressed_meeting_ids:
                raise NotFound("Meeting has been deleted")
            try:
                self._require_portable_commit_sealed(repo, meeting_id)
            except Busy:
                if acquired:
                    self._owners.pop(meeting_id).close()
                raise
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
            try:
                self._require_portable_commit_sealed(repo, meeting_id)
            except Busy:
                self._owners.pop(meeting_id).close()
                raise
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
        with self._publisher_operation(
            declared_identity=meeting_id, revision_of=revision_of
        ) as repo:
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
        MemoryService._require_portable_commit_sealed(repo, meeting_id)
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
        return repo.meeting_detail(meeting, warnings)

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

    def meeting_ids_with_prefix(self, prefix: str) -> tuple[str, ...]:
        """Return only visible canonical identities matching a CLI abbreviation."""
        if not isinstance(prefix, str) or re.fullmatch(r"[a-f0-9][a-f0-9-]{0,35}", prefix) is None:
            raise InvalidInput("Use a canonical UUID or its leading characters")
        library = open_library(self.data_dir)
        try:
            with catalog_read_view(library) as suppression:
                hidden = frozenset(suppression.suppressed_meeting_ids)
                return tuple(
                    identity
                    for (identity,) in library.execute(
                        "SELECT id FROM meetings WHERE id GLOB ? AND status != 'deleting' "
                        "ORDER BY id",
                        (prefix + "*",),
                    )
                    if identity not in hidden and _is_uuid(identity)
                )
        finally:
            library.close()

    def retry_origins(self, meeting_id: str) -> frozenset[str]:
        """Return catalog origins for comparison with configured native candidates."""
        self._identity(meeting_id)
        library = open_library(self.data_dir)
        try:
            with catalog_read_view(library) as suppression:
                if meeting_id in suppression.suppressed_meeting_ids:
                    raise NotFound("Meeting does not exist")
                return frozenset(
                    origin
                    for (origin,) in library.execute(
                        "SELECT i.origin_path FROM ingest_sources i JOIN meetings m "
                        "ON m.id=i.meeting_id WHERE m.id=? AND m.status!='deleting'",
                        (meeting_id,),
                    )
                )
        finally:
            library.close()

    def resolve_citation(self, citation: Citation) -> CitationTarget:
        """Resolve text and playback authority within one guarded catalog view."""
        if not isinstance(citation, Citation):
            raise InvalidInput("Use a citation reference")
        for identity in (citation.meeting_id, citation.revision_id, citation.segment_id):
            self._identity(identity)
        start, end, quote = citation.start_char, citation.end_char, citation.quote
        if quote is not None and (not isinstance(quote, str) or not quote):
            raise InvalidInput("Citation quote must be nonempty text")
        if start is not None or end is not None:
            if type(start) is not int or type(end) is not int or not 0 <= start < end:
                raise InvalidInput("Use valid citation character bounds")

        library = open_library(self.data_dir)
        try:
            with catalog_read_view(library) as suppression:
                if citation.meeting_id in suppression.suppressed_meeting_ids:
                    raise NotFound("Meeting does not exist")
                repo = MemoryRepository(library)
                meeting = repo.get_meeting(citation.meeting_id)
                if meeting is None or meeting.status == MeetingStatus.DELETING:
                    raise NotFound("Meeting does not exist")
                target = CitationTarget(
                    meeting.id,
                    citation.revision_id,
                    citation.segment_id,
                    title=meeting.title,
                    started_at=meeting.started_at,
                    status="missing",
                )
                revision = repo.get_revision(citation.revision_id)
                segment_row = library.execute(
                    "SELECT * FROM segments WHERE id=?", (citation.segment_id,)
                ).fetchone()
                if revision is None:
                    if segment_row is not None or quote is not None or start is not None:
                        raise InvalidInput("Citation evidence does not match its revision")
                    return target
                if (
                    revision.meeting_id != meeting.id
                    or segment_row is None
                    or segment_row["revision_id"] != revision.id
                ):
                    raise InvalidInput(
                        "Citation segment does not belong to its meeting and revision"
                    )
                segment = Segment(**dict(segment_row))
                if start is not None and end is not None:
                    if end > len(segment.text) or (
                        quote is not None and quote != segment.text[start:end]
                    ):
                        raise InvalidInput("Citation evidence does not match its character bounds")
                elif quote is not None and quote not in segment.text:
                    raise InvalidInput("Citation quote does not match its segment")

                speaker = library.execute(
                    "SELECT p.display_name FROM speaker_assignments s "
                    "JOIN meeting_participants l USING (meeting_id, participant_id) "
                    "JOIN participants p ON p.id=s.participant_id "
                    "WHERE s.meeting_id=? AND s.revision_id=? AND s.speaker_key=?",
                    (meeting.id, revision.id, segment.speaker_key),
                ).fetchone()
                audio_path = self._citation_audio(repo, meeting.id, revision.id)
                status = "current" if revision.id == meeting.active_revision_id else "historical"
                return replace(
                    target,
                    text=segment.text,
                    speaker=speaker[0] if speaker is not None else None,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    timing_source=segment.timing_source,
                    audio_path=audio_path,
                    status=status if audio_path is not None else "missing",
                )
        finally:
            library.close()

    def _citation_audio(
        self, repo: MemoryRepository, meeting_id: str, revision_id: str
    ) -> Path | None:
        links = repo.get_revision_audio_assets(revision_id)
        if len(links) != 1:
            return None
        (link,) = links
        if (
            link.meeting_id != meeting_id
            or link.ordinal != 0
            or PurePosixPath(link.source_file).suffix.lower() != ".wav"
        ):
            return None
        asset = repo.get_audio_asset(link.asset_id)
        if asset is None or asset.meeting_id != meeting_id or asset.status != "available":
            return None
        relative = PurePosixPath(asset.relative_path)
        parts = relative.parts
        if (
            relative.as_posix() != asset.relative_path
            or len(parts) != 5
            or parts[:3] != ("meetings", meeting_id, "revisions")
            or not _is_uuid(parts[3])
            or relative.suffix.lower() != ".wav"
        ):
            return None
        try:
            path = managed_path(self.data_dir, asset.relative_path)
            revision = managed_path(
                self.data_dir, f"meetings/{meeting_id}/revisions/{revision_id}", directory=True
            )
            validate_artifact_roles(revision, (link.source_file,))
            validate_artifact_roles(path.parent, (path.name,))
            if path.stat().st_size == asset.size_bytes and file_hash(path) == asset.sha256:
                return path
        except (OSError, ValueError):
            pass
        return None

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
                self._require_portable_commit_sealed(repo, meeting_id)
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
                self._require_portable_commit_sealed(repo, meeting_id)
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
                self._require_portable_commit_sealed(repo, meeting_id)
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
                self._require_portable_commit_sealed(repo, meeting_id)
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
