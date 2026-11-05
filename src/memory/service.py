"""Local immutable file publication with a recoverable SQLite operation journal."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

from here.memory.artifacts import (
    file_hash,
    managed_path,
    normalized_origin,
    read_artifacts,
    read_meeting_state,
    validate_artifact_roles,
    write_durable_json,
)
from here.memory.database import Busy, LibraryConnection, LibraryError, migrate, open_library
from here.memory.models import (
    AudioAsset,
    IngestSource,
    Meeting,
    MeetingStatus,
    PendingOperation,
    Segment,
    TranscriptRevision,
)
from here.memory.repository import MemoryRepository
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


@dataclass(frozen=True)
class RecoveryReport:
    warnings: tuple[object, ...] = ()


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
    except sqlite3.OperationalError:
        return False


class _FileLock:
    def __init__(self, path: Path, *, timeout: float = 0.25) -> None:
        self.path = session_artifact_path(path.parent, path.name)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.stream.seek(0)
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError as exc:
                if time.monotonic() >= deadline:
                    self.stream.close()
                    raise Busy("Library operation is busy; retry locally") from exc
                time.sleep(0.01)

    def close(self) -> None:
        self.stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


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
            library = open_library(self.data_dir)
            migrate(library)
            yield MemoryRepository(library)
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

    def active_revision(self, meeting_id: str | None) -> str | None:
        if not meeting_id:
            return None
        library = open_library(self.data_dir)
        try:
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
            repo = MemoryRepository(library)
            if meeting_id is not None:
                try:
                    canonical = str(UUID(meeting_id)) == meeting_id
                except ValueError:
                    canonical = False
                if canonical:
                    meeting = repo.get_meeting(meeting_id)
                    return (meeting.id, meeting.active_revision_id) if meeting else None
            rows = library.execute(
                "SELECT DISTINCT m.id,m.active_revision_id FROM ingest_sources i "
                "JOIN meetings m ON m.id=i.meeting_id WHERE i.origin_path=?",
                (normalized_origin(session_dir),),
            ).fetchall()
            if len(rows) > 1:
                raise ValueError("Legacy origin conflict: more than one catalog identity")
            return (rows[0][0], rows[0][1]) if rows else None
        finally:
            library.close()

    def ingest_session(self, session_dir: Path, *, revision_of: str | None = None) -> IngestResult:
        identity: str | None = None
        operation: PendingOperation | None = None
        staging_rel: str | None = None
        # Opening the catalog is a typed storage boundary; CLI must distinguish a
        # broken database from a rejected source or publication conflict.
        with self._operation() as repo:
            try:
                source = read_artifacts(session_dir)
                previous_origin = repo.get_ingest_source(source.origin_path, source.fingerprint)
                identity = source.metadata.meeting_id or (
                    previous_origin.meeting_id if previous_origin else str(uuid4())
                )
                if source.metadata.meeting_id is None and revision_of is not None:
                    parent = repo.get_revision(revision_of)
                    if parent is None:
                        return IngestResult(
                            "rejected", warnings=("Revision conflict: missing parent",)
                        )
                    identity = parent.meeting_id
                if previous_origin:
                    return IngestResult("unchanged", previous_origin.meeting_id)
                previous = repo.get_meeting(identity)
                active = (
                    repo.get_revision(previous.active_revision_id)
                    if (previous and previous.active_revision_id)
                    else None
                )
                if active and active.content_hash == source.content_hash:
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
                        "rejected", identity, ("Identity conflict: content differs",)
                    )
                if revision_of is not None and (not active or revision_of != active.id):
                    return IngestResult("rejected", identity, ("Revision conflict: stale retry",))
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
                    "SELECT sha256 FROM audio_assets WHERE meeting_id=?", (identity,)
                ).fetchall()
                known_hashes = {row[0] for row in existing_audio}
                assets = []
                audio_names = {item["name"] for item in source.audio}
                for audio in source.audio:
                    if audio["sha256"] in known_hashes:
                        continue
                    values = dict(audio)
                    name = values.pop("name")
                    assets.append(
                        AudioAsset(
                            id=str(uuid4()),
                            meeting_id=identity,
                            relative_path=f"{destination_rel}/{name}",
                            **values,
                        )
                    )
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
                meeting = Meeting(
                    id=identity,
                    legacy_session_id=metadata.session_id,
                    title=source.directory.name,
                    started_at=_utc_timestamp(metadata.started_at),
                    completed_at=_utc_timestamp(metadata.completed_at),
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
                warnings: tuple[object, ...] = (
                    (repo.library.storage_warning,) if repo.library.storage_warning else ()
                )
                if meeting.started_at is None or meeting.completed_at is None:
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
                    repo.insert_segments(segments)
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
                                        meeting.title,
                                    ),
                                )
                    else:
                        warnings += (FtsUnavailable(),)
                    repo.activate_revision(identity, revision_id)
                self._write_receipt(repo, operation)
                self._write_state(repo, identity)
                with repo.transaction():
                    repo.transition_pending_operation(operation_id, "complete")
                return IngestResult("revised" if active else "inserted", identity, warnings)
            except Exception as exc:
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
                return IngestResult("rejected", identity, (str(exc),))

    def _reconcile_operation(self, repo: MemoryRepository, operation: PendingOperation) -> None:
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
        write_durable_json(
            session_artifact_path(directory, f"{meeting.row_version}-{uuid4()}.json"),
            {"schema_version": 1, "meeting": asdict(meeting)},
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
            "schema_version": 1,
            "meeting": asdict(meeting),
            "revision": asdict(revision),
            "segments": [asdict(segment) for segment in repo.get_segments(revision.id)],
            "audio_assets": [
                dict(row)
                for row in repo.library.execute(
                    "SELECT * FROM audio_assets WHERE meeting_id=?", (meeting.id,)
                )
            ],
            "ingest_sources": [
                dict(row)
                for row in repo.library.execute(
                    "SELECT * FROM ingest_sources WHERE revision_id=?", (revision.id,)
                )
            ],
            "hashes": payload["hashes"],
        }
        write_durable_json(receipt, snapshot)

    def _rebuild(self, repo: MemoryRepository, warnings: list[object]) -> None:
        meetings_root = managed_path(self.data_dir, "meetings", directory=True)
        if not meetings_root.exists():
            return
        for directory in meetings_root.iterdir():
            try:
                identity = str(UUID(directory.name))
                if identity != directory.name:
                    raise ValueError("Invalid managed meeting UUID")
                validate_session_directory(directory)
                # Existing catalog rows stay authoritative. Only absent meetings
                # are restored; uncertain unreceipted publications are not adopted.
                if repo.get_meeting(identity) is not None:
                    continue
                revisions_path = managed_path(
                    self.data_dir, f"meetings/{identity}/revisions", directory=True
                )
                if not revisions_path.exists():
                    continue
                snapshots = []
                for revision_dir in revisions_path.iterdir():
                    revision_id = str(UUID(revision_dir.name))
                    validate_session_directory(revision_dir)
                    receipt = session_artifact_path(revision_dir, "catalog.json")
                    if not receipt.exists():
                        warnings.append(f"Uncertain publication retained: {identity}/{revision_id}")
                        continue
                    snapshot = json.loads(receipt.read_text(encoding="utf-8"))
                    if (
                        snapshot.get("schema_version") != 1
                        or snapshot["meeting"]["id"] != identity
                        or snapshot["revision"]["meeting_id"] != identity
                        or snapshot["revision"]["id"] != revision_id
                    ):
                        raise ValueError("Invalid managed publication receipt")
                    validate_artifact_roles(revision_dir, snapshot["hashes"])
                    for name, digest in snapshot["hashes"].items():
                        if file_hash(session_artifact_path(revision_dir, name)) != digest:
                            raise ValueError("Managed revision hash mismatch during rebuild")
                    for asset in snapshot["audio_assets"]:
                        path = managed_path(self.data_dir, asset["relative_path"])
                        if asset["meeting_id"] != identity or not path.is_relative_to(directory):
                            raise ValueError("Managed audio identity mismatch")
                        if path.suffix.lower() != ".wav":
                            raise ValueError("Managed audio asset must have a WAV role")
                        validate_artifact_roles(path.parent, (path.name,))
                        if (
                            path.stat().st_size != asset["size_bytes"]
                            or file_hash(path) != asset["sha256"]
                        ):
                            raise ValueError("Managed audio hash or identity mismatch")
                    for segment in snapshot["segments"]:
                        if segment["revision_id"] != revision_id:
                            raise ValueError("Managed segment revision mismatch")
                    for origin in snapshot["ingest_sources"]:
                        if origin["meeting_id"] != identity or origin["revision_id"] != revision_id:
                            raise ValueError("Managed source identity mismatch")
                    snapshots.append(snapshot)
                if not snapshots:
                    continue
                snapshots.sort(key=lambda item: item["revision"]["revision_number"])
                meeting_values = snapshots[-1]["meeting"]
                states = managed_path(self.data_dir, f"meetings/{identity}/states", directory=True)
                if states.exists():
                    for entry in states.iterdir():
                        try:
                            state = read_meeting_state(states, entry.name, identity)
                        except (OSError, ValueError, TypeError, KeyError) as exc:
                            warnings.append(str(exc))
                            continue
                        if state["row_version"] >= meeting_values["row_version"]:
                            meeting_values = state
                meeting = Meeting(**meeting_values)
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
                        for values in snapshot["ingest_sources"]:
                            repo.insert_ingest_source(IngestSource(**values))
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
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                warnings.append(str(exc))

    def reconcile(self) -> RecoveryReport:
        warnings: list[object] = []
        with self._operation() as repo:
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
            for identity in identities:
                if identity is None:
                    continue
                try:
                    if str(UUID(identity)) != identity:
                        raise ValueError("Invalid owner identity")
                    owner_path = managed_path(self.data_dir, f".owners/{identity}.lock")
                    if not owner_path.exists():
                        continue
                    probe = _FileLock(owner_path, timeout=0)
                    probe.close()
                except Busy:
                    live.add(identity)
                except (OSError, ValueError) as exc:
                    warnings.append(str(exc))
                    # An invalid owner path is not proof the job has stopped.
                    live.add(identity)
            for row in rows:
                if row["meeting_id"] in live:
                    continue
                try:
                    self._reconcile_operation(repo, PendingOperation(**dict(row)))
                except Exception as exc:
                    warnings.append(str(exc))
            self._rebuild(repo, warnings)
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
            with repo.transaction():
                for row in active_rows:
                    if row[0] not in live:
                        repo.library.execute(
                            "UPDATE meetings SET status='interrupted', row_version=row_version+1 "
                            "WHERE id=? AND status IN ('recording','processing')",
                            (row[0],),
                        )
            for row in active_rows:
                if row[0] not in live:
                    self._write_state(repo, row[0])
        return RecoveryReport(tuple(warnings))

    def begin_capture(self, meeting_id: str, *, started_at: str | None = None) -> None:
        if str(UUID(meeting_id)) != meeting_id:
            raise ValueError("Invalid capture identity")
        directory = managed_path(self.data_dir, ".owners", directory=True)
        directory.mkdir(parents=True, exist_ok=True)
        if meeting_id not in self._owners:
            self._owners[meeting_id] = _FileLock(directory / f"{meeting_id}.lock")
        with self._operation() as repo:
            with repo.transaction():
                if repo.get_meeting(meeting_id) is None:
                    repo.insert_meeting(
                        Meeting(
                            id=meeting_id,
                            started_at=_utc_timestamp(started_at),
                            status=MeetingStatus.RECORDING,
                        )
                    )
            self._write_state(repo, meeting_id)

    def set_status(self, meeting_id: str, status: str) -> None:
        status = MeetingStatus(status)
        with self._operation() as repo:
            with repo.transaction():
                repo.library.execute(
                    "UPDATE meetings SET status=?, row_version=row_version+1 WHERE id=?",
                    (status, meeting_id),
                )
            self._write_state(repo, meeting_id)

    def begin_processing(self, meeting_id: str) -> None:
        """Own the local processing lifetime, preserving any prior active evidence."""
        if str(UUID(meeting_id)) != meeting_id:
            raise ValueError("Invalid processing identity")
        directory = managed_path(self.data_dir, ".owners", directory=True)
        directory.mkdir(parents=True, exist_ok=True)
        if meeting_id not in self._owners:
            self._owners[meeting_id] = _FileLock(directory / f"{meeting_id}.lock")
        with self._operation() as repo:
            with repo.transaction():
                if repo.get_meeting(meeting_id) is None:
                    repo.insert_meeting(Meeting(id=meeting_id, status=MeetingStatus.PROCESSING))
                else:
                    repo.library.execute(
                        "UPDATE meetings SET status='processing', "
                        "row_version=row_version+1 WHERE id=?",
                        (meeting_id,),
                    )
            self._write_state(repo, meeting_id)

    def finish_attempt(
        self, meeting_id: str | None, status: str, *, revision_of: str
    ) -> IngestResult:
        """A failed/cancelled retry changes state, never the completed evidence."""
        if status not in {"failed", "cancelled"}:
            raise ValueError("Only unsuccessful retry outcomes use state-only publication")
        with self._operation() as repo:
            with repo.transaction():
                if meeting_id is None:
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
                repo.library.execute(
                    "UPDATE meetings SET status=?, row_version=row_version+1 WHERE id=?",
                    (status, meeting_id),
                )
            self._write_state(repo, meeting_id)
        return IngestResult("unchanged", meeting_id)

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

    def show(self, meeting_id: str) -> dict[str, object] | None:
        with self._operation() as repo:
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
            return result
