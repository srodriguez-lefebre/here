"""Export selected catalog evidence through a memory-owned publication scope."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, BinaryIO, Literal
from uuid import UUID
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from here.memory.database import IntegrityFailure, LibraryError
from here.memory.models import (
    AudioAsset,
    MeetingStatus,
    Participant,
    RevisionAudioAsset,
    Segment,
)
from here.memory.repository import (
    _PortableExportRows,
    _PortableLifecycleEvent,
    _PortableMeetingRows,
)
from here.output.paths import UnsafeSessionPath, validate_artifact_filename
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from here.memory.service import MemoryService, _PortableExportView


_MAX_ENTRIES = 10_000
_MAX_UNCOMPRESSED_BYTES = 20 * 1024**3
_AUDIO_BLOCK_BYTES = 1024 * 1024
_MANIFEST_PATH = "manifest.json"

# Upper bounds for local and central headers, ZIP64 extras, and the end records.
_ZIP_ENTRY_OVERHEAD = 128
_ZIP_END_OVERHEAD = 128


class ExportCancelled(LibraryError):
    """The caller stopped the export before publication."""


class InsufficientSpace(LibraryError):
    """The destination cannot accommodate the package."""


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size: Annotated[int, Field(ge=0)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    exporter_version: str
    exported_at: datetime
    meeting_ids: tuple[str, ...]
    audio_included: bool
    participant_ids: tuple[str, ...]
    answer_ids: tuple[str, ...]
    omitted: tuple[str, ...]
    files: tuple[ManifestFile, ...]


class ExportResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    included_answer_ids: tuple[str, ...]
    omitted_answer_ids: tuple[str, ...]
    payload_bytes: int
    archive_bytes: int
    warnings: tuple[str, ...] = ()


class _EventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_status: MeetingStatus | None = Field(default=None, alias="from")
    to: MeetingStatus | None = None
    fields: Annotated[list[Literal["title", "notes"]], Field(max_length=2)] = Field(
        default_factory=list
    )


def _canonical_archive_id(value: str) -> str:
    try:
        if str(UUID(value)) == value:
            return value
    except (AttributeError, TypeError, ValueError):
        pass
    raise IntegrityFailure("Selected catalog evidence has an invalid identity")


def _audio_entry(asset: AudioAsset) -> str:
    meeting_id = _canonical_archive_id(asset.meeting_id)
    asset_id = _canonical_archive_id(asset.id)
    return f"meetings/{meeting_id}/audio/{asset_id}.wav"


def _audio_metadata(asset: AudioAsset, include_audio: bool) -> dict[str, object]:
    return {
        "id": asset.id,
        "meeting_id": asset.meeting_id,
        "size_bytes": asset.size_bytes,
        "sha256": asset.sha256,
        "codec": asset.codec,
        "sample_rate": asset.sample_rate,
        "channels": asset.channels,
        "duration_ms": asset.duration_ms,
        "status": asset.status if include_audio else "missing",
        "path": _audio_entry(asset) if include_audio else None,
    }


def _audio_link(link: RevisionAudioAsset) -> dict[str, object]:
    try:
        validate_artifact_filename(link.source_file)
    except (TypeError, UnsafeSessionPath) as error:
        raise IntegrityFailure("Selected revision has an invalid audio source label") from error
    if Path(link.source_file).suffix.lower() != ".wav":
        raise IntegrityFailure("Selected revision has an invalid audio source label")
    return {
        "meeting_id": link.meeting_id,
        "revision_id": link.revision_id,
        "asset_id": link.asset_id,
        "ordinal": link.ordinal,
        "source_file": link.source_file,
    }


def _event_metadata(event: _PortableLifecycleEvent) -> dict[str, object]:
    if event.kind not in {
        "status_changed",
        "metadata_updated",
        "participant_linked",
        "speaker_assigned",
    }:
        raise IntegrityFailure("Selected meeting has an unknown lifecycle event")
    try:
        payload = _EventPayload.model_validate_json(event.payload)
    except ValidationError as error:
        raise IntegrityFailure("Selected meeting has an invalid lifecycle event payload") from error
    return {
        "id": event.id,
        "meeting_id": event.meeting_id,
        "kind": event.kind,
        "occurred_at": event.occurred_at,
        "recorded_duration_ms": event.recorded_duration_ms,
        "payload": payload.model_dump(mode="json", by_alias=True, exclude_unset=True),
    }


def _meeting_metadata(rows: _PortableMeetingRows, include_audio: bool) -> dict[str, object]:
    meeting = rows.meeting
    return {
        "id": meeting.id,
        "legacy_session_id": meeting.legacy_session_id,
        "title": meeting.title,
        "notes": meeting.notes,
        "started_at": meeting.started_at,
        "completed_at": meeting.completed_at,
        "duration_ms": meeting.duration_ms,
        "paused_ms": meeting.paused_ms,
        "status": meeting.status,
        "active_revision_id": meeting.active_revision_id,
        "created_at": meeting.created_at,
        "updated_at": meeting.updated_at,
        "lifecycle_events": [_event_metadata(event) for event in rows.lifecycle_events],
        "audio_assets": [_audio_metadata(asset, include_audio) for asset in rows.audio_assets],
    }


def _segment_metadata(segment: Segment) -> dict[str, object]:
    return {
        "id": segment.id,
        "revision_id": segment.revision_id,
        "ordinal": segment.ordinal,
        "text": segment.text,
        "timing_source": segment.timing_source,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "speaker_key": segment.speaker_key,
    }


def _meeting_payloads(
    rows: _PortableMeetingRows, include_audio: bool
) -> Iterator[tuple[str, object]]:
    prefix = f"meetings/{_canonical_archive_id(rows.meeting.id)}"
    yield f"{prefix}/meeting.json", _meeting_metadata(rows, include_audio)
    links: dict[str, list[RevisionAudioAsset]] = defaultdict(list)
    for link in rows.detail.revision_audio_assets:
        links[link.revision_id].append(link)
    segments: dict[str, list[Segment]] = defaultdict(list)
    for segment in rows.detail.segments:
        segments[segment.revision_id].append(segment)
    for revision in rows.detail.revisions:
        if type(revision.revision_number) is not int or revision.revision_number < 1:
            raise IntegrityFailure("Selected transcript has an invalid revision number")
        revision_path = f"{prefix}/revisions/{revision.revision_number}"
        yield (
            f"{revision_path}/revision.json",
            {
                "id": revision.id,
                "meeting_id": revision.meeting_id,
                "revision_number": revision.revision_number,
                "raw_text": revision.raw_text,
                "presentation_text": revision.presentation_text,
                "provider": revision.provider,
                "model": revision.model,
                "created_at": revision.created_at,
                "content_hash": revision.content_hash,
                "status": revision.status,
                "audio_links": [_audio_link(link) for link in links[revision.id]],
            },
        )
        yield (
            f"{revision_path}/segments.json",
            [_segment_metadata(segment) for segment in segments[revision.id]],
        )


def _participants(rows: _PortableExportRows) -> tuple[Participant, ...]:
    participants = {
        participant.id: participant
        for meeting in rows.meetings
        for participant in meeting.detail.participants
    }
    return tuple(participants[identity] for identity in sorted(participants))


def _json_payloads(rows: _PortableExportRows, include_audio: bool) -> Iterator[tuple[str, bytes]]:
    links = []
    assignments = []
    for meeting in rows.meetings:
        yield from (
            (path, _json_bytes(document))
            for path, document in _meeting_payloads(meeting, include_audio)
        )
        linked = {link.participant_id for link in meeting.detail.participant_links}
        links.extend(
            {
                "meeting_id": link.meeting_id,
                "participant_id": link.participant_id,
                "role": link.role,
            }
            for link in meeting.detail.participant_links
        )
        for assignment in meeting.detail.speaker_assignments:
            if assignment.participant_id is not None and assignment.participant_id not in linked:
                raise IntegrityFailure("Selected meeting has an unlinked speaker assignment")
            assignments.append(
                {
                    "meeting_id": assignment.meeting_id,
                    "revision_id": assignment.revision_id,
                    "speaker_key": assignment.speaker_key,
                    "participant_id": assignment.participant_id,
                }
            )
    yield (
        "participants.json",
        _json_bytes(
            {
                "participants": [
                    {"id": participant.id, "display_name": participant.display_name}
                    for participant in _participants(rows)
                ],
                "meeting_participants": links,
            }
        ),
    )
    yield "speaker_assignments.json", _json_bytes(assignments)


def _json_bytes(document: object) -> bytes:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _check_cancelled(cancellation: Callable[[], bool]) -> None:
    if cancellation():
        raise ExportCancelled("Portable export cancelled")


def _entry(path: str, payload: bytes) -> ManifestFile:
    return ManifestFile(path=path, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _check_limits(entries: int, uncompressed_bytes: int) -> None:
    if entries > _MAX_ENTRIES or uncompressed_bytes > _MAX_UNCOMPRESSED_BYTES:
        raise LibraryError("Selection exceeds the portable package size or entry limit")


@contextmanager
def _archive_disk_errors() -> Iterator[None]:
    try:
        yield
    except OSError as error:
        if error.errno == errno.ENOSPC:
            raise InsufficientSpace("Not enough space for the portable package") from error
        raise LibraryError("Could not write the portable package") from error


@contextmanager
def _open_archive(stream: BinaryIO) -> Iterator[ZipFile]:
    with _archive_disk_errors():
        archive = ZipFile(stream, mode="w", compression=ZIP_STORED)
    try:
        yield archive
    except BaseException:
        # The owner removes this unpublished file. Preserve the primary error.
        with suppress(OSError):
            archive.close()
        raise
    else:
        with _archive_disk_errors():
            archive.close()


def _write_payload(archive: ZipFile, member: ZipInfo, payload: bytes) -> None:
    with _archive_disk_errors():
        archive.writestr(member, payload)


def _write_audio(
    archive: ZipFile,
    member: ZipInfo,
    view: _PortableExportView,
    asset: AudioAsset,
    progress: Callable[[int], None],
    cancellation: Callable[[], bool],
    written: int,
) -> int:
    member.file_size = asset.size_bytes
    digest = hashlib.sha256()
    remaining = asset.size_bytes
    with view.open_audio(asset) as audio:
        with _archive_disk_errors():
            output = archive.open(member, mode="w", force_zip64=True)
        try:
            while True:
                _check_cancelled(cancellation)
                try:
                    block = audio.read(min(_AUDIO_BLOCK_BYTES, remaining + 1))
                except OSError as error:
                    raise IntegrityFailure("Could not read the selected managed audio") from error
                if not block:
                    break
                remaining -= len(block)
                if remaining < 0:
                    raise IntegrityFailure("Managed audio size changed during export")
                digest.update(block)
                with _archive_disk_errors():
                    output.write(block)
                written += len(block)
                progress(written)
        except BaseException:
            with suppress(OSError):
                output.close()
            raise
        else:
            with _archive_disk_errors():
                output.close()
    if remaining or digest.hexdigest() != asset.sha256:
        raise IntegrityFailure("Managed audio no longer matches its recorded size and hash")
    return written


class PortabilityService:
    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    def export(
        self,
        ids: tuple[str, ...],
        destination: Path,
        include_audio: bool,
        progress: Callable[[int], None],
        cancellation: Callable[[], bool],
    ) -> ExportResult:
        _check_cancelled(cancellation)
        with self._memory._portable_export_view(ids) as view:
            rows = view.rows
            entries = []
            payload_bytes = 0
            for path, payload in _json_payloads(rows, include_audio):
                _check_cancelled(cancellation)
                entries.append(_entry(path, payload))
                payload_bytes += len(payload)
                _check_limits(len(entries) + 1, payload_bytes)
            audio = tuple(asset for meeting in rows.meetings for asset in meeting.audio_assets)
            if include_audio:
                for asset in audio:
                    _check_cancelled(cancellation)
                    if asset.status != "available":
                        raise IntegrityFailure("Selected managed audio is unavailable")
                    try:
                        entry = ManifestFile(
                            path=_audio_entry(asset), size=asset.size_bytes, sha256=asset.sha256
                        )
                    except ValidationError as error:
                        raise IntegrityFailure(
                            "Selected managed audio metadata is invalid"
                        ) from error
                    entries.append(entry)
                    payload_bytes += asset.size_bytes
                    _check_limits(len(entries) + 1, payload_bytes)
            try:
                exporter_version = version("here")
            except PackageNotFoundError as error:
                raise LibraryError("The installed exporter version is unavailable") from error
            manifest = Manifest(
                exporter_version=exporter_version,
                exported_at=datetime.now(timezone.utc),
                meeting_ids=tuple(meeting.meeting.id for meeting in rows.meetings),
                audio_included=include_audio,
                participant_ids=tuple(participant.id for participant in _participants(rows)),
                answer_ids=(),
                omitted=(),
                files=tuple(entries),
            )
            manifest_bytes = manifest.model_dump_json().encode("utf-8")
            total_bytes = payload_bytes + len(manifest_bytes)
            _check_limits(len(entries) + 1, total_bytes)
            paths = (*(entry.path for entry in entries), _MANIFEST_PATH)
            if len(set(paths)) != len(paths):
                raise IntegrityFailure("Selected catalog evidence has duplicate package members")
            reserved_bytes = (
                total_bytes
                + _ZIP_END_OVERHEAD
                + sum(_ZIP_ENTRY_OVERHEAD + 2 * len(path.encode("utf-8")) for path in paths)
            )
            stamp = manifest.exported_at
            dos_time = (
                min(max(stamp.year, 1980), 2107),
                stamp.month,
                stamp.day,
                stamp.hour,
                stamp.minute,
                stamp.second,
            )
            _check_cancelled(cancellation)
            with view._open_export_destination(destination) as temporary:
                if include_audio and temporary.free_space() < reserved_bytes:
                    raise InsufficientSpace("Not enough space for the portable package")
                with temporary.open_stream() as stream:
                    with _open_archive(stream) as archive:
                        written = 0
                        for path, payload in _json_payloads(rows, include_audio):
                            _check_cancelled(cancellation)
                            _write_payload(archive, ZipInfo(path, dos_time), payload)
                            written += len(payload)
                            progress(written)
                        if include_audio:
                            for asset in audio:
                                written = _write_audio(
                                    archive,
                                    ZipInfo(_audio_entry(asset), dos_time),
                                    view,
                                    asset,
                                    progress,
                                    cancellation,
                                    written,
                                )
                        _check_cancelled(cancellation)
                        _write_payload(archive, ZipInfo(_MANIFEST_PATH, dos_time), manifest_bytes)
                    with _archive_disk_errors():
                        stream.flush()
                        os.fsync(stream.fileno())
                        archive_bytes = stream.tell()
                view.publish(temporary, cancelled=cancellation)
        return ExportResult(
            path=destination,
            included_answer_ids=manifest.answer_ids,
            omitted_answer_ids=manifest.omitted,
            payload_bytes=payload_bytes,
            archive_bytes=archive_bytes,
            warnings=view.warnings,
        )
