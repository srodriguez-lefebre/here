"""Stage and validate portable meeting data before the library may publish it."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, BinaryIO, Literal, TypeVar
from uuid import UUID
from zipfile import BadZipFile, ZipFile

import soundfile as sf  # type: ignore[import-untyped]
from here.memory.artifacts import managed_path, portable_stage_io_path
from here.memory.database import LibraryError
from here.memory.models import (
    AudioAsset,
    Meeting,
    MeetingDetail,
    MeetingParticipant,
    MeetingStatus,
    Participant,
    RevisionAudioAsset,
    Segment,
    SpeakerAssignment,
    TranscriptRevision,
)
from here.memory.portable_package import InvalidPackage
from here.memory.repository import _PortableLifecycleEvent, _PortableMeetingRows
from here.output.paths import UnsafeSessionPath, validate_artifact_filename
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from here.memory.portability import Manifest
    from here.memory.service import _PortableImportStage


_BLOCK_BYTES = 1024 * 1024
_MAX_SQLITE_INTEGER = (1 << 63) - 1
_Id = Annotated[
    str, Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
]
_Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_Nonnegative = Annotated[int, Field(ge=0, le=_MAX_SQLITE_INTEGER)]
_Positive = Annotated[int, Field(gt=0, le=_MAX_SQLITE_INTEGER)]
_Outcome = Literal["inserted", "unchanged", "conflict", "rejected"]
_INVALID_UNICODE = re.compile(r"[\ud800-\udfff]")


class ImportCancelled(LibraryError):
    """The caller stopped the import before publishing the next meeting."""


@dataclass(frozen=True, slots=True)
class _PortableImportFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _PortableImportInventory:
    meeting_ids: tuple[str, ...]
    files: tuple[_PortableImportFile, ...]


@dataclass(frozen=True, slots=True)
class _PortableImportedMeeting:
    detail: MeetingDetail
    lifecycle_events: tuple[_PortableLifecycleEvent, ...]
    audio_payloads: tuple[tuple[AudioAsset, str], ...]


@dataclass(frozen=True, slots=True)
class _ValidatedPortableImport:
    stage: _PortableImportStage
    manifest_sha256: str
    include_audio: bool
    meetings: tuple[_PortableImportedMeeting, ...]


@dataclass(frozen=True, slots=True)
class _PortableMeetingOutcome:
    meeting_id: str
    status: _Outcome
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ImportReport:
    meetings: tuple[_PortableMeetingOutcome, ...]
    pending_recovery_ids: tuple[str, ...] = ()
    cleanup_pending: bool = False


class _PortableDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)


class _EventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    from_status: MeetingStatus | None = Field(default=None, alias="from")
    to: MeetingStatus | None = None
    fields: Annotated[list[Literal["title", "notes"]], Field(max_length=2)] = Field(
        default_factory=list
    )


class _EventDocument(_PortableDocument):
    id: _Id
    meeting_id: _Id
    kind: Literal["status_changed", "metadata_updated", "participant_linked", "speaker_assigned"]
    occurred_at: str
    recorded_duration_ms: _Nonnegative | None
    payload: dict[str, object]


class _AssetDocument(_PortableDocument):
    id: _Id
    meeting_id: _Id
    size_bytes: _Nonnegative
    sha256: _Digest
    codec: str
    sample_rate: _Positive
    channels: _Positive
    duration_ms: _Nonnegative
    status: Literal["available", "missing"]
    path: str | None


class _MeetingDocument(_PortableDocument):
    id: _Id
    legacy_session_id: str | None
    title: str
    notes: str
    started_at: str | None
    completed_at: str | None
    duration_ms: _Nonnegative | None
    paused_ms: _Nonnegative
    status: Literal["completed", "failed", "cancelled", "interrupted"]
    active_revision_id: _Id | None
    created_at: str
    updated_at: str
    lifecycle_events: list[object]
    audio_assets: list[object]


class _LinkDocument(_PortableDocument):
    meeting_id: _Id
    revision_id: _Id
    asset_id: _Id
    ordinal: _Nonnegative
    source_file: str


class _RevisionDocument(_PortableDocument):
    id: _Id
    meeting_id: _Id
    revision_number: _Positive
    raw_text: str
    presentation_text: str | None
    provider: str
    model: str
    created_at: str
    content_hash: _Digest
    status: Literal["complete", "partial"]
    audio_links: list[object]


class _SegmentDocument(_PortableDocument):
    id: _Id
    revision_id: _Id
    ordinal: _Nonnegative
    text: str
    timing_source: Literal["provider", "imported_explicit", "absent"]
    start_ms: _Nonnegative | None
    end_ms: _Nonnegative | None
    speaker_key: str | None


class _ParticipantDocument(_PortableDocument):
    id: _Id
    display_name: str


class _ParticipantLinkDocument(_PortableDocument):
    meeting_id: _Id
    participant_id: _Id
    role: str | None


class _ParticipantsDocument(_PortableDocument):
    participants: list[object]
    meeting_participants: list[object]


class _AssignmentDocument(_PortableDocument):
    meeting_id: _Id
    revision_id: _Id
    speaker_key: str
    participant_id: _Id | None


_Document = TypeVar("_Document", bound=_PortableDocument)
_Row = TypeVar("_Row")


def import_inventory(manifest: Manifest) -> _PortableImportInventory:
    if manifest.answer_ids:
        raise InvalidPackage("This version cannot yet import portable answers")
    return _PortableImportInventory(
        manifest.meeting_ids,
        tuple(_PortableImportFile(file.path, file.size, file.sha256) for file in manifest.files),
    )


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise ImportCancelled("Portable import cancelled")


def stage_package_files(
    archive: ZipFile,
    directory: Path,
    inventory: _PortableImportInventory,
    progress: Callable[[int], None],
    cancelled: Callable[[], bool],
) -> None:
    directory = portable_stage_io_path(directory)
    written = 0
    for entry in inventory.files:
        _check_cancelled(cancelled)
        try:
            destination = managed_path(directory, entry.path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination = managed_path(directory, entry.path)
            member = archive.open(entry.path)
        except (BadZipFile, EOFError, KeyError, OSError, UnsafeSessionPath, zlib.error):
            raise InvalidPackage("Could not open a registered portable payload") from None
        digest = hashlib.sha256()
        remaining = entry.size
        with member, _create_staged_file(destination) as output:
            while True:
                _check_cancelled(cancelled)
                try:
                    block = member.read(min(_BLOCK_BYTES, remaining + 1))
                except (BadZipFile, EOFError, OSError, zlib.error):
                    raise InvalidPackage("Could not decode a registered portable payload") from None
                if not block:
                    break
                remaining -= len(block)
                if remaining < 0:
                    raise InvalidPackage("Portable payload exceeds its manifest size")
                try:
                    output.write(block)
                except OSError:
                    raise LibraryError("Could not write a portable staging file") from None
                digest.update(block)
                written += len(block)
                progress(written)
            if remaining or digest.hexdigest() != entry.sha256:
                raise InvalidPackage("Portable payload size or SHA-256 does not match the manifest")
            try:
                output.flush()
                os.fsync(output.fileno())
            except OSError:
                raise LibraryError("Could not flush a portable staging file") from None


def _create_staged_file(destination: Path) -> BinaryIO:
    try:
        return destination.open("xb")
    except OSError:
        raise LibraryError("Could not create a registered portable staging file") from None


def _unique_json(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise InvalidPackage("Portable payload contains a duplicate JSON key")
        result[name] = value
    return result


def _read_json(directory: Path, relative: str) -> object:
    try:
        path = managed_path(directory, relative)
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_unique_json)
    except (OSError, ValueError, RecursionError):
        raise InvalidPackage("Could not read a registered portable JSON payload") from None


def _document(model: type[_Document], value: object) -> _Document:
    if not isinstance(value, dict) or value.keys() != model.model_fields.keys():
        raise InvalidPackage("Portable payload fields do not match format 1")
    try:
        return model.model_validate(value)
    except (ValidationError, UnicodeError):
        raise InvalidPackage("Portable payload has invalid format-1 values") from None


def _objects(value: object) -> list[object]:
    if not isinstance(value, list):
        raise InvalidPackage("Portable payload must contain a JSON array")
    return value


def _check_dates(*timestamps: str | None) -> None:
    for value in timestamps:
        if value is not None:
            try:
                datetime.fromisoformat(value)
            except ValueError:
                raise InvalidPackage("Portable evidence has an invalid timestamp") from None


def _check_utf8(*values: str | None) -> None:
    for value in values:
        if value is not None and _INVALID_UNICODE.search(value):
            raise InvalidPackage("Portable evidence contains invalid Unicode")


def _events(meeting: _MeetingDocument) -> tuple[_PortableLifecycleEvent, ...]:
    events = []
    for item in meeting.lifecycle_events:
        event = _document(_EventDocument, item)
        if event.meeting_id != meeting.id or event.payload.keys() - {"from", "to", "fields"}:
            raise InvalidPackage("Portable lifecycle event does not belong to its meeting")
        _check_dates(event.occurred_at)
        try:
            payload = _EventPayload.model_validate(event.payload)
        except ValidationError:
            raise InvalidPackage("Portable lifecycle event has invalid details") from None
        events.append(
            _PortableLifecycleEvent(
                event.id,
                event.meeting_id,
                event.kind,
                event.occurred_at,
                event.recorded_duration_ms,
                payload.model_dump_json(by_alias=True, exclude_unset=True),
            )
        )
    return tuple(sorted(events, key=lambda event: (event.occurred_at, event.id)))


def _segments(
    directory: Path, relative: str, revision_id: str, cancelled: Callable[[], bool]
) -> tuple[Segment, ...]:
    segments = []
    ordinals = set()
    for item in _objects(_read_json(directory, relative)):
        _check_cancelled(cancelled)
        segment = _document(_SegmentDocument, item)
        _check_utf8(segment.text, segment.speaker_key)
        start, end = segment.start_ms, segment.end_ms
        if (
            segment.revision_id != revision_id
            or segment.ordinal in ordinals
            or (start is not None and end is not None and end < start)
            or (segment.timing_source == "absent") != (start is None and end is None)
        ):
            raise InvalidPackage("Portable segment timing or revision relationship is invalid")
        ordinals.add(segment.ordinal)
        segments.append(Segment(**segment.model_dump()))
    return tuple(sorted(segments, key=lambda segment: segment.ordinal))


def _audio_links(revision: _RevisionDocument) -> tuple[RevisionAudioAsset, ...]:
    links = []
    labels = set()
    for item in revision.audio_links:
        link = _document(_LinkDocument, item)
        _check_utf8(link.source_file)
        try:
            validate_artifact_filename(link.source_file)
        except UnsafeSessionPath:
            raise InvalidPackage("Portable revision has an unsafe audio source label") from None
        if (
            link.meeting_id != revision.meeting_id
            or link.revision_id != revision.id
            or Path(link.source_file).suffix.lower() != ".wav"
            or link.source_file in labels
        ):
            raise InvalidPackage("Portable revision audio relationship is invalid")
        labels.add(link.source_file)
        links.append(RevisionAudioAsset(**link.model_dump()))
    links.sort(key=lambda link: link.ordinal)
    if any(link.ordinal != ordinal for ordinal, link in enumerate(links)):
        raise InvalidPackage("Portable revision audio ordinals are invalid")
    return tuple(links)


def _verify_wav(directory: Path, relative: str, asset: AudioAsset) -> None:
    try:
        with managed_path(directory, relative).open("rb") as stream:
            info = sf.info(stream)
    except (OSError, RuntimeError, ValueError):
        raise InvalidPackage("Portable audio could not be validated as WAV") from None
    if (
        info.format not in {"WAV", "WAVEX", "RF64"}
        or (info.subtype or info.format) != asset.codec
        or info.samplerate != asset.sample_rate
        or info.channels != asset.channels
        or round(info.duration * 1000) != asset.duration_ms
    ):
        raise InvalidPackage("Portable WAV properties do not match its declared audio asset")


def _audio_assets(
    directory: Path,
    meeting: _MeetingDocument,
    links: tuple[RevisionAudioAsset, ...],
    entries: dict[str, _PortableImportFile],
    include_audio: bool,
    cancelled: Callable[[], bool],
) -> tuple[tuple[AudioAsset, ...], tuple[tuple[AudioAsset, str], ...]]:
    first_revisions: dict[str, str] = {}
    for link in links:
        first_revisions.setdefault(link.asset_id, link.revision_id)
    assets: dict[str, AudioAsset] = {}
    payloads = []
    for item in meeting.audio_assets:
        _check_cancelled(cancelled)
        document = _document(_AssetDocument, item)
        _check_utf8(document.codec)
        revision_id = first_revisions.get(document.id)
        if document.meeting_id != meeting.id or revision_id is None or document.id in assets:
            raise InvalidPackage(
                "Portable audio asset does not belong to its meeting and revisions"
            )
        filename = f"audio_{UUID(document.id).hex}.wav"
        local = f"meetings/{meeting.id}/revisions/{revision_id}/{filename}"
        asset = AudioAsset(relative_path=local, **document.model_dump(exclude={"path"}))
        if include_audio:
            role = f"meetings/{meeting.id}/audio/{asset.id}.wav"
            entry = entries.get(role)
            if (
                document.path != role
                or asset.status != "available"
                or entry is None
                or (entry.size, entry.sha256) != (asset.size_bytes, asset.sha256)
            ):
                raise InvalidPackage("Portable audio asset does not match the package inventory")
            _verify_wav(directory, role, asset)
            payloads.append((asset, role))
        elif document.path is not None or asset.status != "missing":
            raise InvalidPackage("No-audio package contains an inconsistent audio asset")
        assets[asset.id] = asset
    if set(assets) != first_revisions.keys():
        raise InvalidPackage("Portable audio links reference a missing asset")
    return (
        tuple(assets[asset_id] for asset_id in sorted(assets)),
        tuple(sorted(payloads, key=lambda payload: payload[0].id)),
    )


def _participants_by_meeting(
    directory: Path, manifest: Manifest, cancelled: Callable[[], bool]
) -> tuple[dict[str, Participant], dict[str, list[MeetingParticipant]]]:
    roster = _document(_ParticipantsDocument, _read_json(directory, "participants.json"))
    participants: dict[str, Participant] = {}
    for item in roster.participants:
        _check_cancelled(cancelled)
        person = _document(_ParticipantDocument, item)
        _check_utf8(person.display_name)
        if not person.display_name.strip() or person.id in participants:
            raise InvalidPackage("Portable participant names or identities are invalid")
        participants[person.id] = Participant(person.id, person.display_name)
    links: dict[str, list[MeetingParticipant]] = defaultdict(list)
    meeting_ids = set(manifest.meeting_ids)
    link_keys: set[tuple[str, str]] = set()
    for item in roster.meeting_participants:
        _check_cancelled(cancelled)
        link = _document(_ParticipantLinkDocument, item)
        _check_utf8(link.role)
        key = (link.meeting_id, link.participant_id)
        if (
            link.meeting_id not in meeting_ids
            or link.participant_id not in participants
            or key in link_keys
        ):
            raise InvalidPackage("Portable participant link does not resolve uniquely")
        link_keys.add(key)
        links[link.meeting_id].append(MeetingParticipant(**link.model_dump()))
    if participants.keys() != set(manifest.participant_ids) or participants.keys() != {
        participant_id for _, participant_id in link_keys
    }:
        raise InvalidPackage("Portable participants do not match the manifest and meeting links")
    return participants, links


def _assignments_by_meeting(
    directory: Path, meeting_ids: set[str], cancelled: Callable[[], bool]
) -> dict[str, list[SpeakerAssignment]]:
    assignments: dict[str, list[SpeakerAssignment]] = defaultdict(list)
    keys: set[tuple[str, str]] = set()
    for item in _objects(_read_json(directory, "speaker_assignments.json")):
        _check_cancelled(cancelled)
        assignment = _document(_AssignmentDocument, item)
        _check_utf8(assignment.speaker_key)
        key = (assignment.revision_id, assignment.speaker_key)
        if assignment.meeting_id not in meeting_ids or not assignment.speaker_key or key in keys:
            raise InvalidPackage("Portable speaker assignment does not resolve uniquely")
        keys.add(key)
        assignments[assignment.meeting_id].append(SpeakerAssignment(**assignment.model_dump()))
    return assignments


def _meeting_bundle(
    directory: Path,
    identity: str,
    entries: dict[str, _PortableImportFile],
    revision_roles: list[tuple[int, str]],
    participant_rows: tuple[Participant, ...],
    participant_links: tuple[MeetingParticipant, ...],
    speaker_assignments: tuple[SpeakerAssignment, ...],
    include_audio: bool,
    cancelled: Callable[[], bool],
) -> _PortableImportedMeeting:
    prefix = f"meetings/{identity}"
    meeting = _document(_MeetingDocument, _read_json(directory, f"{prefix}/meeting.json"))
    if meeting.id != identity:
        raise InvalidPackage("Portable meeting does not match its manifest identity")
    _check_dates(meeting.started_at, meeting.completed_at, meeting.created_at, meeting.updated_at)
    _check_utf8(meeting.legacy_session_id, meeting.title, meeting.notes)
    revisions = []
    segments: list[Segment] = []
    links: list[RevisionAudioAsset] = []
    for number, role in sorted(revision_roles):
        _check_cancelled(cancelled)
        revision = _document(_RevisionDocument, _read_json(directory, role))
        if revision.meeting_id != identity or revision.revision_number != number:
            raise InvalidPackage("Portable revision does not match its meeting and revision number")
        _check_dates(revision.created_at)
        _check_utf8(
            revision.raw_text, revision.presentation_text, revision.provider, revision.model
        )
        revisions.append(
            TranscriptRevision(origin="", **revision.model_dump(exclude={"audio_links"}))
        )
        links.extend(_audio_links(revision))
        segment_path = f"{prefix}/revisions/{number}/segments.json"
        segments.extend(_segments(directory, segment_path, revision.id, cancelled))
    if meeting.active_revision_id is not None and all(
        revision.id != meeting.active_revision_id for revision in revisions
    ):
        raise InvalidPackage("Portable meeting active revision does not belong to that meeting")
    speakers = {(segment.revision_id, segment.speaker_key) for segment in segments}
    linked = {link.participant_id for link in participant_links}
    if any(
        (assignment.revision_id, assignment.speaker_key) not in speakers
        or (assignment.participant_id is not None and assignment.participant_id not in linked)
        for assignment in speaker_assignments
    ):
        raise InvalidPackage("Portable speaker assignment does not match its scoped evidence")
    assets, payloads = _audio_assets(
        directory, meeting, tuple(links), entries, include_audio, cancelled
    )
    detail = MeetingDetail(
        meeting=Meeting(
            **meeting.model_dump(exclude={"audio_assets", "lifecycle_events", "status"}),
            status=MeetingStatus(meeting.status),
        ),
        audio_assets=assets,
        revisions=tuple(revisions),
        segments=tuple(segments),
        revision_audio_assets=tuple(links),
        participants=participant_rows,
        participant_links=participant_links,
        speaker_assignments=speaker_assignments,
    )
    return _PortableImportedMeeting(detail, _events(meeting), payloads)


def _check_unique(items: Iterable[_Row], identity: Callable[[_Row], str]) -> None:
    seen: set[str] = set()
    for item in items:
        key = identity(item)
        if key in seen:
            raise InvalidPackage("Portable evidence repeats an entity identity")
        seen.add(key)


def validate_staged_import(
    stage: _PortableImportStage,
    manifest: Manifest,
    manifest_bytes: bytes,
    inventory: _PortableImportInventory,
    cancelled: Callable[[], bool],
) -> _ValidatedPortableImport:
    _check_cancelled(cancelled)
    directory = portable_stage_io_path(stage.directory)
    participants, links = _participants_by_meeting(directory, manifest, cancelled)
    assignments = _assignments_by_meeting(directory, set(manifest.meeting_ids), cancelled)
    entries = {entry.path: entry for entry in inventory.files}
    revision_roles: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for path in entries:
        if path.endswith("/revision.json"):
            _, meeting_id, _, number, _ = path.split("/")
            revision_roles[meeting_id].append((int(number), path))
    meetings = []
    for identity in manifest.meeting_ids:
        _check_cancelled(cancelled)
        meeting_links = tuple(sorted(links[identity], key=lambda item: item.participant_id))
        meetings.append(
            _meeting_bundle(
                directory,
                identity,
                entries,
                revision_roles[identity],
                tuple(participants[link.participant_id] for link in meeting_links),
                meeting_links,
                tuple(
                    sorted(
                        assignments[identity], key=lambda row: (row.revision_id, row.speaker_key)
                    )
                ),
                manifest.audio_included,
                cancelled,
            )
        )
    _check_unique((row for item in meetings for row in item.detail.revisions), lambda row: row.id)
    _check_unique((row for item in meetings for row in item.detail.segments), lambda row: row.id)
    _check_unique(
        (row for item in meetings for row in item.detail.audio_assets), lambda row: row.id
    )
    _check_unique((row for item in meetings for row in item.lifecycle_events), lambda row: row.id)
    if {role for item in meetings for _, role in item.audio_payloads} != {
        path for path in entries if path.endswith(".wav")
    }:
        raise InvalidPackage("Portable audio payloads must match exactly the selected assets")
    _check_cancelled(cancelled)
    return _ValidatedPortableImport(
        stage,
        hashlib.sha256(manifest_bytes).hexdigest(),
        manifest.audio_included,
        tuple(meetings),
    )


def _project_detail(detail: MeetingDetail, include_audio: bool) -> MeetingDetail:
    return replace(
        detail,
        meeting=replace(detail.meeting, row_version=1),
        revisions=tuple(replace(row, origin="") for row in detail.revisions),
        audio_assets=tuple(
            replace(row, relative_path="", status=row.status if include_audio else "missing")
            for row in detail.audio_assets
        ),
        warnings=(),
    )


def matches_portable_meeting(
    incoming: _PortableImportedMeeting, existing: _PortableMeetingRows, *, include_audio: bool
) -> bool:
    if _project_detail(incoming.detail, include_audio) != _project_detail(
        existing.detail, include_audio
    ):
        return False
    if len(incoming.lifecycle_events) != len(existing.lifecycle_events):
        return False
    try:
        return all(
            replace(left, payload="") == replace(right, payload="")
            and json.loads(left.payload)
            == json.loads(right.payload, object_pairs_hook=_unique_json)
            for left, right in zip(
                incoming.lifecycle_events, existing.lifecycle_events, strict=True
            )
        )
    except (InvalidPackage, ValueError, RecursionError):
        return False
