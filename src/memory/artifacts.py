"""Read source artifacts without changing their origin or inferring evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast
from uuid import UUID

import soundfile as sf  # type: ignore[import-untyped]
from here.memory.models import (
    AudioAsset,
    IngestSource,
    Meeting,
    MeetingStatus,
    Participant,
    RevisionAudioAsset,
    Segment,
    TranscriptRevision,
)
from here.output.metadata import SessionMetadata
from here.output.paths import (
    UnsafeSessionPath,
    session_artifact_path,
    staged_artifact_path,
    validate_artifact_filename,
    validate_session_directory,
)

SESSION_ARTIFACT_NAMES = frozenset(
    {
        "session.json",
        "transcript.txt",
        "transcript.md",
        "raw_transcript.txt",
        "segments.json",
        "chunks.json",
        "events.json",
        "errors.json",
        "audio.wav",
    }
)


def declared_artifact_roles(metadata: SessionMetadata) -> set[str]:
    audio = {item.audio_file for item in metadata.capture_sources if item.audio_file}
    if metadata.recoverable_audio:
        audio.add(metadata.recoverable_audio)
    if any(Path(name).suffix.lower() != ".wav" for name in audio):
        raise ValueError("Session audio references must name WAV artifacts")
    allowed = set(SESSION_ARTIFACT_NAMES) | audio
    if any(name not in allowed for name in metadata.output_files):
        raise ValueError("Session manifest references an unknown artifact role")
    return allowed


def validate_artifact_roles(
    directory: Path, names: Iterable[str], *, require_audio_declaration: bool = False
) -> None:
    """A persisted hash/cleanup plan cannot itself grant a file an artifact role."""
    remaining = []
    for name in names:
        if not isinstance(name, str):
            raise ValueError("Artifact names must be strings")
        session_artifact_path(directory, name)
        if require_audio_declaration and Path(name).suffix.lower() == ".wav":
            remaining.append(name)
            continue
        if name in SESSION_ARTIFACT_NAMES or re.fullmatch(
            r"(?:audio_[0-9a-f]{32}|source_[0-9]{2,})\.wav", name
        ):
            continue
        if Path(name).suffix.lower() != ".wav":
            raise ValueError("Persisted plan references an unknown artifact role")
        remaining.append(name)
    if remaining:
        # Nonstandard v1 WAV names need an explicit validated session/audio role.
        # Reading this fixed protocol manifest never dereferences those references.
        path = session_artifact_path(directory, "session.json")
        if not path.exists():
            raise ValueError("Unrecognized WAV has no explicit session audio role")
        metadata = SessionMetadata.model_validate_json(path.read_text(encoding="utf-8-sig"))
        allowed = declared_artifact_roles(metadata)
        if require_audio_declaration:
            allowed = {item.audio_file for item in metadata.capture_sources if item.audio_file}
            if metadata.recoverable_audio:
                allowed.add(metadata.recoverable_audio)
            if "audio.wav" in metadata.output_files:
                allowed.add("audio.wav")
        if metadata.schema_version != 1 or not set(remaining) <= allowed:
            raise ValueError("Unrecognized WAV is not declared session audio")


def _state_uuid(value: object) -> bool:
    try:
        return isinstance(value, str) and str(UUID(value)) == value
    except ValueError:
        return False


def _state_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _meeting_state_version(name: str) -> int:
    filename = Path(name)
    version, separator, identity = filename.stem.partition("-")
    try:
        valid = (
            filename.suffix == ".json"
            and separator
            and int(version) > 0
            and str(UUID(identity)) == identity
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Unrecognized meeting-state entry retained")
    return int(version)


def read_meeting_snapshot(directory: Path, name: str, meeting_id: str) -> dict[str, Any]:
    """Validate the existing state envelope and its additive lifecycle supplements."""
    version = _meeting_state_version(name)
    path = session_artifact_path(directory, name)
    state = json.loads(path.read_text(encoding="utf-8"))
    return _validate_meeting_snapshot(state, version, meeting_id)


def _validate_meeting_snapshot(state: Any, version: int, meeting_id: str) -> dict[str, Any]:
    meeting = state.get("meeting") if isinstance(state, dict) else None
    if (
        not isinstance(state, dict)
        or state.get("schema_version") not in {1, 2}
        or not isinstance(meeting, dict)
        or meeting.get("id") != meeting_id
        or meeting.get("row_version") != version
    ):
        raise ValueError("Invalid managed meeting-state envelope")
    if state["schema_version"] == 1:
        return state
    expected = {
        "schema_version",
        "meeting",
        "participants",
        "meeting_participants",
        "speaker_assignments",
        "lifecycle_events",
        "audio_assets",
        "ingest_sources",
        "source_kind",
    }
    if set(state) != expected or state["source_kind"] not in {"session", "legacy_audio_only"}:
        raise ValueError("Invalid lifecycle snapshot envelope")
    Meeting(**meeting)
    if (
        not _state_uuid(meeting_id)
        or any(
            not isinstance(meeting.get(key), str)
            for key in ("title", "notes", "created_at", "updated_at")
        )
        or type(meeting.get("row_version")) is not int
        or meeting["row_version"] < 1
        or type(meeting.get("paused_ms")) is not int
        or meeting["paused_ms"] < 0
        or any(
            meeting.get(key) is not None and not isinstance(meeting[key], str)
            for key in ("legacy_session_id", "started_at", "completed_at")
        )
        or meeting.get("duration_ms") is not None
        and (type(meeting["duration_ms"]) is not int or meeting["duration_ms"] < 0)
        or meeting.get("active_revision_id") is not None
        and not _state_uuid(meeting["active_revision_id"])
    ):
        raise ValueError("Invalid lifecycle meeting values")
    MeetingStatus(meeting["status"])
    for key in ("created_at", "updated_at", "started_at", "completed_at"):
        if meeting[key] is not None:
            datetime.fromisoformat(meeting[key])
    columns = {
        "participants": {"id", "display_name"},
        "meeting_participants": {"meeting_id", "participant_id", "role"},
        "speaker_assignments": {"meeting_id", "revision_id", "speaker_key", "participant_id"},
        "lifecycle_events": {
            "id",
            "meeting_id",
            "kind",
            "occurred_at",
            "recorded_duration_ms",
            "payload",
        },
        "audio_assets": set(AudioAsset.__dataclass_fields__),
        "ingest_sources": set(IngestSource.__dataclass_fields__),
    }
    for table, field_names in columns.items():
        entries = state[table]
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict) or set(entry) != field_names for entry in entries
        ):
            raise ValueError("Invalid lifecycle snapshot table shape")
        if table != "participants" and any(entry["meeting_id"] != meeting_id for entry in entries):
            raise ValueError("Lifecycle snapshot relationship belongs to another meeting")
    participants = {}
    for entry in state["participants"]:
        participant = Participant(**entry)
        if (
            not _state_uuid(participant.id)
            or not isinstance(participant.display_name, str)
            or not participant.display_name.strip()
            or participant.id in participants
        ):
            raise ValueError("Invalid lifecycle participant")
        participants[participant.id] = participant
    linked = set()
    for entry in state["meeting_participants"]:
        if (
            entry["participant_id"] not in participants
            or entry["participant_id"] in linked
            or entry["role"] is not None
            and not isinstance(entry["role"], str)
        ):
            raise ValueError("Invalid lifecycle participant link")
        linked.add(entry["participant_id"])
    if set(participants) != linked:
        raise ValueError("Lifecycle participant snapshot must contain linked identities only")
    assigned = set()
    for entry in state["speaker_assignments"]:
        scope_key = (entry["revision_id"], entry["speaker_key"])
        if (
            not _state_uuid(entry["revision_id"])
            or not isinstance(entry["speaker_key"], str)
            or not entry["speaker_key"]
            or scope_key in assigned
            or entry["participant_id"] is not None
            and entry["participant_id"] not in linked
        ):
            raise ValueError("Invalid lifecycle speaker assignment")
        assigned.add(scope_key)
    events = set()
    for entry in state["lifecycle_events"]:
        if (
            not _state_uuid(entry["id"])
            or entry["id"] in events
            or entry["kind"]
            not in {"status_changed", "metadata_updated", "participant_linked", "speaker_assigned"}
            or not isinstance(entry["occurred_at"], str)
            or not isinstance(entry["payload"], str)
            or entry["recorded_duration_ms"] is not None
            and (
                type(entry["recorded_duration_ms"]) is not int or entry["recorded_duration_ms"] < 0
            )
        ):
            raise ValueError("Invalid lifecycle event")
        datetime.fromisoformat(entry["occurred_at"])
        payload = json.loads(entry["payload"])
        if not isinstance(payload, dict) or set(payload) - {"from", "to", "fields"}:
            raise ValueError("Invalid lifecycle event payload")
        for key in ("from", "to"):
            if payload.get(key) is not None:
                MeetingStatus(payload[key])
        if "fields" in payload and (
            not isinstance(payload["fields"], list)
            or len(payload["fields"]) > 2
            or any(
                not isinstance(value, str) or value not in {"title", "notes"}
                for value in payload["fields"]
            )
        ):
            raise ValueError("Invalid lifecycle event edited fields")
        events.add(entry["id"])
    assets = set()
    for entry in state["audio_assets"]:
        asset = AudioAsset(**entry)
        if (
            not _state_uuid(asset.id)
            or asset.id in assets
            or not _state_digest(asset.sha256)
            or asset.status not in {"available", "missing", "corrupt"}
            or not isinstance(asset.relative_path, str)
            or not isinstance(asset.codec, str)
            or any(
                type(entry[key]) is not int or entry[key] < 0
                for key in ("size_bytes", "duration_ms")
            )
            or any(
                type(entry[key]) is not int or entry[key] <= 0
                for key in ("sample_rate", "channels")
            )
        ):
            raise ValueError("Invalid lifecycle audio asset")
        assets.add(asset.id)
    origins = set()
    for entry in state["ingest_sources"]:
        source = IngestSource(**entry)
        origin_key = (source.origin_path, source.origin_fingerprint)
        if (
            not _state_uuid(source.id)
            or not isinstance(source.origin_path, str)
            or not source.origin_path
            or not _state_digest(source.origin_fingerprint)
            or source.revision_id is not None
            and not _state_uuid(source.revision_id)
            or origin_key in origins
        ):
            raise ValueError("Invalid lifecycle origin")
        origins.add(origin_key)
    return cast(dict[str, Any], state)


def read_meeting_state(directory: Path, name: str, meeting_id: str) -> dict[str, Any]:
    """Keep the schema-1 caller contract while validating both envelope versions."""
    return cast(dict[str, Any], read_meeting_snapshot(directory, name, meeting_id)["meeting"])


PORTABLE_GROUP_NAME = ".portable-group.json"
PORTABLE_RECEIPT_NAME = "portable-catalog.json"
_PORTABLE_WAV = re.compile(r"audio_[0-9a-f]{32}\.wav")


@dataclass(frozen=True, slots=True)
class PortableGroup:
    operation_id: str
    meeting_id: str
    revision_ids: tuple[str, ...]
    group_sha256: str | None = None

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "operation_id": self.operation_id,
            "meeting_id": self.meeting_id,
            "revision_ids": list(self.revision_ids),
        }


@dataclass(frozen=True, slots=True)
class PortableReceipt:
    operation_id: str
    meeting_id: str
    revision_id: str
    hashes: dict[str, str]
    receipt_sha256: str
    snapshot: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PortableCompletionMember:
    revision_id: str
    receipt_sha256: str
    hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class PortableCompletion:
    operation_id: str
    meeting_id: str
    members: tuple[PortableCompletionMember, ...]
    group_sha256: str
    state_name: str
    state_sha256: str
    state: dict[str, Any] | None = None

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "operation_id": self.operation_id,
            "meeting_id": self.meeting_id,
            "members": [asdict(member) for member in self.members],
            "group_sha256": self.group_sha256,
            "state_name": self.state_name,
            "state_sha256": self.state_sha256,
        }


def portable_completion_name(operation_id: str) -> str:
    if not _state_uuid(operation_id):
        raise ValueError("Invalid portable import operation identity")
    return f"portable-group-{operation_id}.json"


def _portable_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate key in portable publication proof")
        value[key] = item
    return value


def _portable_document(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_portable_json_object)
    if not isinstance(value, dict):
        raise ValueError("Portable publication proof must be an object")
    return value


def _portable_role_document(directory: Path, name: str) -> tuple[dict[str, Any], str]:
    raw = session_artifact_path(directory, name).read_bytes()
    return _portable_document(raw), hashlib.sha256(raw).hexdigest()


def _portable_fields(value: object, names: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("Invalid portable publication proof fields")
    return value


def _portable_envelope(
    value: dict[str, Any], *, version: int, operation_id: str, meeting_id: str
) -> None:
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != version
        or not _state_uuid(value["operation_id"])
        or value["operation_id"] != operation_id
        or not _state_uuid(value["meeting_id"])
        or value["meeting_id"] != meeting_id
    ):
        raise ValueError("Portable publication proof has a mismatched envelope")


def _portable_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(name, str)
        or _PORTABLE_WAV.fullmatch(name) is None
        or not _state_digest(digest)
        for name, digest in value.items()
    ):
        raise ValueError("Invalid portable revision artifact hashes")
    return dict(value)


def read_portable_group(meeting_dir: Path, meeting_id: str) -> PortableGroup:
    """Read the separately managed registration for an ordered import group."""
    value, digest = _portable_role_document(meeting_dir, PORTABLE_GROUP_NAME)
    _portable_fields(value, {"schema_version", "operation_id", "meeting_id", "revision_ids"})
    _portable_envelope(value, version=1, operation_id=value["operation_id"], meeting_id=meeting_id)
    revisions = value["revision_ids"]
    if (
        meeting_dir.name != meeting_id
        or not isinstance(revisions, list)
        or any(not _state_uuid(revision_id) for revision_id in revisions)
        or len(revisions) != len(set(revisions))
    ):
        raise ValueError("Invalid portable import group membership")
    return PortableGroup(value["operation_id"], meeting_id, tuple(revisions), digest)


def _portable_model_fields(value: object, model: type) -> dict[str, Any]:
    return _portable_fields(value, {item.name for item in fields(model)})


def _portable_table(value: object, model: type) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Portable receipt table must be an array")
    return [_portable_model_fields(row, model) for row in value]


def _portable_receipt_audio(
    value: dict[str, Any], group: PortableGroup, revision_id: str
) -> dict[str, str]:
    audio = {}
    local_wavs = {}
    revision_ids = set(group.revision_ids)
    for asset in _portable_table(value["audio_assets"], AudioAsset):
        identity = asset["id"]
        if (
            not _state_uuid(identity)
            or identity in audio
            or asset["meeting_id"] != group.meeting_id
            or not _state_digest(asset["sha256"])
            or asset["status"] not in ("available", "missing")
            or not isinstance(asset["relative_path"], str)
        ):
            raise ValueError("Invalid portable receipt audio identity")
        basename = f"audio_{UUID(identity).hex}.wav"
        parts = asset["relative_path"].split("/")
        if (
            len(parts) != 5
            or parts[:3] != ["meetings", group.meeting_id, "revisions"]
            or parts[3] not in revision_ids
            or parts[4] != basename
        ):
            raise ValueError("Portable receipt audio is outside its registered group")
        if parts[3] == revision_id and asset["status"] == "available":
            local_wavs[basename] = asset["sha256"]
        audio[identity] = asset

    linked = set()
    names = set()
    for ordinal, link in enumerate(
        _portable_table(value["revision_audio_assets"], RevisionAudioAsset)
    ):
        if (
            link["meeting_id"] != group.meeting_id
            or link["revision_id"] != revision_id
            or not _state_uuid(link["asset_id"])
            or link["asset_id"] not in audio
            or type(link["ordinal"]) is not int
            or link["ordinal"] != ordinal
            or not isinstance(link["source_file"], str)
            or link["source_file"] in names
        ):
            raise ValueError("Invalid portable receipt audio link")
        validate_artifact_filename(link["source_file"])
        linked.add(link["asset_id"])
        names.add(link["source_file"])
    if linked != set(audio):
        raise ValueError("Portable receipt audio is missing its revision links")
    return local_wavs


def read_portable_receipt(revision_dir: Path, group: PortableGroup) -> PortableReceipt:
    """Read a group-bound receipt without interpreting ordinary publication formats."""
    revision_id = revision_dir.name
    value, digest = _portable_role_document(revision_dir, PORTABLE_RECEIPT_NAME)
    _portable_fields(
        value,
        {
            "schema_version",
            "operation_id",
            "meeting_id",
            "revision_id",
            "meeting",
            "revision",
            "segments",
            "audio_assets",
            "revision_audio_assets",
            "ingest_sources",
            "hashes",
        },
    )
    _portable_envelope(
        value, version=4, operation_id=group.operation_id, meeting_id=group.meeting_id
    )
    meeting = _portable_model_fields(value["meeting"], Meeting)
    revision = _portable_model_fields(value["revision"], TranscriptRevision)
    if (
        revision_id not in group.revision_ids
        or value["revision_id"] != revision_id
        or meeting["id"] != group.meeting_id
        or revision["meeting_id"] != group.meeting_id
        or revision["id"] != revision_id
        or type(revision["revision_number"]) is not int
        or revision["revision_number"] < 1
        or not _state_digest(revision["content_hash"])
    ):
        raise ValueError("Portable receipt revision identity disagrees with its group")
    segments = _portable_table(value["segments"], Segment)
    segment_ids = set()
    for ordinal, segment in enumerate(segments):
        if (
            segment["revision_id"] != revision_id
            or not _state_uuid(segment["id"])
            or segment["id"] in segment_ids
            or type(segment["ordinal"]) is not int
            or segment["ordinal"] != ordinal
            or segment["speaker_key"] is not None
            and not isinstance(segment["speaker_key"], str)
        ):
            raise ValueError("Invalid portable receipt segment identity")
        segment_ids.add(segment["id"])
    sources = _portable_table(value["ingest_sources"], IngestSource)
    if sources and (
        revision_id != group.revision_ids[0]
        or len(sources) != 1
        or sources[0]["meeting_id"] != group.meeting_id
        or sources[0]["revision_id"] is not None
        or sources[0]["origin_path"] != revision["origin"]
        or not _state_uuid(sources[0]["id"])
        or not _state_digest(sources[0]["origin_fingerprint"])
    ):
        raise ValueError("Invalid portable receipt origin")
    hashes = _portable_hashes(value["hashes"])
    if hashes != _portable_receipt_audio(value, group, revision_id):
        raise ValueError("Portable receipt artifact hashes disagree with its audio")
    return PortableReceipt(group.operation_id, group.meeting_id, revision_id, hashes, digest, value)


def read_portable_completion(states_dir: Path, group: PortableGroup) -> PortableCompletion:
    """Read completion evidence and validate the exact measured lifecycle snapshot."""
    name = portable_completion_name(group.operation_id)
    value, _ = _portable_role_document(states_dir, name)
    _portable_fields(
        value,
        {
            "schema_version",
            "operation_id",
            "meeting_id",
            "members",
            "group_sha256",
            "state_name",
            "state_sha256",
        },
    )
    _portable_envelope(
        value, version=1, operation_id=group.operation_id, meeting_id=group.meeting_id
    )
    state_name = value["state_name"]
    if (
        not _state_digest(value["group_sha256"])
        or value["group_sha256"] != group.group_sha256
        or not isinstance(state_name, str)
        or re.fullmatch(r"[1-9][0-9]*-[0-9a-f-]{36}\.json", state_name) is None
        or not _state_digest(value["state_sha256"])
        or not isinstance(value["members"], list)
    ):
        raise ValueError("Invalid portable import completion binding")
    state_version = _meeting_state_version(state_name)
    members = []
    for item in value["members"]:
        member = _portable_fields(item, {"revision_id", "receipt_sha256", "hashes"})
        if not _state_uuid(member["revision_id"]) or not _state_digest(member["receipt_sha256"]):
            raise ValueError("Invalid portable import completion member")
        members.append(
            PortableCompletionMember(
                member["revision_id"], member["receipt_sha256"], _portable_hashes(member["hashes"])
            )
        )
    if tuple(member.revision_id for member in members) != group.revision_ids:
        raise ValueError("Portable completion membership differs from its registration")

    raw = session_artifact_path(states_dir, state_name).read_bytes()
    state_sha256 = hashlib.sha256(raw).hexdigest()
    if state_sha256 != value["state_sha256"]:
        raise ValueError("Portable lifecycle snapshot digest mismatch")
    state = _validate_meeting_snapshot(_portable_document(raw), state_version, group.meeting_id)
    allowed_revisions = set(group.revision_ids)
    active_revision = state["meeting"]["active_revision_id"]
    if (
        type(state["schema_version"]) is not int
        or state["schema_version"] != 2
        or active_revision is not None
        and active_revision not in allowed_revisions
    ):
        raise ValueError("Portable completion requires a complete matching lifecycle snapshot")
    if any(item["revision_id"] not in allowed_revisions for item in state["speaker_assignments"]):
        raise ValueError("Portable lifecycle assignment refers to a revision outside the group")
    sources = state["ingest_sources"]
    if len(sources) != 1 or sources[0]["revision_id"] is not None:
        raise ValueError("Portable lifecycle snapshot requires one meeting-level source")
    return PortableCompletion(
        group.operation_id,
        group.meeting_id,
        tuple(members),
        value["group_sha256"],
        state_name,
        state_sha256,
        state,
    )


def validate_portable_group_members(
    group: PortableGroup, completion: PortableCompletion, receipts: Iterable[PortableReceipt]
) -> None:
    """Validate every registered member before the caller restores any catalog rows."""
    state = completion.state
    if (
        completion.operation_id != group.operation_id
        or completion.meeting_id != group.meeting_id
        or completion.group_sha256 != group.group_sha256
        or state is None
        or tuple(member.revision_id for member in completion.members) != group.revision_ids
    ):
        raise ValueError("Portable completion binding differs from its registration")
    receipts_by_id = {}
    for receipt in receipts:
        if (
            receipt.operation_id != group.operation_id
            or receipt.meeting_id != group.meeting_id
            or receipt.revision_id in receipts_by_id
        ):
            raise ValueError("Mixed or duplicated portable group receipts")
        receipts_by_id[receipt.revision_id] = receipt
    if set(receipts_by_id) != set(group.revision_ids):
        raise ValueError("Portable group is missing its exact receipt set")

    previous_number = 0
    state_assets = {asset["id"]: asset for asset in state["audio_assets"]}
    first_revision_by_asset = {}
    state_origin = state["ingest_sources"][0]
    speakers_by_revision: dict[str, set[str]] = {}
    for member in completion.members:
        receipt = receipts_by_id[member.revision_id]
        if (
            member.receipt_sha256 != receipt.receipt_sha256
            or member.hashes != receipt.hashes
            or receipt.snapshot["meeting"] != state["meeting"]
            or receipt.snapshot["revision"]["origin"] != state_origin["origin_path"]
        ):
            raise ValueError("Portable receipt evidence differs from its completion proof")
        number = receipt.snapshot["revision"]["revision_number"]
        if number <= previous_number:
            raise ValueError("Portable receipt revision order differs from its registration")
        previous_number = number
        speakers_by_revision[member.revision_id] = {
            segment["speaker_key"]
            for segment in receipt.snapshot["segments"]
            if segment["speaker_key"] is not None
        }
        for asset in receipt.snapshot["audio_assets"]:
            if state_assets.get(asset["id"]) != asset:
                raise ValueError("Portable receipt audio differs from its lifecycle snapshot")
            first_revision_by_asset.setdefault(asset["id"], member.revision_id)
        sources = receipt.snapshot["ingest_sources"]
        if sources and sources != [state_origin]:
            raise ValueError("Portable receipt source differs from its lifecycle snapshot")
    if set(first_revision_by_asset) != set(state_assets):
        raise ValueError("Portable lifecycle snapshot contains unreceipted audio")
    for asset_id, asset in state_assets.items():
        if asset["relative_path"].split("/")[3] != first_revision_by_asset[asset_id]:
            raise ValueError("Portable audio owner is not its first referencing revision")
    for assignment in state["speaker_assignments"]:
        if assignment["speaker_key"] not in speakers_by_revision[assignment["revision_id"]]:
            raise ValueError("Portable lifecycle assignment has no matching scoped speaker")


def write_durable_json(path: Path, value: object) -> None:
    """Flush a small owned journal/receipt before its same-volume atomic replace."""
    with staged_artifact_path(path) as staged:
        with staged.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())


def managed_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    """Validate every nested component using the shared output path policy."""
    if not relative or "\\" in relative or PureWindowsPath(relative).drive:
        raise UnsafeSessionPath("Invalid managed relative path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or any(p in {".", ".."} for p in parts):
        raise UnsafeSessionPath("Invalid managed relative path")
    current = root
    validate_session_directory(current)
    for index, part in enumerate(parts):
        # Validate a directory name as a filename before creating or descending it.
        if index < len(parts) - 1 or directory:
            if not (current / part).exists():
                session_artifact_path(current, part)
            current = current / part
            validate_session_directory(current)
        else:
            current = session_artifact_path(current, part)
    return current


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_origin(directory: Path) -> str:
    directory = directory.absolute()
    validate_session_directory(directory)
    return os.path.normcase(str(directory.resolve()))


@dataclass(frozen=True)
class PendingIntent:
    revision_of: str | None
    intent: str
    publication_id: str | None = None
    source_kind: str = "session"
    hashes: tuple[tuple[str, str], ...] | None = None


def read_pending_intent(directory: Path, meeting_id: str | None) -> PendingIntent:
    """Read only the typed application-owned replay envelope, never provider work."""
    pending = json.loads(
        session_artifact_path(directory, "memory_pending.json").read_text(encoding="utf-8")
    )
    if not isinstance(pending, dict):
        raise ValueError("Invalid pending catalog envelope")
    expected = pending.get("revision_of")
    intent = pending.get("intent", "ingest")
    version = pending.get("schema_version")
    publication_id = pending.get("publication_id")
    source_kind = pending.get("source_kind", "session")
    if (
        type(version) is not int
        or version not in {1, 2, 3}
        or pending.get("meeting_id") != meeting_id
        or not isinstance(intent, str)
        or intent not in {"ingest", "status"}
        or expected is not None
        and (not isinstance(expected, str) or str(UUID(expected)) != expected)
        or intent == "status"
        and expected is None
        or version in {2, 3}
        and (not isinstance(publication_id, str) or str(UUID(publication_id)) != publication_id)
        or not isinstance(source_kind, str)
        or source_kind not in {"session", "legacy_audio_only"}
    ):
        raise ValueError("Invalid pending catalog intent")
    hashes = pending.get("hashes") if version == 3 else None
    if version == 3:
        if (
            not isinstance(hashes, dict)
            or "session.json" not in hashes
            or any(
                not isinstance(name, str)
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for name, digest in hashes.items()
            )
        ):
            raise ValueError("Invalid pending catalog evidence seal")
        validate_artifact_roles(directory, hashes)
    return PendingIntent(
        expected,
        intent,
        publication_id if version in {2, 3} else None,
        source_kind,
        tuple(sorted(hashes.items())) if hashes is not None else None,
    )


def publication_hashes(
    directory: Path, metadata: SessionMetadata, *, staged_metadata: Path | None = None
) -> dict[str, str]:
    """Measure exactly the local evidence authorized for final catalog publication.

    The writer supplies its own staged metadata, so the seal measures actual bytes,
    including platform newline encoding, before replacing the final manifest.
    """
    declared_artifact_roles(metadata)
    names = {"session.json", *metadata.output_files}
    names.update(item.audio_file for item in metadata.capture_sources if item.audio_file)
    if metadata.recoverable_audio:
        names.add(metadata.recoverable_audio)
    paths = {name: session_artifact_path(directory, name) for name in names}
    for name in ("raw_transcript.txt", "transcript.txt", "segments.json"):
        path = session_artifact_path(directory, name)
        if path.exists():
            paths[name] = path
    if not metadata.recoverable_audio:
        audio = session_artifact_path(directory, "audio.wav")
        if audio.exists():
            paths["audio.wav"] = audio
    if staged_metadata is not None:
        paths["session.json"] = staged_metadata
    return {name: file_hash(path) for name, path in paths.items() if path.is_file()}


def validate_pending_evidence(
    directory: Path, metadata: SessionMetadata, pending: PendingIntent
) -> None:
    if pending.hashes is not None and (
        metadata.status == "pending"
        or publication_hashes(directory, metadata) != dict(pending.hashes)
    ):
        raise ValueError("Pending catalog evidence is uncommitted or changed; intent retained")


def _milliseconds(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return round(value * 1000) if math.isfinite(value) and value >= 0 else None


@dataclass(frozen=True)
class SessionArtifacts:
    directory: Path
    metadata: SessionMetadata
    files: tuple[tuple[str, str], ...]
    audio: tuple[dict[str, Any], ...]
    raw_text: str
    presentation_text: str
    segments: tuple[dict[str, Any], ...]
    origin_path: str
    fingerprint: str
    content_hash: str


def read_artifacts(directory: Path) -> SessionArtifacts:
    return _read_artifacts(directory)


def read_retry_artifacts(directory: Path) -> SessionArtifacts:
    """Trusted explicit M1 reprocessing keeps the source's optional opaque ID."""
    return _read_artifacts(directory, trusted_legacy=True)


def _read_artifacts(directory: Path, *, trusted_legacy: bool = False) -> SessionArtifacts:
    directory = directory.absolute()
    validate_session_directory(directory)
    metadata = SessionMetadata.model_validate_json(
        session_artifact_path(directory, "session.json").read_text(encoding="utf-8-sig")
    )
    if metadata.schema_version != 1:
        raise ValueError("Unsupported session schema")
    if (
        not trusted_legacy
        and metadata.meeting_id is not None
        and str(UUID(metadata.meeting_id)) != metadata.meeting_id
    ):
        raise ValueError("Invalid meeting UUID")
    audio_references = {p.audio_file for p in metadata.capture_sources if p.audio_file}
    if metadata.recoverable_audio:
        audio_references.add(metadata.recoverable_audio)
    declared_artifact_roles(metadata)
    names = ["session.json", *metadata.output_files, *audio_references]
    # Validate ALL manifest references before any referenced file is read.
    paths = {name: session_artifact_path(directory, name) for name in names}
    for name in ("raw_transcript.txt", "transcript.txt", "segments.json"):
        path = session_artifact_path(directory, name)
        if path.exists():
            paths[name] = path
    audio_names = [metadata.recoverable_audio] if metadata.recoverable_audio else []
    if trusted_legacy and audio_names and not paths[audio_names[0]].exists():
        # An explicit M1 retry may reconstruct a missing derived mix from its
        # validated raw sources; preserve those as the original revision evidence.
        audio_names = []
    if not audio_names and session_artifact_path(directory, "audio.wav").exists():
        audio_names = ["audio.wav"]
        paths["audio.wav"] = session_artifact_path(directory, "audio.wav")
    if not audio_names:
        audio_names = [p.audio_file for p in metadata.capture_sources if p.audio_file]
    audio = []
    for name in dict.fromkeys(audio_names):
        path = paths[name]
        info = sf.info(path)
        audio.append(
            dict(
                name=name,
                sha256=file_hash(path),
                size_bytes=path.stat().st_size,
                codec=info.subtype or info.format,
                sample_rate=info.samplerate,
                channels=info.channels,
                duration_ms=round(info.duration * 1000),
            )
        )
    if not audio:
        raise ValueError("Session has no recoverable audio")
    presentation = paths.get("transcript.txt")
    raw = paths.get("raw_transcript.txt") or presentation
    raw_text = raw.read_text(encoding="utf-8-sig") if raw and raw.exists() else ""
    presentation_text = (
        presentation.read_text(encoding="utf-8-sig")
        if presentation and presentation.exists()
        else raw_text
    )
    evidence = []
    segment_path = paths.get("segments.json")
    if segment_path and segment_path.exists():
        document = json.loads(segment_path.read_text(encoding="utf-8-sig"))
        if document.get("schema_version", 1) != 1:
            raise ValueError("Unsupported segment schema")
        for ordinal, item in enumerate(document["segments"]):
            start, end = _milliseconds(item.get("start")), _milliseconds(item.get("end"))
            if start is not None and end is not None and end < start:
                start = end = None
            timing = item.get("timing_source") or "provider"
            if start is None and end is None or timing == "absent":
                start = end = None
                timing = "absent"
            if timing not in {"provider", "imported_explicit", "absent"}:
                raise ValueError("Unknown segment timing source")
            speaker = item.get("speaker")
            scope = item.get("speaker_scope") or f"chunk:{item.get('chunk_index', 'session')}"
            key = json.dumps([scope, speaker], ensure_ascii=False) if speaker else None
            evidence.append(
                dict(
                    ordinal=ordinal,
                    text=item["text"],
                    start_ms=start,
                    end_ms=end,
                    timing_source=timing,
                    speaker_key=key,
                )
            )
    else:
        evidence = [
            dict(
                ordinal=i,
                text=text.strip(),
                start_ms=None,
                end_ms=None,
                timing_source="absent",
                speaker_key=None,
            )
            for i, text in enumerate(raw_text.split("\n\n"))
            if text.strip()
        ]
    files = tuple(sorted((name, file_hash(path)) for name, path in paths.items() if path.is_file()))
    fingerprint = hashlib.sha256(json.dumps(files).encode()).hexdigest()
    content = [
        raw_text,
        presentation_text,
        evidence,
        [a["sha256"] for a in audio],
        metadata.status,
        metadata.transcription_model,
    ]
    content_hash = hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()
    return SessionArtifacts(
        directory,
        metadata,
        files,
        tuple(audio),
        raw_text,
        presentation_text,
        tuple(evidence),
        normalized_origin(directory),
        fingerprint,
        content_hash,
    )
