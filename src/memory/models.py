"""Immutable catalog values; source timing and request speaker scopes stay explicit."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MeetingStatus(StrEnum):
    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    DELETING = "deleting"


@dataclass(frozen=True, slots=True)
class Meeting:
    id: str
    title: str = ""
    legacy_session_id: str | None = None
    notes: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    paused_ms: int = 0
    status: MeetingStatus = MeetingStatus.PROCESSING
    active_revision_id: str | None = None
    row_version: int = 1
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class AudioAsset:
    id: str
    meeting_id: str
    relative_path: str
    size_bytes: int
    sha256: str
    codec: str
    sample_rate: int
    channels: int
    duration_ms: int
    status: str = "available"


@dataclass(frozen=True, slots=True)
class RevisionAudioAsset:
    meeting_id: str
    revision_id: str
    asset_id: str
    ordinal: int
    source_file: str


@dataclass(frozen=True, slots=True)
class TranscriptRevision:
    id: str
    meeting_id: str
    revision_number: int
    raw_text: str
    provider: str
    model: str
    origin: str
    content_hash: str
    status: str = "complete"
    presentation_text: str | None = None
    created_at: str = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class Segment:
    id: str
    revision_id: str
    ordinal: int
    text: str
    timing_source: str = "absent"
    start_ms: int | None = None
    end_ms: int | None = None
    speaker_key: str | None = None


@dataclass(frozen=True, slots=True)
class IngestSource:
    id: str
    origin_path: str
    origin_fingerprint: str
    meeting_id: str
    revision_id: str | None = None


@dataclass(frozen=True, slots=True)
class PendingOperation:
    id: str
    kind: str
    phase: str
    meeting_id: str | None = None
    staging_path: str | None = None
    destination_path: str | None = None
    payload_json: str = "{}"
