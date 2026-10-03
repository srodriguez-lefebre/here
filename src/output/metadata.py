from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from here.recording.models import RecordedAudioSource, RecordingSession
from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    label: str
    device_name: str | None = None
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float


class CaptureSourceMetadata(SourceMetadata):
    audio_file: str | None = None


class TranscriptSegmentMetadata(BaseModel):
    text: str
    start: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    speaker: str | None = None
    chunk_index: int | None = None
    speaker_scope: str | None = None


class TranscriptSegmentDocument(BaseModel):
    schema_version: int = Field(default=1)
    semantics: Literal["provider_evidence"] = "provider_evidence"
    timing_reference: Literal["recording_seconds"] = "recording_seconds"
    speaker_identity_scope: Literal["request"] = "request"
    segments: list[TranscriptSegmentMetadata]


class SessionMetadata(BaseModel):
    schema_version: int = Field(default=1)
    session_id: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    status: Literal["pending", "completed", "failed", "cancelled"] = "completed"
    failure_stage: str | None = None
    recoverable_audio: str | None = None
    sources: list[SourceMetadata]
    capture_sources: list[CaptureSourceMetadata] = Field(default_factory=list)
    transcription_model: str
    cleanup_model: str
    cleanup_enabled: bool
    alt_model_used: bool
    live_pipeline_attempted: bool
    live_pipeline_used: bool
    fallback_used: bool
    output_files: list[str]
    total_paused_seconds: float = 0.0


class SessionEventMetadata(BaseModel):
    kind: str
    occurred_at: datetime
    recorded_duration_seconds: float
    total_paused_seconds: float = 0.0
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class SessionEventMetadataDocument(BaseModel):
    schema_version: int = Field(default=1)
    events: list[SessionEventMetadata]


class ChunkMetadata(BaseModel):
    index: int
    mode: Literal["live", "offline"]
    start_seconds: float | None
    end_seconds: float | None
    duration_seconds: float | None
    source_count: int
    transcription_started_at: datetime | None
    transcription_finished_at: datetime | None
    status: Literal["completed", "failed"]
    error: str | None = None


class ChunkMetadataDocument(BaseModel):
    schema_version: int = Field(default=1)
    chunks: list[ChunkMetadata]


class ErrorMetadata(BaseModel):
    stage: str
    type: str
    message: str
    cause_type: str | None = None
    cause_message: str | None = None
    retryable: bool
    occurred_at: datetime


class ErrorMetadataDocument(BaseModel):
    schema_version: int = Field(default=1)
    errors: list[ErrorMetadata]


def source_metadata(source: RecordedAudioSource) -> SourceMetadata:
    return SourceMetadata(
        label=source.label,
        device_name=source.device_name,
        sample_rate=source.sample_rate,
        channels=source.channels,
        frames=source.frames,
        duration_seconds=source.duration_seconds,
    )


def capture_metadata(session: RecordingSession) -> list[CaptureSourceMetadata]:
    return [
        CaptureSourceMetadata(**source_metadata(source).model_dump()) for source in session.sources
    ]


def build_session_metadata(
    *,
    session_id: str,
    session: RecordingSession,
    completed_at: datetime,
    transcription_model: str,
    cleanup_model: str,
    cleanup_enabled: bool,
    alt_model_used: bool,
    live_pipeline_attempted: bool,
    live_pipeline_used: bool,
    fallback_used: bool,
    status: Literal["pending", "completed", "failed", "cancelled"] = "completed",
    failure_stage: str | None = None,
    recoverable_audio: str | None = None,
    started_at: datetime | None = None,
    total_paused_seconds: float = 0.0,
    output_files: list[str],
    capture_sources: list[CaptureSourceMetadata] | None = None,
) -> SessionMetadata:
    duration_seconds = session.duration_seconds
    return SessionMetadata(
        session_id=session_id,
        started_at=started_at or completed_at - timedelta(seconds=duration_seconds),
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        status=status,
        failure_stage=failure_stage,
        recoverable_audio=recoverable_audio,
        sources=[source_metadata(source) for source in session.sources],
        capture_sources=capture_sources or [],
        transcription_model=transcription_model,
        cleanup_model=cleanup_model,
        cleanup_enabled=cleanup_enabled,
        alt_model_used=alt_model_used,
        live_pipeline_attempted=live_pipeline_attempted,
        live_pipeline_used=live_pipeline_used,
        fallback_used=fallback_used,
        output_files=output_files,
        total_paused_seconds=total_paused_seconds,
    )
