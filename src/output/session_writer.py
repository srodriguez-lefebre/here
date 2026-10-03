from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from here.output.markdown import render_transcript_markdown
from here.output.metadata import (
    CaptureSourceMetadata,
    ChunkMetadata,
    ChunkMetadataDocument,
    ErrorMetadata,
    ErrorMetadataDocument,
    SessionEventMetadata,
    SessionEventMetadataDocument,
    SessionMetadata,
    TranscriptSegmentDocument,
    build_session_metadata,
)
from here.recording.models import RecordingSession
from here.transcription.segments import TranscriptSegment

TRANSCRIPT_ENCODING = "utf-8-sig"
METADATA_ENCODING = "utf-8"
MARKDOWN_ENCODING = "utf-8"
TRANSCRIPT_FILE = "transcript.txt"
MARKDOWN_FILE = "transcript.md"
METADATA_FILE = "session.json"
CHUNKS_FILE = "chunks.json"
AUDIO_FILE = "audio.wav"
ERRORS_FILE = "errors.json"
EVENTS_FILE = "events.json"
SEGMENTS_FILE = "segments.json"


@dataclass(slots=True)
class SessionArtifactPaths:
    session_dir: Path
    transcript_path: Path
    markdown_path: Path
    metadata_path: Path
    chunks_path: Path
    errors_path: Path
    events_path: Path
    audio_path: Path | None
    metadata: SessionMetadata
    chunks: ChunkMetadataDocument
    errors: ErrorMetadataDocument
    segments_path: Path | None = None


def _session_id_from_datetime(value: datetime) -> str:
    return value.strftime("%Y%m%d_%H%M%S")


def _reserve_session_dir(target_dir: Path, session_id: str) -> tuple[str, Path]:
    candidate_id = session_id
    candidate_dir = target_dir / candidate_id
    suffix = 2
    while candidate_dir.exists():
        candidate_id = f"{session_id}_{suffix}"
        candidate_dir = target_dir / candidate_id
        suffix += 1
    candidate_dir.mkdir(parents=True)
    return candidate_id, candidate_dir


def create_session_dir(target_dir: Path, completed_at: datetime) -> tuple[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    return _reserve_session_dir(target_dir, _session_id_from_datetime(completed_at))


def _reserve_artifact_path(destination: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=suffix
    )
    os.close(descriptor)
    return Path(name)


def _stage_document(destination: Path, content: str) -> Path:
    staged = _reserve_artifact_path(destination, ".stage")
    try:
        staged.write_text(content, encoding=METADATA_ENCODING)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _publish_metadata_and_segments(
    metadata_path: Path, metadata_json: str, segments_json: str | None
) -> None:
    """Publish metadata last and roll back segment changes on handled I/O failure.

    This is not a multi-file transaction against process termination or concurrent writers.
    """
    segments_path = metadata_path.parent / SEGMENTS_FILE
    staged_paths: list[Path] = []
    backup: Path | None = None
    backup_moved = False
    segment_published = False
    committed = False
    try:
        staged_segments = None
        if segments_json is not None:
            staged_segments = _stage_document(segments_path, segments_json)
            staged_paths.append(staged_segments)
        staged_metadata = _stage_document(metadata_path, metadata_json)
        staged_paths.append(staged_metadata)

        if segments_path.is_symlink() or segments_path.exists():
            backup = _reserve_artifact_path(segments_path, ".backup")
            # Rename the entry itself, without reading through an external link.
            segments_path.replace(backup)
            backup_moved = True
        if staged_segments is not None:
            staged_segments.replace(segments_path)
            segment_published = True
        staged_metadata.replace(metadata_path)
        committed = True
    except OSError:
        if backup_moved:
            assert backup is not None
            try:
                backup.replace(segments_path)
            except OSError as rollback_error:
                rollback_error.add_note(f"Previous segment evidence retained at {backup}")
                raise
            backup_moved = False
        elif segment_published:
            segments_path.unlink(missing_ok=True)
        raise
    finally:
        for staged in staged_paths:
            staged.unlink(missing_ok=True)
        if backup is not None and (committed or not backup_moved):
            backup.unlink(missing_ok=True)


def write_session_artifacts(
    *,
    session: RecordingSession,
    target_dir: Path,
    completed_at: datetime,
    transcription_model: str,
    cleanup_model: str,
    cleanup_enabled: bool,
    alt_model_used: bool,
    live_pipeline_attempted: bool,
    live_pipeline_used: bool,
    fallback_used: bool,
    transcript_text: str | None = None,
    chunks: list[ChunkMetadata] | None = None,
    errors: list[ErrorMetadata] | None = None,
    status: Literal["pending", "completed", "failed", "cancelled"] = "completed",
    failure_stage: str | None = None,
    recoverable_audio: str | None = None,
    session_dir: Path | None = None,
    session_id: str | None = None,
    events: list[SessionEventMetadata] | None = None,
    started_at: datetime | None = None,
    total_paused_seconds: float = 0.0,
    capture_sources: list[CaptureSourceMetadata] | None = None,
    segments: list[TranscriptSegment] | None = None,
) -> SessionArtifactPaths:
    if session_dir is None:
        session_id, session_dir = create_session_dir(target_dir, completed_at)
    else:
        session_dir.mkdir(parents=True, exist_ok=True)
        session_id = session_id or session_dir.name

    output_files = [METADATA_FILE, CHUNKS_FILE]
    if segments is not None:
        output_files.append(SEGMENTS_FILE)
    if transcript_text is not None:
        output_files = [TRANSCRIPT_FILE, MARKDOWN_FILE, *output_files]
    if recoverable_audio is not None:
        output_files.append(recoverable_audio)
    if errors:
        output_files.append(ERRORS_FILE)
    if events:
        output_files.append(EVENTS_FILE)
    output_files.extend(source.audio_file for source in capture_sources or [] if source.audio_file)

    metadata = build_session_metadata(
        session_id=session_id,
        session=session,
        completed_at=completed_at,
        transcription_model=transcription_model,
        cleanup_model=cleanup_model,
        cleanup_enabled=cleanup_enabled,
        alt_model_used=alt_model_used,
        live_pipeline_attempted=live_pipeline_attempted,
        live_pipeline_used=live_pipeline_used,
        fallback_used=fallback_used,
        status=status,
        failure_stage=failure_stage,
        recoverable_audio=recoverable_audio,
        output_files=output_files,
        started_at=started_at,
        total_paused_seconds=total_paused_seconds,
        capture_sources=capture_sources,
    )

    transcript_path = session_dir / TRANSCRIPT_FILE
    markdown_path = session_dir / MARKDOWN_FILE
    metadata_path = session_dir / METADATA_FILE
    chunks_path = session_dir / CHUNKS_FILE
    errors_path = session_dir / ERRORS_FILE
    events_path = session_dir / EVENTS_FILE
    audio_path = session_dir / recoverable_audio if recoverable_audio is not None else None
    chunks_document = ChunkMetadataDocument(chunks=chunks or [])
    errors_document = ErrorMetadataDocument(errors=errors or [])
    events_document = SessionEventMetadataDocument(events=events or [])
    segments_path = session_dir / SEGMENTS_FILE if segments is not None else None
    segments_json = (
        TranscriptSegmentDocument(
            segments=[asdict(segment) for segment in segments]
        ).model_dump_json(indent=2)
        if segments is not None
        else None
    )

    if transcript_text is not None:
        transcript_path.write_text(transcript_text, encoding=TRANSCRIPT_ENCODING)
        markdown_path.write_text(
            render_transcript_markdown(metadata, transcript_text),
            encoding=MARKDOWN_ENCODING,
        )
    chunks_path.write_text(
        chunks_document.model_dump_json(indent=2),
        encoding=METADATA_ENCODING,
    )
    if errors:
        errors_path.write_text(
            errors_document.model_dump_json(indent=2),
            encoding=METADATA_ENCODING,
        )
    elif errors_path.exists():
        errors_path.unlink()
    if events:
        events_path.write_text(
            events_document.model_dump_json(indent=2),
            encoding=METADATA_ENCODING,
        )
    elif events_path.exists():
        events_path.unlink()

    _publish_metadata_and_segments(metadata_path, metadata.model_dump_json(indent=2), segments_json)

    return SessionArtifactPaths(
        session_dir=session_dir,
        transcript_path=transcript_path,
        markdown_path=markdown_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path,
        errors_path=errors_path,
        events_path=events_path,
        audio_path=audio_path,
        metadata=metadata,
        chunks=chunks_document,
        errors=errors_document,
        segments_path=segments_path,
    )
