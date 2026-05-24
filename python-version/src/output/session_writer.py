from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from here.output.markdown import render_transcript_markdown
from here.output.metadata import (
    ChunkMetadata,
    ChunkMetadataDocument,
    ErrorMetadata,
    ErrorMetadataDocument,
    SessionMetadata,
    build_session_metadata,
)
from here.recording.models import RecordingSession

TRANSCRIPT_ENCODING = "utf-8-sig"
METADATA_ENCODING = "utf-8"
MARKDOWN_ENCODING = "utf-8"
TRANSCRIPT_FILE = "transcript.txt"
MARKDOWN_FILE = "transcript.md"
METADATA_FILE = "session.json"
CHUNKS_FILE = "chunks.json"
AUDIO_FILE = "audio.wav"
ERRORS_FILE = "errors.json"


@dataclass(slots=True)
class SessionArtifactPaths:
    session_dir: Path
    transcript_path: Path
    markdown_path: Path
    metadata_path: Path
    chunks_path: Path
    errors_path: Path
    audio_path: Path
    metadata: SessionMetadata
    chunks: ChunkMetadataDocument
    errors: ErrorMetadataDocument


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
    status: str = "completed",
    failure_stage: str | None = None,
    recoverable_audio: str | None = None,
    session_dir: Path | None = None,
    session_id: str | None = None,
) -> SessionArtifactPaths:
    if session_dir is None:
        session_id, session_dir = create_session_dir(target_dir, completed_at)
    else:
        session_dir.mkdir(parents=True, exist_ok=True)
        session_id = session_id or session_dir.name

    output_files = [METADATA_FILE, CHUNKS_FILE]
    if transcript_text is not None:
        output_files = [TRANSCRIPT_FILE, MARKDOWN_FILE, *output_files]
    if recoverable_audio is not None:
        output_files.append(recoverable_audio)
    if errors:
        output_files.append(ERRORS_FILE)

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
    )

    transcript_path = session_dir / TRANSCRIPT_FILE
    markdown_path = session_dir / MARKDOWN_FILE
    metadata_path = session_dir / METADATA_FILE
    chunks_path = session_dir / CHUNKS_FILE
    errors_path = session_dir / ERRORS_FILE
    audio_path = session_dir / AUDIO_FILE
    chunks_document = ChunkMetadataDocument(chunks=chunks or [])
    errors_document = ErrorMetadataDocument(errors=errors or [])

    if transcript_text is not None:
        transcript_path.write_text(transcript_text, encoding=TRANSCRIPT_ENCODING)
        markdown_path.write_text(
            render_transcript_markdown(metadata, transcript_text),
            encoding=MARKDOWN_ENCODING,
        )
    metadata_path.write_text(
        metadata.model_dump_json(indent=2),
        encoding=METADATA_ENCODING,
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

    return SessionArtifactPaths(
        session_dir=session_dir,
        transcript_path=transcript_path,
        markdown_path=markdown_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path,
        errors_path=errors_path,
        audio_path=audio_path,
        metadata=metadata,
        chunks=chunks_document,
        errors=errors_document,
    )
