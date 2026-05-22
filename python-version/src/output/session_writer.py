from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from here.output.markdown import render_transcript_markdown
from here.output.metadata import SessionMetadata, build_session_metadata
from here.recording.models import RecordingSession

TRANSCRIPT_ENCODING = "utf-8-sig"
METADATA_ENCODING = "utf-8"
MARKDOWN_ENCODING = "utf-8"
TRANSCRIPT_FILE = "transcript.txt"
MARKDOWN_FILE = "transcript.md"
METADATA_FILE = "session.json"


@dataclass(slots=True)
class SessionArtifactPaths:
    session_dir: Path
    transcript_path: Path
    markdown_path: Path
    metadata_path: Path
    metadata: SessionMetadata


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


def write_session_artifacts(
    *,
    session: RecordingSession,
    target_dir: Path,
    transcript_text: str,
    completed_at: datetime,
    transcription_model: str,
    cleanup_model: str,
    cleanup_enabled: bool,
    alt_model_used: bool,
    live_pipeline_attempted: bool,
    live_pipeline_used: bool,
    fallback_used: bool,
) -> SessionArtifactPaths:
    target_dir.mkdir(parents=True, exist_ok=True)
    session_id, session_dir = _reserve_session_dir(target_dir, _session_id_from_datetime(completed_at))
    output_files = [TRANSCRIPT_FILE, MARKDOWN_FILE, METADATA_FILE]
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
        output_files=output_files,
    )

    transcript_path = session_dir / TRANSCRIPT_FILE
    markdown_path = session_dir / MARKDOWN_FILE
    metadata_path = session_dir / METADATA_FILE

    transcript_path.write_text(transcript_text, encoding=TRANSCRIPT_ENCODING)
    markdown_path.write_text(
        render_transcript_markdown(metadata, transcript_text),
        encoding=MARKDOWN_ENCODING,
    )
    metadata_path.write_text(
        metadata.model_dump_json(indent=2),
        encoding=METADATA_ENCODING,
    )

    return SessionArtifactPaths(
        session_dir=session_dir,
        transcript_path=transcript_path,
        markdown_path=markdown_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )
