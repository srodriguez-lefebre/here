from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import pytest
from here.output.metadata import ChunkMetadata
from here.output.session_writer import (
    AUDIO_FILE,
    CHUNKS_FILE,
    MARKDOWN_FILE,
    METADATA_FILE,
    SEGMENTS_FILE,
    TRANSCRIPT_ENCODING,
    TRANSCRIPT_FILE,
    SessionArtifactPaths,
    write_session_artifacts,
)
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.segments import TranscriptSegment


def _session(tmp_path: Path) -> RecordingSession:
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"not real wav")
    return RecordingSession(
        sources=[
            RecordedAudioSource(
                path=audio_path,
                sample_rate=16000,
                channels=1,
                frames=48000,
                label="microphone",
                device_name="Asterisk Nova",
            )
        ]
    )


def test_write_session_artifacts_creates_folder_text_markdown_and_metadata(
    tmp_path: Path,
) -> None:
    artifacts = write_session_artifacts(
        session=_session(tmp_path),
        target_dir=tmp_path / "transcriptions",
        transcript_text="Speaker 1: hola",
        completed_at=datetime(2026, 5, 22, 10, 30, 0),
        transcription_model="gpt-4o-transcribe-diarize",
        cleanup_model="gpt-4.1-mini",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=True,
        live_pipeline_used=True,
        fallback_used=False,
        chunks=[
            ChunkMetadata(
                index=1,
                mode="live",
                start_seconds=0.0,
                end_seconds=3.0,
                duration_seconds=3.0,
                source_count=1,
                transcription_started_at=datetime(2026, 5, 22, 10, 29, 0),
                transcription_finished_at=datetime(2026, 5, 22, 10, 29, 5),
                status="completed",
            )
        ],
    )

    assert artifacts.session_dir.name == "20260522_103000"
    assert artifacts.transcript_path.name == TRANSCRIPT_FILE
    assert artifacts.markdown_path.name == MARKDOWN_FILE
    assert artifacts.metadata_path.name == METADATA_FILE
    assert artifacts.chunks_path.name == CHUNKS_FILE
    assert artifacts.audio_path is None
    assert artifacts.transcript_path.read_text(encoding=TRANSCRIPT_ENCODING) == "Speaker 1: hola"

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["session_id"] == "20260522_103000"
    assert metadata["duration_seconds"] == 3.0
    assert metadata["sources"] == [
        {
            "label": "microphone",
            "device_name": "Asterisk Nova",
            "sample_rate": 16000,
            "channels": 1,
            "frames": 48000,
            "duration_seconds": 3.0,
        }
    ]
    assert metadata["output_files"] == [TRANSCRIPT_FILE, MARKDOWN_FILE, METADATA_FILE, CHUNKS_FILE]
    assert "source.wav" not in artifacts.metadata_path.read_text(encoding="utf-8")

    chunks = json.loads(artifacts.chunks_path.read_text(encoding="utf-8"))
    assert chunks["schema_version"] == 1
    assert chunks["chunks"][0]["mode"] == "live"
    assert chunks["chunks"][0]["source_count"] == 1
    assert chunks["chunks"][0]["status"] == "completed"
    assert "Speaker 1: hola" not in artifacts.chunks_path.read_text(encoding="utf-8")
    assert "gpt-4o-transcribe-diarize" not in artifacts.chunks_path.read_text(encoding="utf-8")

    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "# Recording 2026-05-22 10:30" in markdown
    assert "- Session ID: `20260522_103000`" in markdown
    assert "- microphone (Asterisk Nova): 1 channel(s), 16000 Hz, 3s" in markdown
    assert "Speaker 1: hola" in markdown


def test_write_session_artifacts_uses_unique_session_folder(tmp_path: Path) -> None:
    target_dir = tmp_path / "transcriptions"
    (target_dir / "20260522_103000").mkdir(parents=True)

    artifacts = write_session_artifacts(
        session=_session(tmp_path),
        target_dir=target_dir,
        transcript_text="hola",
        completed_at=datetime(2026, 5, 22, 10, 30, 0),
        transcription_model="gpt-4o-transcribe-diarize",
        cleanup_model="gpt-4.1-mini",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=False,
        live_pipeline_used=False,
        fallback_used=False,
    )

    assert artifacts.session_dir.name == "20260522_103000_2"
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["session_id"] == "20260522_103000_2"
    chunks = json.loads(artifacts.chunks_path.read_text(encoding="utf-8"))
    assert chunks == {"schema_version": 1, "chunks": []}


def test_write_session_artifacts_derives_audio_path_from_recoverable_audio(
    tmp_path: Path,
) -> None:
    artifacts = write_session_artifacts(
        session=_session(tmp_path),
        target_dir=tmp_path / "transcriptions",
        completed_at=datetime(2026, 5, 22, 10, 30, 0),
        transcription_model="gpt-4o-transcribe-diarize",
        cleanup_model="gpt-4.1-mini",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=False,
        live_pipeline_used=False,
        fallback_used=False,
        recoverable_audio=AUDIO_FILE,
    )

    assert artifacts.audio_path == artifacts.session_dir / AUDIO_FILE
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["recoverable_audio"] == AUDIO_FILE
    assert AUDIO_FILE in metadata["output_files"]


def _write_segment_artifacts(
    tmp_path: Path,
    *,
    segments: list[TranscriptSegment] | None,
    session_dir: Path | None = None,
    status: Literal["pending", "completed", "failed", "cancelled"] = "completed",
) -> SessionArtifactPaths:
    return write_session_artifacts(
        session=_session(tmp_path),
        target_dir=tmp_path / "transcriptions",
        completed_at=datetime(2026, 5, 22, 10, 30, 0),
        transcription_model="gpt-4o-transcribe-diarize",
        cleanup_model="gpt-4.1-mini",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=False,
        live_pipeline_used=False,
        fallback_used=False,
        segments=segments,
        session_dir=session_dir,
        status=status,
    )


@pytest.mark.parametrize("status", ["failed", "cancelled", "pending", "completed"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["fresh", "rewrite"])
def test_session_without_segment_evidence_has_no_segment_artifact(
    tmp_path: Path,
    status: Literal["pending", "completed", "failed", "cancelled"],
    rewrite: bool,
) -> None:
    session_dir = None
    if rewrite:
        previous = _write_segment_artifacts(
            tmp_path, segments=[TranscriptSegment("old provider evidence", 0.0, 1.0, "A")]
        )
        assert previous.segments_path is not None
        previous_document = json.loads(previous.segments_path.read_text(encoding="utf-8"))
        assert previous_document["segments"][0]["text"] == "old provider evidence"
        assert previous.metadata.status == "completed"
        session_dir = previous.session_dir

    artifacts = _write_segment_artifacts(
        tmp_path, segments=None, session_dir=session_dir, status=status
    )

    if rewrite:
        assert artifacts.session_dir == session_dir
    assert artifacts.segments_path is None
    assert SEGMENTS_FILE not in artifacts.metadata.output_files
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == status
    assert SEGMENTS_FILE not in metadata["output_files"]
    assert not (artifacts.session_dir / SEGMENTS_FILE).exists()


@pytest.mark.parametrize("rewrite", [False, True], ids=["fresh", "rewrite"])
def test_explicit_empty_segment_evidence_writes_current_document(
    tmp_path: Path, rewrite: bool
) -> None:
    session_dir = None
    if rewrite:
        previous = _write_segment_artifacts(
            tmp_path, segments=[TranscriptSegment("old provider evidence", 0.0, 1.0, "A")]
        )
        assert previous.segments_path is not None
        assert previous.segments_path.exists()
        session_dir = previous.session_dir

    artifacts = _write_segment_artifacts(tmp_path, segments=[], session_dir=session_dir)

    if rewrite:
        assert artifacts.session_dir == session_dir
    assert artifacts.segments_path == artifacts.session_dir / SEGMENTS_FILE
    assert SEGMENTS_FILE in artifacts.metadata.output_files
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert SEGMENTS_FILE in metadata["output_files"]
    document = json.loads(artifacts.segments_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["semantics"] == "provider_evidence"
    assert document["segments"] == []
