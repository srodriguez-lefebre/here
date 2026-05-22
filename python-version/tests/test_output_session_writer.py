from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from here.output.session_writer import (
    MARKDOWN_FILE,
    METADATA_FILE,
    TRANSCRIPT_ENCODING,
    TRANSCRIPT_FILE,
    write_session_artifacts,
)
from here.recording.models import RecordedAudioSource, RecordingSession


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
    )

    assert artifacts.session_dir.name == "20260522_103000"
    assert artifacts.transcript_path.name == TRANSCRIPT_FILE
    assert artifacts.markdown_path.name == MARKDOWN_FILE
    assert artifacts.metadata_path.name == METADATA_FILE
    assert artifacts.transcript_path.read_text(encoding=TRANSCRIPT_ENCODING) == "Speaker 1: hola"

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["session_id"] == "20260522_103000"
    assert metadata["duration_seconds"] == 3.0
    assert metadata["sources"] == [
        {
            "label": "microphone",
            "sample_rate": 16000,
            "channels": 1,
            "frames": 48000,
            "duration_seconds": 3.0,
        }
    ]
    assert metadata["output_files"] == [TRANSCRIPT_FILE, MARKDOWN_FILE, METADATA_FILE]
    assert "source.wav" not in artifacts.metadata_path.read_text(encoding="utf-8")

    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "# Recording 2026-05-22 10:30" in markdown
    assert "- Session ID: `20260522_103000`" in markdown
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
