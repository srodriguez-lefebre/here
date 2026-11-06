"""Copilot C1: ordinary precommit callback failures roll back source evidence."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone

import pytest
from memory_fixtures import MEETING_ID, write_marked_wav


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, OSError])
def test_real_writer_restores_previous_evidence_when_precommit_callback_raises(
    tmp_path, error_type
):
    output = importlib.import_module("here.output.session_writer")
    metadata_types = importlib.import_module("here.output.metadata")
    models = importlib.import_module("here.recording.models")
    segment = importlib.import_module("here.transcription.segments").TranscriptSegment
    directory = tmp_path / "session"
    directory.mkdir()
    audio = directory / "audio.wav"
    write_marked_wav(audio)
    session = models.RecordingSession(
        [models.RecordedAudioSource(audio, 8000, 1, 16000, "synthetic")], meeting_id=MEETING_ID
    )
    options = {
        "session": session,
        "target_dir": tmp_path,
        "session_dir": directory,
        "session_id": "synthetic-rollback",
        "completed_at": datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        "transcription_model": "synthetic-model",
        "cleanup_model": "synthetic-cleanup",
        "cleanup_enabled": False,
        "alt_model_used": False,
        "live_pipeline_attempted": False,
        "live_pipeline_used": False,
        "fallback_used": False,
        "recoverable_audio": "audio.wav",
    }
    output.write_session_artifacts(
        **options,
        raw_transcript_text="Synthetic original source evidence.",
        segments=[segment("Synthetic original segment.")],
        status="failed",
        errors=[
            metadata_types.ErrorMetadata(
                stage="offline_transcription",
                type="OSError",
                message="Synthetic old evidence.",
                retryable=True,
                occurred_at=options["completed_at"],
            )
        ],
    )
    canary = directory / "unknown-synthetic.bin"
    canary.write_bytes(b"Synthetic unknown sibling must survive rollback.")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    original = error_type("Synthetic evidence seal rejected before metadata commit")
    callback_calls = []

    def reject(staged, prospective):
        callback_calls.append(staged)
        assert prospective.status == "completed" and staged.is_file()
        assert (directory / "session.json").read_bytes() == before["session.json"]
        assert (directory / "raw_transcript.txt").read_text(
            encoding=output.TRANSCRIPT_ENCODING
        ) == "Synthetic new source evidence."
        assert (
            json.loads((directory / "segments.json").read_text(encoding="utf-8"))["segments"][0][
                "text"
            ]
            == "Synthetic new segment."
        )
        assert not (directory / "errors.json").exists()
        raise original

    with pytest.raises(error_type) as raised:
        output.write_session_artifacts(
            **options,
            raw_transcript_text="Synthetic new source evidence.",
            segments=[segment("Synthetic new segment.")],
            status="completed",
            before_metadata_commit=reject,
        )
    assert raised.value is original and len(callback_calls) == 1
    assert (directory / "session.json").read_bytes() == before["session.json"]
    assert (directory / "raw_transcript.txt").read_bytes() == before["raw_transcript.txt"]
    assert (directory / "segments.json").read_bytes() == before["segments.json"]
    assert (directory / "errors.json").read_bytes() == before["errors.json"]
    assert audio.read_bytes() == before["audio.wav"]
    assert canary.read_bytes() == before[canary.name]
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
