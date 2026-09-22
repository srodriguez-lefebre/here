from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from here.application.processing import (
    ProcessingCancelled,
    SessionProcessingFailed,
    SessionProcessor,
)
from here.output.metadata import SessionEventMetadata
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.client import TranscriptionResult


def make_session(tmp_path: Path) -> RecordingSession:
    audio_path = tmp_path / "capture.wav"
    audio = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)
    sf.write(audio_path, audio, 16000)
    return RecordingSession(
        sources=[
            RecordedAudioSource(
                path=audio_path,
                sample_rate=16000,
                channels=1,
                frames=1600,
                label="microphone",
                device_name="Test microphone",
            )
        ]
    )


def test_process_materializes_audio_transcribes_and_persists_events(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    calls = 0

    def transcribe(recording: RecordingSession, **kwargs: object) -> TranscriptionResult:
        nonlocal calls
        del recording, kwargs
        calls += 1
        return TranscriptionResult(raw_text="hola", final_text="Hola")

    processor = SessionProcessor(transcribe=transcribe, retry_delays=())
    started_at = datetime.now().astimezone()
    events = [
        SessionEventMetadata(
            kind="paused",
            occurred_at=started_at,
            recorded_duration_seconds=0.05,
            total_paused_seconds=0.0,
        )
    ]

    artifacts = processor.process(
        session,
        tmp_path / "sessions",
        events=events,
        started_at=started_at,
        total_paused_seconds=12.5,
    )

    assert calls == 1
    assert artifacts.audio_path is not None and artifacts.audio_path.exists()
    assert artifacts.transcript_path.read_text(encoding="utf-8-sig") == "Hola"
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["started_at"] == started_at.isoformat()
    assert metadata["total_paused_seconds"] == 12.5
    persisted_events = json.loads(artifacts.events_path.read_text(encoding="utf-8"))
    assert persisted_events["events"][0]["kind"] == "paused"
    assert not session.sources[0].path.exists()


def test_processing_cancellation_preserves_audio_as_recoverable_session(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    cancellation = threading.Event()
    cancellation.set()
    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: pytest.fail("transcription must not start")
    )

    with pytest.raises(ProcessingCancelled) as exc_info:
        processor.process(
            session,
            tmp_path / "sessions",
            cancel_event=cancellation,
        )

    session_dir = exc_info.value.session_dir
    assert (session_dir / "audio.wav").exists()
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "cancelled"
    assert metadata["failure_stage"] == "processing_cancelled"
    assert metadata["recoverable_audio"] == "audio.wav"
    assert not session.sources[0].path.exists()


def test_cancellation_arriving_during_transcription_is_persisted(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    cancellation = threading.Event()

    def transcribe(*args: object, **kwargs: object) -> TranscriptionResult:
        del args, kwargs
        cancellation.set()
        return TranscriptionResult(raw_text="late", final_text="Late")

    processor = SessionProcessor(transcribe=transcribe, retry_delays=())

    with pytest.raises(ProcessingCancelled) as exc_info:
        processor.process(
            session,
            tmp_path / "sessions",
            cancel_event=cancellation,
        )

    metadata = json.loads(
        (exc_info.value.session_dir / "session.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "cancelled"
    assert not (exc_info.value.session_dir / "transcript.txt").exists()


def test_offline_transcription_retries_three_times_then_succeeds(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    attempts = 0
    delays: list[float] = []

    def transcribe(recording: RecordingSession, **kwargs: object) -> TranscriptionResult:
        nonlocal attempts
        del recording, kwargs
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary")
        return TranscriptionResult(raw_text="ok", final_text="OK")

    processor = SessionProcessor(
        transcribe=transcribe,
        retry_delays=(0.5, 1.0),
        sleeper=delays.append,
    )

    artifacts = processor.process(session, tmp_path / "sessions")

    assert attempts == 3
    assert delays == [0.5, 1.0]
    assert artifacts.metadata.status == "completed"


def test_terminal_transcription_failure_is_persisted_and_recoverable(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider down")),
        retry_delays=(),
    )

    with pytest.raises(SessionProcessingFailed) as exc_info:
        processor.process(session, tmp_path / "sessions")

    assert exc_info.value.recoverable
    session_dir = exc_info.value.session_dir
    assert (session_dir / "audio.wav").exists()
    errors = json.loads((session_dir / "errors.json").read_text(encoding="utf-8"))
    assert errors["errors"][0]["stage"] == "offline_transcription"
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["recoverable_audio"] == "audio.wav"


def test_retry_reuses_recoverable_audio_and_clears_persisted_errors(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    failing = SessionProcessor(
        transcribe=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("temporary")),
        retry_delays=(),
    )
    with pytest.raises(SessionProcessingFailed) as exc_info:
        failing.process(session, tmp_path / "sessions")

    recovered = SessionProcessor(
        transcribe=lambda *args, **kwargs: TranscriptionResult(
            raw_text="recovered",
            final_text="Recovered",
        ),
        retry_delays=(),
    ).retry(exc_info.value.session_dir)

    assert recovered.metadata.status == "completed"
    assert recovered.transcript_path.read_text(encoding="utf-8-sig") == "Recovered"
    assert not recovered.errors_path.exists()


def test_retry_preserves_existing_lifecycle_events(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    occurred_at = datetime.now().astimezone()
    events = [
        SessionEventMetadata(
            kind="paused",
            occurred_at=occurred_at,
            recorded_duration_seconds=0.05,
        )
    ]
    failing = SessionProcessor(
        transcribe=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("temporary")),
        retry_delays=(),
    )
    with pytest.raises(SessionProcessingFailed) as exc_info:
        failing.process(session, tmp_path / "sessions", events=events)

    recovered = SessionProcessor(
        transcribe=lambda *args, **kwargs: TranscriptionResult(
            raw_text="recovered",
            final_text="Recovered",
        ),
        retry_delays=(),
    ).retry(exc_info.value.session_dir)

    persisted = json.loads(recovered.events_path.read_text(encoding="utf-8"))
    assert [event["kind"] for event in persisted["events"]] == ["paused"]


def test_failed_retry_preserves_existing_lifecycle_events(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    event = SessionEventMetadata(
        kind="paused",
        occurred_at=datetime.now().astimezone(),
        recorded_duration_seconds=0.05,
    )
    first_failure = SessionProcessor(
        transcribe=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("first")),
        retry_delays=(),
    )
    with pytest.raises(SessionProcessingFailed) as exc_info:
        first_failure.process(session, tmp_path / "sessions", events=[event])

    with pytest.raises(SessionProcessingFailed):
        SessionProcessor(
            transcribe=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("second")),
            retry_delays=(),
        ).retry(exc_info.value.session_dir)

    persisted = json.loads(
        (exc_info.value.session_dir / "events.json").read_text(encoding="utf-8")
    )
    assert [item["kind"] for item in persisted["events"]] == ["paused"]


def test_cancelled_retry_updates_session_and_records_event(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    event = SessionEventMetadata(
        kind="paused",
        occurred_at=datetime.now().astimezone(),
        recorded_duration_seconds=0.05,
    )
    first_failure = SessionProcessor(
        transcribe=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("first")),
        retry_delays=(),
    )
    with pytest.raises(SessionProcessingFailed) as exc_info:
        first_failure.process(session, tmp_path / "sessions", events=[event])
    cancellation = threading.Event()
    cancellation.set()

    with pytest.raises(ProcessingCancelled):
        SessionProcessor(retry_delays=()).retry(
            exc_info.value.session_dir,
            cancel_event=cancellation,
        )

    metadata = json.loads(
        (exc_info.value.session_dir / "session.json").read_text(encoding="utf-8")
    )
    persisted = json.loads(
        (exc_info.value.session_dir / "events.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "cancelled"
    assert [item["kind"] for item in persisted["events"]] == [
        "paused",
        "processing_cancelled",
    ]
