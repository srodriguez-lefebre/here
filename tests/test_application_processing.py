from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
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
from here.transcription.segments import TranscriptSegment


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


@pytest.mark.parametrize("offset_hours", [0, -3])
def test_process_materializes_audio_transcribes_and_persists_events(
    tmp_path: Path, offset_hours: int
) -> None:
    session = make_session(tmp_path)
    calls = 0

    def transcribe(recording: RecordingSession, **kwargs: object) -> TranscriptionResult:
        nonlocal calls
        del recording, kwargs
        calls += 1
        return TranscriptionResult(raw_text="hola", final_text="Hola")

    processor = SessionProcessor(transcribe=transcribe, retry_delays=())
    started_at = datetime(2026, 10, 3, 5, 15, 53, 734566, timezone(timedelta(hours=offset_hours)))
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
    assert datetime.fromisoformat(metadata["started_at"]) == started_at
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

    metadata = json.loads((exc_info.value.session_dir / "session.json").read_text(encoding="utf-8"))
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

    persisted = json.loads((exc_info.value.session_dir / "events.json").read_text(encoding="utf-8"))
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

    metadata = json.loads((exc_info.value.session_dir / "session.json").read_text(encoding="utf-8"))
    persisted = json.loads((exc_info.value.session_dir / "events.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "cancelled"
    assert [item["kind"] for item in persisted["events"]] == [
        "paused",
        "processing_cancelled",
    ]


@pytest.fixture
def completed_transcript_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run real producers/persistence; replace only provider work and failed live work."""
    import here.cli as cli_module
    import here.live_processing as live_module
    import here.transcription.client as client_module
    import here.transcription.service as service_module
    from here.transcription.client import AudioTranscription

    class FailedLive:
        def complete(self):
            raise RuntimeError("synthetic live failure requiring offline fallback")

        def chunk_metadata(self):
            return []

        def cleanup(self):
            pass

    def run(path: str, evidence_kind: str) -> Path:
        bounds = {
            "timed": (0.0, 0.05),
            "untimed": (None, None),
            "reversed": (0.05, 0.0),
            "nonfinite": (float("nan"), float("inf")),
        }
        segments = (
            []
            if evidence_kind == "text-only"
            else [TranscriptSegment("original words", *bounds[evidence_kind], speaker="A")]
        )
        provider_text = "original words" if evidence_kind == "text-only" else "A: original words"
        for module in (service_module, live_module):
            monkeypatch.setattr(module, "build_client", lambda: object())
            monkeypatch.setattr(
                module,
                "resolve_transcription_models",
                lambda **kwargs: ("synthetic-transcribe", "synthetic-cleanup", True),
            )
            monkeypatch.setattr(
                module,
                "transcribe_audio_file",
                lambda *args, **kwargs: AudioTranscription(provider_text, segments),
            )
        monkeypatch.setattr(client_module, "cleanup_transcript", lambda *args: "Cleaned display")
        session = make_session(tmp_path)
        target = tmp_path / "sessions"
        processor = SessionProcessor(
            transcribe=service_module.transcribe_recording_session, retry_delays=()
        )
        if path == "file":
            cli_module._transcribe_audio_path(session.sources[0].path, target)
            assert session.sources[0].path.exists(), "file transcription must retain its input"
            (directory,) = target.iterdir()
            return directory
        if path == "retry":
            failing = SessionProcessor(
                transcribe=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
                retry_delays=(),
            )
            with pytest.raises(SessionProcessingFailed) as failure:
                failing.process(session, target)
            return processor.retry(failure.value.session_dir).session_dir
        controller = None
        if path == "live":
            controller = live_module.LiveTranscriptionController(expected_source_count=1)
            audio, sample_rate = sf.read(session.sources[0].path, dtype="float32")
            controller.submit_block("microphone", audio, sample_rate, 1)
        elif path == "fallback":
            controller = FailedLive()
        try:
            return processor.process(session, target, live_controller=controller).session_dir
        finally:
            if path == "live":
                controller.cleanup()

    return run


@pytest.mark.parametrize("path", ["live", "offline", "fallback", "file", "retry"])
@pytest.mark.parametrize(
    "evidence_kind", ["timed", "untimed", "reversed", "nonfinite", "text-only"]
)
def test_every_transcript_path_persists_segment_evidence_and_timing_source(
    completed_transcript_path, path: str, evidence_kind: str
) -> None:
    directory = completed_transcript_path(path, evidence_kind)
    document = json.loads((directory / "segments.json").read_text(encoding="utf-8"))
    assert document["timing_reference"] == "recording_seconds"
    assert len(document["segments"]) == 1
    (segment,) = document["segments"]
    assert segment["text"] == "original words"
    assert segment["speaker"] == (None if evidence_kind == "text-only" else "A")
    assert (segment["start"], segment["end"]) == (
        (0.0, 0.05) if evidence_kind == "timed" else (None, None)
    )
    assert segment["timing_source"] == ("provider" if evidence_kind == "timed" else "absent")
    metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    assert "segments.json" in metadata["output_files"]
    assert metadata["live_pipeline_used"] == (path == "live")
    assert metadata["fallback_used"] == (path == "fallback")


@pytest.mark.parametrize("path", ["live", "offline", "fallback", "file", "retry"])
def test_every_transcript_path_persists_raw_text_separately_from_cleanup(
    completed_transcript_path, path: str
) -> None:
    directory = completed_transcript_path(path, "timed")
    raw = directory / "raw_transcript.txt"
    assert raw.exists(), "provider text must survive independently of cleaned display text"
    assert raw.read_text(encoding="utf-8-sig") == "A: original words"
    assert (directory / "transcript.txt").read_text(encoding="utf-8-sig") == "Cleaned display"
    metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    assert "raw_transcript.txt" in metadata["output_files"]


@pytest.mark.parametrize("path", ["live", "offline"])
def test_five_minute_wall_pause_keeps_second_segment_on_recorded_wav_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    import here.live_processing as live_module
    import here.transcription.service as service_module
    from here.audio.models import ChunkingConfig
    from here.transcription.client import AudioTranscription

    sample_rate = 16000
    audio = np.zeros(2 * sample_rate, dtype=np.float32)
    audio[sample_rate // 4 : sample_rate // 2] = 0.25
    audio[5 * sample_rate // 4 : 3 * sample_rate // 2] = 0.5
    source = tmp_path / "two-recorded-seconds.wav"
    sf.write(source, audio, sample_rate, subtype="PCM_16")
    session = RecordingSession(
        [RecordedAudioSource(source, sample_rate, 1, len(audio), "microphone")]
    )
    config = ChunkingConfig(
        target_sample_rate=sample_rate,
        max_chunk_bytes=2 * sample_rate,
        live_chunk_seconds=1,
        overlap_seconds=0,
        silence_search_seconds=0.0,
    )
    responses = iter(
        [
            AudioTranscription("first marker", [TranscriptSegment("first marker", 0.25, 0.5)]),
            AudioTranscription("second marker", [TranscriptSegment("second marker", 0.25, 0.5)]),
        ]
    )
    for module in (service_module, live_module):
        monkeypatch.setattr(module, "build_client", lambda: object())
        monkeypatch.setattr(
            module,
            "resolve_transcription_models",
            lambda **kwargs: ("synthetic-transcribe", "synthetic-cleanup", False),
        )
        monkeypatch.setattr(
            module, "transcribe_audio_file", lambda *args, **kwargs: next(responses)
        )
    controller = None
    if path == "live":
        controller = live_module.LiveTranscriptionController(
            expected_source_count=1, chunking_config=config
        )
        controller.submit_block("microphone", audio[:sample_rate], sample_rate, 1)
        controller.submit_block("microphone", audio[sample_rate:], sample_rate, 1)
    started = datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc)
    events = [
        SessionEventMetadata(
            kind="paused",
            occurred_at=started + timedelta(seconds=1),
            recorded_duration_seconds=1.0,
        ),
        SessionEventMetadata(
            kind="resumed",
            occurred_at=started + timedelta(seconds=301),
            recorded_duration_seconds=1.0,
            total_paused_seconds=300.0,
        ),
    ]

    def transcribe(recording, **kwargs):
        return service_module.transcribe_recording_session(
            recording, chunking_config=config, **kwargs
        )

    processor = SessionProcessor(transcribe=transcribe, retry_delays=())
    try:
        artifacts = processor.process(
            session,
            tmp_path / "sessions",
            live_controller=controller,
            events=events,
            started_at=started,
            total_paused_seconds=300.0,
        )
    finally:
        if controller is not None:
            controller.cleanup()

    evidence = json.loads(artifacts.segments_path.read_text(encoding="utf-8"))["segments"]
    assert [segment["text"] for segment in evidence] == ["first marker", "second marker"]
    assert [(segment["start"], segment["end"]) for segment in evidence] == [
        (0.25, 0.5),
        (1.25, 1.5),
    ]
    assert [segment["chunk_index"] for segment in evidence] == [1, 2]
    with sf.SoundFile(artifacts.audio_path) as wav:
        assert wav.frames == 2 * sample_rate
        wav.seek(round(evidence[1]["start"] * wav.samplerate))
        assert wav.read(1)[0] > 0.1, "the seek must land inside the second recorded marker"
    assert artifacts.metadata.total_paused_seconds == 300.0
