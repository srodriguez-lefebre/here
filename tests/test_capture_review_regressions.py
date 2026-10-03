import json
import threading
from types import SimpleNamespace

import here.live_processing as live
import here.recording.windows as windows
import numpy as np
import pytest
import soundfile as sf
from here.application import ApplicationState, HereApplicationController, StartRequest
from here.application.processing import SessionProcessor
from here.audio.models import ChunkingConfig
from here.recording.models import CaptureFailed, RecordedAudioSource, RecordingSession
from here.transcription.client import AudioTranscription, TranscriptionResult
from here.transcription.segments import TranscriptSegment


def capture_after_stall(monkeypatch, path, delay, sample_rate, sink, label):
    stop = threading.Event()
    frames = [0]
    errors = []

    class Stream:
        def get_read_available(self):
            return 4

        def read(self, count, **kwargs):
            stop.set()
            return np.full(count, 1234, dtype=np.int16).tobytes()

    times = iter([delay, delay + 0.001])
    monkeypatch.setattr(
        windows, "time", SimpleNamespace(perf_counter=lambda: next(times), sleep=windows.time.sleep)
    )
    with sf.SoundFile(
        path, mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16"
    ) as writer:
        windows._capture_windows_stream_to_file(
            Stream(),
            chunk=4,
            sample_rate=sample_rate,
            channels=1,
            writer=writer,
            stop_event=stop,
            errors=errors,
            label=label,
            written_frames=frames,
            start_time=0.0,
            block_sink=sink,
        )
    assert errors == []
    return RecordedAudioSource(path, sample_rate, 1, frames[0], label, f"Synthetic {label}")


def setup_provider(monkeypatch):
    monkeypatch.setattr(live, "build_client", lambda: object())
    monkeypatch.setattr(
        live, "resolve_transcription_models", lambda **kwargs: ("model", "cleanup", False)
    )


@pytest.mark.parametrize("delays", [(2.5,), (2.5, 1.5)])
def test_stalled_capture_live_evidence_matches_authoritative_recording_time(
    monkeypatch, tmp_path, delays
):
    setup_provider(monkeypatch)
    uploaded = []

    def provider(client, path, model, **kwargs):
        audio, rate = sf.read(path)
        uploaded.append(audio.copy())
        segments = [
            TranscriptSegment(f"speech at {second}", float(second), float(second + 1))
            for second in range(len(audio) // rate)
            if np.any(audio[second * rate : (second + 1) * rate])
        ]
        return AudioTranscription("speech", segments)

    monkeypatch.setattr(live, "transcribe_audio_file", provider)
    pipeline = live.LiveTranscriptionController(
        expected_source_count=len(delays),
        chunking_config=ChunkingConfig(
            target_sample_rate=4, live_chunk_seconds=20, overlap_seconds=0, silence_search_seconds=0
        ),
    )
    handed_off = {}

    def sink(label, block, rate, channels):
        handed_off.setdefault(label, []).append(block.copy())
        pipeline.submit_block(label, block, rate, channels)

    try:
        sources = [
            capture_after_stall(
                monkeypatch, tmp_path / f"source-{index}.wav", delay, 4, sink, f"source-{index}"
            )
            for index, delay in enumerate(delays)
        ]
        session = RecordingSession(sources)
        artifacts = SessionProcessor(retry_delays=()).process(
            session, tmp_path / "sessions", live_controller=pipeline
        )
        evidence = json.loads(artifacts.segments_path.read_text())["segments"]
        assert [segment["start"] for segment in evidence] == (
            [2.0] if len(delays) == 1 else [1.0, 2.0]
        )
        for source in sources:
            data = np.concatenate(handed_off[source.label])
            assert len(data) == source.frames
            assert np.flatnonzero(data[:, 0])[0] / source.sample_rate == int(
                delays[int(source.label[-1])]
            )
        assert len(uploaded[0]) == 12
        assert artifacts.metadata.live_pipeline_used
        assert not artifacts.metadata.fallback_used
    finally:
        pipeline.abort()
        pipeline.cleanup()


def test_catchup_silence_pressure_preserves_primary_audio_and_falls_back(monkeypatch, tmp_path):
    setup_provider(monkeypatch)
    gate = threading.Event()
    chunker = live.LiveTranscriptionController._chunker_worker

    def gated_chunker(self):
        gate.wait(5)
        chunker(self)

    monkeypatch.setattr(live.LiveTranscriptionController, "_chunker_worker", gated_chunker)
    pipeline = live.LiveTranscriptionController(expected_source_count=1)
    observed = []

    def offline(session, **kwargs):
        audio, rate = sf.read(session.sources[0].path)
        observed.append((len(audio), np.flatnonzero(audio)[0] / rate))
        return TranscriptionResult(
            "full audio", "full audio", segments=[TranscriptSegment("speech", 3.0, 3.01)]
        )

    try:
        source = capture_after_stall(
            monkeypatch, tmp_path / "source.wav", 3.005, 400, pipeline.submit_block, "mic"
        )
        assert source.frames == 1204
        primary, rate = sf.read(source.path)
        assert np.flatnonzero(primary)[0] / rate == 3.0
        assert pipeline._error is not None, (
            "catch-up delivery must exceed the finite handoff budget"
        )
        assert pipeline._capture_queue.qsize() == 256
        gate.set()
        artifacts = SessionProcessor(transcribe=offline, retry_delays=()).process(
            RecordingSession([source]), tmp_path / "sessions", live_controller=pipeline
        )
        assert observed[0][0] == 48160
        # Linear resampling can interpolate across one original source frame.
        assert observed[0][1] == pytest.approx(3.0, abs=1 / 400)
        assert artifacts.metadata.fallback_used
        assert not artifacts.metadata.live_pipeline_used
        assert sf.info(artifacts.audio_path).frames == 48160
    finally:
        gate.set()
        pipeline.abort()
        pipeline.cleanup()


@pytest.mark.parametrize("cancel_during", ["abort", "persistence"])
def test_capture_failure_cancel_cutover_is_atomic(monkeypatch, tmp_path, cancel_during):
    path = tmp_path / "synthetic.wav"
    sf.write(path, np.full(32, 0.25), 8000)
    session = RecordingSession([RecordedAudioSource(path, 8000, 1, 32, "mic", "Synthetic")])
    entered = threading.Event()
    release = threading.Event()
    capture_cancelled = threading.Event()
    failure = CaptureFailed(session, OSError("device lost"))

    def barrier():
        entered.set()
        assert release.wait(3)

    def abort():
        if cancel_during == "abort":
            barrier()

    def capture_factory(request, sink):
        def wait():
            raise failure

        return type(
            "Capture",
            (),
            {"wait": staticmethod(wait), "cancel": staticmethod(capture_cancelled.set)},
        )()

    processor = SessionProcessor(retry_delays=())
    preserve = processor._preserve_sources

    def preserve_with_barrier(*args):
        if cancel_during == "persistence":
            barrier()
        return preserve(*args)

    monkeypatch.setattr(processor, "_preserve_sources", preserve_with_barrier)
    controller = HereApplicationController(
        capture_factory=capture_factory,
        live_factory=lambda *args: type(
            "Live", (), {"abort": staticmethod(abort), "cleanup": staticmethod(lambda: None)}
        )(),
        processor=processor,
    )
    target = tmp_path / "sessions"
    try:
        controller.start(StartRequest(output_dir=target))
        assert entered.wait(2)
        if cancel_during == "abort":
            assert controller.snapshot.state is ApplicationState.RECORDING
        else:
            assert controller.snapshot.state is ApplicationState.PROCESSING, (
                "recovery must atomically leave destructive-cancel states before persistence"
            )
        controller.cancel()
    finally:
        release.set()
    snapshot = controller.wait_until_terminal(3)
    assert snapshot.state is ApplicationState.CANCELLED
    assert not path.exists()
    if cancel_during == "abort":
        assert capture_cancelled.is_set()
        assert not snapshot.recoverable
        assert snapshot.session_dir is None
        assert not target.exists(), "accepted destructive cancel must not create any session"
    else:
        assert not capture_cancelled.is_set()
        assert snapshot.recoverable
        metadata = json.loads((snapshot.session_dir / "session.json").read_text())
        assert metadata["status"] == "cancelled"
        assert sf.info(snapshot.session_dir / "audio.wav").frames == 64
        errors = json.loads((snapshot.session_dir / "errors.json").read_text())["errors"]
        assert errors[0]["cause_message"] == "device lost"


def test_normal_capture_return_honors_cancel_at_processing_cutover(tmp_path):
    path = tmp_path / "synthetic.wav"
    sf.write(path, np.full(32, 0.25), 8000)
    session = RecordingSession([RecordedAudioSource(path, 8000, 1, 32, "mic", "Synthetic")])
    returned = threading.Event()
    entered = threading.Event()
    release = threading.Event()

    def wait():
        returned.set()
        return session

    controller = HereApplicationController(
        capture_factory=lambda *args: SimpleNamespace(wait=wait, cancel=lambda: None),
        live_factory=lambda *args: SimpleNamespace(
            abort=lambda: None,
            cleanup=lambda: None,
            complete=lambda: TranscriptionResult("text", "text"),
        ),
        processor=SessionProcessor(
            transcribe=lambda *args, **kwargs: TranscriptionResult("text", "text"), retry_delays=()
        ),
    )
    original_lock = controller._lock

    class CutoverLock:
        def __enter__(self):
            original_lock.acquire()

        def __exit__(self, *args):
            original_lock.release()
            if (
                returned.is_set()
                and not entered.is_set()
                and threading.current_thread().name == "here-application-job"
            ):
                entered.set()
                assert release.wait(3)

    controller._lock = CutoverLock()
    target = tmp_path / "sessions"
    try:
        controller.start(StartRequest(output_dir=target))
        assert entered.wait(2)
        state_at_cancel = controller.snapshot.state
        controller.cancel()
    finally:
        release.set()
    snapshot = controller.wait_until_terminal(3)
    assert snapshot.state is ApplicationState.CANCELLED
    assert not path.exists()
    if state_at_cancel is ApplicationState.RECORDING:
        assert not snapshot.recoverable
        assert not target.exists()
    else:
        assert state_at_cancel is ApplicationState.PROCESSING
        assert snapshot.recoverable
        assert (
            json.loads((snapshot.session_dir / "session.json").read_text())["status"] == "cancelled"
        )
