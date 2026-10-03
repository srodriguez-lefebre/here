import threading
import time

import here.live_processing as live
import numpy as np
import pytest
import soundfile as sf
from here.application.processing import SessionProcessor
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.client import TranscriptionResult


@pytest.mark.parametrize("queue_kind", ["capture", "jobs"])
def test_queue_pressure_disables_live_and_terminal_calls_do_not_block(
    monkeypatch, tmp_path, queue_kind
):
    gate = threading.Event()
    original_chunker = live.LiveTranscriptionController._chunker_worker
    original_transcriber = live.LiveTranscriptionController._transcriber_worker

    def gated_chunker(self):
        gate.wait(3)
        original_chunker(self)

    def gated_transcriber(self):
        gate.wait(3)
        original_transcriber(self)

    monkeypatch.setattr(live, "build_client", lambda: object())
    monkeypatch.setattr(
        live, "resolve_transcription_models", lambda **kwargs: ("model", "cleanup", False)
    )
    monkeypatch.setattr(live.LiveTranscriptionController, "_chunker_worker", gated_chunker)
    monkeypatch.setattr(live.LiveTranscriptionController, "_transcriber_worker", gated_transcriber)
    controller = live.LiveTranscriptionController(expected_source_count=1)
    try:
        started = time.monotonic()
        if queue_kind == "capture":
            for _ in range(257):
                controller.submit_block("mic", np.ones((2, 1)), 8000, 1)
            assert controller._capture_queue.qsize() <= 256
        else:
            for index in range(9):
                source = RecordedAudioSource(tmp_path / f"job-{index}.wav", 8000, 1, 2, "mic")
                controller._enqueue_live_job([live.SourceSegment(index, source, 0.0)])
            assert controller._chunk_queue.qsize() <= 8
        assert time.monotonic() - started < 1
        assert controller._error is not None
        assert "queue" in str(controller._error).lower()
        gate.set()
        terminal = threading.Thread(target=controller.abort, daemon=True)
        terminal.start()
        terminal.join(3)
        assert not terminal.is_alive(), (
            "abort must not wait on sentinel insertion into a full queue"
        )
        with pytest.raises(RuntimeError, match="Live transcription failed"):
            controller.complete()
        assert not controller._chunker_thread.is_alive()
        assert not controller._transcriber_thread.is_alive()
    finally:
        gate.set()
        controller.abort()
        controller.cleanup()
    assert not controller.working_dir.exists()


def test_capture_pressure_fallback_transcribes_complete_primary_audio(monkeypatch, tmp_path):
    gate = threading.Event()
    original_chunker = live.LiveTranscriptionController._chunker_worker

    def gated_chunker(self):
        gate.wait(3)
        original_chunker(self)

    monkeypatch.setattr(live, "build_client", lambda: object())
    monkeypatch.setattr(
        live, "resolve_transcription_models", lambda **kwargs: ("model", "cleanup", False)
    )
    monkeypatch.setattr(live.LiveTranscriptionController, "_chunker_worker", gated_chunker)
    controller = live.LiveTranscriptionController(expected_source_count=1)
    path = tmp_path / "primary.wav"
    block = np.full((32, 1), 0.25, dtype=np.float32)
    with sf.SoundFile(path, mode="w", samplerate=8000, channels=1) as writer:
        for _ in range(300):
            writer.write(block)
            controller.submit_block("mic", block, 8000, 1)
    assert controller._error is not None
    gate.set()
    session = RecordingSession([RecordedAudioSource(path, 8000, 1, 9600, "mic", "Synthetic")])
    observed = []

    def offline(material, **kwargs):
        observed.append(sf.info(material.sources[0].path).frames)
        return TranscriptionResult("complete offline", "complete offline")

    try:
        artifacts = SessionProcessor(transcribe=offline, retry_delays=()).process(
            session, tmp_path / "sessions", live_controller=controller
        )
        assert observed == [19200]
        assert sf.info(artifacts.audio_path).frames == 19200
        assert artifacts.metadata.fallback_used
        assert not artifacts.metadata.live_pipeline_used
        assert artifacts.transcript_path.read_text(encoding="utf-8-sig") == "complete offline"
    finally:
        gate.set()
        controller.abort()
        controller.cleanup()


def test_full_job_queue_does_not_wait_for_provider_and_cleanup_waits_for_workers(
    monkeypatch, tmp_path
):
    from here.audio.models import ChunkingConfig

    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(live, "build_client", lambda: object())
    monkeypatch.setattr(
        live, "resolve_transcription_models", lambda **kwargs: ("model", "cleanup", False)
    )

    def provider(*args, **kwargs):
        entered.set()
        release.wait(8)
        return "in flight text"

    monkeypatch.setattr(live, "transcribe_audio_file", provider)
    controller = live.LiveTranscriptionController(
        expected_source_count=1,
        chunking_config=ChunkingConfig(
            target_sample_rate=4, live_chunk_seconds=2, overlap_seconds=1, silence_search_seconds=0
        ),
    )
    try:
        controller.submit_block("mic", np.full(8, 0.25, dtype=np.float32), 4, 1)
        assert entered.wait(2)
        for index in range(9):
            path = tmp_path / f"pending-{index}.wav"
            sf.write(path, np.full(4, 0.25), 4)
            controller._enqueue_live_job(
                [live.SourceSegment(index + 2, RecordedAudioSource(path, 4, 1, 4, "mic"), 0.0)]
            )
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="Live transcription failed"):
            controller.complete()
        assert time.monotonic() - started < 3
        controller.cleanup()
        assert controller.working_dir.exists(), "in-flight provider still owns its chunk workspace"
        assert controller._transcriber_thread.is_alive()
        release.set()
        controller._transcriber_thread.join(3)
        controller._chunker_thread.join(3)
        assert not controller.working_dir.exists()
        assert controller._result is None
    finally:
        release.set()
        controller.abort()
        controller.cleanup()


@pytest.mark.parametrize("frames", [4, 16])
def test_missing_second_source_exceeds_buffer_budget_and_falls_back(monkeypatch, frames):
    from here.audio.models import ChunkingConfig

    monkeypatch.setattr(live, "build_client", lambda: object())
    monkeypatch.setattr(
        live, "resolve_transcription_models", lambda **kwargs: ("model", "cleanup", False)
    )
    controller = live.LiveTranscriptionController(
        expected_source_count=2,
        chunking_config=ChunkingConfig(
            target_sample_rate=4, live_chunk_seconds=2, overlap_seconds=1, silence_search_seconds=0
        ),
    )
    consumed = threading.Event()
    original_append = live.BufferedSourceState.append_block

    def append(state, data):
        original_append(state, data)
        consumed.set()

    monkeypatch.setattr(live.BufferedSourceState, "append_block", append)
    try:
        controller.submit_block("mic", np.full(frames, 0.25, dtype=np.float32), 4, 1)
        assert consumed.wait(2)
        with pytest.raises(RuntimeError, match="Live transcription failed") as caught:
            controller.complete()
        assert ("buffer" if frames == 16 else "source") in str(caught.value.__cause__).lower()
        assert controller._result is None
        assert controller._source_states == {}
    finally:
        controller.abort()
        controller.cleanup()
