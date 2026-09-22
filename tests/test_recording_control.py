from __future__ import annotations

import threading
import time
from pathlib import Path

import here.recording.windows as windows_module
import numpy as np
import pytest
from here.recording.models import RecordedAudioSource, RecordingSession


class FakeStream:
    def get_read_available(self) -> int:
        return 1

    def read(self, frames: int, *, exception_on_overflow: bool) -> bytes:
        del frames, exception_on_overflow
        return np.array([1200], dtype=np.int16).tobytes()


class FakeWriter:
    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []

    def write(self, block: np.ndarray) -> None:
        self.blocks.append(block.copy())


def test_capture_pause_discards_input_without_writing_silence() -> None:
    stop_event = threading.Event()
    pause_event = threading.Event()
    pause_event.set()
    writer = FakeWriter()
    written_frames = [0]
    errors: list[Exception] = []
    thread = threading.Thread(
        target=windows_module._capture_windows_stream_to_file,
        kwargs={
            "stream": FakeStream(),
            "chunk": 1,
            "sample_rate": 20,
            "channels": 1,
            "writer": writer,
            "stop_event": stop_event,
            "pause_event": pause_event,
            "errors": errors,
            "label": "microphone",
            "written_frames": written_frames,
        },
    )
    thread.start()
    time.sleep(0.04)

    assert writer.blocks == []
    pause_event.clear()
    deadline = time.monotonic() + 1
    while not writer.blocks and time.monotonic() < deadline:
        time.sleep(0.005)
    stop_event.set()
    thread.join(1)

    assert not errors
    assert len(writer.blocks) >= 1
    assert written_frames[0] == len(writer.blocks)


def test_programmatic_handle_stop_returns_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = threading.Event()
    expected = RecordingSession(sources=[])

    def controlled(mode: str, **kwargs: object) -> RecordingSession:
        assert mode == "microphone"
        kwargs["ready_event"].set()
        kwargs["stop_event"].wait()
        stopped.set()
        return expected

    monkeypatch.setattr(windows_module, "_record_windows_controlled", controlled)
    handle = windows_module.start_windows_recording("microphone")

    handle.stop()

    assert handle.wait(1) is expected
    assert stopped.is_set()


def test_programmatic_handle_cancel_deletes_temporary_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "raw.wav"
    audio_path.write_bytes(b"raw")
    session = RecordingSession(
        sources=[
            RecordedAudioSource(
                path=audio_path,
                sample_rate=16000,
                channels=1,
                frames=1,
                label="microphone",
            )
        ]
    )

    def controlled(mode: str, **kwargs: object) -> RecordingSession:
        del mode
        kwargs["ready_event"].set()
        kwargs["stop_event"].wait()
        return session

    monkeypatch.setattr(windows_module, "_record_windows_controlled", controlled)
    handle = windows_module.start_windows_recording("microphone")

    handle.cancel()

    with pytest.raises(RuntimeError, match="cancelled"):
        handle.wait(1)
    assert not audio_path.exists()
