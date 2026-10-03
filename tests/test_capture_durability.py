from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import here.recording.models as models
import here.recording.windows as windows
import numpy as np
import pytest
import soundfile as sf
from here.application import ApplicationState, HereApplicationController, StartRequest
from here.application.processing import SessionProcessingFailed, SessionProcessor
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.client import TranscriptionResult


def audio_session(tmp_path: Path) -> RecordingSession:
    path = tmp_path / "synthetic.wav"
    sf.write(path, np.full((1600, 2), 0.25), 8000, subtype="PCM_16")
    return RecordingSession(
        [RecordedAudioSource(path, 8000, 2, 1600, "microphone", "Synthetic Mic")]
    )


@pytest.mark.parametrize("cancel", [False, True])
def test_reader_failure_preserves_closed_material_except_explicit_cancel(
    monkeypatch, tmp_path, cancel
):
    paths = []
    writers = []
    cause = OSError("synthetic reader failed")
    monkeypatch.setitem(
        sys.modules,
        "pyaudiowpatch",
        SimpleNamespace(PyAudio=lambda: SimpleNamespace(terminate=lambda: None)),
    )
    monkeypatch.setattr(
        windows, "_get_default_windows_input_device", lambda: {"name": "Synthetic Mic"}
    )
    monkeypatch.setattr(
        windows, "_get_default_windows_loopback_device", lambda: {"name": "Synthetic Loopback"}
    )
    monkeypatch.setattr(windows, "_open_windows_input_stream", lambda *args: (object(), 8000, 1))

    def open_writer(rate, channels):
        path = tmp_path / f"source-{len(paths)}.wav"
        writer = sf.SoundFile(path, mode="w", samplerate=rate, channels=channels, subtype="PCM_16")
        paths.append(path)
        writers.append(writer)
        return path, writer

    def reader(stream, **kwargs):
        if kwargs["label"] == "microphone":
            kwargs["writer"].write(np.full((32, 1), 1234, dtype=np.int16))
            kwargs["written_frames"][0] = 32
        else:
            kwargs["errors"].append(cause)
        kwargs["stop_event"].set()

    monkeypatch.setattr(windows, "open_temp_soundfile", open_writer)
    monkeypatch.setattr(windows, "_capture_windows_stream_to_file", reader)
    cancelled = threading.Event()
    if cancel:
        cancelled.set()
    arguments = dict(
        stop_event=threading.Event(),
        pause_event=threading.Event(),
        cancel_event=cancelled,
        ready_event=threading.Event(),
    )
    if cancel:
        assert windows._record_windows_controlled("both", **arguments).sources == []
        assert all(not path.exists() for path in paths)
    else:
        with pytest.raises(RuntimeError) as caught:
            windows._record_windows_controlled("both", **arguments)
        assert hasattr(caught.value, "session"), "capture error must expose useful partial material"
        assert caught.value.__cause__ is cause
        assert all(writer.closed for writer in writers)
        session = caught.value.session
        assert len(session.sources) == 2
        assert session.sources[0].frames == 32
        assert sf.info(session.sources[0].path).frames == 32
        assert session.sources[1].device_name == "Synthetic Loopback"


def test_controller_persists_capture_failure_and_retries_without_recording(monkeypatch, tmp_path):
    session = audio_session(tmp_path)
    cause = OSError("device lost")
    failure_type = getattr(models, "CaptureFailed", None)
    assert failure_type is not None, "partial capture needs a typed failure"
    failure = failure_type(session, cause)
    live = SimpleNamespace(abort=lambda: None, cleanup=lambda: None)

    def capture_factory(request, sink):
        def wait():
            raise failure from cause

        return SimpleNamespace(wait=wait)

    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: TranscriptionResult("recovered", "recovered"),
        retry_delays=(),
    )
    controller = HereApplicationController(
        capture_factory=capture_factory, live_factory=lambda *args: live, processor=processor
    )
    controller.start(StartRequest(output_dir=tmp_path / "sessions"))
    snapshot = controller.wait_until_terminal(3)
    assert snapshot.state is ApplicationState.FAILED
    assert snapshot.recoverable
    assert snapshot.last_error.stage == "capture"
    assert snapshot.session_dir is not None
    metadata = json.loads((snapshot.session_dir / "session.json").read_text())
    assert metadata["status"] == "failed"
    assert not (snapshot.session_dir / "transcript.txt").exists()
    assert sf.info(snapshot.session_dir / "audio.wav").frames == 3200
    errors = json.loads((snapshot.session_dir / "errors.json").read_text())["errors"]
    assert errors[0]["cause_message"] == "device lost"
    controller.retry()
    assert controller.wait_until_terminal(3).state is ApplicationState.COMPLETED


def test_normalization_failure_preserves_raw_audio_for_retry(monkeypatch, tmp_path):
    import here.application.processing as processing

    session = audio_session(tmp_path)
    normalize = processing.materialize_normalized_session
    monkeypatch.setattr(
        processing,
        "materialize_normalized_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("normalizer failed")),
    )
    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: TranscriptionResult("recovered", "recovered"),
        retry_delays=(),
    )
    with pytest.raises(SessionProcessingFailed) as caught:
        processor.process(session, tmp_path / "sessions")
    assert caught.value.recoverable, "saved raw audio must remain retryable"
    metadata = json.loads((caught.value.session_dir / "session.json").read_text())
    assert metadata["capture_sources"][0]["device_name"] == "Synthetic Mic"
    assert (
        sf.info(caught.value.session_dir / metadata["capture_sources"][0]["audio_file"]).frames
        == 1600
    )
    monkeypatch.setattr(processing, "materialize_normalized_session", normalize)
    assert processor.retry(caught.value.session_dir).metadata.status == "completed"


def test_handle_exposes_partial_failure_and_cancel_removes_it(monkeypatch, tmp_path):
    for cancel in (False, True):
        session = audio_session(tmp_path)
        cause = OSError("reader lost")

        def controlled(mode, **kwargs):
            kwargs["ready_event"].set()
            kwargs["stop_event"].wait()
            raise models.CaptureFailed(session, cause) from cause

        monkeypatch.setattr(windows, "_record_windows_controlled", controlled)
        handle = windows.start_windows_recording("microphone")
        if cancel:
            handle.cancel()
            with pytest.raises(RuntimeError, match="cancelled"):
                handle.wait(2)
            assert not session.sources[0].path.exists()
        else:
            handle.stop()
            with pytest.raises(models.CaptureFailed) as caught:
                handle.wait(2)
            assert caught.value.session is session
            assert caught.value.__cause__ is cause
            assert sf.info(session.sources[0].path).frames == 1600


def test_capture_writer_finalization_error_retains_material(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "pyaudiowpatch",
        SimpleNamespace(PyAudio=lambda: SimpleNamespace(terminate=lambda: None)),
    )
    monkeypatch.setattr(
        windows, "_get_default_windows_input_device", lambda: {"name": "Synthetic Mic"}
    )
    monkeypatch.setattr(windows, "_open_windows_input_stream", lambda *args: (object(), 8000, 1))
    cause = OSError("finalize failed")
    path = tmp_path / "finalized.wav"
    writer = sf.SoundFile(path, mode="w", samplerate=8000, channels=1)

    class FailingClose:
        def write(self, data):
            writer.write(data)

        def close(self):
            writer.close()
            raise cause

    monkeypatch.setattr(windows, "open_temp_soundfile", lambda *args: (path, FailingClose()))

    def reader(stream, **kwargs):
        kwargs["writer"].write(np.ones((32, 1)))
        kwargs["written_frames"][0] = 32
        kwargs["stop_event"].set()

    monkeypatch.setattr(windows, "_capture_windows_stream_to_file", reader)
    with pytest.raises(models.CaptureFailed) as caught:
        windows._record_windows_controlled(
            "microphone",
            stop_event=threading.Event(),
            pause_event=threading.Event(),
            cancel_event=threading.Event(),
            ready_event=threading.Event(),
        )
    assert caught.value.__cause__ is cause
    assert writer.closed
    assert sf.info(path).frames == 32
