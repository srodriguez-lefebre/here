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

    original_open = windows.CaptureJournal.open_writer

    def open_writer(journal, **kwargs):
        writer = original_open(journal, **kwargs)
        paths.append(writer.path)
        writers.append(writer._writer)
        return writer

    def reader(stream, **kwargs):
        if kwargs["label"] == "microphone":
            kwargs["writer"].write(np.full((32, 1), 1234, dtype=np.int16))
            kwargs["written_frames"][0] = 32
        else:
            kwargs["errors"].append(cause)
        kwargs["stop_event"].set()

    monkeypatch.setattr(windows.CaptureJournal, "open_writer", open_writer)
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
    writers = []
    original_close = windows.CaptureWriter.close

    def failing_close(writer):
        original_close(writer)
        writers.append(writer)
        raise cause

    monkeypatch.setattr(windows.CaptureWriter, "close", failing_close)

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
    assert writers[0]._writer.closed
    assert sf.info(writers[0].path).frames == 32


@pytest.mark.parametrize(
    "unsafe", ["../outside.wav", "/outside.wav", "C:\\outside.wav", "nested/../../outside.wav"]
)
def test_retry_rejects_raw_audio_paths_outside_session(monkeypatch, tmp_path, unsafe):
    import here.application.processing as processing

    session = audio_session(tmp_path)
    monkeypatch.setattr(
        processing,
        "materialize_normalized_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("failed")),
    )
    processor = SessionProcessor(retry_delays=())
    with pytest.raises(SessionProcessingFailed) as caught:
        processor.process(session, tmp_path / "sessions")
    metadata_path = caught.value.session_dir / "session.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["capture_sources"][0]["audio_file"] = unsafe
    metadata_path.write_text(json.dumps(metadata))
    observed = []
    monkeypatch.setattr(
        processing,
        "materialize_normalized_session",
        lambda *args, **kwargs: (
            observed.append(args) or (_ for _ in ()).throw(OSError("opened unsafe path"))
        ),
    )
    with pytest.raises(ValueError, match="session"):
        processor.retry(caught.value.session_dir)
    assert observed == [], "unsafe source must be rejected before normalization opens it"
    assert session.sources[0].path.exists()


@pytest.mark.parametrize(
    "name,mode",
    [
        ("record_mic_windows", "microphone"),
        ("record_os_windows", "system_audio"),
        ("record_both_windows", "both"),
    ],
)
def test_legacy_enter_adapter_uses_shared_controlled_capture(monkeypatch, name, mode):
    session = RecordingSession([])
    calls = []
    stopped = []
    handle = SimpleNamespace(stop=lambda: stopped.append(True), wait=lambda: session)
    monkeypatch.setattr(
        windows,
        "start_windows_recording",
        lambda actual_mode, **kwargs: calls.append((actual_mode, kwargs)) or handle,
    )
    monkeypatch.setattr(
        windows,
        "_get_default_windows_input_device",
        lambda: (_ for _ in ()).throw(RuntimeError("legacy capture path reached")),
    )
    monkeypatch.setattr(
        windows,
        "_get_default_windows_loopback_device",
        lambda: (_ for _ in ()).throw(RuntimeError("legacy capture path reached")),
    )
    monkeypatch.setattr("builtins.input", lambda: "")
    sink = object()
    assert getattr(windows, name)(block_sink=sink) is session
    assert calls == [(mode, {"block_sink": sink})]
    assert stopped == [True]


def test_enter_adapter_interrupt_cancels_shared_handle(monkeypatch):
    cancelled = []
    handle = SimpleNamespace(
        cancel=lambda: cancelled.append(True),
        wait=lambda: (_ for _ in ()).throw(RuntimeError("cancelled")),
    )
    monkeypatch.setattr(windows, "start_windows_recording", lambda *args, **kwargs: handle)
    monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        windows.record_mic_windows()
    assert cancelled == [True]


def test_retry_rejects_source_symlink_resolving_outside_session(monkeypatch, tmp_path):
    import here.application.processing as processing

    session = audio_session(tmp_path)
    normalize = processing.materialize_normalized_session
    monkeypatch.setattr(
        processing,
        "materialize_normalized_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("failed")),
    )
    processor = SessionProcessor(retry_delays=())
    with pytest.raises(SessionProcessingFailed) as caught:
        processor.process(session, tmp_path / "sessions")
    raw_path = caught.value.session_dir / "source_01.wav"
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"synthetic outside sentinel")
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == raw_path:
            return outside
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(processing, "materialize_normalized_session", normalize)
    with pytest.raises(ValueError, match="resolve inside the session"):
        processor.retry(caught.value.session_dir)
    assert outside.read_bytes() == b"synthetic outside sentinel"
