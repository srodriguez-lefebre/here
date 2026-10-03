import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import pytest
from here.audio_helper import OwnedHelper
from here.recording.isolated import WindowsRecordingHandle
from here.recording.journal import CaptureJournal
from here.recording.models import CaptureFailed

FAKE_BACKEND = """
import sys, time, types
behavior = BEHAVIOR
class Stream:
    reads = 0
    def get_read_available(self): return 1024
    def read(self, frames, **kwargs):
        self.reads += 1
        if behavior == "read": time.sleep(60)
        if behavior == "error": raise RuntimeError("Synthetic device disconnected")
        if behavior in ("error_after_audio", "gated_error_after_audio") and self.reads > 2:
            if behavior == "gated_error_after_audio":
                from pathlib import Path
                while not Path(__file__).with_suffix(".fail").exists(): time.sleep(0.01)
            raise OSError("Synthetic original device failure")
        return b"\\x01\\x00" * frames
    def stop_stream(self):
        if behavior == "close":
            from pathlib import Path
            Path(__file__).with_suffix(".closing").touch()
            time.sleep(60)
    def close(self): pass
class Audio:
    def get_default_input_device_info(self):
        if behavior == "enumerate": time.sleep(60)
        return {"name": "Actual fake USB", "index": 7, "maxInputChannels": 1,
                "defaultSampleRate": 48000}
    def get_default_wasapi_loopback(self): return self.get_default_input_device_info()
    def open(self, **kwargs):
        if behavior == "open": time.sleep(60)
        return Stream()
    def terminate(self): pass
sys.modules["pyaudiowpatch"] = types.SimpleNamespace(PyAudio=Audio, paInt16=8)
from here.recording.helper_worker import run
raise SystemExit(run())
"""


def factory(tmp_path, behavior="normal"):
    script = tmp_path / "fake_hardware.py"
    script.write_text(FAKE_BACKEND.replace("BEHAVIOR", repr(behavior)))
    helpers = []

    def launch(action, params):
        helper = OwnedHelper(action, params, command=[sys.executable, str(script)])
        helpers.append(helper)
        return helper

    return launch, helpers


def test_actual_opened_descriptors_and_stop_save_from_real_child(tmp_path):
    launch, helpers = factory(tmp_path)
    blocks = threading.Event()
    handle = WindowsRecordingHandle(
        "microphone",
        sessions_root=tmp_path / "sessions",
        helper_factory=launch,
        block_sink=lambda *args: blocks.set(),
        timeout=3,
    )
    assert blocks.wait(3)
    handle.stop()
    session = handle.wait(4)
    assert handle.opened_sources[0].device_name == "Actual fake USB"
    assert handle.opened_sources[0].device_index == 7
    assert session.sources[0].frames > 0
    assert handle.live_error is None
    assert helpers[0].process.poll() == 0
    assert not helpers[0].reader.is_alive()


@pytest.mark.parametrize("behavior", ["enumerate", "open"])
def test_stalled_open_has_owner_until_reaped_and_keeps_journal(tmp_path, behavior):
    launch, helpers = factory(tmp_path, behavior)
    with pytest.raises(TimeoutError, match="opening"):
        WindowsRecordingHandle(
            "microphone", sessions_root=tmp_path / "sessions", helper_factory=launch, timeout=1.5
        )
    assert helpers[0].process.poll() is not None
    assert not helpers[0].reader.is_alive()
    assert list((tmp_path / "sessions" / ".captures").glob("*/journal.json"))
    capture_id = next((tmp_path / "sessions" / ".captures").iterdir()).name
    journal = CaptureJournal.load(tmp_path / "sessions", capture_id)
    errors = [event for event in journal.document.events if event["kind"] == "capture_error"]
    assert len(errors) == 1
    assert errors[0]["details"] == {
        "error_type": "TimeoutError",
        "error_message": "Timed out while opening Windows audio devices",
    }


@pytest.mark.parametrize("behavior", ["read", "close"])
def test_stalled_read_or_close_reaps_and_preserves_capture(tmp_path, behavior):
    from here.application.processing import SessionProcessor

    launch, helpers = factory(tmp_path, behavior)
    audio = threading.Event()
    handle = WindowsRecordingHandle(
        "microphone",
        sessions_root=tmp_path / "sessions",
        helper_factory=launch,
        block_sink=lambda *args: audio.set(),
        timeout=1.5,
    )
    if behavior == "close":
        assert audio.wait(3)
        handle.stop()
    with pytest.raises(CaptureFailed) as caught:
        handle.wait(4)
    assert helpers[0].process.poll() is not None
    assert handle._journal.directory.exists()
    journal = CaptureJournal.load(handle._journal.root, handle._journal.document.capture_id)
    errors = [event for event in journal.document.events if event["kind"] == "capture_error"]
    assert len(errors) == 1
    assert errors[0]["details"] == {
        "error_type": "TimeoutError",
        "error_message": (
            "Windows audio reader stopped responding"
            if behavior == "read"
            else "Timed out closing Windows audio devices"
        ),
    }
    artifacts = SessionProcessor().persist_capture_failure(caught.value, journal.root)
    assert [
        (error.type, error.message, error.occurred_at)
        for error in artifacts.errors.errors
        if error.stage == "capture"
    ] == [
        (
            "TimeoutError",
            errors[0]["details"]["error_message"],
            datetime.fromisoformat(errors[0]["occurred_at"]),
        )
    ]


def test_stop_uses_close_deadline_after_real_child_enters_cleanup(tmp_path, monkeypatch):
    import here.recording.isolated as isolated

    now = [0.0]
    monkeypatch.setattr(
        isolated,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0],
            sleep=time.sleep,
        ),
    )
    launch, helpers = factory(tmp_path, "close")
    audio, closing = threading.Event(), threading.Event()
    handle = WindowsRecordingHandle(
        "microphone",
        sessions_root=tmp_path / "sessions",
        helper_factory=launch,
        block_sink=lambda *args: audio.set(),
        timeout=1.5,
    )
    original_get = helpers[0].get
    closed_polls = []

    def get(kind):
        if kind == "tick" and (tmp_path / "fake_hardware.closing").exists():
            # Freeze progress once real stop_stream has been entered. Two polls
            # establish unchanged counts before advancing the owner's clock.
            closed_polls.append(True)
            if len(closed_polls) >= 2:
                closing.set()
            return {"counts": {}}
        return original_get(kind)

    monkeypatch.setattr(helpers[0], "get", get)
    try:
        assert audio.wait(3)
        handle.stop()
        assert closing.wait(3)
        now[0] = 2.0
        with pytest.raises(CaptureFailed, match="Timed out closing Windows audio devices"):
            handle.wait(4)
    finally:
        if helpers[0].process.poll() is None:
            helpers[0].process.kill()
        handle._thread.join(4)
    assert not handle._thread.is_alive()
    assert not helpers[0].reader.is_alive()


def test_unexpected_child_death_keeps_parent_cause(tmp_path):
    from here.application.processing import SessionProcessor

    launch, helpers = factory(tmp_path)
    audio = threading.Event()
    handle = WindowsRecordingHandle(
        "microphone",
        sessions_root=tmp_path / "sessions",
        helper_factory=launch,
        block_sink=lambda *args: audio.set(),
        timeout=3,
    )
    try:
        assert audio.wait(3)
        helpers[0].process.kill()
        with pytest.raises(CaptureFailed, match="ended unexpectedly") as caught:
            handle.wait(4)
    finally:
        if helpers[0].process.poll() is None:
            helpers[0].process.kill()
        handle._thread.join(4)
    assert not handle._thread.is_alive()
    assert not helpers[0].reader.is_alive()
    journal = CaptureJournal.load(handle._journal.root, handle._journal.document.capture_id)
    errors = [event for event in journal.document.events if event["kind"] == "capture_error"]
    assert len(errors) == 1
    assert errors[0]["details"] == {
        "error_type": "RuntimeError",
        "error_message": "Windows audio helper ended unexpectedly",
    }
    artifacts = SessionProcessor().persist_capture_failure(caught.value, journal.root)
    (error,) = artifacts.errors.errors
    assert (error.type, error.message, error.occurred_at) == (
        "RuntimeError",
        "Windows audio helper ended unexpectedly",
        datetime.fromisoformat(errors[0]["occurred_at"]),
    )


def test_blocked_live_consumer_does_not_block_primary_and_is_owned(tmp_path):
    launch, helpers = factory(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def sink(*args):
        entered.set()
        assert release.wait(8)

    handle = WindowsRecordingHandle(
        "microphone",
        sessions_root=tmp_path / "sessions",
        helper_factory=launch,
        block_sink=sink,
        timeout=3,
    )
    try:
        assert entered.wait(2)
        time.sleep(2)
        checkpoint = CaptureJournal.load(
            handle._journal.root, handle._journal.document.capture_id
        ).recording_session()
        assert checkpoint.sources[0].frames >= 48000
        handle.stop()
        with pytest.raises(TimeoutError):
            handle.wait(0.2)
        assert handle._thread.is_alive()
    finally:
        release.set()
        handle.stop()
    session = handle.wait(4)
    assert session.sources[0].frames >= checkpoint.sources[0].frames
    assert handle.live_error is not None
    assert helpers[0].process.poll() == 0


def test_cancel_deletes_only_owned_capture_after_child_close(tmp_path):
    launch, helpers = factory(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    other = root / "keep.txt"
    other.write_text("owned elsewhere")
    handle = WindowsRecordingHandle("microphone", sessions_root=root, helper_factory=launch)
    handle.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        handle.wait(4)
    assert not handle._journal.directory.exists()
    assert other.read_text() == "owned elsewhere"
    assert helpers[0].process.poll() is not None


def test_spontaneous_device_failure_exits_child_cleanly(tmp_path):
    launch, helpers = factory(tmp_path, "error")
    try:
        handle = WindowsRecordingHandle(
            "microphone", sessions_root=tmp_path / "sessions", helper_factory=launch, timeout=3
        )
    except CaptureFailed as exc:
        assert "Synthetic device disconnected" in str(exc)
    else:
        with pytest.raises(CaptureFailed, match="Synthetic device disconnected"):
            handle.wait(4)
    assert helpers[0].process.returncode == 1


def test_original_child_error_survives_parent_and_local_recovery(tmp_path):
    from here.application.recovery import RecoveryService
    from here.output.metadata import ErrorMetadataDocument

    launch, helpers = factory(tmp_path, "error_after_audio")
    root = tmp_path / "sessions"
    try:
        handle = WindowsRecordingHandle(
            "microphone", sessions_root=root, helper_factory=launch, timeout=3
        )
    except CaptureFailed:
        pass
    else:
        with pytest.raises(CaptureFailed):
            handle.wait(4)
    assert helpers[0].process.returncode == 1
    capture_id = next((root / ".captures").iterdir()).name
    journal = CaptureJournal.load(root, capture_id)
    errors = [event for event in journal.document.events if event["kind"] == "capture_error"]
    assert journal.document.state == "interrupted"
    assert len(errors) == 1
    assert errors[0]["details"] == {
        "error_type": "OSError",
        "error_message": "Synthetic original device failure",
    }
    recovery = RecoveryService(root)
    (candidate,) = recovery.discover()
    assert candidate.can_retry
    directory = recovery.materialize(candidate)
    metadata = ErrorMetadataDocument.model_validate_json((directory / "errors.json").read_text())
    assert [(error.type, error.message) for error in metadata.errors] == [
        ("OSError", "Synthetic original device failure")
    ]
    (recovered,) = recovery.discover()
    assert recovered.error_summary == "Synthetic original device failure"


def test_running_controller_preserves_original_child_error_and_owns_closure(tmp_path):
    import soundfile as sf
    from here.application import (
        ApplicationState,
        EventKind,
        HereApplicationController,
        StartRequest,
    )
    from here.application.processing import SessionProcessor
    from here.application.recovery import RecoveryService
    from here.output.metadata import ErrorMetadataDocument, SessionEventMetadataDocument

    launch, helpers = factory(tmp_path, "gated_error_after_audio")
    root = tmp_path / "sessions"
    handles, events = [], []
    recording = threading.Event()

    def capture(request, sink):
        handle = WindowsRecordingHandle(
            "microphone", sessions_root=root, helper_factory=launch, block_sink=sink, timeout=3
        )
        handles.append(handle)
        return handle

    def observe(event):
        events.append(event)
        if event.state is ApplicationState.RECORDING:
            recording.set()

    controller = HereApplicationController(
        capture_factory=capture,
        live_factory=lambda *args: SimpleNamespace(
            submit_block=lambda *args: None, abort=lambda: None, cleanup=lambda: None
        ),
        processor=SessionProcessor(),
    )
    controller.subscribe(observe)
    try:
        controller.start(StartRequest(output_dir=root))
        assert recording.wait(5)
        (tmp_path / "fake_hardware.fail").touch()
        snapshot = controller.wait_until_terminal(6)
        assert snapshot.state is ApplicationState.FAILED
        assert snapshot.worker_complete and snapshot.recoverable
        assert helpers[0].process.returncode == 1
        assert not helpers[0].reader.is_alive()
        assert not handles[0]._thread.is_alive()
    finally:
        (tmp_path / "fake_hardware.fail").touch()
        controller.wait_until_terminal(6)
        for helper in helpers:
            helper.close()

    assert not handles[0]._journal.directory.exists()
    directory = snapshot.session_dir
    (error,) = ErrorMetadataDocument.model_validate_json(
        (directory / "errors.json").read_text()
    ).errors
    capture_events = SessionEventMetadataDocument.model_validate_json(
        (directory / "events.json").read_text()
    ).events
    (original,) = [event for event in capture_events if event.kind == "capture_error"]
    assert (error.type, error.message, error.occurred_at) == (
        "OSError",
        "Synthetic original device failure",
        original.occurred_at,
    )
    assert error.cause_type is None and error.cause_message is None
    assert (
        snapshot.last_error.error_type,
        snapshot.last_error.message,
        snapshot.last_error.occurred_at,
    ) == ("OSError", "Synthetic original device failure", original.occurred_at)
    (event,) = [event for event in events if event.kind is EventKind.ERROR_RECORDED]
    assert event.error == snapshot.last_error
    assert events[-1].kind is EventKind.WORKER_COMPLETED
    assert sf.info(directory / "source_01.wav").frames > 0
    assert sf.info(directory / "audio.wav").frames > 0
    (candidate,) = RecoveryService(root).discover()
    assert candidate.can_retry
    assert candidate.error_summary == "Synthetic original device failure"
    assert candidate.capture_id == handles[0]._journal.document.capture_id


def test_slow_pipe_never_blocks_primary_writer_and_rejects_partial_live(tmp_path, monkeypatch):
    import here.audio_helper as ipc

    original = ipc._read
    release = threading.Event()
    blocked = threading.Event()
    release.set()

    def gated_read(stream):
        if not release.is_set():
            blocked.set()
            assert release.wait(8)
        return original(stream)

    monkeypatch.setattr(ipc, "_read", gated_read)
    launch, helpers = factory(tmp_path)
    handle = WindowsRecordingHandle(
        "microphone", sessions_root=tmp_path / "sessions", helper_factory=launch, timeout=4
    )
    try:
        release.clear()
        assert blocked.wait(2)
        time.sleep(2)
        session = CaptureJournal.load(
            handle._journal.root, handle._journal.document.capture_id
        ).recording_session()
        assert session.sources[0].frames >= 48000
        handle.pause()
        handle.stop()
    finally:
        release.set()
        handle.stop()
    saved = handle.wait(5)
    assert saved.sources[0].frames >= session.sources[0].frames
    assert handle.live_error is not None
    assert helpers[0].get("result")["value"]["live_dropped"]
