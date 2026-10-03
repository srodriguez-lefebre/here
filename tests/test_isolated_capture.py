import sys
import threading
import time

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
        if behavior == "error_after_audio" and self.reads > 2:
            raise OSError("Synthetic original device failure")
        return b"\\x01\\x00" * frames
    def stop_stream(self):
        if behavior == "close": time.sleep(60)
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
    launch, helpers = factory(tmp_path, behavior)
    handle = WindowsRecordingHandle(
        "microphone", sessions_root=tmp_path / "sessions", helper_factory=launch, timeout=1.5
    )
    if behavior == "close":
        time.sleep(0.6)
        handle.stop()
    with pytest.raises(CaptureFailed):
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


def test_unexpected_child_death_keeps_parent_cause(tmp_path):
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
        with pytest.raises(CaptureFailed, match="ended unexpectedly"):
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
