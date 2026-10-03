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
from pathlib import Path
behavior = BEHAVIOR
def marker(name): return Path(__file__).with_suffix("." + name)
class Stream:
    reads = 0
    def __init__(self, index): self.index = index
    def gate(self, operation):
        if marker(f"block-{operation}-{self.index}").exists():
            marker(f"blocked-{operation}-{self.index}").touch()
            time.sleep(60)
    def get_read_available(self):
        self.gate("available")
        return 0 if marker("no-data").exists() else 1024
    def read(self, frames, **kwargs):
        self.gate("read")
        self.reads += 1
        if behavior == "read": time.sleep(60)
        if behavior == "error": raise RuntimeError("Synthetic device disconnected")
        if behavior in ("error_after_audio", "gated_error_after_audio") and self.reads > 2:
            if behavior == "gated_error_after_audio":
                from pathlib import Path
                while not Path(__file__).with_suffix(".fail").exists(): time.sleep(0.01)
            raise OSError("Synthetic original device failure")
        if marker(f"paused-{self.index}").exists(): marker(f"drained-{self.index}").touch()
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
    def get_default_wasapi_loopback(self):
        return {**self.get_default_input_device_info(), "index": 8}
    def open(self, **kwargs):
        if behavior == "open": time.sleep(60)
        return Stream(kwargs["input_device_index"])
    def terminate(self): pass
sys.modules["pyaudiowpatch"] = types.SimpleNamespace(PyAudio=Audio, paInt16=8)
from here.recording.journal import CaptureJournal, CaptureWriter
checkpoint = CaptureWriter.checkpoint
event = CaptureJournal.event
def source_index(label): return 7 if label == "microphone" else 8
def checkpointed(writer):
    checkpoint(writer)
    marker(f"checkpoint-{source_index(writer.source.label)}").touch()
def observed_event(journal, kind, **details):
    event(journal, kind, **details)
    if kind in ("paused", "resumed"):
        marker(f"{kind}-{source_index(details['source'])}").touch()
    if kind == "scheduling_gap":
        marker(f"gap-done-{source_index(details['source'])}").touch()
CaptureWriter.checkpoint = checkpointed
CaptureJournal.event = observed_event
if behavior == "catchup":
    import here.recording.windows as windows
    capture = windows._capture_windows_stream_to_file
    write = CaptureWriter.write
    def delayed_capture(*args, **kwargs):
        kwargs["start_time"] -= 4.0
        capture(*args, **kwargs)
    def slow_write(writer, data):
        index = source_index(writer.source.label)
        if marker(f"block-write-{index}").exists():
            marker(f"blocked-write-{index}").touch()
            time.sleep(60)
        if not marker(f"gap-done-{index}").exists():
            marker(f"gap-start-{index}").touch()
            time.sleep(0.025)
        write(writer, data)
    windows._capture_windows_stream_to_file = delayed_capture
    CaptureWriter.write = slow_write
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


@pytest.mark.parametrize("state", ["recording", "paused", "resumed"])
def test_stop_uses_close_deadline_after_real_child_enters_cleanup(tmp_path, monkeypatch, state):
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
        if state != "recording":
            pause_and_drain(handle, tmp_path)
        if state == "resumed":
            handle.resume()
            wait_for(lambda: (tmp_path / "fake_hardware.resumed-7").exists())
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


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.01)
    raise AssertionError("Synthetic capture condition did not arrive")


def journal_for(handle):
    return CaptureJournal.load(handle._journal.root, handle._journal.document.capture_id)


def pause_and_drain(handle, tmp_path, indexes=(7,)):
    handle.pause()
    # Child-side markers follow the real pause event and a completed drain read.
    # Polling a journal during atomic replacement interferes with Windows writes.
    try:
        wait_for(lambda: all((tmp_path / f"fake_hardware.drained-{i}").exists() for i in indexes))
    except AssertionError as exc:
        markers = {
            index: {
                kind: (tmp_path / f"fake_hardware.{kind}-{index}").exists()
                for kind in ("checkpoint", "paused", "drained")
            }
            for index in indexes
        }
        helper = handle._helper
        before_reap = {
            "exit_code": helper.process.poll(),
            "done": handle._done.is_set(),
            "owner_error": repr(handle._error),
            "child_error": helper.get("error"),
            "tick": helper.get("tick"),
        }
        reap_capture(handle, [helper])
        # Only inspect durable state after the child and its owner have stopped.
        # Preserve the pre-reap status so our cleanup is not mistaken for the cause.
        try:
            capture_errors = [
                event
                for event in journal_for(handle).document.events
                if event["kind"] == "capture_error"
            ]
        except Exception as journal_error:
            capture_errors = repr(journal_error)
        raise AssertionError(
            f"Pause drain failed: markers={markers}; before_reap={before_reap}; "
            f"reaped_exit_code={helper.process.returncode}; capture_errors={capture_errors}"
        ) from exc


def reap_capture(handle, helpers):
    if helpers[0].process.poll() is None:
        helpers[0].process.kill()
    handle._thread.join(4)
    assert not handle._thread.is_alive()
    assert not helpers[0].reader.is_alive()
    assert not helpers[0].writer.is_alive()


@pytest.mark.parametrize("finish", ["complete", "stop", "pause"])
def test_healthy_catchup_writes_outlive_reader_timeout(tmp_path, finish):
    import numpy as np
    import soundfile as sf

    launch, helpers = factory(tmp_path, "catchup")
    blocks = []
    handle = WindowsRecordingHandle(
        "microphone",
        sessions_root=tmp_path / "sessions",
        helper_factory=launch,
        block_sink=lambda label, data, *args: blocks.append(data.copy()),
        timeout=1.5,
    )
    try:
        wait_for(lambda: blocks)
        first = wait_for(lambda: helpers[0].get("tick"))["progress"].get("microphone", 0)
        assert not handle._done.wait(2.1), "Healthy inserted-silence writes timed out"
        assert not (tmp_path / "fake_hardware.gap-done-7").exists()
        assert helpers[0].get("tick")["progress"]["microphone"] > first
        if finish == "complete":
            wait_for(lambda: (tmp_path / "fake_hardware.gap-done-7").exists(), timeout=6)
            wait_for(lambda: any(np.any(block) for block in blocks))
        elif finish == "pause":
            pause_and_drain(handle, tmp_path)
            assert not handle._done.wait(0.1)
        started = time.monotonic()
        handle.stop()
        saved = handle.wait(4)
        assert time.monotonic() - started < 1.5
        assert helpers[0].process.returncode == 0
        assert handle.live_error is None
        data, _ = sf.read(saved.sources[0].path, dtype="int16", always_2d=True)
        np.testing.assert_array_equal(data, np.concatenate(blocks))
        assert saved.sources[0].frames == len(data)
    finally:
        reap_capture(handle, helpers)
    gaps = [
        event for event in journal_for(handle).document.events if event["kind"] == "scheduling_gap"
    ]
    assert gaps
    assert gaps[0]["details"]["inserted_silence_frames"] >= 60 * 1024
    assert not np.any(data[: gaps[0]["details"]["inserted_silence_frames"]])
    if finish != "complete":
        assert len(data) < 180 * 1024, "Stop/pause failed to interrupt catch-up per block"


@pytest.mark.parametrize("mode", ["microphone", "both"])
def test_actual_blocked_catchup_write_still_times_out(tmp_path, mode):
    launch, helpers = factory(tmp_path, "catchup")
    handle = WindowsRecordingHandle(
        mode, sessions_root=tmp_path / "sessions", helper_factory=launch, timeout=1.5
    )
    try:
        wait_for(lambda: (tmp_path / "fake_hardware.checkpoint-7").exists())
        (tmp_path / "fake_hardware.block-write-7").touch()
        wait_for(lambda: (tmp_path / "fake_hardware.blocked-write-7").exists())
        if mode == "both":
            first = wait_for(lambda: helpers[0].get("tick"))["progress"].get("system audio", 0)
        assert handle._done.wait(3), "A blocked WAV write lost its finite deadline"
        with pytest.raises(CaptureFailed, match="reader stopped responding"):
            handle.wait(0)
        if mode == "both":
            assert helpers[0].get("tick")["progress"]["system audio"] > first
    finally:
        reap_capture(handle, helpers)
    assert journal_for(handle).document.state == "interrupted"


@pytest.mark.parametrize(
    ("moment", "operation", "mode", "stalled_index"),
    [
        ("active", "read", "microphone", 7),
        ("paused", "read", "microphone", 7),
        ("paused", "available", "microphone", 7),
        ("active", "read", "both", 7),
        ("paused", "read", "both", 7),
        ("paused", "read", "both", 8),
    ],
)
def test_pause_cannot_hide_an_actual_stalled_reader(
    tmp_path, moment, operation, mode, stalled_index
):
    from here.application.recovery import RecoveryService
    from here.output.metadata import ErrorMetadataDocument

    launch, helpers = factory(tmp_path)
    handle = WindowsRecordingHandle(
        mode, sessions_root=tmp_path / "sessions", helper_factory=launch, timeout=1.5
    )
    indexes = (7, 8) if mode == "both" else (7,)
    try:
        wait_for(
            lambda: all((tmp_path / f"fake_hardware.checkpoint-{i}").exists() for i in indexes)
        )
        if moment == "paused":
            pause_and_drain(handle, tmp_path, indexes)
        (tmp_path / f"fake_hardware.block-{operation}-{stalled_index}").touch()
        wait_for(lambda: (tmp_path / f"fake_hardware.blocked-{operation}-{stalled_index}").exists())
        if moment == "active":
            handle.pause()
        assert handle._done.wait(3), "Pause removed the stalled reader's finite deadline"
        with pytest.raises(CaptureFailed, match="reader stopped responding"):
            handle.wait(0)
        assert helpers[0].process.poll() is not None
    finally:
        reap_capture(handle, helpers)
    journal = journal_for(handle)
    assert journal.document.state == "interrupted"
    assert all(source.frames >= 1024 for source in journal.document.sources)
    (error,) = [event for event in journal.document.events if event["kind"] == "capture_error"]
    assert error["details"] == {
        "error_type": "TimeoutError",
        "error_message": "Windows audio reader stopped responding",
    }
    recovery = RecoveryService(journal.root)
    (candidate,) = recovery.discover()
    assert candidate.can_retry
    directory = recovery.materialize(candidate)
    (saved_error,) = ErrorMetadataDocument.model_validate_json(
        (directory / "errors.json").read_text()
    ).errors
    assert saved_error.type == "TimeoutError"
    assert saved_error.message == error["details"]["error_message"]
    assert saved_error.occurred_at == datetime.fromisoformat(error["occurred_at"])


@pytest.mark.parametrize("no_data", [False, True])
@pytest.mark.parametrize("finish", ["stop", "resume"])
def test_healthy_pause_outlives_watchdog_without_audio(tmp_path, no_data, finish):
    launch, helpers = factory(tmp_path)
    live_frames = []
    handle = WindowsRecordingHandle(
        "microphone",
        sessions_root=tmp_path / "sessions",
        helper_factory=launch,
        block_sink=lambda label, data, *args: live_frames.append(len(data)),
        timeout=1.5,
    )
    try:
        wait_for(lambda: live_frames)
        pause_and_drain(handle, tmp_path)
        journal = journal_for(handle)
        frames = journal.document.sources[0].frames
        wait_for(lambda: (helpers[0].get("tick") or {}).get("counts") == {"microphone": frames})
        wait_for(lambda: sum(live_frames) == frames)
        if no_data:
            (tmp_path / "fake_hardware.no-data").touch()
        raw = journal.recording_session().sources[0].path
        before = raw.read_bytes()
        paused_events = journal.document.events
        assert not handle._done.wait(2.1), "A healthy paused reader was timed out"
        assert sum(live_frames) == frames
        assert raw.read_bytes() == before
        assert journal_for(handle).document.events == paused_events
        assert helpers[0].get("tick")["counts"] == {"microphone": frames}
        if finish == "resume":
            (tmp_path / "fake_hardware.no-data").unlink(missing_ok=True)
            # Rapid commands must not turn an old pause acknowledgement into health.
            for _ in range(5):
                handle.resume()
                handle.pause()
            handle.resume()
            wait_for(lambda: sum(live_frames) > frames)
        handle.stop()
        saved = handle.wait(4)
        assert saved.sources[0].frames == sum(live_frames)
        if finish == "stop":
            assert saved.sources[0].frames == frames
        else:
            assert saved.sources[0].frames > frames
        assert handle.live_error is None
        assert helpers[0].process.returncode == 0
    finally:
        reap_capture(handle, helpers)


@pytest.mark.parametrize("delivery", ["stale", "missing", "rollback"])
def test_paused_reader_requires_new_progress_despite_child_heartbeats(
    tmp_path, monkeypatch, delivery
):
    launch, helpers = factory(tmp_path)
    handle = WindowsRecordingHandle(
        "microphone", sessions_root=tmp_path / "sessions", helper_factory=launch, timeout=1.5
    )
    try:
        wait_for(lambda: (tmp_path / "fake_hardware.checkpoint-7").exists())
        pause_and_drain(handle, tmp_path)
        frozen = wait_for(lambda: helpers[0].get("tick"))
        original_get = helpers[0].get
        polls = 0

        def get(kind):
            nonlocal polls
            if kind != "tick":
                return original_get(kind)
            polls += 1
            if polls % 2 == 0:
                if delivery == "missing":
                    return None
                if delivery == "rollback":
                    return {**frozen, "progress": {"microphone": 0}}
            return frozen

        monkeypatch.setattr(helpers[0], "get", get)
        assert handle._done.wait(3), "Cached or missing ticks renewed paused reader health"
        with pytest.raises(CaptureFailed, match="reader stopped responding"):
            handle.wait(0)
        assert polls > 2
    finally:
        reap_capture(handle, helpers)


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
