import json
import threading
from pathlib import Path

import numpy as np
import pytest
from here.application import (
    ApplicationState,
    EventKind,
    HereApplicationController,
    InvalidApplicationCommand,
    StartRequest,
)
from here.application.processing import SessionProcessor
from here.application.recovery import RecoveryService
from here.output.paths import UnsafeSessionPath
from here.recording.journal import CaptureJournal
from here.transcription.client import TranscriptionResult
from loguru import logger


@pytest.fixture
def capture(tmp_path):
    journal = CaptureJournal.create(tmp_path / "sessions")
    writer = journal.open_writer(label="mic", sample_rate=8000, channels=1)
    writer.write(np.full(80, 1234, dtype=np.int16))
    writer.close()
    return journal


def prepare_retry(capture):
    def interrupted_provider(*args, **kwargs):
        raise SystemExit("interrupted before completed publication")

    with pytest.raises(SystemExit, match="interrupted before completed"):
        SessionProcessor(transcribe=interrupted_provider).process(
            capture.recording_session(), capture.root
        )
    assert json.loads((capture.destination / "session.json").read_text())["status"] == "pending"


def inject_cleanup_fault(monkeypatch, capture, fault):
    source_path = capture.directory / "source_01.wav"
    journal_path = capture.directory / "journal.json"
    if fault in {"source_unlink", "journal_unlink"}:
        target = source_path if fault == "source_unlink" else journal_path
        unlink = Path.unlink

        def deny_unlink(path, *args, **kwargs):
            if path == target:
                raise PermissionError(13, "synthetic capture cleanup denial", str(path))
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", deny_unlink)
        return

    replace = Path.replace

    def damage_journal_after_commit(path, target):
        result = replace(path, target)
        if Path(target) == capture.destination / "session.json":
            if json.loads(Path(target).read_text())["status"] == "completed":
                if fault == "journal_media":
                    source_path.write_bytes(b"damaged original journal WAV")
                else:
                    assert fault == "journal_path"
                    document = json.loads(journal_path.read_text())
                    document["sources"][0]["audio_file"] = "../../user.wav"
                    journal_path.write_text(json.dumps(document))
        return result

    monkeypatch.setattr(Path, "replace", damage_journal_after_commit)


@pytest.mark.parametrize("operation", ["process", "retry"])
@pytest.mark.parametrize(
    "fault", ["source_unlink", "journal_unlink", "journal_media", "journal_path"]
)
def test_completed_artifacts_survive_capture_cleanup_failure(
    capture, monkeypatch, operation, fault
):
    if operation == "retry":
        prepare_retry(capture)
    owned_unknown = capture.directory / "notes.keep"
    owned_unknown.write_bytes(b"unknown entry must remain")
    external = capture.root / "user.wav"
    external.write_bytes(b"external input must remain")
    warnings = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    observed_audio = []

    def transcribe(session, **kwargs):
        observed_audio.append(session.sources[0].path.read_bytes())
        return TranscriptionResult("raw provider words", "Completed transcript")

    processor = SessionProcessor(transcribe=transcribe, retry_delays=())
    inject_cleanup_fault(monkeypatch, capture, fault)
    try:
        artifacts = (
            processor.process(capture.recording_session(), capture.root)
            if operation == "process"
            else processor.retry(capture.destination)
        )
    finally:
        logger.remove(sink)

    assert artifacts.metadata.status == "completed"
    assert artifacts.metadata.meeting_id == capture.document.capture_id
    assert json.loads(artifacts.metadata_path.read_text())["status"] == "completed"
    assert artifacts.transcript_path.read_text(encoding="utf-8-sig") == "Completed transcript"
    assert artifacts.audio_path.read_bytes() == observed_audio[0]
    assert owned_unknown.read_bytes() == b"unknown entry must remain"
    assert external.read_bytes() == b"external input must remain"
    assert (capture.directory / "journal.json").exists()
    assert RecoveryService(capture.root).discover() == []
    assert any(
        "committed" in message and str(capture.destination) in message for message in warnings
    )
    if fault == "journal_media":
        with pytest.raises(RuntimeError):
            capture.discard()
    elif fault == "journal_path":
        with pytest.raises(UnsafeSessionPath):
            capture.discard()
        assert external.read_bytes() == b"external input must remain"


@pytest.mark.parametrize("operation", ["process", "retry"])
@pytest.mark.parametrize("fault", ["source_unlink", "journal_media", "journal_path"])
def test_controller_emits_persisted_after_capture_cleanup_failure(
    capture, monkeypatch, operation, fault
):
    closure_entered = threading.Event()
    closure_release = threading.Event()
    stopped = threading.Event()
    session = capture.recording_session()

    class Capture:
        def stop(self):
            stopped.set()

        def wait(self):
            assert stopped.wait(3)
            return session

    class Live:
        def complete(self):
            return TranscriptionResult("live", "Completed transcript")

        def abort(self):
            pass

        def cleanup(self):
            pass

        def wait_closed(self):
            closure_entered.set()
            assert closure_release.wait(3)

    if operation == "retry":
        prepare_retry(capture)
    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: TranscriptionResult("offline", "Completed transcript"),
        retry_delays=(),
    )
    controller = HereApplicationController(
        capture_factory=lambda *args: Capture(),
        live_factory=lambda *args: Live(),
        processor=processor,
    )
    events = []
    controller.subscribe(events.append)
    inject_cleanup_fault(monkeypatch, capture, fault)
    try:
        if operation == "process":
            controller.start(StartRequest(output_dir=capture.root))
            controller.stop()
            assert closure_entered.wait(3)
            assert controller.snapshot.state is ApplicationState.COMPLETED
            assert not controller.snapshot.worker_complete
            with pytest.raises(InvalidApplicationCommand):
                controller.start(StartRequest(output_dir=capture.root))
        else:
            controller.retry(capture.destination)
            controller.wait_until_terminal(3)
        persisted = [event for event in events if event.kind is EventKind.SESSION_PERSISTED]
        assert len(persisted) == 1
        assert persisted[0].session_dir == capture.destination
    finally:
        closure_release.set()
        stopped.set()
        controller.wait_until_terminal(3)

    assert controller.snapshot.state is ApplicationState.COMPLETED
    assert controller.snapshot.worker_complete
    assert controller.snapshot.last_error is None
    assert controller.snapshot.session_dir == capture.destination
    assert events[-1].kind is EventKind.WORKER_COMPLETED
    assert json.loads((capture.destination / "session.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("operation", ["discard", "session_cleanup", "cancelled_processing"])
def test_uncommitted_capture_cleanup_stays_strict(capture, monkeypatch, operation):
    session = capture.recording_session()
    source_path = capture.directory / "source_01.wav"
    original = source_path.read_bytes()
    inject_cleanup_fault(monkeypatch, capture, "source_unlink")

    with pytest.raises(PermissionError, match="synthetic capture cleanup denial"):
        if operation == "discard":
            capture.discard()
        elif operation == "session_cleanup":
            session.cleanup()
        else:
            cancelled = threading.Event()
            cancelled.set()
            SessionProcessor(transcribe=lambda *a, **kw: pytest.fail("provider called")).process(
                session, capture.root, cancel_event=cancelled
            )
    assert source_path.read_bytes() == original
    assert (capture.directory / "journal.json").exists()
    if operation == "cancelled_processing":
        assert (
            json.loads((capture.destination / "session.json").read_text())["status"] == "cancelled"
        )


@pytest.mark.parametrize("operation", ["process", "retry"])
def test_completed_metadata_publication_failure_stays_strict(capture, monkeypatch, operation):
    if operation == "retry":
        prepare_retry(capture)
    original_source = (capture.directory / "source_01.wav").read_bytes()
    replace = Path.replace

    def fail_completed_metadata(path, target):
        if Path(target) == capture.destination / "session.json":
            if json.loads(path.read_text())["status"] == "completed":
                raise OSError("completed metadata publication denied")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_completed_metadata)
    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: TranscriptionResult("raw", "Completed transcript"),
        retry_delays=(),
    )
    with pytest.raises(OSError, match="completed metadata publication denied"):
        if operation == "process":
            processor.process(capture.recording_session(), capture.root)
        else:
            processor.retry(capture.destination)

    assert (capture.directory / "source_01.wav").read_bytes() == original_source
    assert (capture.directory / "journal.json").exists()
    assert json.loads((capture.destination / "session.json").read_text())["status"] == "pending"
    (candidate,) = RecoveryService(capture.root).discover()
    assert candidate.session_dir == capture.destination
    assert candidate.can_retry
