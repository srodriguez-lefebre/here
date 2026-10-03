import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from here.application import ApplicationState, EventKind, HereApplicationController, StartRequest
from here.application.processing import SessionProcessor
from here.application.recovery import RecoveryService
from here.output.metadata import ErrorMetadata, ErrorMetadataDocument, SessionEventMetadata
from here.recording.journal import CaptureJournal
from here.recording.models import CaptureFailed, RecordedAudioSource, RecordingSession


def make_capture(root):
    journal = CaptureJournal.create(root)
    writer = journal.open_writer(label="mic", sample_rate=8000, channels=1)
    writer.write(np.full(80, 1234, dtype=np.int16))
    writer.close()
    return journal


@pytest.mark.parametrize("message", ["original device failure", ""])
@pytest.mark.parametrize("empty_override", [False, True])
def test_persistence_uses_current_journal_errors_in_order(tmp_path, message, empty_override):
    journal = make_capture(tmp_path)
    session = journal.recording_session()
    raw = session.sources[0].path.read_bytes()
    journal.finish(OSError(message))
    journal.event("capture_error", error_type="TimeoutError", error_message="device close failed")
    occurred = [datetime.fromisoformat(event["occurred_at"]) for event in journal.document.events]
    foreign_event = SessionEventMetadata(
        kind="capture_error",
        occurred_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        recorded_duration_seconds=0,
        details={"error_type": "RuntimeError", "error_message": "unrelated capture failure"},
    )
    # The attached document predates the real failures; even a stale cached error
    # or a caller-supplied lifecycle event must not replace current journal evidence.
    session.journal.document.events.append(foreign_event.model_dump(mode="json"))
    original_errors = [] if empty_override else None
    artifacts = SessionProcessor().persist_capture_failure(
        CaptureFailed(session, RuntimeError("IPC transport wrapper")),
        tmp_path,
        events=[foreign_event],
        original_errors=original_errors,
    )

    errors = ErrorMetadataDocument.model_validate_json(artifacts.errors_path.read_text()).errors
    assert [(error.type, error.message, error.occurred_at) for error in errors] == [
        ("OSError", message, occurred[0]),
        ("TimeoutError", "device close failed", occurred[1]),
    ]
    assert artifacts.errors.errors == errors
    assert original_errors == ([] if empty_override else None)
    assert all(error.cause_type is None and error.cause_message is None for error in errors)
    assert artifacts.metadata.status == "failed"
    assert artifacts.metadata.meeting_id == journal.document.capture_id
    assert (artifacts.session_dir / "source_01.wav").read_bytes() == raw
    assert sf.info(artifacts.audio_path).frames == 160
    assert not journal.directory.exists()
    (candidate,) = RecoveryService(tmp_path).discover()
    assert candidate.can_retry and candidate.capture_id == journal.document.capture_id
    assert candidate.error_summary == "device close failed"


def test_restart_recovery_preserves_empty_original_capture_message(tmp_path):
    journal = make_capture(tmp_path)
    journal.finish(OSError(""))
    occurred = datetime.fromisoformat(journal.document.events[-1]["occurred_at"])
    recovery = RecoveryService(tmp_path)
    (candidate,) = recovery.discover()

    directory = recovery.materialize(candidate)

    (error,) = ErrorMetadataDocument.model_validate_json(
        (directory / "errors.json").read_text()
    ).errors
    assert (error.type, error.message, error.occurred_at) == ("OSError", "", occurred)
    assert recovery.discover()[0].can_retry


def test_explicit_capture_errors_take_priority_without_mutating_caller_list(tmp_path):
    journal = make_capture(tmp_path)
    journal.finish(OSError("journal device failure"))
    original_errors = [
        ErrorMetadata(
            stage="capture",
            type="SelectedDeviceError",
            message="",
            retryable=False,
            occurred_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
    ]
    before = [error.model_dump() for error in original_errors]

    def unavailable_normalizer(*args, **kwargs):
        raise OSError("normalization unavailable")

    artifacts = SessionProcessor(normalize=unavailable_normalizer).persist_capture_failure(
        CaptureFailed(journal.recording_session(), RuntimeError("IPC wrapper")),
        tmp_path,
        original_errors=original_errors,
    )

    assert [error.model_dump() for error in original_errors] == before
    assert artifacts.errors.errors[0].model_dump() == before[0]
    assert [(error.stage, error.type, error.message) for error in artifacts.errors.errors] == [
        ("capture", "SelectedDeviceError", ""),
        ("recoverable_audio", "OSError", "normalization unavailable"),
    ]
    assert artifacts.audio_path is None
    pcm, rate = sf.read(artifacts.session_dir / "source_01.wav", dtype="int16")
    assert rate == 8000
    np.testing.assert_array_equal(pcm, [1234] * 80)
    assert RecoveryService(tmp_path).discover()[0].can_retry


@pytest.mark.parametrize(
    ("kind", "details"),
    [
        ("capture_stopped", {"error_type": "OSError", "error_message": "not a failure"}),
        ("capture_error", {}),
        ("capture_error", {"error_type": "OSError"}),
        ("capture_error", {"error_type": None, "error_message": "missing type"}),
        ("capture_error", {"error_type": "", "error_message": "missing type"}),
        ("capture_error", {"error_type": 42, "error_message": "invalid type"}),
        ("capture_error", {"error_type": "OSError", "error_message": None}),
        ("capture_error", {"error_type": "OSError", "error_message": 42}),
    ],
)
def test_unusable_journal_error_keeps_capture_failure_fallback(tmp_path, kind, details):
    journal = make_capture(tmp_path)
    journal.event(kind, **details)
    failure = CaptureFailed(journal.recording_session(), TimeoutError("parent capture timeout"))

    artifacts = SessionProcessor().persist_capture_failure(failure, tmp_path)

    (error,) = artifacts.errors.errors
    assert (error.type, error.message, error.cause_type, error.cause_message) == (
        "CaptureFailed",
        "Audio capture failed: parent capture timeout",
        "TimeoutError",
        "parent capture timeout",
    )


def test_no_journal_keeps_wrapper_fallback_and_preserves_audio(tmp_path):
    source = tmp_path / "capture.wav"
    sf.write(source, np.full(80, 1234, dtype=np.int16), 8000, subtype="PCM_16")
    raw = source.read_bytes()
    failure = CaptureFailed(
        RecordingSession([RecordedAudioSource(source, 8000, 1, 80, "mic")]),
        OSError("in-process device failure"),
    )
    unrelated_event = SessionEventMetadata(
        kind="capture_error",
        occurred_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        recorded_duration_seconds=0,
        details={"error_type": "RuntimeError", "error_message": "unrelated capture"},
    )

    artifacts = SessionProcessor().persist_capture_failure(
        failure, tmp_path / "sessions", events=[unrelated_event]
    )

    (error,) = artifacts.errors.errors
    assert (error.type, error.message, error.cause_type, error.cause_message) == (
        "CaptureFailed",
        "Audio capture failed: in-process device failure",
        "OSError",
        "in-process device failure",
    )
    assert not source.exists()
    assert (artifacts.session_dir / "source_01.wav").read_bytes() == raw
    assert sf.info(artifacts.audio_path).frames == 160
    assert RecoveryService(tmp_path / "sessions").discover()[0].can_retry


@pytest.mark.parametrize("message", ["original device failure", ""])
def test_controller_exposes_persisted_capture_error_after_journal_cleanup(tmp_path, message):
    journal = make_capture(tmp_path)
    journal.finish(OSError(message))
    journal.event("capture_error", error_type="TimeoutError", error_message="device close failed")
    occurred = datetime.fromisoformat(journal.document.events[0]["occurred_at"])
    failure = CaptureFailed(journal.recording_session(), RuntimeError("IPC transport wrapper"))

    def fail_capture():
        raise failure

    def unavailable_normalizer(*args, **kwargs):
        raise OSError("normalization unavailable")

    controller = HereApplicationController(
        capture_factory=lambda *args: SimpleNamespace(wait=fail_capture),
        live_factory=lambda *args: SimpleNamespace(abort=lambda: None, cleanup=lambda: None),
        processor=SessionProcessor(normalize=unavailable_normalizer),
    )
    events = []
    controller.subscribe(events.append)
    controller.start(StartRequest(output_dir=tmp_path))
    snapshot = controller.wait_until_terminal(3)

    assert snapshot.state is ApplicationState.FAILED
    assert snapshot.recoverable and snapshot.worker_complete
    assert not journal.directory.exists()
    assert (
        snapshot.last_error.stage,
        snapshot.last_error.error_type,
        snapshot.last_error.message,
        snapshot.last_error.occurred_at,
    ) == ("capture", "OSError", message, occurred)
    (event,) = [event for event in events if event.kind is EventKind.ERROR_RECORDED]
    assert event.error == snapshot.last_error
    assert events[-1].kind is EventKind.WORKER_COMPLETED
    errors = json.loads((snapshot.session_dir / "errors.json").read_text())["errors"]
    assert [(error["type"], error["message"]) for error in errors] == [
        ("OSError", message),
        ("TimeoutError", "device close failed"),
        ("OSError", "normalization unavailable"),
    ]


@pytest.mark.parametrize("error_document", ["missing", "empty", "other-stage"])
def test_older_processor_without_persisted_capture_error_keeps_ui_fallback(
    tmp_path, error_document
):
    artifacts = SimpleNamespace(session_dir=tmp_path / "session")
    if error_document != "missing":
        artifacts.errors = ErrorMetadataDocument(
            errors=[]
            if error_document == "empty"
            else [
                ErrorMetadata(
                    stage="recoverable_audio",
                    type="OSError",
                    message="normalization unavailable",
                    retryable=True,
                    occurred_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                )
            ]
        )

    def fail_capture():
        raise CaptureFailed(RecordingSession([]), OSError("original in-process failure"))

    controller = HereApplicationController(
        capture_factory=lambda *args: SimpleNamespace(wait=fail_capture),
        live_factory=lambda *args: SimpleNamespace(abort=lambda: None, cleanup=lambda: None),
        processor=SimpleNamespace(persist_capture_failure=lambda *args, **kwargs: artifacts),
    )
    events = []
    controller.subscribe(events.append)
    controller.start(StartRequest(output_dir=tmp_path))
    snapshot = controller.wait_until_terminal(3)

    assert snapshot.state is ApplicationState.FAILED and snapshot.worker_complete
    assert (
        snapshot.last_error.stage,
        snapshot.last_error.error_type,
        snapshot.last_error.message,
    ) == (
        "capture",
        "CaptureFailed",
        "Audio capture failed: original in-process failure",
    )
    (event,) = [event for event in events if event.kind is EventKind.ERROR_RECORDED]
    assert event.error == snapshot.last_error
