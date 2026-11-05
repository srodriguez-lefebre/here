"""Native capture ownership starts before ready and survives catalog unavailability."""

from __future__ import annotations

import importlib
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest


def _configure(tmp_path, monkeypatch):
    root, sessions = tmp_path / "data", tmp_path / "sessions"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Capture recovery must not allocate provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    return root, sessions, memory


def _real_child_factory(tmp_path, behavior):
    # Reuse the already-reviewed real OwnedHelper/PCM backend and bootstrap ack.
    # Only its disposable generated script is adapted so child source is exactly
    # the same package tree as the parent, including mapped immutable RED runs.
    harness = importlib.import_module("test_isolated_capture")
    launch, helpers = harness.factory(tmp_path, behavior)
    package = importlib.import_module("here")
    source = str(Path(next(iter(package.__path__))).resolve())
    script = tmp_path / "fake_hardware.py"
    original = script.read_text(encoding="utf-8")
    script.write_text(
        "import sys, types\n"
        "package = types.ModuleType('here')\n"
        f"package.__path__ = [{source!r}]\n"
        "sys.modules['here'] = package\n" + original,
        encoding="utf-8",
    )
    return launch, helpers


def _snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def _candidate(recovery_module, journal):
    session = journal.recording_session()
    return recovery_module.RecoveryCandidate(
        journal.destination,
        journal.document.started_at.strftime("%Y%m%d_%H%M%S"),
        session.duration_seconds,
        "interrupted",
        "Synthetic prior recovery selection",
        session.duration_seconds > 0,
        journal.document.capture_id,
    )


@pytest.mark.parametrize("operation", ["discover", "materialize"])
def test_published_journal_is_owned_before_capture_constructor_returns(
    tmp_path, monkeypatch, operation
):
    root, sessions, _memory = _configure(tmp_path, monkeypatch)
    isolated = importlib.import_module("here.recording.isolated")
    journal_type = importlib.import_module("here.recording.journal").CaptureJournal
    recovery_module = importlib.import_module("here.application.recovery")
    database = importlib.import_module("here.memory.database")
    launch, helpers = _real_child_factory(tmp_path, "enumerate")
    entered, release = threading.Event(), threading.Event()
    observed, failures = {}, []

    def gated_factory(action, parameters):
        journal = journal_type.load(sessions, parameters["capture_id"])
        # A closed, real PCM checkpoint makes recovery meaningful without touching
        # a concurrently replacing journal or claiming measured hardware/audio.
        writer = journal.open_writer(label="synthetic", sample_rate=8000, channels=1)
        writer.write(np.full((1024, 1), 1234, dtype=np.int16))
        writer.close()
        observed["journal"] = journal
        entered.set()
        assert release.wait(20), "Test did not release the pre-ready factory"
        return launch(action, parameters)

    def open_capture():
        try:
            isolated.WindowsRecordingHandle(
                "microphone", sessions_root=sessions, helper_factory=gated_factory, timeout=3
            )
        except BaseException as exc:
            failures.append(exc)

    opening = threading.Thread(target=open_capture, name="synthetic-pre-ready-owner")
    opening.start()
    try:
        assert entered.wait(10), "Real handle did not publish its journal before helper_factory"
        assert opening.is_alive() and not helpers
        journal = observed["journal"]
        stale = _candidate(recovery_module, journal)
        before = _snapshot(journal.directory)
        with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
            assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 0
        recovery = recovery_module.RecoveryService(sessions)
        accesses = []
        original_load = journal_type.load
        original_metadata = recovery_module.read_session_metadata

        def tracked_load(*args, **kwargs):
            accesses.append("journal/media inspection")
            return original_load(*args, **kwargs)

        def tracked_metadata(*args, **kwargs):
            accesses.append("metadata inspection")
            return original_metadata(*args, **kwargs)

        with monkeypatch.context() as watch:
            watch.setattr(journal_type, "load", staticmethod(tracked_load))
            watch.setattr(recovery_module, "read_session_metadata", tracked_metadata)
            if operation == "discover":
                candidates = recovery.discover()
                assert not any(c.capture_id == stale.capture_id for c in candidates), (
                    "Recovery offered the live journal before capture factory returned"
                )
            else:
                with pytest.raises(database.Busy):
                    recovery.materialize(stale)
            assert accesses == [], "Ownership must be probed before metadata/media inspection"
        assert _snapshot(journal.directory) == before
        assert not journal.destination.exists()
    finally:
        release.set()
        opening.join(15)
        assert not opening.is_alive(), "Native constructor did not finish after factory release"
        for helper in helpers:
            if helper.process.poll() is None:
                helper.close()
            assert helper.process.poll() is not None and not helper.reader.is_alive()

    # The deliberately blocked real child opening reaches its ordinary deadline;
    # only after actual helper reaping can its retained material become recovery.
    assert len(helpers) == 1 and len(failures) == 1
    assert isinstance(failures[0], importlib.import_module("here.recording.models").CaptureFailed)
    assert isinstance(failures[0].cause, TimeoutError)
    candidates = recovery.discover()
    (available,) = [c for c in candidates if c.capture_id == stale.capture_id]
    assert available.can_retry
    directory = recovery.materialize(available)
    assert (directory / "audio.wav").is_file()
    assert (directory / "session.json").is_file()
    assert (directory / "source_01.wav").read_bytes() == before["source_01.wav"]


def test_native_owner_survives_reaped_helper_through_persistence_when_catalog_is_busy(
    tmp_path, monkeypatch
):
    root, sessions, memory = _configure(tmp_path, monkeypatch)
    application = importlib.import_module("here.application")
    processing = importlib.import_module("here.application.processing")
    isolated = importlib.import_module("here.recording.isolated")
    recovery_module = importlib.import_module("here.application.recovery")
    controller_harness = importlib.import_module("test_application_controller")
    launch, helpers = _real_child_factory(tmp_path, "error_after_audio")
    entered, release = threading.Event(), threading.Event()
    observed = {}

    def normalize(session, directory, **kwargs):
        observed["session"] = session
        entered.set()
        assert release.wait(20), "Test did not release real failure persistence"
        return processing.materialize_normalized_session(session, directory, **kwargs)

    def no_provider(*args, **kwargs):
        pytest.fail("Persisting a capture failure must not invoke the transcriber")

    def capture(request, sink):
        handle = isolated.WindowsRecordingHandle(
            "microphone", sessions_root=sessions, helper_factory=launch, block_sink=sink, timeout=3
        )
        observed["capture"] = handle
        return handle

    processor = processing.SessionProcessor(
        normalize=normalize, transcribe=no_provider, memory_service=memory
    )
    controller = application.HereApplicationController(
        capture_factory=capture,
        live_factory=lambda *args: controller_harness.FakeLive(),
        processor=processor,
        memory_service=memory,
    )
    with closing(sqlite3.connect(root / "library.sqlite3")) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        controller.start(application.StartRequest(output_dir=sessions))
        try:
            assert entered.wait(15), "Actual capture failure never entered file persistence"
            assert helpers and all(h.process.poll() is not None for h in helpers)
            assert all(not h.reader.is_alive() for h in helpers)
            assert blocker.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 0
            # Real DB contention prevented catalog ownership; native ownership must
            # still bridge handle.wait -> actual file normalization/publication.
            capture = observed["capture"]
            journal_type = importlib.import_module("here.recording.journal").CaptureJournal
            journal = journal_type.load(sessions, capture.meeting_id)
            before = _snapshot(journal.directory)
            recovery = recovery_module.RecoveryService(sessions)
            stale = _candidate(recovery_module, journal)
            assert not any(c.capture_id == capture.meeting_id for c in recovery.discover()), (
                "Reaped helper exposed its journal while controller persistence still owned it"
            )
            with pytest.raises(importlib.import_module("here.memory.database").Busy):
                recovery.materialize(stale)
            assert _snapshot(journal.directory) == before
        finally:
            release.set()
            terminal = controller.wait_until_terminal(15)
            blocker.rollback()
    assert terminal.worker_complete and terminal.state.value == "failed"
    assert terminal.session_dir is not None
    assert (terminal.session_dir / "audio.wav").is_file()
    assert (terminal.session_dir / "session.json").is_file()
    recovered = recovery_module.RecoveryService(sessions).discover()
    assert any(c.capture_id == observed["capture"].meeting_id and c.can_retry for c in recovered)
