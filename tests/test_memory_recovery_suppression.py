"""Confirmed deletion suppresses retained native recovery without consuming its audio."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import threading
from contextlib import closing, contextmanager, nullcontext
from pathlib import Path
from uuid import UUID

import pytest
import soundfile as sf
import test_copilot_legacy_catalog_failure as legacy_guards


def _forbidden(*args, **kwargs):
    pytest.fail("Synthetic recovery must not allocate provider or network work")


@pytest.fixture(autouse=True)
def synthetic_environment(tmp_path, monkeypatch):
    data, sessions = tmp_path / "data", tmp_path / "sessions"
    safe = {
        "HERE_DATA_DIR": str(data),
        "HERE_ENV_FILE": str(tmp_path / "absent.env"),
        "TRANSCRIPTIONS_DIR": str(sessions),
        "OPENAI_API_KEY": "synthetic-w3-recovery-key",
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CLEANUP_ENABLED": "false",
    }
    for name in (
        "HERE_PROVIDER_SMOKE_KEY",
        "HERE_WHEEL_PYTHON",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in safe.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", _forbidden)
    monkeypatch.setattr("here.application.processing.transcribe_recording_session", _forbidden)

    # The existing real OwnedHelper harness gets an explicit environment, never
    # the caller's credentials/provider/wheel variables or real settings file.
    child_environment = {
        name: os.environ[name]
        for name in ("SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP", "COMSPEC")
        if name in os.environ
    }
    child_environment.update(safe)
    original_popen = subprocess.Popen

    def safe_popen(*args, **kwargs):
        kwargs["env"] = dict(child_environment)
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", safe_popen)
    return data, sessions


@pytest.fixture
def native_capture(tmp_path):
    """Reuse the real native owner/journal/PCM child with its existing fake hardware."""
    from here.recording.isolated import WindowsRecordingHandle
    from test_memory_capture_ownership import _real_child_factory

    handles, helpers = [], []

    def capture(name, sessions, sink):
        hardware = tmp_path / name
        hardware.mkdir()
        launch, children = _real_child_factory(hardware, "normal")
        # Block transport in the synthetic child before its package imports too.
        script = hardware / "fake_hardware.py"
        script.write_text(
            "import socket\n"
            "def no_network(*args, **kwargs):\n"
            "    raise AssertionError('Synthetic W3 child attempted network')\n"
            "socket.socket.connect = no_network\n"
            "socket.socket.connect_ex = no_network\n"
            "socket.create_connection = no_network\n" + script.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        try:
            handle = WindowsRecordingHandle(
                "microphone",
                sessions_root=sessions,
                block_sink=sink,
                helper_factory=launch,
                timeout=3,
            )
        finally:
            helpers.extend(children)
        handles.append(handle)
        return handle

    try:
        yield capture
    finally:
        try:
            for handle in reversed(handles):
                try:
                    handle.stop()
                    handle.wait(8)
                finally:
                    handle.close_ownership()
        finally:
            for helper in helpers:
                helper.close()
                assert helper.process.poll() is not None
                assert not helper.reader.is_alive() and not helper.writer.is_alive()


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _files(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@contextmanager
def _observe_capture_material(monkeypatch, journal):
    """Permit owner/registry/journal metadata; observe reconstruction and WAV reads."""
    from here.recording.journal import CaptureJournal

    reads = []
    original_session, original_info, original_open = (
        CaptureJournal.recording_session,
        sf.info,
        Path.open,
    )
    protected = {journal.directory.absolute(), journal.destination.absolute()}

    def is_audio(path):
        try:
            actual = Path(path).absolute()
        except TypeError:
            return False
        return actual.parent in protected and actual.suffix.lower() == ".wav"

    def recording_session(current, *args, **kwargs):
        if current.document.capture_id == journal.document.capture_id:
            reads.append("native recording reconstruction")
        return original_session(current, *args, **kwargs)

    def info(path, *args, **kwargs):
        if is_audio(path):
            reads.append("WAV inspection")
        return original_info(path, *args, **kwargs)

    def opened(path, *args, **kwargs):
        if is_audio(path):
            reads.append("WAV open")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as watch:
        watch.setattr(CaptureJournal, "recording_session", recording_session)
        watch.setattr(sf, "info", info)
        watch.setattr(Path, "open", opened)
        yield reads


def _assert_closed_native(handle, data, sessions):
    from here.memory.service import MemoryService
    from here.output.ownership import FileLease
    from here.recording.journal import CaptureJournal

    assert not handle._thread.is_alive()
    assert handle._helper.process.poll() is not None
    assert not handle._helper.reader.is_alive() and not handle._helper.writer.is_alive()
    MemoryService.check_owner(data, handle.meeting_id)
    with closing(FileLease(CaptureJournal.owner_path(sessions, handle.meeting_id), create=False)):
        pass


def test_confirmed_delete_hides_retained_completed_native_capture_before_audio_inspection(
    tmp_path, monkeypatch, synthetic_environment, native_capture
):
    """Catch journal fallback bypassing a real completed meeting's durable suppression."""
    from here.application import ApplicationState, HereApplicationController, StartRequest
    from here.application.models import SourceMode
    from here.application.processing import SessionProcessor
    from here.application.recovery import RecoveryService
    from here.memory.service import MemoryService
    from here.recording.journal import CaptureJournal
    from here.transcription.client import TranscriptionResult
    from loguru import logger
    from test_application_controller import FakeLive
    from test_committed_capture_cleanup import inject_cleanup_fault

    data, sessions = synthetic_environment
    memory = MemoryService(data, sessions_root=sessions)
    audio_ready = threading.Event()
    captures, warnings = [], []

    class SyntheticLive(FakeLive):
        def complete(self):
            return TranscriptionResult(
                "Synthetic W3 raw evidence", "Synthetic W3 completed evidence"
            )

    with monkeypatch.context() as cleanup_fault:

        def capture(request, sink):
            def received(*args):
                sink(*args)
                audio_ready.set()

            handle = native_capture("deleted-hardware", request.output_dir, received)
            captures.append(handle)
            journal = CaptureJournal.load(sessions, handle.meeting_id)
            inject_cleanup_fault(cleanup_fault, journal, "source_unlink")
            return handle

        controller = HereApplicationController(
            capture_factory=capture,
            live_factory=lambda *args: SyntheticLive(),
            processor=SessionProcessor(transcribe=_forbidden, memory_service=memory),
            memory_service=memory,
        )
        sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
        try:
            controller.start(StartRequest(output_dir=sessions, source_mode=SourceMode.MICROPHONE))
            assert audio_ready.wait(15), "Synthetic native capture produced no PCM"
            controller.stop()
            outcome = controller.wait_until_terminal(15)
            assert outcome.worker_complete and outcome.state is ApplicationState.COMPLETED
            assert outcome.last_error is None
        finally:
            try:
                if controller.snapshot.state in {
                    ApplicationState.PREPARING,
                    ApplicationState.RECORDING,
                    ApplicationState.PAUSED,
                }:
                    controller.stop()
                controller.wait_until_terminal(15)
            finally:
                logger.remove(sink_id)

    (handle,) = captures
    _assert_closed_native(handle, data, sessions)
    journal = CaptureJournal.load(sessions, handle.meeting_id)
    identity, destination = journal.document.capture_id, journal.destination
    assert str(UUID(identity)) == identity == handle.meeting_id
    assert outcome.session_dir == destination
    assert any(
        "metadata committed" in message and str(destination) in message for message in warnings
    )
    metadata = json.loads((destination / "session.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed" and metadata["meeting_id"] == identity
    assert journal.recording_session().duration_seconds > 0
    detail = memory.get_meeting(identity)
    assert detail.meeting.status == "completed" and detail.meeting.active_revision_id is not None
    (active,) = [
        revision
        for revision in detail.revisions
        if revision.id == detail.meeting.active_revision_id
    ]
    assert active.meeting_id == identity
    assert active.origin == os.path.normcase(os.path.normpath(str(destination.absolute())))
    assert memory.retry_parent(destination, identity) == (
        identity,
        detail.meeting.active_revision_id,
    )
    origins = _rows(
        data,
        "SELECT origin_path,origin_fingerprint FROM ingest_sources WHERE meeting_id=?",
        (identity,),
    )
    assert origins
    source_before, journal_before = _files(destination), _files(journal.directory)
    assert {"source_01.wav", "journal.json"} <= journal_before.keys()

    # This sibling uses the same real native fixture but has no completed source
    # publication; its stopped, unsuppressed journal is genuinely recoverable.
    sibling_audio = threading.Event()
    sibling = native_capture("healthy-hardware", sessions, lambda *args: sibling_audio.set())
    assert sibling_audio.wait(10)
    sibling.stop()
    assert sibling.wait(8).duration_seconds > 0
    _assert_closed_native(sibling, data, sessions)
    healthy = CaptureJournal.load(sessions, sibling.meeting_id)
    healthy_before = _files(healthy.directory)

    result = memory.delete_meeting(identity)
    assert result.status == "completed" and result.meeting_id == identity
    tombstone = json.loads((data / ".deletions" / f"{identity}.json").read_text(encoding="utf-8"))
    assert tombstone["meeting_id"] == identity and tombstone["phase"] == "complete"
    recorded_origins = {
        (entry["origin_path"], entry["origin_fingerprint"]) for entry in tombstone["origins"]
    }
    assert recorded_origins == set(origins)
    assert memory.show(identity) is None
    assert _rows(data, "SELECT id FROM meetings WHERE id=?", (identity,)) == []
    assert _files(destination) == source_before and _files(journal.directory) == journal_before

    with _observe_capture_material(monkeypatch, journal) as reads:
        candidates = RecoveryService(sessions).discover()
    assert _files(destination) == source_before and _files(journal.directory) == journal_before
    assert _files(healthy.directory) == healthy_before
    (available,) = [item for item in candidates if item.capture_id == sibling.meeting_id]
    assert available.session_dir == healthy.destination and available.can_retry
    assert available.recorded_duration_seconds > 0
    leaked = [
        (item.capture_id, str(item.session_dir), item.can_retry)
        for item in candidates
        if item.capture_id == identity or item.session_dir == destination
    ]
    assert (leaked, reads) == ([], []), "Deleted native recovery was exposed or inspected"


@pytest.mark.parametrize("source_id", [None, legacy_guards.OPAQUE_ID], ids=["none", "opaque"])
def test_legacy_recovery_resolves_actual_origin_owner_before_audio(
    tmp_path, monkeypatch, synthetic_environment, source_id
):
    from here.application.recovery import RecoveryService
    from here.memory.database import Busy
    from here.memory.service import MemoryService

    data, sessions = synthetic_environment
    source = legacy_guards._failed_session(sessions / "legacy", source_id)
    memory = MemoryService(data, sessions_root=sessions)
    before = _files(source.directory)
    # This is the existing explicit trusted-retry preparation, not ordinary
    # import (which correctly rejects opaque IDs) or a manufactured owner row.
    identity, parent = memory.prepare_retry(source.directory, source_id)
    assert str(UUID(identity)) == identity and identity != source_id
    assert memory.retry_parent(source.directory, source_id) == (identity, parent)
    assert _files(source.directory) == before
    recovery = RecoveryService(sessions)
    (stale,) = recovery.discover()
    assert stale.capture_id == source_id and stale.can_retry
    private = tmp_path / "unrelated-synthetic.txt"
    private.write_bytes(b"Synthetic unrelated material")
    memory.begin_processing(identity)
    try:
        with legacy_guards._observe_material(monkeypatch, source, private) as reads:
            assert recovery.discover() == []
            with pytest.raises(Busy):
                recovery.materialize(stale)
        assert reads == [] and _files(source.directory) == before
    finally:
        memory.finish_capture(identity)
    (available,) = recovery.discover()
    assert available.capture_id == source_id and available.can_retry
    assert recovery.materialize(available) == source.directory
    assert _files(source.directory) == before


def test_corrupt_legacy_catalog_remains_metadata_only(tmp_path, monkeypatch):
    # Reuse one accepted C5 representative, including its healthy sibling and
    # stale-selection refusal, without copying the accepted A privacy matrix.
    legacy_guards.test_corrupt_catalog_keeps_legacy_session_visible_without_assuming_ownership(
        tmp_path, monkeypatch, legacy_guards.OPAQUE_ID
    )


def test_ambiguous_legacy_catalog_remains_metadata_only(tmp_path, monkeypatch):
    legacy_guards.test_ambiguous_real_legacy_origin_mapping_is_a_visible_nonretryable_warning(
        tmp_path, monkeypatch
    )


@pytest.mark.parametrize("owner", ["busy", "unsafe"])
def test_native_owner_guards_reconstruction_before_audio(
    monkeypatch, synthetic_environment, native_capture, owner
):
    from here.application.recovery import RecoveryService
    from here.output.ownership import FileLease
    from here.recording.journal import CaptureJournal

    data, sessions = synthetic_environment
    audio_ready = threading.Event()
    handle = native_capture("guard-hardware", sessions, lambda *args: audio_ready.set())
    assert audio_ready.wait(10)
    handle.stop()
    assert handle.wait(8).duration_seconds > 0
    _assert_closed_native(handle, data, sessions)
    journal = CaptureJournal.load(sessions, handle.meeting_id)
    before = _files(journal.directory)
    owner_path = CaptureJournal.owner_path(sessions, handle.meeting_id)
    if owner == "unsafe":
        owner_path.unlink()
        owner_path.mkdir()  # A directory is never an ordinary OS lease file.
    guard = (
        closing(FileLease(owner_path, timeout=0, create=False))
        if owner == "busy"
        else nullcontext()
    )
    with guard, _observe_capture_material(monkeypatch, journal) as reads:
        assert not any(
            item.capture_id == handle.meeting_id for item in RecoveryService(sessions).discover()
        )
    assert reads == [] and _files(journal.directory) == before
    assert not journal.destination.exists()
