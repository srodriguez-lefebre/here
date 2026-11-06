"""Whole-group GR3: accepted failure-persistence cancellation has one durable outcome."""

from __future__ import annotations

import gc
import importlib
import json
import sqlite3
import threading
from contextlib import closing
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
from memory_fixtures import write_marked_wav


def _rows(root, sql):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql).fetchall()


@pytest.mark.parametrize("identity_mode", ["canonical", "optional-none"])
def test_real_capture_failure_cancel_updates_one_catalog_identity_and_survives_rebuild(
    tmp_path, monkeypatch, identity_mode
):
    root, sessions = tmp_path / "data", tmp_path / "sessions"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Capture-failure cancellation allocated provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    application = importlib.import_module("here.application")
    processing = importlib.import_module("here.application.processing")
    journal_type = importlib.import_module("here.recording.journal").CaptureJournal
    models = importlib.import_module("here.recording.models")
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    cause = OSError("Synthetic original device lost during capture.")
    if identity_mode == "canonical":
        journal = journal_type.create(sessions, owned=False)
        writer = journal.open_writer(label="synthetic", sample_rate=8000, channels=1)
        writer.write(np.full((1024, 1), 1234, dtype=np.int16))
        writer.close()
        journal.finish(cause)
        session = journal.recording_session()
    else:
        source = tmp_path / "optional-source.wav"
        write_marked_wav(source)
        session = models.RecordingSession(
            [models.RecordedAudioSource(source, 8000, 1, 16000, "synthetic")]
        )
    source_id = session.meeting_id
    raw_bytes = session.sources[0].path.read_bytes()
    unrelated = tmp_path / "unrelated-external.wav"
    unrelated.write_bytes(raw_bytes)
    failure = models.CaptureFailed(session, cause)
    capture_cancelled = threading.Event()

    def wait_capture():
        raise failure

    capture = SimpleNamespace(meeting_id=source_id, wait=wait_capture, cancel=capture_cancelled.set)
    live = SimpleNamespace(abort=lambda: None, cleanup=lambda: None)
    processor = processing.SessionProcessor(
        transcribe=forbidden, retry_delays=(), memory_service=memory
    )
    entered, release = threading.Event(), threading.Event()
    real_preserve = processor._preserve_sources

    def preserve_with_barrier(*args, **kwargs):
        entered.set()
        assert release.wait(20), "Test did not release actual capture-failure persistence"
        return real_preserve(*args, **kwargs)

    monkeypatch.setattr(processor, "_preserve_sources", preserve_with_barrier)
    controller = application.HereApplicationController(
        capture_factory=lambda *args: capture,
        live_factory=lambda *args: live,
        processor=processor,
        memory_service=memory,
    )
    controller.start(application.StartRequest(output_dir=sessions))
    try:
        assert entered.wait(15)
        assert controller.snapshot.state.value == "processing"
        assert _rows(root, "SELECT id,status FROM meetings") == (
            [(source_id, "processing")] if source_id is not None else []
        )
        controller.cancel()
    finally:
        release.set()
        terminal = controller.wait_until_terminal(20)
    assert terminal.worker_complete and terminal.state.value == "cancelled"
    assert terminal.recoverable and terminal.last_error is None
    assert not capture_cancelled.is_set()
    directory = terminal.session_dir
    assert directory is not None
    metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    assert metadata["meeting_id"] == source_id and metadata["status"] == "cancelled"
    assert (directory / "source_01.wav").read_bytes() == raw_bytes
    assert (directory / metadata["recoverable_audio"]).is_file()
    errors = json.loads((directory / "errors.json").read_text(encoding="utf-8"))["errors"]
    assert any(
        error["type"] == "OSError"
        and error["message"] == str(cause)
        or error.get("cause_type") == "OSError"
        and error.get("cause_message") == str(cause)
        for error in errors
    )
    assert unrelated.read_bytes() == raw_bytes
    meetings = _rows(root, "SELECT id,status FROM meetings")
    assert len(meetings) == 1, "Accepted cancellation allocated a second meeting identity"
    identity = meetings[0][0]
    assert str(UUID(identity)) == identity
    if source_id is not None:
        assert identity == source_id
    assert meetings == [(identity, "cancelled")]
    assert terminal.memory_warning is None
    assert not (directory / "memory_pending.json").exists()
    revisions = _rows(
        root, "SELECT id,meeting_id,revision_number,raw_text,content_hash FROM transcript_revisions"
    )
    assert len(revisions) == 1 and revisions[0][1:3] == (identity, 1)
    active = _rows(root, "SELECT active_revision_id FROM meetings")
    assert active == [(revisions[0][0],)]
    importlib.import_module("here.application.recovery").RecoveryService(sessions).discover()
    memory.reconcile()
    assert _rows(root, "SELECT id,status FROM meetings") == [(identity, "cancelled")]
    assert not (directory / "memory_pending.json").exists()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    del processor, controller, memory
    gc.collect()
    assert root.resolve().is_relative_to(tmp_path.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    importlib.import_module("here.memory.service").MemoryService(root).reconcile()
    assert _rows(root, "SELECT id,status FROM meetings") == [(identity, "cancelled")]
    assert (
        _rows(
            root,
            "SELECT id,meeting_id,revision_number,raw_text,content_hash FROM transcript_revisions",
        )
        == revisions
    )
    assert _rows(root, "SELECT active_revision_id FROM meetings") == active
    assert (directory / "source_01.wav").read_bytes() == raw_bytes
    assert unrelated.read_bytes() == raw_bytes
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
