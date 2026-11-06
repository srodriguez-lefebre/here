"""Copilot C2: lifecycle and default processing use the injected catalog root."""

from __future__ import annotations

import importlib
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest


def _rows(root, sql):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql).fetchall()


@pytest.mark.parametrize("processor_mode", ["default", "caller-injected"])
def test_controller_processes_into_custom_memory_root_with_real_session_processor(
    tmp_path, monkeypatch, processor_mode
):
    default_root, custom_root = tmp_path / "default-data", tmp_path / "custom-data"
    sessions = tmp_path / "sessions"
    monkeypatch.setenv("HERE_DATA_DIR", str(default_root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic controller allocated real provider/network transport")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    application = importlib.import_module("here.application")
    processing = importlib.import_module("here.application.processing")
    client = importlib.import_module("here.transcription.client")
    segment = importlib.import_module("here.transcription.segments").TranscriptSegment
    memory = importlib.import_module("here.memory.service").MemoryService(custom_root)
    journal = importlib.import_module("here.recording.journal").CaptureJournal.create(sessions)
    writer = journal.open_writer(label="synthetic", sample_rate=8000, channels=1)
    writer.write(np.full((1024, 1), 1234, dtype=np.int16))
    writer.close()
    journal.finish()
    session = journal.recording_session()
    capture = SimpleNamespace(meeting_id=session.meeting_id, wait=lambda: session)

    def unavailable_live():
        raise RuntimeError("Synthetic live route unavailable; use explicit offline provider")

    live = SimpleNamespace(
        complete=unavailable_live,
        chunk_metadata=lambda: [],
        abort=lambda: None,
        cleanup=lambda: None,
    )
    options = {"memory_service": memory}
    if processor_mode == "caller-injected":
        options["processor"] = processing.SessionProcessor(memory_service=memory)
    controller = application.HereApplicationController(
        capture_factory=lambda *args: capture, live_factory=lambda *args: live, **options
    )
    calls = []
    text = "Synthetic transcript belongs in the controller's injected catalog."

    def provider(recording, **kwargs):
        calls.append(recording.meeting_id)
        assert recording.sources[0].path.is_file()
        return client.TranscriptionResult(raw_text=text, final_text=text, segments=[segment(text)])

    # Only the external transcriber boundary is replaced. The default constructor,
    # actual processor, controller, writer and catalog remain in the exercised path.
    monkeypatch.setattr(controller._processor, "_transcribe", provider)
    assert not (default_root / "library.sqlite3").exists()
    controller.start(application.StartRequest(output_dir=sessions))
    terminal = controller.wait_until_terminal(20)
    assert terminal.worker_complete and terminal.state.value == "completed"
    assert terminal.last_error is None and calls == [session.meeting_id]
    custom_meetings = _rows(custom_root, "SELECT id,status,active_revision_id FROM meetings")
    default_meetings = (
        _rows(default_root, "SELECT id,status,active_revision_id FROM meetings")
        if (default_root / "library.sqlite3").exists()
        else []
    )
    assert len(custom_meetings) == 1 and custom_meetings[0][1] == "completed", (
        f"Injected catalog was not finalized; other root contains {default_meetings}"
    )
    revisions = _rows(
        custom_root, "SELECT id,meeting_id,revision_number,raw_text FROM transcript_revisions"
    )
    assert len(revisions) == 1 and revisions[0][1:] == (session.meeting_id, 1, text)
    assert custom_meetings == [(session.meeting_id, "completed", revisions[0][0])]
    assert default_meetings == [] and not (default_root / "library.sqlite3").exists()
    assert terminal.memory_warning is None
    assert not (terminal.session_dir / "memory_pending.json").exists()
    with closing(sqlite3.connect(custom_root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
