"""Copilot C3: rejected completed replay stays visible without provider retry."""

from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
from contextlib import closing

import pytest
from memory_fixtures import MEETING_ID, make_session
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QComboBox, QPushButton


def _rows(root, sql):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql).fetchall()


def _setup_pending(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-recovery-warning-key")
    monkeypatch.setenv("CLEANUP_ENABLED", "false")

    def forbidden(*args, **kwargs):
        pytest.fail("Completed library recovery invoked provider/network transport")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    source = make_session(tmp_path / "sessions" / "recorded", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    previous = _rows(root, "SELECT id,meeting_id,raw_text,content_hash FROM transcript_revisions")
    processing = importlib.import_module("here.application.processing")
    client = importlib.import_module("here.transcription.client")
    segment = importlib.import_module("here.transcription.segments").TranscriptSegment
    calls = []

    def provider(*args, **kwargs):
        calls.append("explicit retry")
        text = "Synthetic completed result waiting for its local catalog publication."
        return client.TranscriptionResult(raw_text=text, final_text=text, segments=[segment(text)])

    blocker = sqlite3.connect(root / "library.sqlite3")
    blocker.execute("BEGIN IMMEDIATE")
    try:
        artifacts = processing.SessionProcessor(
            transcribe=provider, retry_delays=(), memory_service=memory
        ).retry(source.directory)
        assert artifacts.metadata.status == "completed" and calls == ["explicit retry"]
        marker = source.directory / "memory_pending.json"
        document = json.loads(marker.read_text(encoding="utf-8"))
        assert document["schema_version"] == 3 and document["revision_of"] == previous[0][0]
        for name, digest in document["hashes"].items():
            assert hashlib.sha256((source.directory / name).read_bytes()).hexdigest() == digest
    except BaseException:
        blocker.rollback()
        blocker.close()
        raise
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(processing, "transcribe_recording_session", forbidden)
    return root, source, previous, calls, blocker


@pytest.mark.parametrize("rejection", ["sqlite-busy", "tampered-seal"])
def test_completed_replay_rejection_is_visible_and_cannot_offer_provider_retry(
    tmp_path, monkeypatch, qtbot, rejection
):
    root, source, previous, calls, blocker = _setup_pending(tmp_path, monkeypatch)
    try:
        if rejection == "tampered-seal":
            blocker.rollback()
            (source.directory / "raw_transcript.txt").write_text(
                "Synthetic changed local evidence no longer matches its final seal.",
                encoding="utf-8-sig",
            )
        before = {
            path.name: path.read_bytes() for path in source.directory.iterdir() if path.is_file()
        }
        recovery = importlib.import_module("here.application.recovery").RecoveryService(
            source.directory.parent
        )
        first = recovery.discover()
        second = recovery.discover()
        assert {
            path.name: path.read_bytes() for path in source.directory.iterdir() if path.is_file()
        } == before
        assert (
            _rows(root, "SELECT id,meeting_id,raw_text,content_hash FROM transcript_revisions")
            == previous
        )
        assert _rows(root, "SELECT id,status,active_revision_id FROM meetings") == [
            (MEETING_ID, "completed", previous[0][0])
        ]
        assert calls == ["explicit retry"]
        assert len(first) == len(second) == 1, (
            "Completed catalog failure disappeared from discovery"
        )
        candidate = second[0]
        assert candidate.session_dir == source.directory and candidate.capture_id == MEETING_ID
        assert candidate.status == "completed" and candidate.can_retry is False
        assert isinstance(candidate.error_summary, str) and candidate.error_summary.strip()
        contract = importlib.import_module("here.ui.contract")
        core = importlib.import_module("here.ui.fake_controller").FakeApplicationController()
        retry_calls = []
        monkeypatch.setattr(core, "retry", lambda *args: retry_calls.append(args))

        def no_credentials(*args, **kwargs):
            pytest.fail("A library-only warning requested provider credentials")

        monkeypatch.setattr(contract, "require_provider_key", no_credentials)
        adapter = contract.ApplicationUiAdapter(
            core, source.directory.parent, recovery_service=recovery
        )
        settings = QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
        preferences = importlib.import_module("here.ui.preferences").VisualPreferences(settings)
        window = importlib.import_module("here.ui.main_window").MainWindow(adapter, preferences)
        qtbot.addWidget(window)
        window.show()
        window.set_recovery(second)
        selector = window.findChild(QComboBox, "recoverySelector")
        button = window.findChild(QPushButton, "recoverButton")
        assert selector.currentData() == candidate
        assert candidate.error_summary in selector.currentText()
        assert not button.isEnabled()
        selected = QSignalSpy(window.recoveryRequested)
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        assert selected.count() == 0
        with pytest.raises(RuntimeError):
            adapter.retry_candidate(candidate)
        assert retry_calls == [] and calls == ["explicit retry"]
    finally:
        blocker.rollback()
        blocker.close()


def test_control_successful_local_replay_and_completed_without_marker_stay_hidden(
    tmp_path, monkeypatch
):
    root, source, previous, calls, blocker = _setup_pending(tmp_path, monkeypatch)
    blocker.rollback()
    blocker.close()
    recovery = importlib.import_module("here.application.recovery").RecoveryService(
        source.directory.parent
    )
    assert recovery.discover() == []
    assert not (source.directory / "memory_pending.json").exists()
    assert recovery.discover() == []
    revisions = _rows(root, "SELECT id,meeting_id,raw_text,content_hash FROM transcript_revisions")
    assert len(revisions) == 2 and previous[0] in revisions
    assert _rows(root, "SELECT status FROM meetings") == [("completed",)]
    assert calls == ["explicit retry"]
