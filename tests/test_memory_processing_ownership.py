"""Durable processing state and real ownership span provider waits and failures."""

from __future__ import annotations

import importlib
import json
import sqlite3
import threading
from contextlib import closing

import pytest
from memory_fixtures import MEETING_ID, make_session, write_marked_wav


def _configure(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-processing-key")
    return root


def _memory(root):
    return importlib.import_module("here.memory.service").MemoryService(root)


def _query(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3", timeout=3)) as connection:
        return connection.execute(sql, parameters).fetchall()


def _result(text):
    client = importlib.import_module("here.transcription.client")
    segments = importlib.import_module("here.transcription.segments")
    return client.TranscriptionResult(
        raw_text=text,
        final_text=text,
        segments=[segments.TranscriptSegment(text, start=0.0, end=1.0)],
    )


def _forbid_provider(*args, **kwargs):
    pytest.fail("Local catalog recovery invoked a provider or network")


@pytest.mark.parametrize(
    "producer,fails",
    [
        ("file", False),
        ("controller-retry", False),
        ("controller-retry", True),
    ],
    ids=["file-completed", "retry-completed", "retry-failed"],
)
def test_provider_wait_keeps_durable_processing_and_owner_until_outcome(
    tmp_path,
    monkeypatch,
    producer,
    fails,
):
    root = _configure(tmp_path, monkeypatch)
    memory = _memory(root)
    processing = importlib.import_module("here.application.processing")
    application = importlib.import_module("here.application")
    recovery = importlib.import_module("here.application.recovery")
    database = importlib.import_module("here.memory.database")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    previous = None
    if producer == "controller-retry":
        source = make_session(sessions / "recorded", meeting_id=MEETING_ID)
        assert memory.ingest_session(source.directory).status == "inserted"
        previous = _query(root, "SELECT id,raw_text FROM transcript_revisions")[0]
    else:
        audio = tmp_path / "external.wav"
        write_marked_wav(audio)
    entered, release = threading.Event(), threading.Event()
    observed = {}
    outcome = {}

    def provider(session, **kwargs):
        assert session.sources and session.sources[0].path.is_file()
        directory = session.sources[0].path.parent
        metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        observed.update(directory=directory, meeting_id=metadata["meeting_id"])
        entered.set()
        assert release.wait(20), "Test did not release the synthetic provider"
        if fails:
            raise RuntimeError("Synthetic retry provider failure")
        return _result("Fresh processing evidence")

    processor = processing.SessionProcessor(
        transcribe=provider,
        retry_delays=(),
        memory_service=memory,
    )
    controller = None
    worker = None
    if producer == "controller-retry":
        controller = application.HereApplicationController(
            processor=processor, memory_service=memory
        )
        controller.retry(source.directory)
    else:

        def run_file():
            try:
                outcome["artifacts"] = processor.transcribe_audio_path(audio, sessions)
            except BaseException as error:
                outcome["error"] = error

        worker = threading.Thread(target=run_file, name="synthetic-file-producer")
        worker.start()
    competitor = _memory(root)
    try:
        assert entered.wait(15), repr(outcome.get("error"))
        identity = observed["meeting_id"]
        expected_active = previous[0] if previous else None
        assert _query(
            root, "SELECT status,active_revision_id FROM meetings WHERE id=?", (identity,)
        ) == [("processing", expected_active)]
        with pytest.raises(database.Busy):
            competitor.begin_capture(identity)
        recovery.RecoveryService(sessions).discover()
        _memory(root).reconcile()
        assert _query(
            root, "SELECT status,active_revision_id FROM meetings WHERE id=?", (identity,)
        ) == [("processing", expected_active)]
        if previous:
            assert _query(root, "SELECT id,raw_text FROM transcript_revisions") == [previous]
    finally:
        release.set()
        if worker is not None:
            worker.join(20)
            assert not worker.is_alive()
        if controller is not None:
            controller.wait_until_terminal(20)
        if observed.get("meeting_id"):
            competitor.finish_capture(observed["meeting_id"])
    assert "error" not in outcome, repr(outcome.get("error"))
    identity = observed["meeting_id"]
    expected_status = "failed" if fails else "completed"
    assert _query(root, "SELECT status FROM meetings WHERE id=?", (identity,)) == [
        (expected_status,)
    ]
    if controller is not None:
        assert controller.snapshot.state.value == expected_status
    if fails:
        assert _query(root, "SELECT id,raw_text FROM transcript_revisions") == [previous]
        assert _query(root, "SELECT active_revision_id FROM meetings") == [(previous[0],)]
    else:
        assert _query(
            root,
            "SELECT r.raw_text FROM transcript_revisions r JOIN meetings m "
            "ON m.active_revision_id=r.id WHERE m.id=?",
            (identity,),
        ) == [("Fresh processing evidence",)]
        assert _query(root, "SELECT COUNT(*) FROM transcript_revisions") == [
            (2 if previous else 1,)
        ]
    # An independent service can acquire the actual ownership lock after worker teardown.
    competitor.begin_capture(identity)
    competitor.finish_capture(identity)
    assert _query(root, "SELECT status FROM meetings WHERE id=?", (identity,)) == [
        (expected_status,)
    ]


def test_busy_failed_retry_replays_only_failed_state_and_preserves_completed_evidence(
    tmp_path,
    monkeypatch,
):
    root = _configure(tmp_path, monkeypatch)
    memory = _memory(root)
    processing = importlib.import_module("here.application.processing")
    application = importlib.import_module("here.application")
    source = make_session(tmp_path / "sessions" / "recorded", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    prior = _query(root, "SELECT id,raw_text,content_hash FROM transcript_revisions")[0]
    audio_before = source.audio.read_bytes()
    entered, release = threading.Event(), threading.Event()

    def failed_provider(session, **kwargs):
        assert session.sources[0].path.is_file()
        entered.set()
        assert release.wait(20), "Test did not release the synthetic failing provider"
        raise RuntimeError("Synthetic provider failed after prior completed revision")

    processor = processing.SessionProcessor(
        transcribe=failed_provider,
        retry_delays=(),
        memory_service=memory,
    )
    controller = application.HereApplicationController(processor=processor, memory_service=memory)
    controller.retry(source.directory)
    try:
        assert entered.wait(15)
        # Reserve the DB only after the provider starts: preflight and owner acquisition are real.
        with closing(
            sqlite3.connect(root / "library.sqlite3", isolation_level=None, timeout=3)
        ) as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            try:
                release.set()
                assert controller.wait_until_terminal(20).state.value == "failed"
                assert _query(
                    root, "SELECT id,raw_text,content_hash FROM transcript_revisions"
                ) == [prior]
                assert _query(root, "SELECT active_revision_id FROM meetings") == [(prior[0],)]
                marker = source.directory / "memory_pending.json"
                assert marker.is_file(), "Failed retry lost its durable state-only recovery intent"
                intent = json.loads(marker.read_text(encoding="utf-8"))
                assert intent["meeting_id"] == MEETING_ID
                assert intent["revision_of"] == prior[0]
                assert prior[1] not in marker.read_text(encoding="utf-8")
            finally:
                blocker.rollback()
    finally:
        release.set()
        controller.wait_until_terminal(20)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(processing, "transcribe_recording_session", _forbid_provider)
    monkeypatch.setattr("here.transcription.client.build_client", _forbid_provider)
    monkeypatch.setattr("socket.socket.connect", _forbid_provider)
    recovery = importlib.import_module("here.application.recovery").RecoveryService(
        source.directory.parent
    )
    recovery.discover()
    recovery.discover()
    assert not marker.exists()
    assert _query(root, "SELECT id,status,active_revision_id FROM meetings") == [
        (MEETING_ID, "failed", prior[0])
    ]
    assert _query(root, "SELECT id,raw_text,content_hash FROM transcript_revisions") == [prior]
    assert _query(root, "SELECT COUNT(*) FROM audio_assets") == [(1,)]
    assert source.audio.read_bytes() == audio_before
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
