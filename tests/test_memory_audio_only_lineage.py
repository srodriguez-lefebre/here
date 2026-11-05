"""Pre-manifest M1 audio recovery reserves identity without inventing capture dates."""

from __future__ import annotations

import gc
import importlib
import json
import sqlite3
from contextlib import closing
from uuid import UUID

import pytest
from memory_fixtures import write_marked_wav


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-audio-only-key")
    monkeypatch.setenv("CLEANUP_ENABLED", "false")

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic audio-only recovery allocated provider/network transport")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    directory = tmp_path / "sessions" / "legacy-audio-only"
    directory.mkdir(parents=True)
    write_marked_wav(directory / "audio.wav")
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    return root, memory, directory


def _result(text):
    client = importlib.import_module("here.transcription.client")
    segment = importlib.import_module("here.transcription.segments").TranscriptSegment
    return client.TranscriptionResult(raw_text=text, final_text=text, segments=[segment(text)])


def _revisions(root):
    return _rows(
        root,
        "SELECT id,meeting_id,revision_number,raw_text,content_hash "
        "FROM transcript_revisions ORDER BY revision_number",
    )


def test_audio_only_retry_reserves_one_owned_identity_and_preserves_unknown_dates_on_rebuild(
    tmp_path, monkeypatch
):
    root, memory, directory = _setup(tmp_path, monkeypatch)
    audio = directory / "audio.wav"
    before_audio = audio.read_bytes()
    assert set(path.name for path in directory.iterdir()) == {"audio.wav"}
    assert _rows(root, "SELECT COUNT(*) FROM meetings") == [(0,)]
    processing = importlib.import_module("here.application.processing")
    service = importlib.import_module("here.memory.service")
    database = importlib.import_module("here.memory.database")
    observed = []

    def provider(session, **kwargs):
        snapshot = {
            "source_id": session.meeting_id,
            "meetings": _rows(
                root, "SELECT id,status,active_revision_id,started_at,completed_at FROM meetings"
            ),
            "revisions": _revisions(root),
            "origins": _rows(root, "SELECT meeting_id,revision_id FROM ingest_sources"),
            "owned": False,
        }
        if len(snapshot["meetings"]) == 1:
            independent = service.MemoryService(root)
            identity = snapshot["meetings"][0][0]
            try:
                independent.begin_processing(identity)
            except database.Busy:
                snapshot["owned"] = True
            finally:
                independent.finish_capture(identity)
        observed.append(snapshot)
        return _result(f"Synthetic audio-only explicit result {len(observed)}.")

    processor = processing.SessionProcessor(
        transcribe=provider, retry_delays=(), memory_service=memory
    )
    first = processor.retry(directory)
    assert first.metadata.status == "completed" and first.metadata.meeting_id is None
    assert len(observed) == 1
    boundary = observed[0]
    assert len(boundary["meetings"]) == 1
    identity, status, active, started, completed = boundary["meetings"][0]
    assert str(UUID(identity)) == identity and status == "processing"
    assert active is None and boundary["revisions"] == []
    assert boundary["origins"] == [(identity, None)]
    assert started is None and completed is None
    assert boundary["source_id"] is None and boundary["owned"]
    first_revisions = _revisions(root)
    assert len(first_revisions) == 1
    assert first_revisions[0][1:4] == (identity, 1, "Synthetic audio-only explicit result 1.")
    # The compatibility writer has processing-derived dates; they cannot become
    # evidence for when this orphaned WAV was actually captured.
    compatibility = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    assert compatibility["started_at"] and compatibility["completed_at"]
    assert _rows(root, "SELECT started_at,completed_at FROM meetings") == [(None, None)]
    second = processor.retry(directory)
    assert second.metadata.status == "completed" and second.metadata.meeting_id is None
    assert len(observed) == 2
    assert observed[1]["source_id"] is None and observed[1]["owned"]
    assert observed[1]["meetings"] == [(identity, "processing", first_revisions[0][0], None, None)]
    assert observed[1]["revisions"] == first_revisions
    revisions = _revisions(root)
    assert len(revisions) == 2 and revisions[0] == first_revisions[0]
    assert revisions[1][1:4] == (identity, 2, "Synthetic audio-only explicit result 2.")
    assert revisions[1][0] != revisions[0][0]
    assert _rows(root, "SELECT DISTINCT meeting_id FROM ingest_sources") == [(identity,)]
    assert _rows(
        root, "SELECT id,status,active_revision_id,started_at,completed_at FROM meetings"
    ) == [(identity, "completed", revisions[1][0], None, None)]
    assert _rows(root, "SELECT timing_source,start_ms,end_ms FROM segments") == [
        ("absent", None, None),
        ("absent", None, None),
    ]
    assert audio.read_bytes() == before_audio
    assert not (directory / "memory_pending.json").exists()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    memory.reconcile()
    assert _revisions(root) == revisions
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    del processor, memory
    gc.collect()
    assert root.resolve().is_relative_to(tmp_path.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    service.MemoryService(root).reconcile()
    assert _revisions(root) == revisions
    assert _rows(
        root, "SELECT id,status,active_revision_id,started_at,completed_at FROM meetings"
    ) == [(identity, "completed", revisions[1][0], None, None)]
    assert audio.read_bytes() == before_audio and len(observed) == 2
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_audio_only_busy_reservation_aborts_before_provider_normalization_or_manifest(
    tmp_path, monkeypatch
):
    root, memory, directory = _setup(tmp_path, monkeypatch)
    processing = importlib.import_module("here.application.processing")
    database = importlib.import_module("here.memory.database")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    calls = []

    def provider(*args, **kwargs):
        calls.append("provider")
        return _result("Unexpected work before audio-only identity reservation.")

    def normalize(*args, **kwargs):
        calls.append("normalize")
        return processing.materialize_normalized_session(*args, **kwargs)

    processor = processing.SessionProcessor(
        transcribe=provider, normalize=normalize, retry_delays=(), memory_service=memory
    )
    with closing(sqlite3.connect(root / "library.sqlite3")) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(database.Busy):
                processor.retry(directory)
        finally:
            blocker.rollback()
    assert calls == []
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    assert _rows(root, "SELECT COUNT(*) FROM meetings") == [(0,)]
    assert _rows(root, "SELECT COUNT(*) FROM transcript_revisions") == [(0,)]
