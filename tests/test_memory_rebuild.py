"""Managed evidence rebuild and failed file-transcription persistence contracts."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from memory_fixtures import MEETING_ID, make_corpus, write_marked_wav
from typer.testing import CliRunner


def _memory(root):
    return importlib.import_module("here.memory.service").MemoryService(root)


def _configure(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-rebuild-secret-must-not-be-persisted")
    return root


def _state(root):
    queries = {
        "meetings": "SELECT id,legacy_session_id,status,active_revision_id,duration_ms,paused_ms "
        "FROM meetings ORDER BY id",
        "revisions": "SELECT id,meeting_id,revision_number,raw_text,content_hash,status "
        "FROM transcript_revisions ORDER BY meeting_id,revision_number",
        "segments": "SELECT id,revision_id,ordinal,text,start_ms,end_ms,timing_source,speaker_key "
        "FROM segments ORDER BY revision_id,ordinal",
        "assets": "SELECT id,meeting_id,relative_path,sha256,size_bytes,codec,sample_rate,channels,"
        "duration_ms,status FROM audio_assets ORDER BY id",
    }
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        return {key: connection.execute(query).fetchall() for key, query in queries.items()}


def test_fresh_catalog_rebuilds_both_committed_revisions_from_managed_evidence_only(
    tmp_path,
    monkeypatch,
):
    root = _configure(tmp_path, monkeypatch)
    memory = _memory(root)
    external = tmp_path / "external-origins"
    corpus = make_corpus(external)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Rebuild used network"))
    first = memory.ingest_session(corpus.identified.directory)
    assert (first.status, first.meeting_id) == ("inserted", MEETING_ID)
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        previous = connection.execute("SELECT active_revision_id FROM meetings").fetchone()[0]
    revised = memory.ingest_session(corpus.conflicting_identity.directory, revision_of=previous)
    assert (revised.status, revised.meeting_id) == ("revised", MEETING_ID)
    before = _state(root)
    assert len(before["meetings"]) == 1
    assert [row[2] for row in before["revisions"]] == [1, 2]
    assert len(before["segments"]) == 3
    assert before["meetings"][0][3] == before["revisions"][1][0]
    assert before["assets"][0][3] == corpus.identified.audio_sha256
    for artifact in (root / "meetings").rglob("*"):
        if artifact.is_file():
            assert b"synthetic-rebuild-secret-must-not-be-persisted" not in artifact.read_bytes()
    del memory
    gc.collect()
    # The service owns operation-scoped connections; close our final snapshot/checkpoint too.
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    hidden_external = tmp_path / "unavailable-origins"
    external.rename(hidden_external)
    original_open = Path.open

    def managed_only_open(path, *args, **kwargs):
        if path.absolute().is_relative_to(hidden_external.absolute()):
            pytest.fail("Rebuild read an external origin instead of managed committed evidence")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", managed_only_open)
    # Remove only the four explicitly owned SQLite files, never meeting material or backups.
    assert root.resolve().is_relative_to(tmp_path.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        database_file = root / ("library.sqlite3" + suffix)
        assert database_file.absolute().parent == root.absolute()
        assert not database_file.is_symlink()
        database_file.unlink(missing_ok=True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    processing = importlib.import_module("here.application.processing")
    monkeypatch.setattr(
        processing,
        "transcribe_recording_session",
        lambda *args, **kwargs: pytest.fail("Rebuild invoked transcription"),
    )
    restored = _memory(root)
    restored.reconcile()
    assert _state(root) == before
    restored.reconcile()
    assert _state(root) == before
    for _identity, _meeting, relative_path, digest, *_rest in before["assets"]:
        assert hashlib.sha256((root / relative_path).read_bytes()).hexdigest() == digest


def test_file_provider_failure_still_catalogs_failed_session_and_preserves_original_exit(
    tmp_path,
    monkeypatch,
):
    root = _configure(tmp_path, monkeypatch)
    cli = importlib.import_module("here.cli")
    audio = tmp_path / "external.wav"
    write_marked_wav(audio)
    original_audio = audio.read_bytes()
    sessions = tmp_path / "sessions"

    def provider_failure(session, **kwargs):
        # Only the provider fails, after real normalization publishes an owned WAV.
        assert session.sources
        normalized = session.sources[0].path
        assert normalized.is_file() and normalized != audio
        assert normalized.parent.parent == sessions
        assert (normalized.parent / "session.json").is_file()
        raise RuntimeError("Synthetic provider transport unavailable")

    monkeypatch.setattr(cli, "transcribe_recording_session", provider_failure)
    monkeypatch.setattr(
        "socket.socket.connect", lambda *args: pytest.fail("Real provider was called")
    )
    result = CliRunner().invoke(cli.app, ["trans", str(audio), "--output-dir", str(sessions)])
    assert result.exit_code == 1, result.output + repr(result.exception)
    directories = [path for path in sessions.iterdir() if not path.name.startswith(".")]
    assert len(directories) == 1
    saved = directories[0]
    assert (saved / "audio.wav").is_file()
    metadata = json.loads((saved / "session.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["failure_stage"] == "offline_transcription"
    assert metadata["recoverable_audio"] == "audio.wav"
    errors = json.loads((saved / "errors.json").read_text(encoding="utf-8"))["errors"]
    assert any(error["message"] == "Synthetic provider transport unavailable" for error in errors)
    assert audio.read_bytes() == original_audio
    assert (root / "library.sqlite3").is_file(), "Failed file artifacts bypassed the catalog"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        meeting = connection.execute("SELECT id,status FROM meetings").fetchone()
        assert meeting == (metadata["meeting_id"], "failed")
        asset = connection.execute(
            "SELECT meeting_id,relative_path,sha256,status FROM audio_assets"
        ).fetchone()
        assert asset[0] == meeting[0]
        assert asset[3] == "available"
        assert hashlib.sha256((root / asset[1]).read_bytes()).hexdigest() == asset[2]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
