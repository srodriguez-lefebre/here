"""A v1 origin without an embedded UUID retains its assigned catalog lineage."""

from __future__ import annotations

import importlib
import json
import sqlite3
from contextlib import closing
from uuid import UUID

import pytest
from memory_fixtures import TRANSCRIPT_TEXT, make_session


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def test_explicit_legacy_retry_reuses_catalog_uuid_without_rewriting_v1_identity(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-legacy-retry-key")
    module = importlib.import_module("here.memory.service")
    memory = module.MemoryService(root)
    source = make_session(tmp_path / "sessions" / "legacy")
    original_metadata = source.metadata.read_bytes()
    original_audio = source.audio.read_bytes()
    assert json.loads(original_metadata).get("meeting_id") is None
    imported = memory.ingest_session(source.directory)
    assert imported.status == "inserted"
    identity = imported.meeting_id
    assert str(UUID(identity)) == identity
    assert source.metadata.read_bytes() == original_metadata
    previous = _rows(
        root, "SELECT id,meeting_id,revision_number,raw_text,content_hash FROM transcript_revisions"
    )[0]
    assert previous[1:4] == (identity, 1, TRANSCRIPT_TEXT)
    previous_segments = _rows(
        root,
        "SELECT id,revision_id,ordinal,text,start_ms,end_ms "
        "FROM segments WHERE revision_id=? ORDER BY ordinal",
        (previous[0],),
    )
    processing = importlib.import_module("here.application.processing")
    client = importlib.import_module("here.transcription.client")
    segments = importlib.import_module("here.transcription.segments")
    explicit_text = "Explicit legacy retry evidence."
    provider_calls = []

    def provider(session, **kwargs):
        assert session.sources[0].path.is_file()
        provider_calls.append(session.sources[0].path)
        text = explicit_text if len(provider_calls) == 1 else "Unexpected provider replay evidence."
        return client.TranscriptionResult(
            raw_text=text,
            final_text=text,
            segments=[segments.TranscriptSegment(text, start=0.25, end=1.0)],
        )

    monkeypatch.setattr(
        "socket.socket.connect", lambda *args: pytest.fail("Real provider was called")
    )
    processor = processing.SessionProcessor(
        transcribe=provider, retry_delays=(), memory_service=memory
    )
    result = processor.retry(source.directory)
    assert result.metadata.status == "completed"
    assert result.metadata.meeting_id is None
    assert json.loads(source.metadata.read_text(encoding="utf-8")).get("meeting_id") is None
    assert source.audio.read_bytes() == original_audio
    assert _rows(root, "SELECT id FROM meetings") == [(identity,)]
    revisions = _rows(
        root,
        "SELECT id,meeting_id,revision_number,raw_text,content_hash "
        "FROM transcript_revisions ORDER BY revision_number",
    )
    assert len(revisions) == 2
    assert revisions[0] == previous
    assert revisions[1][0] != previous[0]
    assert revisions[1][1:4] == (identity, 2, explicit_text)
    assert _rows(root, "SELECT active_revision_id FROM meetings") == [(revisions[1][0],)]
    assert (
        _rows(
            root,
            "SELECT id,revision_id,ordinal,text,start_ms,end_ms FROM segments "
            "WHERE revision_id=? ORDER BY ordinal",
            (previous[0],),
        )
        == previous_segments
    )
    assert _rows(
        root, "SELECT text,start_ms,end_ms FROM segments WHERE revision_id=?", (revisions[1][0],)
    ) == [(explicit_text, 250, 1000)]
    assert _rows(root, "SELECT COUNT(*) FROM audio_assets") == [(1,)]

    def forbid_replay(*args, **kwargs):
        pytest.fail("Local recovery replayed the transcription provider")

    monkeypatch.setattr(processing, "transcribe_recording_session", forbid_replay)
    monkeypatch.setattr(client, "build_client", forbid_replay)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    memory.reconcile()
    importlib.import_module("here.application.recovery").RecoveryService(
        source.directory.parent
    ).discover()
    assert (
        _rows(
            root,
            "SELECT id,meeting_id,revision_number,raw_text,content_hash "
            "FROM transcript_revisions ORDER BY revision_number",
        )
        == revisions
    )
    assert len(provider_calls) == 1
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
