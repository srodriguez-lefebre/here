"""Explicit M1 producers resolve unmapped identities before mutation/provider work."""

from __future__ import annotations

import importlib
import json
import sqlite3
from contextlib import closing
from uuid import UUID

import pytest
from memory_fixtures import TRANSCRIPT_TEXT, make_session

OPAQUE_ID = "synthetic-m1-opaque-identity"
NEW_TEXT = "Synthetic explicit producer revision after local legacy import."


def _configure(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unmapped-key")
    monkeypatch.setenv("CLEANUP_ENABLED", "false")

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic producer allocated a real provider/network connection")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    return root


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _result():
    client = importlib.import_module("here.transcription.client")
    segments = importlib.import_module("here.transcription.segments")
    return client.TranscriptionResult(
        raw_text=NEW_TEXT,
        final_text=NEW_TEXT,
        segments=[segments.TranscriptSegment(NEW_TEXT)],
    )


@pytest.mark.parametrize("source_id", [None, OPAQUE_ID], ids=["none", "opaque"])
@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_unmapped_legacy_is_imported_and_owned_before_explicit_file_provider(
    tmp_path, monkeypatch, source_id, outcome
):
    root = _configure(tmp_path, monkeypatch)
    service = importlib.import_module("here.memory.service")
    processing = importlib.import_module("here.application.processing")
    database = importlib.import_module("here.memory.database")
    memory = service.MemoryService(root)
    source = make_session(tmp_path / "sessions" / "unmapped", meeting_id=source_id)
    original_audio = source.audio.read_bytes()
    assert _rows(root, "SELECT COUNT(*) FROM meetings") == [(0,)]
    observed = []
    original_failure = RuntimeError("Synthetic unmapped provider failed after identity resolution")

    def provider(session, **kwargs):
        # Capture the real DB snapshot at the actual provider boundary. Assertions
        # run after the producer so its exception policy cannot mask the oracle.
        record = {
            "meetings": _rows(root, "SELECT id,status,active_revision_id FROM meetings"),
            "revisions": _rows(
                root,
                "SELECT id,meeting_id,revision_number,raw_text,content_hash "
                "FROM transcript_revisions ORDER BY revision_number",
            ),
            "segments": _rows(root, "SELECT id,revision_id,ordinal,text FROM segments ORDER BY id"),
            "source_id": session.meeting_id,
            "metadata_id": json.loads(source.metadata.read_text(encoding="utf-8")).get(
                "meeting_id"
            ),
            "owner_blocked": False,
        }
        if len(record["meetings"]) == 1:
            identity = record["meetings"][0][0]
            independent = service.MemoryService(root)
            try:
                independent.begin_processing(identity)
            except database.Busy:
                record["owner_blocked"] = True
            finally:
                independent.finish_capture(identity)
        observed.append(record)
        if outcome == "failed":
            raise original_failure
        return _result()

    processor = processing.SessionProcessor(
        transcribe=provider, retry_delays=(), memory_service=memory
    )
    if outcome == "failed":
        with pytest.raises(processing.SessionProcessingFailed) as failed:
            processor.transcribe_audio_path(source.audio, source.directory.parent)
        assert failed.value.__cause__ is original_failure
    else:
        assert (
            processor.transcribe_audio_path(source.audio, source.directory.parent).metadata.status
            == "completed"
        )
    assert len(observed) == 1
    boundary = observed[0]
    assert len(boundary["meetings"]) == 1, "Local import must resolve identity before provider"
    identity, status, parent = boundary["meetings"][0]
    assert str(UUID(identity)) == identity
    assert status == "processing" and parent is not None, (
        "The original v1 revision must already be active before explicit reprocessing"
    )
    assert boundary["owner_blocked"], "Independent processor acquired the live producer's identity"
    assert boundary["source_id"] == boundary["metadata_id"] == source_id
    assert len(boundary["revisions"]) == 1
    previous = boundary["revisions"][0]
    assert previous[:4] == (parent, identity, 1, TRANSCRIPT_TEXT)
    assert _rows(root, "SELECT id,status FROM meetings") == [(identity, outcome)]
    assert _rows(
        root,
        "SELECT id,meeting_id,revision_number,raw_text,content_hash "
        "FROM transcript_revisions WHERE id=?",
        (parent,),
    ) == [previous]
    assert (
        _rows(
            root,
            "SELECT id,revision_id,ordinal,text FROM segments WHERE revision_id=? ORDER BY id",
            (parent,),
        )
        == boundary["segments"]
    )
    revisions = _rows(
        root,
        "SELECT id,revision_number,raw_text FROM transcript_revisions ORDER BY revision_number",
    )
    if outcome == "completed":
        assert [(r[1], r[2]) for r in revisions] == [(1, TRANSCRIPT_TEXT), (2, NEW_TEXT)]
        assert revisions[1][0] != parent
        assert _rows(root, "SELECT active_revision_id FROM meetings") == [(revisions[1][0],)]
    else:
        assert revisions == [(parent, 1, TRANSCRIPT_TEXT)]
        assert _rows(root, "SELECT active_revision_id FROM meetings") == [(parent,)]
    assert json.loads(source.metadata.read_text(encoding="utf-8")).get("meeting_id") == source_id
    assert source.audio.read_bytes() == original_audio
    assert not (source.directory / "memory_pending.json").exists()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    importlib.import_module("here.application.recovery").RecoveryService(
        source.directory.parent
    ).discover()
    memory.reconcile()
    assert len(observed) == 1
    assert _rows(root, "SELECT id,status FROM meetings") == [(identity, outcome)]
    assert not (source.directory / "memory_pending.json").exists()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("source_id", [None, OPAQUE_ID], ids=["none", "opaque"])
@pytest.mark.parametrize("fault", ["busy", "corrupt"])
def test_unmapped_identity_catalog_failure_aborts_before_source_mutation_or_provider(
    tmp_path, monkeypatch, source_id, fault
):
    root = _configure(tmp_path, monkeypatch)
    processing = importlib.import_module("here.application.processing")
    database = importlib.import_module("here.memory.database")
    source = make_session(tmp_path / "sessions" / "unmapped", meeting_id=source_id)
    # Selecting a distinct validated WAV exercises the normalization boundary too.
    selected = source.directory / "raw.wav"
    selected.write_bytes(source.audio.read_bytes())
    before = _snapshot(source.directory)
    calls = []
    normalizations = []

    def provider(*args, **kwargs):
        calls.append("provider")
        return _result()

    def normalize(*args, **kwargs):
        normalizations.append("normalize")
        return processing.materialize_normalized_session(*args, **kwargs)

    memory = None
    blocker = None
    if fault == "busy":
        memory = importlib.import_module("here.memory.service").MemoryService(root)
        blocker = sqlite3.connect(root / "library.sqlite3")
        blocker.execute("BEGIN IMMEDIATE")
        expected_error = database.Busy
    else:
        root.mkdir(parents=True, exist_ok=True)
        assert root.resolve().is_relative_to(tmp_path.resolve())
        (root / "library.sqlite3").write_bytes(b"Synthetic invalid SQLite library; no user data.")
        expected_error = database.IntegrityFailure
    processor = processing.SessionProcessor(
        transcribe=provider, normalize=normalize, retry_delays=(), memory_service=memory
    )
    error = None
    try:
        try:
            processor.transcribe_audio_path(selected, source.directory.parent)
        except Exception as exc:
            error = exc
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
    assert calls == [], "Provider ran before the unmapped identity could be resolved"
    assert normalizations == [], "Audio normalization ran before safe identity resolution"
    assert _snapshot(source.directory) == before
    assert isinstance(error, expected_error), repr(error)
    assert not (source.directory / "memory_pending.json").exists()
