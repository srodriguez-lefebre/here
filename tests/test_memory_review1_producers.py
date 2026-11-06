"""Round-one review regressions for producer identity, state and ownership."""

from __future__ import annotations

import gc
import importlib
import json
import sqlite3
import threading
from contextlib import closing
from uuid import UUID

import pytest
from memory_fixtures import MEETING_ID, SEGMENT_TEXTS, TRANSCRIPT_TEXT, make_session
from typer.testing import CliRunner


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-review-provider-key")
    monkeypatch.setenv("TRANSCRIPTION_MODEL", "synthetic-transcription-model")
    monkeypatch.setenv("CLEANUP_MODEL", "synthetic-cleanup-model")
    monkeypatch.setenv("CLEANUP_ENABLED", "false")
    return root, importlib.import_module("here.memory.service").MemoryService(root)


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _result(text, *, original_segments=False):
    client = importlib.import_module("here.transcription.client")
    segment_type = importlib.import_module("here.transcription.segments").TranscriptSegment
    if original_segments:
        evidence = [
            segment_type(
                text,
                start=float(index),
                end=float(index + 1),
                speaker="speaker_0",
                chunk_index=index,
                speaker_scope=f"synthetic-request-{index}",
            )
            for index, text in enumerate(SEGMENT_TEXTS)
        ]
    else:
        evidence = [segment_type(text, start=0.0, end=1.0)]
    return client.TranscriptionResult(raw_text=text, final_text=text, segments=evidence)


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_existing_legacy_cli_keeps_catalog_lineage_and_optional_manifest_identity(
    tmp_path,
    monkeypatch,
    outcome,
):
    root, memory = _setup(tmp_path, monkeypatch)
    source = make_session(tmp_path / "sessions" / "legacy")
    audio_before = source.audio.read_bytes()
    assert json.loads(source.metadata.read_text()).get("meeting_id") is None
    imported = memory.ingest_session(source.directory)
    assert imported.status == "inserted"
    assert str(UUID(imported.meeting_id)) == imported.meeting_id
    previous = _rows(root, "SELECT id,raw_text FROM transcript_revisions")[0]
    cli = importlib.import_module("here.cli")

    def provider(session, **kwargs):
        assert session.sources[0].path.is_file()
        if outcome == "failed":
            raise OSError("Synthetic legacy CLI provider failure")
        return _result("Changed legacy CLI evidence")

    monkeypatch.setattr(cli, "transcribe_recording_session", provider)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Real network used"))
    result = CliRunner().invoke(cli.app, ["trans", str(source.audio)])
    assert result.exit_code == (0 if outcome == "completed" else 1), result.output + repr(
        result.exception
    )
    assert json.loads(source.metadata.read_text()).get("meeting_id") is None
    assert source.audio.read_bytes() == audio_before
    assert _rows(root, "SELECT id,status FROM meetings") == [(imported.meeting_id, outcome)]
    assert _rows(
        root, "SELECT id,raw_text FROM transcript_revisions WHERE id=?", (previous[0],)
    ) == [previous]
    if outcome == "completed":
        revisions = _rows(
            root,
            "SELECT id,revision_number,raw_text FROM transcript_revisions ORDER BY revision_number",
        )
        assert [row[1:] for row in revisions] == [
            (1, previous[1]),
            (2, "Changed legacy CLI evidence"),
        ]
        assert _rows(root, "SELECT active_revision_id FROM meetings") == [(revisions[1][0],)]
    else:
        assert _rows(root, "SELECT COUNT(*) FROM transcript_revisions") == [(1,)]
        assert _rows(root, "SELECT active_revision_id FROM meetings") == [(previous[0],)]
    assert not (source.directory / "memory_pending.json").exists()
    importlib.import_module("here.application.recovery").RecoveryService(
        source.directory.parent
    ).discover()
    assert _rows(root, "SELECT id,status FROM meetings") == [(imported.meeting_id, outcome)]


def test_two_explicit_identical_retries_complete_new_revisions_after_reconcile_and_rebuild(
    tmp_path,
    monkeypatch,
):
    root, memory = _setup(tmp_path, monkeypatch)
    source = make_session(tmp_path / "sessions" / "recorded", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    original = _rows(root, "SELECT id,raw_text,content_hash FROM transcript_revisions")[0]
    processing = importlib.import_module("here.application.processing")
    processor = processing.SessionProcessor(
        transcribe=lambda *args, **kwargs: _result(TRANSCRIPT_TEXT, original_segments=True),
        retry_delays=(),
        memory_service=memory,
    )
    for _attempt in range(2):
        assert processor.retry(source.directory).metadata.status == "completed"
    after_producers = _rows(root, "SELECT status FROM meetings")
    memory.reconcile()
    after_reconcile = _rows(root, "SELECT status FROM meetings")
    revisions = _rows(
        root,
        "SELECT id,revision_number,raw_text,content_hash FROM transcript_revisions "
        "ORDER BY revision_number",
    )
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    del processor, memory
    gc.collect()
    assert root.resolve().is_relative_to(tmp_path.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Rebuild used network"))
    importlib.import_module("here.memory.service").MemoryService(root).reconcile()
    assert (
        after_producers
        == after_reconcile
        == _rows(root, "SELECT status FROM meetings")
        == [("completed",)]
    )
    assert [row[1] for row in revisions] == [1, 2, 3]
    assert len({row[0] for row in revisions}) == 3
    assert revisions[0][0] == original[0]
    assert all(row[2:] == original[1:] for row in revisions)
    assert (
        _rows(
            root,
            "SELECT id,revision_number,raw_text,content_hash FROM transcript_revisions "
            "ORDER BY revision_number",
        )
        == revisions
    )
    assert _rows(root, "SELECT active_revision_id FROM meetings") == [(revisions[-1][0],)]


def test_second_processor_cannot_mutate_or_call_provider_before_real_owner_releases(
    tmp_path,
    monkeypatch,
):
    root, first_memory = _setup(tmp_path, monkeypatch)
    service = importlib.import_module("here.memory.service")
    second_memory = service.MemoryService(root)
    processing = importlib.import_module("here.application.processing")
    database = importlib.import_module("here.memory.database")
    source = make_session(tmp_path / "sessions" / "recorded", meeting_id=MEETING_ID)
    assert first_memory.ingest_session(source.directory).status == "inserted"
    entered, release, second_called = threading.Event(), threading.Event(), threading.Event()
    first_result = {}

    def first_provider(session, **kwargs):
        entered.set()
        assert release.wait(20)
        return _result("First owned retry evidence")

    def second_provider(session, **kwargs):
        second_called.set()
        return _result("Second owned retry evidence")

    first = processing.SessionProcessor(
        transcribe=first_provider, retry_delays=(), memory_service=first_memory
    )
    second = processing.SessionProcessor(
        transcribe=second_provider, retry_delays=(), memory_service=second_memory
    )

    def run_first():
        try:
            first_result["artifacts"] = first.retry(source.directory)
        except BaseException as error:
            first_result["error"] = error

    worker = threading.Thread(target=run_first, name="first-owned-review-producer")
    worker.start()
    try:
        assert entered.wait(15), repr(first_result.get("error"))
        before = {
            path.name: path.read_bytes() for path in source.directory.iterdir() if path.is_file()
        }
        with pytest.raises(database.Busy):
            second.retry(source.directory)
        assert not second_called.is_set()
        assert {
            path.name: path.read_bytes() for path in source.directory.iterdir() if path.is_file()
        } == before
    finally:
        release.set()
        worker.join(20)
        assert not worker.is_alive()
    assert "error" not in first_result, repr(first_result.get("error"))
    assert first_result["artifacts"].metadata.status == "completed"
    assert second.retry(source.directory).metadata.status == "completed"
    assert second_called.is_set()
    assert _rows(
        root, "SELECT revision_number,raw_text FROM transcript_revisions ORDER BY revision_number"
    ) == [
        (1, TRANSCRIPT_TEXT),
        (2, "First owned retry evidence"),
        (3, "Second owned retry evidence"),
    ]


def test_control_completed_processor_releases_owner_for_independent_service(tmp_path, monkeypatch):
    root, memory = _setup(tmp_path, monkeypatch)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    processing = importlib.import_module("here.application.processing")
    processor = processing.SessionProcessor(
        transcribe=lambda *args, **kwargs: _result("Successful owned evidence"),
        retry_delays=(),
        memory_service=memory,
    )
    assert processor.retry(source.directory).metadata.status == "completed"
    independent = importlib.import_module("here.memory.service").MemoryService(root)
    independent.begin_processing(MEETING_ID)
    try:
        assert _rows(root, "SELECT status FROM meetings") == [("processing",)]
        independent.set_status(MEETING_ID, "completed")
    finally:
        independent.finish_capture(MEETING_ID)
    assert _rows(
        root, "SELECT revision_number,raw_text FROM transcript_revisions ORDER BY revision_number"
    ) == [
        (1, TRANSCRIPT_TEXT),
        (2, "Successful owned evidence"),
    ]
