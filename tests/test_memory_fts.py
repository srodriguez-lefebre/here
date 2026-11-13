"""B5 owns one FTS reindexer; real M3 integration remains a separate gate."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import FrozenInstanceError

import pytest
from here.memory import service as service_module
from here.memory.database import LibraryError
from here.memory.service import FtsUnavailable, MemoryService
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json

PARTIAL_ID = "33333333-3333-4333-8333-333333333333"
SUPPRESSED_ID = "44444444-4444-4444-8444-444444444444"
CANARY = "quartzzephyr"


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Reindex allocated provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)


@pytest.fixture
def index_catalog(tmp_path):
    memory = MemoryService(tmp_path / "data", sessions_root=tmp_path / "sessions")
    previous = None
    for name, identity, text, partial in (
        ("old", MEETING_ID, "Obsolete historical copper", False),
        ("new", MEETING_ID, "Decisión ágil current copper", False),
        ("other", OTHER_MEETING_ID, "Independent copper current", False),
        ("partial", PARTIAL_ID, "Partial copper literal only", True),
        ("suppressed", SUPPRESSED_ID, "Suppressed copper evidence", False),
    ):
        source = make_session(
            tmp_path / name,
            meeting_id=identity,
            transcript_text=text,
            segments=[{"text": text, "start": 0.0, "end": 1.0}],
        )
        if partial:
            document = json.loads(source.metadata.read_text())
            document["status"] = "failed"
            document["started_at"] = "2026-01-01T12:00:00"
            document["completed_at"] = "2026-01-01T12:05:02"
            write_json(source.metadata, document)
        result = memory.ingest_session(
            source.directory, revision_of=previous if name == "new" else None
        )
        assert result.status == ("revised" if name == "new" else "inserted")
        if name == "old":
            previous = memory.active_revision(identity)
    return memory


def _suppress(root, identity=SUPPRESSED_ID):
    directory = root / ".deletions"
    directory.mkdir(exist_ok=True)
    write_json(
        directory / f"{identity}.json",
        {"schema_version": 1, "meeting_id": identity, "origins": [], "phase": "pending"},
    )


def test_reindex_replaces_stale_index_with_only_unsuppressed_active_complete_revisions(
    index_catalog,
):
    memory = index_catalog
    root = memory.data_dir
    _suppress(root)
    partial_before = _rows(
        root,
        "SELECT id,raw_text,status FROM transcript_revisions WHERE meeting_id=?",
        (PARTIAL_ID,),
    )
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute("DELETE FROM segments_fts")
        connection.execute(
            "INSERT INTO segments_fts VALUES ('stale','stale','stale','stale','stale')"
        )
        connection.commit()
    report = memory.reindex()
    assert report.indexed_meetings == 2 and report.indexed_segments == 2
    assert isinstance(report.warnings, tuple)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        report.indexed_segments = -1
    actual = _rows(root, "SELECT meeting_id,revision_id,text FROM segments_fts ORDER BY meeting_id")
    assert actual == [
        (MEETING_ID, memory.active_revision(MEETING_ID), "Decisión ágil current copper"),
        (OTHER_MEETING_ID, memory.active_revision(OTHER_MEETING_ID), "Independent copper current"),
    ]
    assert _rows(root, "SELECT text FROM segments_fts WHERE segments_fts MATCH 'decision'") == [
        ("Decisión ágil current copper",)
    ]
    assert (
        _rows(
            root,
            "SELECT id,raw_text,status FROM transcript_revisions WHERE meeting_id=?",
            (PARTIAL_ID,),
        )
        == partial_before
    )
    assert memory.reindex() == report
    assert (
        _rows(root, "SELECT meeting_id,revision_id,text FROM segments_fts ORDER BY meeting_id")
        == actual
    )


def test_reindex_is_one_transaction_and_keeps_old_index_on_insert_failure(
    index_catalog, monkeypatch
):
    from here.memory.database import LibraryConnection

    memory = index_catalog
    root = memory.data_dir
    before = _rows(root, "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id")
    execute = LibraryConnection.execute
    inserted = []

    def fail_second(self, sql, parameters=()):
        if sql.lstrip().upper().startswith("INSERT INTO SEGMENTS_FTS"):
            inserted.append(parameters)
            if len(inserted) == 2:
                raise sqlite3.OperationalError("synthetic reindex insert failure")
        return execute(self, sql, parameters)

    with monkeypatch.context() as faults:
        faults.setattr(LibraryConnection, "execute", fail_second)
        with pytest.raises(LibraryError):
            memory.reindex()
    assert len(inserted) == 2
    assert _rows(root, "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id") == before


def test_reindex_invalid_registry_never_clears_existing_index(index_catalog):
    memory = index_catalog
    before = _rows(memory.data_dir, "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id")
    _suppress(memory.data_dir)
    (memory.data_dir / ".deletions" / f"{SUPPRESSED_ID}.json").write_text("[]")
    with pytest.raises(LibraryError):
        memory.reindex()
    assert (
        _rows(memory.data_dir, "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id")
        == before
    )


def test_missing_fts_preserves_typed_warning_and_does_not_block_delete_or_reindex(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(service_module, "fts_available", lambda connection: False)
    memory = MemoryService(tmp_path / "data", sessions_root=tmp_path / "sessions")
    source = make_session(tmp_path / "origin", meeting_id=MEETING_ID)
    ingest = memory.ingest_session(source.directory)
    assert ingest.status == "inserted"
    assert any(isinstance(item, FtsUnavailable) for item in ingest.warnings)
    report = memory.reindex()
    assert report.indexed_meetings == report.indexed_segments == 0
    assert any(isinstance(item, FtsUnavailable) for item in report.warnings)
    result = memory.delete_meeting(MEETING_ID)
    assert result.status == "completed"
    assert any(isinstance(item, FtsUnavailable) for item in result.warnings)
    assert memory.show(MEETING_ID) is None
    assert _rows(memory.data_dir, "SELECT name FROM sqlite_master WHERE name='segments_fts'") == []


class _SecureDeleteConnection:
    """Forward real SQLite; only the optional FTS secure-delete instruction can fail."""

    def __init__(self, connection, forced_unavailable, calls):
        self._connection = connection
        self._forced_unavailable = forced_unavailable
        self._calls = calls

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def execute(self, sql, parameters=()):
        if "secure-delete" in sql.lower() or "secure-delete" in parameters:
            self._calls.append(sql)
            if self._forced_unavailable:
                raise sqlite3.OperationalError("unrecognized special insert ID: secure-delete")
        return self._connection.execute(sql, parameters)


@pytest.mark.parametrize("forced_unavailable", [False, True], ids=["native-probe", "fallback"])
def test_delete_probes_fts_secure_delete_and_leaves_no_sqlite_wal_backup_token_residue(
    tmp_path, monkeypatch, forced_unavailable
):
    memory = MemoryService(tmp_path / "data", sessions_root=tmp_path / "sessions")
    source = make_session(
        tmp_path / "origin",
        meeting_id=MEETING_ID,
        transcript_text=CANARY,
        segments=[{"text": CANARY}],
    )
    assert memory.ingest_session(source.directory).status == "inserted"
    root = memory.data_dir
    backup = root / "backups" / "library-purge.sqlite3"
    backup.parent.mkdir()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        with closing(sqlite3.connect(backup)) as copy:
            connection.backup(copy)
    assert CANARY.encode() in backup.read_bytes()
    calls = []
    original_open = service_module.open_library

    def open_with_probe(*args, **kwargs):
        library = original_open(*args, **kwargs)
        library.connection = _SecureDeleteConnection(library.connection, forced_unavailable, calls)
        return library

    monkeypatch.setattr(service_module, "open_library", open_with_probe)
    result = memory.delete_meeting(MEETING_ID)
    assert result.status == "completed"
    assert calls, "The actual FTS capability was assumed instead of probed"
    assert _rows(root, "SELECT * FROM segments_fts WHERE segments_fts MATCH ?", (CANARY,)) == []
    for path in root.rglob("*"):
        if path.is_file():
            assert CANARY.encode() not in path.read_bytes(), path


def test_deleting_transaction_removes_fts_before_any_owned_file_is_removed(
    index_catalog, monkeypatch
):
    from pathlib import Path

    memory = index_catalog
    root = memory.data_dir
    original = Path.unlink
    observations = []

    def unlink(self, *args, **kwargs):
        if self.is_relative_to(root / "meetings" / MEETING_ID):
            observations.append(
                _rows(root, "SELECT status FROM meetings WHERE id=?", (MEETING_ID,))
            )
            assert observations[-1] == [("deleting",)]
            assert (
                _rows(root, "SELECT meeting_id FROM segments_fts WHERE meeting_id=?", (MEETING_ID,))
                == []
            )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert memory.delete_meeting(MEETING_ID).status == "completed"
    assert observations
    assert _rows(
        root, "SELECT meeting_id FROM segments_fts WHERE meeting_id=?", (OTHER_MEETING_ID,)
    ) == [(OTHER_MEETING_ID,)]


def test_control_accepted_partial_fixture_remains_out_of_persistent_fts(index_catalog):
    root = index_catalog.data_dir
    assert _rows(
        root, "SELECT status FROM transcript_revisions WHERE meeting_id=?", (PARTIAL_ID,)
    ) == [("partial",)]
    assert _rows(
        root, "SELECT started_at,completed_at FROM meetings WHERE id=?", (PARTIAL_ID,)
    ) == [(None, None)]
    assert _rows(root, "SELECT * FROM segments_fts WHERE meeting_id=?", (PARTIAL_ID,)) == []


def test_reindex_counts_actual_committed_rows_not_eligible_empty_meetings(index_catalog, tmp_path):
    from dataclasses import fields

    from here.memory import models

    memory = index_catalog
    empty_id = "55555555-5555-4555-8555-555555555555"
    empty = make_session(
        tmp_path / "zero-segment",
        meeting_id=empty_id,
        transcript_text="Synthetic complete text with no structured segments",
        segments=[],
    )
    assert memory.ingest_session(empty.directory).status == "inserted"
    assert _rows(
        memory.data_dir, "SELECT status FROM transcript_revisions WHERE meeting_id=?", (empty_id,)
    ) == [("complete",)]
    assert (
        _rows(
            memory.data_dir,
            "SELECT s.id FROM segments s JOIN transcript_revisions r ON r.id=s.revision_id "
            "WHERE r.meeting_id=?",
            (empty_id,),
        )
        == []
    )
    _suppress(memory.data_dir)
    report = memory.reindex()
    assert isinstance(report, models.ReindexReport) and not hasattr(report, "__dict__")
    assert [field.name for field in fields(models.ReindexReport)] == [
        "indexed_meetings",
        "indexed_segments",
        "warnings",
    ]
    assert models.ReindexReport() == models.ReindexReport(0, 0, ())
    assert (report.indexed_meetings, report.indexed_segments) == (2, 2)
    assert _rows(
        memory.data_dir, "SELECT COUNT(DISTINCT meeting_id),COUNT(*) FROM segments_fts"
    ) == [(2, 2)]
    assert (
        _rows(
            memory.data_dir, "SELECT meeting_id FROM segments_fts WHERE meeting_id=?", (empty_id,)
        )
        == []
    )


def test_reindex_busy_is_typed_error_not_partial_success_report(index_catalog):
    from here.memory.database import Busy
    from here.output.ownership import FileLease

    memory = index_catalog
    before = _rows(memory.data_dir, "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id")
    with closing(FileLease(memory.data_dir / ".memory.lock")):
        with pytest.raises(Busy):
            memory.reindex()
    assert (
        _rows(memory.data_dir, "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id")
        == before
    )


@pytest.mark.parametrize("probe_failure", ["ioerr", "busy", "missing-module"])
def test_h1_reindex_classifies_real_probe_errors_and_preserves_existing_index(
    index_catalog, monkeypatch, probe_failure
):
    from here.memory.database import Busy

    memory = index_catalog
    before = _rows(memory.data_dir, "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id")
    assert before, "The preservation assertion needs a populated real FTS index"
    original_open = service_module.open_library
    calls = []

    class ProbeConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql, parameters=()):
            if sql == "CREATE VIRTUAL TABLE temp.here_fts_probe USING fts5(text)":
                calls.append(sql)
                code, message = {
                    "ioerr": (sqlite3.SQLITE_IOERR, "synthetic disk I/O error"),
                    "busy": (sqlite3.SQLITE_BUSY, "synthetic database is locked"),
                    "missing-module": (sqlite3.SQLITE_ERROR, "no such module: fts5"),
                }[probe_failure]
                error = sqlite3.OperationalError(message)
                error.sqlite_errorcode = code
                raise error
            return self.connection.execute(sql, parameters)

    def open_with_probe(*args, **kwargs):
        library = original_open(*args, **kwargs)
        library.connection = ProbeConnection(library.connection)
        return library

    monkeypatch.setattr(service_module, "open_library", open_with_probe)
    if probe_failure == "missing-module":
        report = memory.reindex()
        assert (report.indexed_meetings, report.indexed_segments) == (0, 0)
        assert len(report.warnings) == 1 and isinstance(report.warnings[0], FtsUnavailable)
    else:
        with pytest.raises(Busy if probe_failure == "busy" else LibraryError) as caught:
            memory.reindex()
        if probe_failure == "ioerr":
            assert not isinstance(caught.value, Busy)
    assert calls == ["CREATE VIRTUAL TABLE temp.here_fts_probe USING fts5(text)"]
    assert (
        _rows(memory.data_dir, "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id")
        == before
    )
