"""Copilot C5: uncertain legacy catalog ownership is visible but cannot retry."""

from __future__ import annotations

import importlib
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pytest
import soundfile as sf
from memory_fixtures import OTHER_MEETING_ID, make_session, write_json

OPAQUE_ID = "synthetic-opaque-legacy-identity"
CORRUPT_BYTES = b"Synthetic corrupt catalog; private-canary-do-not-display; not SQLite."


def _setup(tmp_path, monkeypatch):
    root, sessions = tmp_path / "data", tmp_path / "sessions"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Legacy catalog discovery allocated provider/network transport")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    monkeypatch.setattr("here.application.processing.transcribe_recording_session", forbidden)
    module = importlib.import_module("here.application.recovery")
    return root, sessions, module.RecoveryService(sessions)


def _failed_session(directory, identity):
    source = make_session(directory, meeting_id=identity)
    metadata = json.loads(source.metadata.read_text(encoding="utf-8"))
    metadata["status"] = "failed"
    metadata["failure_stage"] = "offline_transcription"
    write_json(source.metadata, metadata)
    return source


def _snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


@contextmanager
def _observe_material(monkeypatch, source, private):
    """Delegate real readers, allowing only minimal metadata/identity inspection."""
    reads = []
    original_info, original_open = sf.info, Path.open
    journal = importlib.import_module("here.recording.journal").CaptureJournal
    original_load = journal.load

    def protected(path):
        try:
            actual = Path(path).absolute()
        except TypeError:
            return False
        return actual == private.absolute() or (
            actual.parent == source.directory.absolute() and actual.suffix.lower() == ".wav"
        )

    def info(path, *args, **kwargs):
        if protected(path):
            reads.append("WAV inspection")
        return original_info(path, *args, **kwargs)

    def opened(path, *args, **kwargs):
        if protected(path):
            reads.append("WAV/private file open")
        return original_open(path, *args, **kwargs)

    def loaded(*args, **kwargs):
        reads.append("native journal inspection")
        return original_load(*args, **kwargs)

    with monkeypatch.context() as watch:
        watch.setattr(sf, "info", info)
        watch.setattr(Path, "open", opened)
        watch.setattr(journal, "load", staticmethod(loaded))
        yield reads


@pytest.mark.parametrize("identity", [None, OPAQUE_ID], ids=["none", "opaque"])
def test_corrupt_catalog_keeps_legacy_session_visible_without_assuming_ownership(
    tmp_path, monkeypatch, identity
):
    root, sessions, recovery = _setup(tmp_path, monkeypatch)
    source = _failed_session(sessions / "a-legacy", identity)
    good = _failed_session(sessions / "z-known-canonical", OTHER_MEETING_ID)
    # Obtain an actual valid pre-existing selection before catalog corruption.
    selected = recovery.discover()
    (stale,) = [item for item in selected if item.session_dir == source.directory]
    assert stale.can_retry and stale.capture_id == identity
    assert not (root / "library.sqlite3").exists()
    root.mkdir(parents=True, exist_ok=True)
    catalog = root / "library.sqlite3"
    catalog.write_bytes(CORRUPT_BYTES)
    private = tmp_path / "synthetic-private-unrelated.txt"
    private.write_text("Synthetic private sibling remains unread.", encoding="utf-8")
    before, good_before = _snapshot(source.directory), _snapshot(good.directory)
    with _observe_material(monkeypatch, source, private) as reads:
        first = recovery.discover()
        second = recovery.discover()
        # Metadata-only materialization cannot turn uncertain ownership into a
        # provider attempt, even for a selection obtained before corruption.
        with pytest.raises(RuntimeError):
            recovery.materialize(stale)
    assert reads == [], "Unresolved catalog ownership must stop material inspection"
    assert catalog.read_bytes() == CORRUPT_BYTES
    assert _snapshot(source.directory) == before and _snapshot(good.directory) == good_before
    assert private.read_text(encoding="utf-8") == "Synthetic private sibling remains unread."
    known = [item for item in second if item.session_dir == good.directory]
    assert len(known) == 1 and known[0].capture_id == OTHER_MEETING_ID and known[0].can_retry
    first_warning = [item for item in first if item.session_dir == source.directory]
    second_warning = [item for item in second if item.session_dir == source.directory]
    assert len(first_warning) == len(second_warning) == 1, (
        "Valid legacy source disappeared when catalog owner lookup failed"
    )
    candidate = second_warning[0]
    assert candidate.capture_id == identity and candidate.status == "failed"
    assert candidate.can_retry is False
    assert isinstance(candidate.error_summary, str) and 0 < len(candidate.error_summary) <= 512
    assert "private-canary-do-not-display" not in candidate.error_summary
    assert str(root) not in candidate.error_summary and OPAQUE_ID not in candidate.error_summary
    assert not (source.directory / "memory_pending.json").exists()
    contract = importlib.import_module("here.ui.contract")
    core = importlib.import_module("here.ui.fake_controller").FakeApplicationController()
    retried = []
    monkeypatch.setattr(core, "retry", lambda *args: retried.append(args))

    def no_credentials(*args, **kwargs):
        pytest.fail("Uncertain library ownership requested provider credentials")

    monkeypatch.setattr(contract, "require_provider_key", no_credentials)
    adapter = contract.ApplicationUiAdapter(core, sessions, recovery_service=recovery)
    with pytest.raises(RuntimeError):
        adapter.retry_candidate(candidate)
    assert retried == [] and catalog.read_bytes() == CORRUPT_BYTES
    assert _snapshot(source.directory) == before


@pytest.mark.parametrize("identity", [None, OPAQUE_ID], ids=["none", "opaque"])
def test_control_unmapped_legacy_without_catalog_remains_normally_recoverable(
    tmp_path, monkeypatch, identity
):
    root, sessions, recovery = _setup(tmp_path, monkeypatch)
    source = _failed_session(sessions / "unmapped", identity)
    before = _snapshot(source.directory)
    (first,) = recovery.discover()
    assert recovery.discover() == [first]
    assert first.capture_id == identity and first.status == "failed" and first.can_retry
    assert recovery.materialize(first) == source.directory
    assert _snapshot(source.directory) == before
    assert not (root / "library.sqlite3").exists()


def test_control_real_mapped_legacy_owner_remains_hidden_and_blocks_stale_materialization(
    tmp_path, monkeypatch
):
    root, sessions, recovery = _setup(tmp_path, monkeypatch)
    source = _failed_session(sessions / "mapped-legacy", None)
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    inserted = memory.ingest_session(source.directory)
    assert inserted.status == "inserted" and inserted.meeting_id is not None
    (stale,) = recovery.discover()
    assert stale.capture_id is None and stale.can_retry
    before = _snapshot(source.directory)
    private = tmp_path / "synthetic-private-unrelated.txt"
    private.write_bytes(b"Synthetic private source stays unread.")
    database = importlib.import_module("here.memory.database")
    memory.begin_processing(inserted.meeting_id)
    try:
        with _observe_material(monkeypatch, source, private) as reads:
            assert recovery.discover() == recovery.discover() == []
            with pytest.raises(database.Busy):
                recovery.materialize(stale)
        assert reads == []
        assert _snapshot(source.directory) == before
        with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
            assert connection.execute("SELECT id,status FROM meetings").fetchall() == [
                (inserted.meeting_id, "processing")
            ]
    finally:
        memory.finish_capture(inserted.meeting_id)
    (available,) = recovery.discover()
    assert available.session_dir == source.directory and available.capture_id is None
    assert available.can_retry and _snapshot(source.directory) == before


def test_ambiguous_real_legacy_origin_mapping_is_a_visible_nonretryable_warning(
    tmp_path, monkeypatch
):
    from uuid import uuid4

    root, sessions, recovery = _setup(tmp_path, monkeypatch)
    source = _failed_session(sessions / "a-ambiguous-legacy", None)
    good = _failed_session(sessions / "z-known-canonical", OTHER_MEETING_ID)
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    first_import = memory.ingest_session(source.directory)
    second_import = memory.ingest_session(good.directory)
    assert first_import.status == second_import.status == "inserted"
    assert first_import.meeting_id != second_import.meeting_id
    (stale,) = [item for item in recovery.discover() if item.session_dir == source.directory]
    assert stale.capture_id is None and stale.can_retry
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        first_origin = connection.execute(
            "SELECT origin_path,origin_fingerprint FROM ingest_sources WHERE meeting_id=?",
            (first_import.meeting_id,),
        ).fetchone()
        second_origin = connection.execute(
            "SELECT origin_fingerprint,revision_id FROM ingest_sources WHERE meeting_id=?",
            (second_import.meeting_id,),
        ).fetchone()
        assert first_origin[1] != second_origin[0]
        # Both fingerprints and revision parents came from actual validated
        # imports. A real FK-valid second mapping makes this one origin ambiguous.
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO ingest_sources "
            "(id,origin_path,origin_fingerprint,meeting_id,revision_id) VALUES (?,?,?,?,?)",
            (
                str(uuid4()),
                first_origin[0],
                second_origin[0],
                second_import.meeting_id,
                second_origin[1],
            ),
        )
        connection.commit()
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        before_origins = connection.execute("SELECT * FROM ingest_sources ORDER BY id").fetchall()
        before_meetings = connection.execute(
            "SELECT id,status,active_revision_id FROM meetings ORDER BY id"
        ).fetchall()
    with pytest.raises(ValueError, match="Legacy origin conflict"):
        memory.retry_parent(source.directory, None)
    before, good_before = _snapshot(source.directory), _snapshot(good.directory)
    private = tmp_path / "synthetic-private-unrelated.txt"
    private.write_bytes(b"Synthetic ambiguity canary remains unread.")
    with _observe_material(monkeypatch, source, private) as reads:
        first = recovery.discover()
        second = recovery.discover()
        with pytest.raises((ValueError, RuntimeError)):
            recovery.materialize(stale)
    assert reads == []
    assert _snapshot(source.directory) == before and _snapshot(good.directory) == good_before
    assert private.read_bytes() == b"Synthetic ambiguity canary remains unread."
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert (
            connection.execute("SELECT * FROM ingest_sources ORDER BY id").fetchall()
            == before_origins
        )
        assert (
            connection.execute(
                "SELECT id,status,active_revision_id FROM meetings ORDER BY id"
            ).fetchall()
            == before_meetings
        )
    known = [item for item in second if item.session_dir == good.directory]
    assert len(known) == 1 and known[0].capture_id == OTHER_MEETING_ID and known[0].can_retry
    first_warning = [item for item in first if item.session_dir == source.directory]
    second_warning = [item for item in second if item.session_dir == source.directory]
    assert len(first_warning) == len(second_warning) == 1, (
        "Valid legacy source disappeared when its real origin mapping was ambiguous"
    )
    candidate = second_warning[0]
    assert candidate.capture_id is None and candidate.status == "failed"
    assert candidate.can_retry is False
    assert isinstance(candidate.error_summary, str) and 0 < len(candidate.error_summary) <= 512
    assert str(source.directory) not in candidate.error_summary
    contract = importlib.import_module("here.ui.contract")
    core = importlib.import_module("here.ui.fake_controller").FakeApplicationController()
    retried = []
    monkeypatch.setattr(core, "retry", lambda *args: retried.append(args))

    def no_credentials(*args, **kwargs):
        pytest.fail("Ambiguous ownership requested provider credentials")

    monkeypatch.setattr(contract, "require_provider_key", no_credentials)
    adapter = contract.ApplicationUiAdapter(core, sessions, recovery_service=recovery)
    with pytest.raises(RuntimeError):
        adapter.retry_candidate(candidate)
    assert retried == [] and _snapshot(source.directory) == before
