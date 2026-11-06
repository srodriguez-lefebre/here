"""B5 deletion and R50 contracts, synthetic TEMP data only.

Faults target filesystem/SQLite boundaries; no production crash switches are used.
Actual M3 query integration is a separately transferred-source gate.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from uuid import uuid4

import pytest
from here.memory import service as service_module
from here.memory.database import Busy, LibraryConnection, LibraryError, open_library
from here.memory.service import MemoryService, NotFound
from here.output.ownership import FileLease, LeaseBusy
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json

CANARY = "quartzzephyr B5 synthetic purge canary"


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _files(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*") if p.is_file()}


def _delete(memory, identity=MEETING_ID):
    return memory.delete_meeting(identity)


def _assert_result(result, status):
    assert result.status == status
    assert result.meeting_id == MEETING_ID
    assert isinstance(result.reason, str)
    if status != "completed":
        assert result.reason in {
            "invalid_input", "not_found", "conflict", "busy", "hook_failed",
            "purge_failed", "io", "cleanup_pending",
        }
    if status == "pending":
        assert result.error_category == "io"
    assert isinstance(result.warnings, tuple)
    assert result.error_category in {None, "invalid_input", "conflict", "io"}
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.status = "completed" if status != "completed" else "pending"


def _registry(root):
    return root / ".deletions" / f"{MEETING_ID}.json"


def _view(root):
    library = open_library(root)
    try:
        with service_module.catalog_read_view(library) as snapshot:
            return snapshot
    finally:
        library.close()


def _hidden(memory, identity=MEETING_ID):
    assert identity in _view(memory.data_dir).suppressed_meeting_ids
    assert memory.show(identity) is None
    with pytest.raises(NotFound):
        memory.get_meeting(identity)


def _lose_database(root, temp_root):
    assert root.resolve().is_relative_to(temp_root.resolve())
    assert not root.is_symlink() and not root.is_junction()
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.parent == root and not path.is_symlink() and not path.is_junction()
        path.unlink(missing_ok=True)


def _assert_purged(memory):
    root = memory.data_dir
    assert _rows(root, "SELECT id FROM meetings WHERE id=?", (MEETING_ID,)) == []
    for table in ("transcript_revisions", "audio_assets", "revision_audio_assets",
                  "ingest_sources", "meeting_participants", "speaker_assignments",
                  "lifecycle_events", "segments_fts"):
        assert _rows(root, f"SELECT meeting_id FROM {table} WHERE meeting_id=?",
                     (MEETING_ID,)) == [], table
    assert _rows(root, "SELECT COUNT(*) FROM segments") == [(0,)]
    meeting_dir = root / "meetings" / MEETING_ID
    assert not meeting_dir.exists() or not list(meeting_dir.rglob("*"))
    assert _rows(root, "SELECT COUNT(*) FROM deleted_origins")[0][0] >= 1
    assert _rows(root, "SELECT phase FROM pending_operations WHERE kind='delete'") == [
        ("complete",)
    ]
    record = json.loads(_registry(root).read_text(encoding="utf-8"))
    assert set(record) == {"schema_version", "meeting_id", "origins", "phase"}
    assert record["schema_version"] == 1 and record["phase"] == "complete"
    assert record["meeting_id"] == MEETING_ID
    assert record["origins"]
    for origin in record["origins"]:
        assert set(origin) == {"origin_path", "origin_fingerprint"}
    for path in root.rglob("*"):
        if path.is_file():
            contents = path.read_bytes()
            assert CANARY.encode() not in contents, path
            assert b"quartzzephyr" not in contents, path
    assert (root / ".memory.lock").is_file()


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("B5 local deletion/read guard allocated provider/network work")
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)


@pytest.fixture
def catalog(tmp_path):
    source = make_session(tmp_path / "origin", meeting_id=MEETING_ID,
                          transcript_text=CANARY,
                          segments=[{"text": CANARY, "start": 0.0, "end": 1.0,
                                     "speaker": "speaker_0", "speaker_scope": "b5-request"}])
    memory = MemoryService(tmp_path / "data", sessions_root=tmp_path / "sessions")
    assert memory.ingest_session(source.directory).status == "inserted"
    return memory, source


def test_delete_idle_purges_rows_owned_files_backups_and_retains_minimal_tombstone(catalog):
    memory, source = catalog
    before = _files(source.directory)
    backup = memory.data_dir / "backups" / "library-synthetic.sqlite3"
    backup.parent.mkdir()
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as live:
        with closing(sqlite3.connect(backup)) as copy:
            live.backup(copy)
    participant = memory.add_meeting_participant(MEETING_ID, "Synthetic person")
    assert participant is not None
    _assert_result(_delete(memory), "completed")
    _assert_purged(memory)
    assert _files(source.directory) == before
    assert memory.ingest_session(source.directory).status == "rejected"
    memory.reconcile()
    memory.reconcile()
    _assert_purged(memory)


@pytest.mark.parametrize("owner", ["recording", "processing", "external_lease"])
def test_live_real_owner_rejects_before_barrier_without_changing_visibility(catalog, owner):
    memory, source = catalog
    before = _files(source.directory)
    owner_path = memory.data_dir / ".owners" / f"{MEETING_ID}.lock"
    owner_path.parent.mkdir(exist_ok=True)
    if owner == "external_lease":
        lease = FileLease(owner_path)
    else:
        memory.set_status(MEETING_ID, owner)
        lease = None
    try:
        with pytest.raises(Busy):
            _delete(memory)
        assert memory.show(MEETING_ID) is not None
        assert not _registry(memory.data_dir).exists()
        assert _files(source.directory) == before
    finally:
        if lease is not None:
            lease.close()


def test_residual_unlocked_owner_filename_is_not_live_work(catalog):
    memory, _ = catalog
    path = memory.data_dir / ".owners" / f"{MEETING_ID}.lock"
    path.parent.mkdir(exist_ok=True)
    lease = FileLease(path)
    lease.close()
    assert path.exists()
    _assert_result(_delete(memory), "completed")
    _assert_purged(memory)


@pytest.mark.parametrize("unknown", ["unregistered.txt", "unexpected/subdirectory.txt"])
def test_unknown_meeting_material_is_preserved_pending_without_recursive_authority(catalog, unknown):
    memory, source = catalog
    path = memory.data_dir / "meetings" / MEETING_ID / unknown
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unknown quartzzephyr material")
    before = _files(source.directory)
    _assert_result(_delete(memory), "pending")
    _hidden(memory)
    assert path.read_bytes() == b"unknown quartzzephyr material"
    assert json.loads(_registry(memory.data_dir).read_text())["phase"] == "pending"
    memory.reconcile()
    assert path.exists() and _files(source.directory) == before
    path.unlink()
    if path.parent.name == "unexpected":
        path.parent.rmdir()
    memory.reconcile()
    _assert_purged(memory)


@contextmanager
def _deny_windows_delete(path):
    if os.name != "nt":
        pytest.skip("Windows FILE_SHARE_DELETE semantics")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    closed = False
    def release():
        nonlocal closed
        if not closed:
            assert kernel.CloseHandle(handle)
            closed = True
    try:
        yield release
    finally:
        release()


def test_windows_locked_audio_stays_hidden_pending_until_handle_released(catalog):
    memory, _ = catalog
    audio = memory.data_dir / memory.show(MEETING_ID)["audio_assets"][0]["relative_path"]
    with _deny_windows_delete(audio):
        _assert_result(_delete(memory), "pending")
        _hidden(memory)
        assert audio.is_file()
        assert _rows(memory.data_dir, "SELECT status FROM meetings") == [("deleting",)]
        memory.reconcile()
        assert audio.is_file()
    memory.reconcile()
    _assert_purged(memory)


def test_before_delete_hook_releases_real_playback_handle_before_first_purge(catalog):
    memory, _ = catalog
    audio = memory.data_dir / memory.show(MEETING_ID)["audio_assets"][0]["relative_path"]
    with _deny_windows_delete(audio) as release:
        def stop(identity):
            assert identity == MEETING_ID and audio.exists()
            assert not _registry(memory.data_dir).exists()
            release()
        memory.before_delete.append(stop)
        _assert_result(_delete(memory), "completed")
    _assert_purged(memory)


def test_before_delete_hook_failure_never_accepts_deletion(catalog):
    memory, _ = catalog
    before = _files(memory.data_dir / "meetings")
    def fail(identity):
        raise OSError("synthetic player stop failure")
    memory.before_delete.append(fail)
    _assert_result(_delete(memory), "rejected")
    assert not _registry(memory.data_dir).exists()
    assert memory.show(MEETING_ID) is not None
    assert _files(memory.data_dir / "meetings") == before


def test_different_explicit_origin_and_identity_with_identical_audio_remains_allowed(catalog, tmp_path):
    memory, source = catalog
    other = make_session(tmp_path / "explicit-new-origin", meeting_id=OTHER_MEETING_ID)
    assert source.audio.read_bytes() == other.audio.read_bytes()
    _assert_result(_delete(memory), "completed")
    assert memory.ingest_session(source.directory).status == "rejected"
    assert memory.ingest_session(other.directory).status == "inserted"
    assert memory.show(OTHER_MEETING_ID) is not None


def test_same_fingerprint_different_legacy_origin_gets_new_identity_after_delete(tmp_path):
    source = make_session(tmp_path / "legacy")
    duplicate = tmp_path / "explicit-copy"
    shutil.copytree(source.directory, duplicate)
    memory = MemoryService(tmp_path / "data", sessions_root=tmp_path / "sessions")
    first = memory.ingest_session(source.directory)
    assert first.status == "inserted"
    fingerprint = _rows(memory.data_dir, "SELECT origin_fingerprint FROM ingest_sources")[0][0]
    assert memory.delete_meeting(first.meeting_id).status == "completed"
    copied = memory.ingest_session(duplicate)
    assert copied.status == "inserted" and copied.meeting_id != first.meeting_id
    assert _rows(memory.data_dir, "SELECT origin_fingerprint FROM ingest_sources") == [(fingerprint,)]
    assert memory.show(copied.meeting_id) is not None


def test_complete_tombstone_survives_db_loss_and_restored_old_receipts(catalog, tmp_path):
    memory, source = catalog
    old_tree = tmp_path / "saved-owned-tree"
    shutil.copytree(memory.data_dir / "meetings", old_tree)
    origins = _rows(memory.data_dir, "SELECT origin_fingerprint FROM ingest_sources")
    _assert_result(_delete(memory), "completed")
    shutil.copytree(old_tree, memory.data_dir / "meetings", dirs_exist_ok=True)
    _lose_database(memory.data_dir, tmp_path)
    rebuilt = MemoryService(memory.data_dir, sessions_root=memory.data_dir.parent / "sessions")
    _hidden(rebuilt)
    rebuilt.reconcile()
    assert _rows(memory.data_dir, "SELECT origin_fingerprint FROM deleted_origins") == origins
    assert rebuilt.ingest_session(source.directory).status == "rejected"
    _assert_purged(rebuilt)


def test_audio_a_b_a_exact_links_survive_rebuild_then_all_history_is_deleted(tmp_path):
    memory = MemoryService(tmp_path / "data", sessions_root=tmp_path / "sessions")
    sources = []
    hashes = []
    for index in range(3):
        text = f"{CANARY} revision {index}"
        source = make_session(tmp_path / f"origin-{index}", meeting_id=MEETING_ID,
                              transcript_text=text,
                              segments=[{"text": text, "start": 0.0, "end": 1.0}])
        if index == 1:
            contents = bytearray(source.audio.read_bytes())
            contents[-2:] = b"\x34\x12"
            source.audio.write_bytes(contents)
        hashes.append(hashlib.sha256(source.audio.read_bytes()).hexdigest())
        previous = memory.active_revision(MEETING_ID)
        assert memory.ingest_session(source.directory, revision_of=previous).status == (
            "inserted" if index == 0 else "revised")
        sources.append(source)
    before = memory.show(MEETING_ID)
    links = before["revision_audio_assets"]
    revisions = sorted(before["revisions"], key=lambda r: r["revision_number"])
    assets = {a["id"]: a for a in before["audio_assets"]}
    ids = [next(link["asset_id"] for link in links if link["revision_id"] == r["id"])
           for r in revisions]
    assert ids[0] == ids[2] != ids[1]
    assert [assets[identity]["sha256"] for identity in ids] == hashes
    _lose_database(memory.data_dir, tmp_path)
    memory = MemoryService(memory.data_dir, sessions_root=memory.data_dir.parent / "sessions")
    memory.reconcile()
    assert sorted(memory.show(MEETING_ID)["revision_audio_assets"], key=lambda x: x["revision_id"]) == (
        sorted(links, key=lambda x: x["revision_id"]))
    _assert_result(_delete(memory), "completed")
    _lose_database(memory.data_dir, tmp_path)
    memory = MemoryService(memory.data_dir, sessions_root=memory.data_dir.parent / "sessions")
    memory.reconcile()
    _assert_purged(memory)
    assert all(source.audio.exists() for source in sources)


def _record(root, *, identity=MEETING_ID, phase="pending", origins=()):
    directory = root / ".deletions"
    directory.mkdir(exist_ok=True)
    path = directory / f"{identity}.json"
    write_json(path, {"schema_version": 1, "meeting_id": identity,
                      "origins": list(origins), "phase": phase})
    return path


def _origin(root):
    path, fingerprint = _rows(root, "SELECT origin_path,origin_fingerprint FROM ingest_sources")[0]
    return {"origin_path": path, "origin_fingerprint": fingerprint}


@pytest.mark.parametrize("phase", ["pending", "complete"])
def test_read_view_suppresses_identity_union_exact_origin_pair_aliases(catalog, tmp_path, phase):
    memory, _ = catalog
    origin = _origin(memory.data_dir)
    other = make_session(tmp_path / "other", meeting_id=OTHER_MEETING_ID)
    assert memory.ingest_session(other.directory).status == "inserted"
    _record(memory.data_dir, identity=str(uuid4()), phase=phase, origins=[origin])
    snapshot = _view(memory.data_dir)
    assert MEETING_ID in snapshot.suppressed_meeting_ids
    assert OTHER_MEETING_ID not in snapshot.suppressed_meeting_ids
    # The pair is conjunctive: matching only the hash does not suppress the other origin.
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
        connection.execute("UPDATE ingest_sources SET origin_fingerprint=? WHERE meeting_id=?",
                           (origin["origin_fingerprint"], OTHER_MEETING_ID))
        connection.commit()
    assert OTHER_MEETING_ID not in _view(memory.data_dir).suppressed_meeting_ids
    # Matching only the origin with different fingerprint also must remain visible.
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
        connection.execute("UPDATE ingest_sources SET origin_path=?,origin_fingerprint=? "
                           "WHERE meeting_id=?", (origin["origin_path"], "a" * 64, OTHER_MEETING_ID))
        connection.commit()
    assert OTHER_MEETING_ID not in _view(memory.data_dir).suppressed_meeting_ids


def test_read_view_digest_tracks_alias_change_with_unchanged_registry_and_stays_stable(catalog):
    memory, source = catalog
    origin = _origin(memory.data_dir)
    path = _record(memory.data_dir, identity=OTHER_MEETING_ID, origins=[origin])
    raw = path.read_bytes()
    before = _view(memory.data_dir)
    assert before.suppressed_meeting_ids == tuple(sorted((MEETING_ID, OTHER_MEETING_ID)))
    assert _view(memory.data_dir) == before
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
        connection.execute("UPDATE ingest_sources SET origin_fingerprint=?", ("f" * 64,))
        connection.commit()
    after = _view(memory.data_dir)
    assert path.read_bytes() == raw
    assert after.suppressed_meeting_ids == (OTHER_MEETING_ID,)
    assert after.fingerprint != before.fingerprint
    assert _view(memory.data_dir).fingerprint == after.fingerprint
    source.directory.rename(source.directory.with_name("origin-is-offline"))
    assert _view(memory.data_dir) == after  # Never inspect the external origin.


def test_read_view_canonical_digest_ignores_record_order_but_includes_control(catalog):
    memory, _ = catalog
    origin = _origin(memory.data_dir)
    path = _record(memory.data_dir, origins=[origin])
    before = _view(memory.data_dir)
    document = json.loads(path.read_text())
    path.write_text(json.dumps(dict(reversed(list(document.items()))), separators=(",", ":")))
    assert _view(memory.data_dir) == before
    document["phase"] = "complete"
    write_json(path, document)
    after = _view(memory.data_dir)
    assert after.suppressed_meeting_ids == before.suppressed_meeting_ids
    assert after.fingerprint != before.fingerprint


def test_read_view_is_two_field_frozen_sorted_snapshot_and_caller_closes_once(catalog, monkeypatch):
    memory, _ = catalog
    _record(memory.data_dir, identity=OTHER_MEETING_ID)
    _record(memory.data_dir)
    library = open_library(memory.data_dir)
    closed = []
    original_close = LibraryConnection.close
    def counted_close(self):
        closed.append(self)
        original_close(self)
    monkeypatch.setattr(LibraryConnection, "close", counted_close)
    try:
        with service_module.catalog_read_view(library) as snapshot:
            assert {field.name for field in fields(snapshot)} == {
                "suppressed_meeting_ids", "fingerprint"}
            assert snapshot.suppressed_meeting_ids == (MEETING_ID, OTHER_MEETING_ID)
            assert isinstance(snapshot.fingerprint, str) and snapshot.fingerprint
            assert not hasattr(snapshot, "entries")
            with pytest.raises((FrozenInstanceError, AttributeError)):
                snapshot.fingerprint = "modified"
        assert closed == []
        assert library.execute("SELECT 1").fetchone()[0] == 1
    finally:
        library.close()
    assert closed == [library]


def test_read_view_lock_precedes_single_begin_and_is_retained_through_caller_fetch(catalog, monkeypatch):
    memory, _ = catalog
    library = open_library(memory.data_dir)
    statements = []
    lock_failures = []
    def trace(sql):
        statements.append(sql)
        if sql.strip().upper().startswith("BEGIN"):
            try:
                lease = FileLease(memory.data_dir / ".memory.lock", timeout=0)
            except LeaseBusy:
                return
            lease.close()
            lock_failures.append("BEGIN ran before publication lock acquisition")
    library.connection.set_trace_callback(trace)
    def forbidden(*args, **kwargs):
        pytest.fail("Read view opened another connection, migrated, or nested an operation")
    monkeypatch.setattr(service_module, "open_library", forbidden)
    monkeypatch.setattr(service_module, "migrate", forbidden)
    monkeypatch.setattr(MemoryService, "_operation", forbidden)
    try:
        with service_module.catalog_read_view(library):
            assert library.connection.in_transaction
            assert library.execute("SELECT id FROM meetings").fetchall()
            with pytest.raises(LeaseBusy):
                FileLease(memory.data_dir / ".memory.lock", timeout=0)
        assert not library.connection.in_transaction
        assert lock_failures == []
        assert [s.strip().upper() for s in statements if s.upper().startswith("BEGIN")] == ["BEGIN"]
        with closing(FileLease(memory.data_dir / ".memory.lock", timeout=0)):
            pass
    finally:
        library.close()


@pytest.mark.parametrize("failure", ["caller", "sql"])
def test_read_view_exception_rolls_back_only_its_transaction_and_releases_lock(catalog, failure):
    memory, _ = catalog
    library = open_library(memory.data_dir)
    try:
        exception = RuntimeError if failure == "caller" else LibraryError
        with pytest.raises(exception):
            with service_module.catalog_read_view(library):
                if failure == "caller":
                    raise RuntimeError("caller page assembly failed")
                library.connection.execute("SELECT * FROM no_such_b5_table")
        assert not library.connection.in_transaction
        assert library.execute("SELECT 1").fetchone()[0] == 1
        with closing(FileLease(memory.data_dir / ".memory.lock", timeout=0)):
            pass
    finally:
        library.close()


def test_read_view_rejects_foreign_transaction_without_commit_or_rollback(catalog):
    memory, _ = catalog
    library = open_library(memory.data_dir)
    try:
        library.execute("BEGIN")
        library.execute("UPDATE meetings SET title='uncommitted caller value'")
        with pytest.raises(LibraryError):
            with service_module.catalog_read_view(library):
                pytest.fail("Foreign transaction was accepted")
        assert library.connection.in_transaction
        assert library.execute("SELECT title FROM meetings").fetchone()[0] == "uncommitted caller value"
        library.connection.rollback()
        assert library.execute("SELECT title FROM meetings").fetchone()[0] != "uncommitted caller value"
    finally:
        library.close()


def test_read_view_rejects_nested_use_without_destroying_outer_snapshot(catalog):
    memory, _ = catalog
    library = open_library(memory.data_dir)
    try:
        with service_module.catalog_read_view(library) as outer:
            with pytest.raises(LibraryError):
                with service_module.catalog_read_view(library):
                    pytest.fail("Nested read view was accepted")
            assert library.connection.in_transaction
            assert library.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 1
            assert outer.suppressed_meeting_ids == ()
    finally:
        library.close()


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
def test_read_view_busy_lock_never_begins_transaction_and_connection_remains_owned(catalog, journal):
    memory, _ = catalog
    library = open_library(memory.data_dir)
    library.execute(f"PRAGMA journal_mode={journal}")
    statements = []
    library.connection.set_trace_callback(statements.append)
    lease = FileLease(memory.data_dir / ".memory.lock")
    try:
        with pytest.raises(Busy):
            with service_module.catalog_read_view(library):
                pytest.fail("Read view ignored held publication owner")
        assert not library.connection.in_transaction
        assert not any(s.upper().startswith("BEGIN") for s in statements)
        assert library.execute("SELECT 1").fetchone()[0] == 1
    finally:
        lease.close()
        library.close()


def test_rollback_journal_sql_busy_is_typed_and_releases_read_guard(catalog):
    memory, _ = catalog
    library = open_library(memory.data_dir)
    library.execute("PRAGMA journal_mode=DELETE")
    other = sqlite3.connect(memory.data_dir / "library.sqlite3", isolation_level=None)
    other.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(Busy):
            with service_module.catalog_read_view(library):
                library.execute("SELECT * FROM meetings").fetchall()
        assert not library.connection.in_transaction
        with closing(FileLease(memory.data_dir / ".memory.lock", timeout=0)):
            pass
    finally:
        other.rollback()
        other.close()
        assert library.execute("SELECT 1").fetchone()[0] == 1
        library.close()


def test_genuinely_absent_registry_is_empty_read_only_and_never_created(catalog):
    memory, _ = catalog
    registry = memory.data_dir / ".deletions"
    assert not registry.exists()
    before = _files(memory.data_dir / "meetings")
    first = _view(memory.data_dir)
    assert first.suppressed_meeting_ids == ()
    assert _view(memory.data_dir) == first
    assert not registry.exists()
    assert _files(memory.data_dir / "meetings") == before


@pytest.mark.parametrize("damage", ["json", "array", "wrong_identity", "unknown_field", "directory"])
def test_invalid_registry_fails_exposure_ingest_and_rebuild_closed_preserving_material(
    catalog, tmp_path, damage
):
    memory, source = catalog
    path = _record(memory.data_dir, origins=[_origin(memory.data_dir)])
    if damage == "json":
        path.write_text("{broken")
    elif damage == "array":
        write_json(path, [])
    elif damage == "directory":
        path.unlink()
        path.mkdir()
    else:
        document = json.loads(path.read_text())
        document["meeting_id" if damage == "wrong_identity" else "raw_text"] = CANARY
        write_json(path, document)
    before = _files(memory.data_dir / "meetings")
    for read in (lambda: _view(memory.data_dir), lambda: memory.show(MEETING_ID),
                 lambda: memory.get_meeting(MEETING_ID)):
        with pytest.raises(LibraryError):
            read()
    with pytest.raises(LibraryError):
        memory.ingest_session(source.directory)
    with pytest.raises(LibraryError):
        memory.reconcile()
    assert _files(memory.data_dir / "meetings") == before
    _lose_database(memory.data_dir, tmp_path)
    rebuilt = MemoryService(memory.data_dir)
    with pytest.raises(LibraryError):
        rebuilt.reconcile()
    assert _files(memory.data_dir / "meetings") == before


def test_inaccessible_registry_is_not_treated_as_absent(catalog, monkeypatch):
    memory, _ = catalog
    path = _record(memory.data_dir)
    original = Path.open
    def open_path(self, *args, **kwargs):
        if self == path:
            raise PermissionError("Synthetic registry access denied")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(LibraryError):
        _view(memory.data_dir)
    with pytest.raises(LibraryError):
        memory.reconcile()
    assert (memory.data_dir / "meetings" / MEETING_ID).is_dir()


@pytest.mark.parametrize("target", ["registry", "record", "meeting", "owner"])
def test_unsafe_link_cannot_grant_registry_or_purge_authority(catalog, tmp_path, target):
    memory, _ = catalog
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_bytes(b"outside synthetic canary")
    if target == "registry":
        link = memory.data_dir / ".deletions"
        destination = outside
    elif target == "record":
        (memory.data_dir / ".deletions").mkdir()
        link = _registry(memory.data_dir)
        destination = sentinel
    elif target == "owner":
        link = memory.data_dir / ".owners" / f"{MEETING_ID}.lock"
        link.parent.mkdir(exist_ok=True)
        link.unlink(missing_ok=True)
        destination = sentinel
    else:
        # Add an unsafe child; no replacement/removal of the real owned tree.
        link = memory.data_dir / "meetings" / MEETING_ID / "unsafe"
        destination = outside
    try:
        link.symlink_to(destination, target_is_directory=destination.is_dir())
    except OSError:
        pytest.skip("Symlink creation privilege unavailable")
    before = _files(outside)
    if target in {"registry", "record"}:
        with pytest.raises(LibraryError):
            _view(memory.data_dir)
        with pytest.raises(LibraryError):
            memory.reconcile()
    else:
        try:
            result = _delete(memory)
        except (LibraryError, ValueError, OSError):
            pass
        else:
            assert result.status in {"rejected", "pending"}
    assert _files(outside) == before and link.is_symlink()



def _install_fault(monkeypatch, root, boundary, *, crash=False):
    """Cut real durable writes/transactions/removals without production test hooks."""
    from here.memory.repository import MemoryRepository

    original_replace = os.replace
    original_unlink = Path.unlink
    original_execute = LibraryConnection.execute
    original_transaction = MemoryRepository.transaction
    hits = []
    def fault():
        hits.append(boundary)
        if crash:
            os._exit(73)
        raise OSError(f"Synthetic B5 {boundary} fault")
    def replace(source, target, *args, **kwargs):
        target = Path(target)
        is_record = target.parent == root / ".deletions" and target.suffix == ".json"
        phase = json.loads(Path(source).read_text(encoding="utf-8")).get("phase") if is_record else None
        if phase == "pending" and boundary == "before-barrier":
            fault()
        if phase == "complete" and boundary == "before-complete":
            fault()
        result = original_replace(source, target, *args, **kwargs)
        if phase == "pending" and boundary == "after-barrier":
            fault()
        if phase == "complete" and boundary == "after-complete":
            fault()
        return result
    def unlink(self, *args, **kwargs):
        owned = self.is_relative_to(root / "meetings" / MEETING_ID)
        backup = self.parent == root / "backups"
        if backup and boundary == "backup":
            fault()
        result = original_unlink(self, *args, **kwargs)
        if owned and boundary == "mid-files":
            fault()
        return result
    def execute(self, sql, parameters=()):
        upper = sql.strip().upper()
        if boundary == "wal" and "WAL_CHECKPOINT" in upper:
            fault()
        if boundary == "sql-rollback" and upper.startswith("UPDATE MEETINGS") and (
            _registry(root).exists()
        ):
            fault()
        return original_execute(self, sql, parameters)
    @contextmanager
    def transaction(self):
        with original_transaction(self) as repo:
            yield repo
        if boundary == "after-delete-commit" and _registry(root).exists():
            rows = self.library.execute(
                "SELECT phase FROM pending_operations WHERE kind='delete' AND meeting_id=?",
                (MEETING_ID,),
            ).fetchall()
            if rows and rows[0][0] == "prepared":
                fault()
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(LibraryConnection, "execute", execute)
    monkeypatch.setattr(MemoryRepository, "transaction", transaction)
    return hits


@pytest.mark.parametrize("boundary", [
    "before-barrier", "after-barrier", "sql-rollback", "after-delete-commit",
    "mid-files", "before-complete",
])
def test_failure_outcome_respects_durable_barrier_acceptance(catalog, monkeypatch, boundary):
    memory, source = catalog
    original = _files(source.directory)
    before = _files(memory.data_dir / "meetings")
    with monkeypatch.context() as faults:
        hits = _install_fault(faults, memory.data_dir, boundary)
        result = _delete(memory)
        assert hits, "Fault did not reach the intended real boundary"
        _assert_result(result, "rejected" if boundary == "before-barrier" else "pending")
        assert result.error_category == "io"
    if boundary == "before-barrier":
        assert not _registry(memory.data_dir).exists()
        assert memory.show(MEETING_ID) is not None
        assert _files(memory.data_dir / "meetings") == before
    else:
        _hidden(memory)
        assert json.loads(_registry(memory.data_dir).read_text())["phase"] == "pending"
        memory.reconcile()
        memory.reconcile()
        _assert_purged(memory)
    assert _files(source.directory) == original


def test_temporary_barrier_write_failure_preserves_known_orphan_stage_without_suppression(
    catalog, monkeypatch
):
    memory, _ = catalog
    directory = memory.data_dir / ".deletions"
    directory.mkdir()
    from here.output.paths import reserve_artifact_path
    unknown = reserve_artifact_path(_registry(memory.data_dir), ".stage")
    unknown.write_bytes(b"not a deletion record")
    original_open = Path.open
    def fail_staged_open(self, mode="r", *args, **kwargs):
        if self.parent == directory and any(flag in mode for flag in ("w", "x", "a")):
            raise OSError("synthetic staged barrier write failure")
        return original_open(self, mode, *args, **kwargs)
    before = _files(memory.data_dir / "meetings")
    with monkeypatch.context() as faults:
        faults.setattr(Path, "open", fail_staged_open)
        _assert_result(_delete(memory), "rejected")
    assert not _registry(memory.data_dir).exists()
    assert unknown.read_bytes() == b"not a deletion record"
    assert _files(memory.data_dir / "meetings") == before
    # Actual A producer-role orphan is not a barrier and must not change visibility.
    assert _view(memory.data_dir).suppressed_meeting_ids == ()
    assert memory.show(MEETING_ID) is not None
    assert _rows(memory.data_dir, "SELECT id FROM meetings") == [(MEETING_ID,)]


def test_durable_barrier_then_real_sql_busy_returns_pending_and_reconcile_finishes(catalog, monkeypatch):
    memory, _ = catalog
    original_replace = os.replace
    blockers = []
    def replace(source, target, *args, **kwargs):
        result = original_replace(source, target, *args, **kwargs)
        if Path(target) == _registry(memory.data_dir):
            document = json.loads(Path(target).read_text())
            if document["phase"] == "pending" and not blockers:
                connection = sqlite3.connect(memory.data_dir / "library.sqlite3", isolation_level=None)
                connection.execute("BEGIN IMMEDIATE")
                blockers.append(connection)
        return result
    try:
        with monkeypatch.context() as faults:
            faults.setattr(os, "replace", replace)
            result = _delete(memory)
        assert blockers
        _assert_result(result, "pending")
        assert result.error_category == "io" and result.reason == "busy"
        _hidden(memory)
    finally:
        for connection in blockers:
            connection.rollback()
            connection.close()
    memory.reconcile()
    _assert_purged(memory)


@pytest.mark.parametrize("boundary", ["wal", "backup"])
def test_rows_and_files_gone_but_managed_residue_failure_stays_pending(catalog, monkeypatch, boundary):
    memory, _ = catalog
    backup = memory.data_dir / "backups" / "library-before-delete.sqlite3"
    backup.parent.mkdir()
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as live:
        with closing(sqlite3.connect(backup)) as copy:
            live.backup(copy)
    with monkeypatch.context() as faults:
        hits = _install_fault(faults, memory.data_dir, boundary)
        result = _delete(memory)
        assert hits, "Residue fault was not reached"
        _assert_result(result, "pending")
    _hidden(memory)
    assert json.loads(_registry(memory.data_dir).read_text())["phase"] == "pending"
    memory.reconcile()
    _assert_purged(memory)


_CRASH_SCRIPT = r"""
import os, runpy, socket, sys, types
from pathlib import Path
repo, root, boundary, sessions_root = map(str, sys.argv[1:])
package = types.ModuleType('here')
package.__path__ = [str(Path(repo) / 'src')]
sys.modules['here'] = package
sys.path.insert(0, str(Path(repo) / 'tests'))
def forbidden(*args, **kwargs):
    raise AssertionError('B5 child used network')
socket.socket.connect = forbidden
namespace = runpy.run_path(str(Path(repo) / 'tests' / 'test_memory_delete.py'))
import pytest
with pytest.MonkeyPatch.context() as monkeypatch:
    memory = namespace['MemoryService'](Path(root), sessions_root=Path(sessions_root))
    namespace['_install_fault'](monkeypatch, Path(root), boundary, crash=True)
    memory.delete_meeting(namespace['MEETING_ID'])
raise AssertionError('B5 child failed to reach requested crash boundary')
"""


@pytest.mark.parametrize("boundary", ["before-barrier", "after-barrier", "after-delete-commit",
                                      "mid-files", "before-complete", "after-complete"])
@pytest.mark.parametrize("lose_database", [False, True])
def test_subprocess_crash_restart_obeys_r42_acceptance_and_never_resurrects(
    catalog, tmp_path, boundary, lose_database
):
    memory, source = catalog
    original = _files(source.directory)
    env = dict(os.environ)
    env["HERE_ENV_FILE"] = str(tmp_path / "absent.env")
    env["HERE_DATA_DIR"] = str(memory.data_dir)
    env["TRANSCRIPTIONS_DIR"] = str(tmp_path / "sessions")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENAI_API_KEY"] = "synthetic-test-key"
    env.pop("HERE_PROVIDER_SMOKE_KEY", None)
    env.pop("HERE_WHEEL_PYTHON", None)
    process = subprocess.run(
        [sys.executable, "-c", _CRASH_SCRIPT, str(Path(__file__).resolve().parents[1]),
         str(memory.data_dir), boundary, str(tmp_path / "sessions")],
        env=env, capture_output=True, text=True, timeout=25,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.returncode == 73, process.stdout + process.stderr
    accepted = boundary != "before-barrier"
    if accepted:
        assert _registry(memory.data_dir).is_file()
        _hidden(memory)
    else:
        assert not _registry(memory.data_dir).exists()
        assert memory.show(MEETING_ID) is not None
    if lose_database:
        _lose_database(memory.data_dir, tmp_path)
    restarted = MemoryService(memory.data_dir, sessions_root=memory.data_dir.parent / "sessions")
    if accepted:
        _hidden(restarted)
    restarted.reconcile()
    restarted.reconcile()
    if accepted:
        assert restarted.ingest_session(source.directory).status == "rejected"
        _assert_purged(restarted)
    else:
        assert restarted.show(MEETING_ID) is not None
    assert _files(source.directory) == original


def test_private_parser_is_reused_under_writer_lock_without_public_read_context(catalog, monkeypatch):
    memory, source = catalog
    origin = _origin(memory.data_dir)
    _record(memory.data_dir, origins=[origin])
    # If a writer calls the public context, it reacquires the non-reentrant lock.
    def forbidden(*args, **kwargs):
        pytest.fail("Already locked writer called public catalog_read_view")
    monkeypatch.setattr(service_module, "catalog_read_view", forbidden)
    assert memory.ingest_session(source.directory).status == "rejected"
    memory.reindex()
    memory.reconcile()
    assert _rows(memory.data_dir, "SELECT id FROM meetings") == []


def test_synthetic_m4_derived_purge_hook_runs_in_final_sql_transaction(catalog):
    memory, _ = catalog
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
        connection.execute("CREATE TABLE synthetic_m4_citations(meeting_id TEXT, claim TEXT)")
        connection.execute("INSERT INTO synthetic_m4_citations VALUES (?,?)", (MEETING_ID, CANARY))
        connection.commit()
    def purge(repo, identity):
        assert repo.connection.in_transaction and identity == MEETING_ID
        repo.library.execute("DELETE FROM synthetic_m4_citations WHERE meeting_id=?", (identity,))
    memory._derived_row_purgers.append(purge)
    _assert_result(_delete(memory), "completed")
    assert _rows(memory.data_dir, "SELECT * FROM synthetic_m4_citations") == []
    _assert_purged(memory)


def test_synthetic_derived_hook_failure_rolls_back_rows_and_remains_pending(catalog):
    memory, _ = catalog
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
        connection.execute("CREATE TABLE synthetic_m4_citations(meeting_id TEXT, claim TEXT)")
        connection.execute("INSERT INTO synthetic_m4_citations VALUES (?,?)", (MEETING_ID, CANARY))
        connection.commit()
    def fail(repo, identity):
        repo.library.execute("DELETE FROM synthetic_m4_citations WHERE meeting_id=?", (identity,))
        raise sqlite3.OperationalError("synthetic derived purge failed")
    memory._derived_row_purgers.append(fail)
    _assert_result(_delete(memory), "pending")
    assert _rows(memory.data_dir, "SELECT claim FROM synthetic_m4_citations") == [(CANARY,)]
    assert _rows(memory.data_dir, "SELECT id FROM meetings") == [(MEETING_ID,)]
    assert _rows(memory.data_dir, "SELECT COUNT(*) FROM transcript_revisions") == [(1,)]
    assert _rows(memory.data_dir, "SELECT COUNT(*) FROM revision_audio_assets") == [(1,)]
    _hidden(memory)
    memory._derived_row_purgers.clear()
    memory._derived_row_purgers.append(lambda repo, identity: repo.library.execute(
        "DELETE FROM synthetic_m4_citations WHERE meeting_id=?", (identity,)))
    memory.reconcile()
    _assert_purged(memory)



def test_control_accepted_fixture_has_raw_text_exact_audio_and_complete_segment(catalog):
    memory, source = catalog
    shown = memory.show(MEETING_ID)
    assert shown["revisions"][0]["raw_text"] == CANARY
    assert shown["revisions"][0]["status"] == "complete"
    assert len(shown["segments"]) == 1
    asset = shown["audio_assets"][0]
    assert (memory.data_dir / asset["relative_path"]).read_bytes() == source.audio.read_bytes()
    assert _rows(memory.data_dir, "SELECT meeting_id FROM segments_fts") == [(MEETING_ID,)]


def test_restore_old_backup_cannot_expose_or_rebuild_pending_deleted_meeting(catalog, tmp_path):
    memory, source = catalog
    copy = tmp_path / "synthetic-old.sqlite3"
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as live:
        with closing(sqlite3.connect(copy)) as saved:
            live.backup(saved)
    unknown = memory.data_dir / "meetings" / MEETING_ID / "unknown.txt"
    unknown.write_bytes(b"preserve unrelated material")
    _assert_result(_delete(memory), "pending")
    _lose_database(memory.data_dir, tmp_path)
    shutil.copyfile(copy, memory.data_dir / "library.sqlite3")
    assert _rows(memory.data_dir, "SELECT status FROM meetings") == [("completed",)]
    restored = MemoryService(memory.data_dir, sessions_root=memory.data_dir.parent / "sessions")
    _hidden(restored)
    assert restored.ingest_session(source.directory).status == "rejected"
    restored.reconcile()
    assert unknown.read_bytes() == b"preserve unrelated material"
    _hidden(restored)
    unknown.unlink()
    restored.reconcile()
    _assert_purged(restored)


def test_reconcile_restores_origin_tombstone_before_existing_receipt_rebuild(catalog, tmp_path, monkeypatch):
    memory, _ = catalog
    origin = _origin(memory.data_dir)
    _record(memory.data_dir, origins=[origin])
    _lose_database(memory.data_dir, tmp_path)
    restored = MemoryService(memory.data_dir, sessions_root=memory.data_dir.parent / "sessions")
    rebuild = MemoryService._rebuild
    calls = []
    def inspect_before_rebuild(self, repo, *args, **kwargs):
        calls.append(True)
        restored_origins = repo.library.execute(
            "SELECT origin_fingerprint FROM deleted_origins").fetchall()
        assert [tuple(row) for row in restored_origins] == [(origin["origin_fingerprint"],)]
        assert repo.library.execute("SELECT phase FROM pending_operations WHERE kind='delete'").fetchone()[0] in {
            "prepared", "complete"}
        return rebuild(self, repo, *args, **kwargs)
    monkeypatch.setattr(MemoryService, "_rebuild", inspect_before_rebuild)
    restored.reconcile()
    assert calls
    _assert_purged(restored)


def test_r50_alias_resolution_and_caller_fetch_share_same_wal_snapshot(catalog):
    memory, _ = catalog
    origin = _origin(memory.data_dir)
    _record(memory.data_dir, identity=OTHER_MEETING_ID, origins=[origin])
    library = open_library(memory.data_dir)
    assert library.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    try:
        with service_module.catalog_read_view(library) as before:
            assert before.suppressed_meeting_ids == (MEETING_ID, OTHER_MEETING_ID)
            with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as other:
                other.execute("UPDATE ingest_sources SET origin_fingerprint=?", ("b" * 64,))
                other.commit()
            assert library.execute("SELECT origin_fingerprint FROM ingest_sources").fetchone()[0] == (
                origin["origin_fingerprint"])
    finally:
        library.close()
    after = _view(memory.data_dir)
    assert after.suppressed_meeting_ids == (OTHER_MEETING_ID,)
    assert after.fingerprint != before.fingerprint


def test_unknown_registry_role_is_never_opened_as_record_or_granted_purge_authority(catalog, monkeypatch):
    memory, _ = catalog
    directory = memory.data_dir / ".deletions"
    directory.mkdir()
    unknown = directory / "private-looking.env"
    unknown.write_bytes(b"SYNTHETIC_UNKNOWN=never-read")
    original_open = Path.open
    def reject_unknown(self, *args, **kwargs):
        if self == unknown:
            pytest.fail("Registry opened unrecognized role before validating its name")
        return original_open(self, *args, **kwargs)
    before = _files(memory.data_dir / "meetings")
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", reject_unknown)
        with pytest.raises(LibraryError):
            _view(memory.data_dir)
    assert unknown.read_bytes() == b"SYNTHETIC_UNKNOWN=never-read"
    assert _files(memory.data_dir / "meetings") == before


def test_pending_delete_journal_unknown_path_does_not_authorize_external_removal(catalog, tmp_path):
    memory, _ = catalog
    unknown = memory.data_dir / "meetings" / MEETING_ID / "unknown.txt"
    unknown.write_bytes(b"retained")
    _assert_result(_delete(memory), "pending")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside must remain")
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
        connection.execute("UPDATE pending_operations SET destination_path='../outside.txt' WHERE kind='delete'")
        connection.commit()
    memory.reconcile()
    assert outside.read_bytes() == b"outside must remain"
    assert unknown.read_bytes() == b"retained"
    _hidden(memory)
    assert json.loads(_registry(memory.data_dir).read_text())["phase"] == "pending"



@pytest.mark.parametrize("target", ["registry", "meeting"])
def test_windows_junction_never_grants_external_registry_or_cleanup_authority(catalog, tmp_path, target):
    if os.name != "nt":
        pytest.skip("Windows junction semantics")
    memory, _ = catalog
    outside = tmp_path / "junction-target"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_bytes(b"external synthetic material")
    link = (memory.data_dir / ".deletions" if target == "registry" else
            memory.data_dir / "meetings" / MEETING_ID / "junction")
    assert link.absolute().is_relative_to(tmp_path.absolute())
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                            capture_output=True, text=True, timeout=10,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stdout + result.stderr
    assert link.is_junction()
    try:
        if target == "registry":
            with pytest.raises(LibraryError):
                _view(memory.data_dir)
            with pytest.raises(LibraryError):
                memory.reconcile()
        else:
            result = _delete(memory)
            assert result.status in {"rejected", "pending"}
        assert sentinel.read_bytes() == b"external synthetic material"
        assert link.is_junction()
    finally:
        assert link.is_junction() and link.absolute().is_relative_to(tmp_path.absolute())
        link.rmdir()  # Remove only the verified junction itself, never its target.





def test_r51_models_keep_exact_default_order_frozen_slots_at_result_boundary(catalog):
    from here.memory import models
    memory, _ = catalog
    assert [field.name for field in fields(models.DeletionResult)] == [
        "status", "meeting_id", "reason", "warnings", "error_category"]
    default = models.DeletionResult("pending")
    assert (default.meeting_id, default.reason, default.warnings, default.error_category) == (
        None, "", (), None)
    assert not hasattr(default, "__dict__")
    result = _delete(memory)
    assert isinstance(result, models.DeletionResult) and not hasattr(result, "__dict__")
    _assert_result(result, "completed")


def test_before_delete_ack_runs_without_sql_transaction_or_publication_lock(catalog, monkeypatch):
    memory, _ = catalog
    opened = []
    original = service_module.open_library
    def track(*args, **kwargs):
        library = original(*args, **kwargs)
        opened.append(library)
        return library
    monkeypatch.setattr(service_module, "open_library", track)
    acknowledged = []
    def release(identity):
        assert identity == MEETING_ID
        for library in opened:
            try:
                active = library.connection.in_transaction
            except sqlite3.ProgrammingError:
                continue  # Earlier preflight connection was already closed.
            assert not active
        with closing(FileLease(memory.data_dir / ".memory.lock", timeout=0)):
            with closing(sqlite3.connect(memory.data_dir / "library.sqlite3", timeout=0)) as db:
                db.execute("BEGIN IMMEDIATE")
                db.rollback()
        assert not _registry(memory.data_dir).exists()
        acknowledged.append(identity)
    memory.before_delete.append(release)
    _assert_result(_delete(memory), "completed")
    assert acknowledged == [MEETING_ID]


@pytest.mark.parametrize("hook_fails", [False, True], ids=["ack", "failure"])
def test_concurrent_acceptance_during_hook_is_rechecked_and_never_reported_rejected(catalog, hook_fails):
    memory, _ = catalog
    unknown = memory.data_dir / "meetings" / MEETING_ID / "unknown.txt"
    unknown.write_bytes(b"pending cleanup")
    competing = MemoryService(memory.data_dir, sessions_root=memory.data_dir.parent / "sessions")
    def release(identity):
        _assert_result(competing.delete_meeting(identity), "pending")
        if hook_fails:
            raise OSError("private arbitrary callback detail must not become reason")
    memory.before_delete.append(release)
    result = _delete(memory)
    _assert_result(result, "pending")
    if hook_fails:
        assert result.reason == "hook_failed"
    _hidden(memory)
    assert unknown.read_bytes() == b"pending cleanup"
    memory.before_delete.clear()
    unknown.unlink()
    memory.reconcile()
    _assert_purged(memory)


def test_owner_acquired_during_hook_ack_is_revalidated_before_new_barrier(catalog):
    memory, _ = catalog
    owner_path = memory.data_dir / ".owners" / f"{MEETING_ID}.lock"
    owner_path.parent.mkdir(exist_ok=True)
    retained = []
    def acquire_after_preflight(identity):
        retained.append(FileLease(owner_path))
    memory.before_delete.append(acquire_after_preflight)
    try:
        with pytest.raises(Busy):
            _delete(memory)
        assert not _registry(memory.data_dir).exists()
        assert memory.show(MEETING_ID) is not None
    finally:
        for lease in retained:
            lease.close()


def test_prior_barrier_with_missing_sql_rows_keeps_canonical_pending_on_repeated_final_write_failure(
    catalog, monkeypatch
):
    memory, _ = catalog
    with monkeypatch.context() as faults:
        hits = _install_fault(faults, memory.data_dir, "before-complete")
        _assert_result(_delete(memory), "pending")
        assert _rows(memory.data_dir, "SELECT id FROM meetings") == []
        _assert_result(_delete(MemoryService(
            memory.data_dir, sessions_root=memory.data_dir.parent / "sessions")), "pending")
        assert len(hits) >= 2
    _assert_result(_delete(MemoryService(
        memory.data_dir, sessions_root=memory.data_dir.parent / "sessions")), "completed")
    _assert_purged(memory)


def test_prior_valid_barrier_and_publication_lock_busy_returns_pending_not_exception(catalog):
    memory, _ = catalog
    _record(memory.data_dir, origins=[_origin(memory.data_dir)])
    with closing(FileLease(memory.data_dir / ".memory.lock")):
        result = _delete(memory)
    _assert_result(result, "pending")
    assert result.reason == "busy"
    _hidden(memory)


def test_prior_barrier_with_rows_gone_hook_failure_keeps_pending_identity(catalog, monkeypatch):
    memory, _ = catalog
    with monkeypatch.context() as faults:
        _install_fault(faults, memory.data_dir, "before-complete")
        _assert_result(_delete(memory), "pending")
    assert _rows(memory.data_dir, "SELECT id FROM meetings") == []
    def fail(identity):
        raise RuntimeError("arbitrary private failure")
    memory.before_delete.append(fail)
    result = _delete(memory)
    _assert_result(result, "pending")
    assert result.reason == "hook_failed"


def test_known_orphan_stage_is_never_parsed_or_deleted_and_is_reported(catalog, monkeypatch):
    from here.output.paths import reserve_artifact_path
    memory, _ = catalog
    (memory.data_dir / ".deletions").mkdir()
    stage = reserve_artifact_path(_registry(memory.data_dir), ".stage")
    # Even record-shaped bytes at a temporary role cannot establish acceptance.
    write_json(stage, {"schema_version": 1, "meeting_id": MEETING_ID,
                       "origins": [_origin(memory.data_dir)], "phase": "pending"})
    original_open = Path.open
    def prohibit_temp_read(self, mode="r", *args, **kwargs):
        if self == stage and "r" in mode:
            pytest.fail("Known producer stage was read as a deletion record")
        return original_open(self, mode, *args, **kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", prohibit_temp_read)
        assert _view(memory.data_dir).suppressed_meeting_ids == ()
        assert memory.show(MEETING_ID) is not None
        report = memory.reconcile()
    assert report.warnings or report.skipped
    assert stage.exists() and memory.show(MEETING_ID) is not None


def test_unknown_stage_name_fails_closed_without_accepting_or_purging(catalog):
    memory, _ = catalog
    directory = memory.data_dir / ".deletions"
    directory.mkdir()
    unknown = directory / "interrupted.stage"
    unknown.write_bytes(b"unrecognized synthetic material")
    before = _files(memory.data_dir / "meetings")
    with pytest.raises(LibraryError):
        _view(memory.data_dir)
    with pytest.raises(LibraryError):
        memory.reconcile()
    assert _files(memory.data_dir / "meetings") == before
    assert not _registry(memory.data_dir).exists()
    assert unknown.read_bytes() == b"unrecognized synthetic material"


def test_derived_purgers_keep_order_same_transaction_and_relationships_until_final_core_delete(catalog):
    memory, _ = catalog
    detail = memory.show(MEETING_ID)
    revision = detail["meeting"]["active_revision_id"]
    participant = memory.add_meeting_participant(
        MEETING_ID, "Synthetic linked person", speaker_key=detail["segments"][0]["speaker_key"])
    assert participant.speaker_assignments
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as db:
        db.execute("CREATE TABLE synthetic_m4_citations(revision_id TEXT, claim TEXT)")
        db.execute("INSERT INTO synthetic_m4_citations VALUES (?,?)", (revision, CANARY))
        db.commit()
    seen = []
    def first(repo, identity):
        assert repo.connection.in_transaction
        assert repo.get_meeting(identity) is not None
        assert repo.library.execute(
            "SELECT COUNT(*) FROM revision_audio_assets WHERE meeting_id=?", (identity,)
        ).fetchone()[0] == 1
        assert repo.library.execute(
            "SELECT COUNT(*) FROM speaker_assignments WHERE meeting_id=?", (identity,)
        ).fetchone()[0] == 1
        repo.library.execute(
            "DELETE FROM synthetic_m4_citations WHERE revision_id IN "
            "(SELECT id FROM transcript_revisions WHERE meeting_id=?)", (identity,))
        seen.append(("first", repo, repo.connection))
    def second(repo, identity):
        assert seen[-1] == ("first", repo, repo.connection)
        assert repo.connection.in_transaction and repo.get_revision(revision) is not None
        assert repo.library.execute("SELECT COUNT(*) FROM synthetic_m4_citations").fetchone()[0] == 0
        seen.append(("second", repo, repo.connection))
    memory._derived_row_purgers.extend([first, second])
    _assert_result(_delete(memory), "completed")
    assert [entry[0] for entry in seen] == ["first", "second"]
    _assert_purged(memory)



@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_known_stage_role_is_not_safe_when_linked_and_fails_closed(catalog, tmp_path, kind):
    from here.output.paths import reserve_artifact_path
    memory, _ = catalog
    directory = memory.data_dir / ".deletions"
    directory.mkdir()
    stage = reserve_artifact_path(_registry(memory.data_dir), ".stage")
    stage.unlink()
    outside = tmp_path / "linked-metadata.json"
    outside.write_bytes(b"synthetic metadata target must remain")
    try:
        if kind == "symlink":
            stage.symlink_to(outside)
        else:
            stage.hardlink_to(outside)
    except OSError:
        pytest.skip(f"{kind} creation unavailable")
    try:
        with pytest.raises(LibraryError):
            _view(memory.data_dir)
        with pytest.raises(LibraryError):
            memory.reconcile()
        assert stage.exists()
        assert outside.read_bytes() == b"synthetic metadata target must remain"
    finally:
        stage.unlink()
