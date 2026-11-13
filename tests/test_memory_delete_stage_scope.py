"""R60 controls for foreign private stages and observable deletion scope.

These supplement W2's established real-retry fixture; they create no publication
protocol or provider workflow. First-run passes are controls, not invented RED.
"""

from __future__ import annotations

import json
from contextlib import closing
from uuid import uuid4

import pytest
from here.memory.database import Busy
from here.memory.service import MemoryService
from here.output.ownership import FileLease
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session
from test_memory_delete import (
    _assert_purged,
    _assert_result,
    _hidden,
    _install_fault,
    _lose_database,
    catalog as catalog,
)
from test_memory_delete_publication_stage import (
    _files,
    _rows,
    interrupted_retry as interrupted_retry,
    synthetic_environment as synthetic_environment,
)


def _phase(root, identity=MEETING_ID):
    record = json.loads((root / ".deletions" / f"{identity}.json").read_text(encoding="utf-8"))
    assert set(record) == {"schema_version", "meeting_id", "origins", "phase"}
    assert record["schema_version"] == 1 and record["meeting_id"] == identity
    return record["phase"]


def _private_unknown(root):
    directory = root / ".staging" / str(uuid4())
    directory.mkdir(parents=True)
    (directory / "opaque.bin").write_bytes(b"Synthetic R60 private unknown without ownership proof")
    return directory, _files(directory)


def test_sql_proven_foreign_stage_survives_other_meeting_delete_recovery(
    interrupted_retry, monkeypatch,
):
    case = interrupted_retry
    memory = case.memory()
    other = make_session(case.sessions / "other", meeting_id=OTHER_MEETING_ID)
    other_before = _files(other.directory)
    assert memory.ingest_session(other.directory).status == "inserted"
    plan = json.loads(case.stage_files[".publication.json"])
    assert set(plan) == {"schema_version", "operation_id", "hashes"}
    assert plan["operation_id"] == case.operation["id"]
    assert plan["hashes"] == json.loads(case.operation["payload_json"])["hashes"]
    columns = "id,kind,phase,meeting_id,staging_path,destination_path,payload_json"
    foreign_before = _rows(
        case.root, f"SELECT {columns} FROM pending_operations WHERE id=?",
        (case.operation["id"],),
    )
    assert len(foreign_before) == 1 and foreign_before[0][3] == MEETING_ID
    owner = case.root / ".owners" / f"{MEETING_ID}.lock"
    owner.parent.mkdir(exist_ok=True)
    # Global reconcile also visits unrelated prepared publications. An actual
    # canonical lease preserves this foreign operation during that separate work;
    # its meeting identity/paths/plan come from real SQL, never the lease filename.
    with closing(FileLease(owner, timeout=0)):
        with pytest.raises(Busy):
            MemoryService.check_owner(case.root, MEETING_ID)
        with monkeypatch.context() as fault:
            hits = _install_fault(fault, case.root, "before-complete")
            pending = memory.delete_meeting(OTHER_MEETING_ID)
        assert hits and set(hits) == {"before-complete"}
        assert (pending.status, pending.meeting_id, pending.error_category) == (
            "pending", OTHER_MEETING_ID, "io",
        )
        assert _phase(case.root, OTHER_MEETING_ID) == "pending"
        assert _rows(case.root, "SELECT id FROM meetings WHERE id=?", (OTHER_MEETING_ID,)) == []
        _hidden(memory, OTHER_MEETING_ID)
        assert _files(case.stage) == case.stage_files
        assert _rows(
            case.root, f"SELECT {columns} FROM pending_operations WHERE id=?",
            (case.operation["id"],),
        ) == foreign_before
        report = memory.reconcile()
        assert any(item.identity == OTHER_MEETING_ID for item in report.completed)
        result = memory.delete_meeting(OTHER_MEETING_ID)
        assert (result.status, result.meeting_id, result.error_category) == (
            "completed", OTHER_MEETING_ID, None,
        )
        assert _phase(case.root, OTHER_MEETING_ID) == "complete"
        _hidden(memory, OTHER_MEETING_ID)
        assert memory.get_meeting(MEETING_ID).meeting.active_revision_id == case.active_revision
        assert _files(case.stage) == case.stage_files
        assert _rows(
            case.root, f"SELECT {columns} FROM pending_operations WHERE id=?",
            (case.operation["id"],),
        ) == foreign_before
    for origin, before in case.origins:
        assert _files(origin) == before
    assert _files(other.directory) == other_before
    assert not (case.root / "meetings" / OTHER_MEETING_ID).exists()


def test_pending_without_usable_sql_retains_unidentifiable_private_material(
    interrupted_retry, monkeypatch, tmp_path,
):
    case = interrupted_retry
    memory = case.memory()
    with monkeypatch.context() as fault:
        hits = _install_fault(fault, case.root, "after-delete-commit")
        pending = memory.delete_meeting(MEETING_ID)
    assert hits and set(hits) == {"after-delete-commit"}
    _assert_result(pending, "pending")
    assert _phase(case.root) == "pending"
    _hidden(memory)
    assert _files(case.stage) == case.stage_files
    _lose_database(case.root, tmp_path)
    assert not (case.root / "library.sqlite3").exists()
    unknown = case.stage / "opaque.bin"
    unknown.write_bytes(b"Synthetic R60 uncertain private material after loss of SQL authority")
    before = _files(case.stage)
    restarted = case.memory()
    assert _rows(case.root, "SELECT id FROM meetings") == []
    assert _rows(case.root, "SELECT id FROM pending_operations") == []
    _hidden(restarted)
    report = restarted.reconcile()
    result = restarted.delete_meeting(MEETING_ID)
    _assert_result(result, "pending")
    assert result.warnings and (report.warnings or report.skipped)
    assert _phase(case.root) == "pending"
    _hidden(restarted)
    assert _files(case.stage) == before
    assert unknown.read_bytes() == before["opaque.bin"]
    assert json.loads((case.stage / ".publication.json").read_text()) == json.loads(
        case.stage_files[".publication.json"]
    )
    for origin, contents in case.origins:
        assert _files(origin) == contents
    # No test or product removes the uncertainty to manufacture completion.


def test_verified_complete_tombstone_keeps_phase_with_new_private_unknown(catalog):
    memory, source = catalog
    origin_before = _files(source.directory)
    _assert_result(memory.delete_meeting(MEETING_ID), "completed")
    _assert_purged(memory)
    assert _phase(memory.data_dir) == "complete"
    _hidden(memory)
    unknown, before = _private_unknown(memory.data_dir)
    report = memory.reconcile()
    assert report.warnings or report.skipped
    assert _phase(memory.data_dir) == "complete"
    _hidden(memory)
    result = memory.delete_meeting(MEETING_ID)
    _assert_result(result, "completed")
    assert result.error_category is None
    assert _phase(memory.data_dir) == "complete"
    _assert_purged(memory)
    assert _files(unknown) == before
    assert _files(source.directory) == origin_before


def test_preexisting_foreign_private_unknown_does_not_reject_before_barrier(
    catalog, monkeypatch,
):
    memory, source = catalog
    origin_before = _files(source.directory)
    unknown, before = _private_unknown(memory.data_dir)
    assert not (memory.data_dir / ".deletions" / f"{MEETING_ID}.json").exists()
    assert memory.show(MEETING_ID) is not None
    with monkeypatch.context() as fault:
        hits = _install_fault(fault, memory.data_dir, "after-barrier")
        pending = memory.delete_meeting(MEETING_ID)
    assert hits and set(hits) == {"after-barrier"}, "Deletion did not reach durable acceptance"
    _assert_result(pending, "pending")
    assert _phase(memory.data_dir) == "pending"
    _hidden(memory)
    assert _files(unknown) == before
    memory.reconcile()
    result = memory.delete_meeting(MEETING_ID)
    _assert_result(result, "completed")
    assert result.error_category is None
    _assert_purged(memory)
    assert _phase(memory.data_dir) == "complete"
    _hidden(memory)
    assert _files(unknown) == before
    assert _files(source.directory) == origin_before
