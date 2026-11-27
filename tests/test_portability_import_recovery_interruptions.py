"""Restart portable recovery after interruptions of owned filesystem work."""

from __future__ import annotations

from pathlib import Path

import pytest
import test_portability_import_recovery as recovery
from here.memory.artifacts import PORTABLE_RECEIPT_NAME
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID
from test_portability_import_recovery import source as source
from test_portability_import_recovery import target as target

pytestmark = recovery.pytestmark

_RECEIPT_CUTOVER = r"""
if cutover == "before-receipt-replace":
    def cut_before_receipt_publish(move):
        def move_file(self, target):
            receipt = pathlib.Path(target)
            if (
                receipt.name == PORTABLE_RECEIPT_NAME
                and receipt.parent.parent.name == "revisions"
                and receipt.parent.parent.parent.name == receipt_meeting_id
            ):
                assert self.is_file() and self.stat().st_size > 0
                assert not receipt.exists()
                ready.with_suffix(".receipt-source").write_text(str(self), encoding="utf-8")
                pause()
            return move(self, target)
        return move_file

    pathlib.Path.replace = cut_before_receipt_publish(pathlib.Path.replace)
    pathlib.Path.rename = cut_before_receipt_publish(pathlib.Path.rename)
"""


def test_reconcile_retries_interrupted_uncommitted_meeting_removal(
    source: MemoryService,
    target: MemoryService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = recovery._export(source, tmp_path / "uncommitted-removal.here.zip")
    with recovery._paused_import(target, package, "before-sql-commit") as process:
        stage = recovery._import_stage(target)
        destination = recovery._meeting_root(target)
        assert destination.is_dir()
        recovery._kill(process)

    archive = package.read_bytes()
    neighbor = stage.parent / "independent-stage"
    neighbor.write_bytes(b"Unrelated synthetic material")
    assert recovery._journal(target) == [(MEETING_ID, "prepared")]
    owned_roots = {
        recovery._probe_path(destination),
        recovery._probe_path(stage / "work" / MEETING_ID),
    }
    rmdir = Path.rmdir
    interrupted = False

    def remove_directory(path: Path) -> None:
        nonlocal interrupted
        physical = recovery._probe_path(path)
        if not interrupted and physical in owned_roots and not any(physical.iterdir()):
            interrupted = True
            raise OSError("Synthetic interruption removing the empty portable meeting")
        rmdir(path)

    first_restart = recovery._memory(target.data_dir.parent)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "rmdir", remove_directory)
        first_restart.reconcile()
    assert interrupted

    fresh = recovery._memory(target.data_dir.parent)
    for _ in range(2):
        fresh.reconcile()
        assert fresh.show(MEETING_ID) is None
        assert recovery._journal(fresh) == [(MEETING_ID, "reverted")]
        assert not destination.exists()
        assert not stage.exists()
        assert neighbor.read_bytes() == b"Unrelated synthetic material"
        assert package.read_bytes() == archive


def test_reconcile_recovers_receipt_interrupted_at_atomic_publish(
    source: MemoryService,
    target: MemoryService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = recovery._export(source, tmp_path / "receipt-publication.here.zip")
    archive = package.read_bytes()
    committed_history = source.get_meeting(MEETING_ID)
    destination = recovery._meeting_root(target)

    # Extend the existing real Windows subprocess harness with this cutover.
    entry = "\nmemory = service.MemoryService("
    setup, separator, run_import = recovery._CHILD_IMPORT.partition(entry)
    assert separator
    cutover = f"\nreceipt_meeting_id = {MEETING_ID!r}\n" + _RECEIPT_CUTOVER
    with monkeypatch.context() as patch:
        patch.setattr(recovery, "_CHILD_IMPORT", setup + cutover + separator + run_import)
        with recovery._paused_import(target, package, "before-receipt-replace") as process:
            stage = recovery._import_stage(target)
            assert not list(recovery._probe_path(destination).rglob(PORTABLE_RECEIPT_NAME))
            recovery._kill(process)
    assert recovery._journal(target) == [(MEETING_ID, "prepared")]

    source_record = package.parent / "import-ready.receipt-source"
    receipt_source = recovery._probe_path(Path(source_record.read_text(encoding="utf-8")))
    assert receipt_source.is_file()

    neighbor = stage.parent / f".{PORTABLE_RECEIPT_NAME}.synthetic.stage"
    neighbor.write_bytes(b"Unrelated synthetic material")
    fresh = recovery._memory(target.data_dir.parent)
    fresh.reconcile()

    recovery._assert_restored(fresh, committed_history)
    assert recovery._journal(fresh) == [(MEETING_ID, "complete")]
    assert not stage.exists()
    receipt_work_removed = not receipt_source.exists()
    deletion = fresh.delete_meeting(MEETING_ID)
    assert (receipt_work_removed, deletion.status) == (True, "completed")
    assert fresh.show(MEETING_ID) is None
    assert not destination.exists()
    assert neighbor.read_bytes() == b"Unrelated synthetic material"
    assert package.read_bytes() == archive
