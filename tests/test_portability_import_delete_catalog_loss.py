"""Recover an accepted portable deletion after loss of its SQLite catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import test_portability_import_recovery as recovery
from here.memory.artifacts import PORTABLE_RECEIPT_NAME
from here.memory.portability import PortabilityService
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID
from test_portability_import_recovery import source as source
from test_portability_import_recovery import target as target

pytestmark = recovery.pytestmark


def test_reconcile_purges_receiptless_portable_group_after_delete_and_catalog_loss(
    source: MemoryService,
    target: MemoryService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = recovery._export(source, tmp_path / "deletion-recovery.here.zip")
    archive = package.read_bytes()
    destination = recovery._meeting_root(target)
    tombstone = target.data_dir / ".deletions" / f"{MEETING_ID}.json"
    with recovery._paused_import(target, package, "before-first-receipt") as process:
        stage = recovery._import_stage(target)
        recovery._kill(process)

    assert target.show(MEETING_ID) is not None
    published = recovery._probe_path(destination)
    audio = next(published.rglob("*.wav"))
    audio_bytes = audio.read_bytes()
    assert audio_bytes
    interrupted = False

    def interrupt_purge(*_args: object) -> None:
        nonlocal interrupted
        interrupted = True
        raise OSError("Synthetic interruption before the managed meeting purge")

    with monkeypatch.context() as patch:
        patch.setattr(MemoryService, "_purge_delete_files", interrupt_purge)
        result = target.delete_meeting(MEETING_ID)
    assert interrupted
    assert result.status == "pending"
    assert json.loads(tombstone.read_text(encoding="utf-8"))["phase"] == "pending"
    assert not list(published.rglob(PORTABLE_RECEIPT_NAME))
    assert audio.read_bytes() == audio_bytes

    recovery._lose_database(target, tmp_path, process)
    assert not (target.data_dir / "library.sqlite3").exists()
    fresh = recovery._memory(target.data_dir.parent)
    fresh.reconcile()

    assert fresh.show(MEETING_ID) is None
    assert not published.exists()
    assert not stage.exists()
    assert json.loads(tombstone.read_text(encoding="utf-8"))["phase"] == "complete"
    report = PortabilityService(fresh).import_archive(package, lambda _: None, lambda: False)
    assert [(item.meeting_id, item.status) for item in report.meetings] == [
        (MEETING_ID, "rejected"),
    ]
    assert fresh.show(MEETING_ID) is None
    assert not published.exists()
    assert package.read_bytes() == archive
