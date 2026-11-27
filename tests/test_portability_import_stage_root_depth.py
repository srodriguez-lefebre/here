"""Exercise Windows cancellation when the import stage root exceeds MAX_PATH."""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from here.memory.portability import ImportCancelled, PortabilityService
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID
from test_portability_import import _catalog
from test_portability_import_recovery import _export, _probe_path
from test_portability_import_recovery import pytestmark as pytestmark
from test_portability_import_recovery import source as source
from test_portability_import_recovery import target as target

_MAX_PATH = 260
_DATA_ROOT_LENGTH = 220


def test_cancelled_import_cleans_stage_root_past_windows_max_path(
    source: MemoryService, target: MemoryService, tmp_path: Path
) -> None:
    prefix = tmp_path / "data-"
    padding = _DATA_ROOT_LENGTH - len(str(prefix))
    if padding < 0:
        pytest.skip("Temporary base is too deep to isolate the portable staging boundary")
    data = prefix.with_name(prefix.name + "x" * padding)
    staging = data / ".staging"
    assert len(str(staging)) < _MAX_PATH
    assert len(str(data / "library.sqlite3-wal")) < _MAX_PATH

    package = _export(source, tmp_path / "depth.here.zip")
    archive = package.read_bytes()
    # Migrate the empty test catalog at normal depth before isolating staging I/O.
    data.mkdir()
    try:
        with (
            closing(sqlite3.connect(target.data_dir / "library.sqlite3")) as origin_db,
            closing(sqlite3.connect(data / "library.sqlite3")) as deep_db,
        ):
            origin_db.backup(deep_db)
        deep_target = MemoryService(data)
        assert deep_target.show(MEETING_ID) is None
        before = _catalog(deep_target)
        stages: list[Path] = []

        def record_first_write(written: int) -> None:
            assert written > 0
            (stage,) = _probe_path(staging).glob("import-*")
            assert stage.is_dir()
            assert len(str(staging / stage.name)) > _MAX_PATH
            stages.append(stage)

        with pytest.raises(ImportCancelled):
            PortabilityService(deep_target).import_archive(
                package, record_first_write, lambda: bool(stages)
            )

        (stage,) = stages
        assert not stage.exists()
        assert not list(_probe_path(staging).iterdir())
        assert not (data / "meetings").exists()
        assert deep_target.show(MEETING_ID) is None
        assert _catalog(deep_target) == before
        assert package.read_bytes() == archive
    finally:
        # Use the extended path even during teardown of a failed partial import.
        probe = _probe_path(data)
        if probe.exists():
            shutil.rmtree(probe)
