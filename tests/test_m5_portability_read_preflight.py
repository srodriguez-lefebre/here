"""Portable export's R50 boundary and the unchanged portable import writer guard."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest
from here.memory.database import LibraryError
from here.memory.portability import PortabilityService
from here.memory.service import MemoryService
from here.output.ownership import FileLease, LeaseBusy
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="uses native portable export")

_SECOND_TOMBSTONE = "33333333-3333-4333-8333-333333333333"
_FINALS = (f"{OTHER_MEETING_ID}.json", f"{_SECOND_TOMBSTONE}.json")
_STAGE = f".{_SECOND_TOMBSTONE}.json.Abc_1234.stage"
_ROLES = frozenset((*_FINALS, _STAGE))
_EXISTING = b"synthetic destination contents"


@dataclass(frozen=True)
class _RegistryStat:
    original: os.stat_result
    st_ino: int

    @property
    def st_dev(self) -> int:
        return self.original.st_dev or 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


@dataclass
class _RegistryObservation:
    changes: int = 0
    opens: list[tuple[int, str, frozenset[str]]] = field(default_factory=list)


def _seed_registry(root: Path) -> Path:
    directory = root / ".deletions"
    directory.mkdir(parents=True)
    for name in _FINALS:
        record = directory / name
        write_json(
            record,
            {"schema_version": 1, "meeting_id": record.stem, "origins": [], "phase": "complete"},
        )
    (directory / _STAGE).write_bytes(b"synthetic incomplete stage")
    return directory


def _observe_registry(
    monkeypatch: pytest.MonkeyPatch, directory: Path, change_after_enumeration: Callable[[], bool]
) -> _RegistryObservation:
    """Change only the registry's reported identity after real role enumeration."""
    observation = _RegistryObservation()
    checked: set[str] = set()
    original_iterdir = Path.iterdir
    original_lstat = Path.lstat
    original_open = Path.open

    def iterdir(path: Path) -> Iterator[Path]:
        entries = original_iterdir(path)
        if path != directory:
            return entries
        members = tuple(entries)
        checked.clear()
        with pytest.raises(LeaseBusy):
            with closing(FileLease(directory.parent / ".memory.lock", timeout=0, create=False)):
                pytest.fail("Registry was enumerated without the catalog publication lock")
        if change_after_enumeration():
            observation.changes += 1
        return iter(members)

    def lstat(path: Path) -> os.stat_result | _RegistryStat:
        info = original_lstat(path)
        if path == directory:
            return _RegistryStat(info, observation.changes + 1)
        if path.parent == directory:
            checked.add(path.name)
        return info

    def open_path(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.parent == directory:
            observation.opens.append((observation.changes, path.name, frozenset(checked)))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "open", open_path)
    return observation


@pytest.fixture
def selected(tmp_path: Path) -> tuple[MemoryService, Path]:
    sessions = tmp_path / "synthetic-sessions"
    source = make_session(sessions / "selected", meeting_id=MEETING_ID)
    library = MemoryService(tmp_path / "configured-library", sessions_root=sessions)
    assert library.ingest_session(source.directory).status == "inserted"
    return library, _seed_registry(library.data_dir)


@pytest.mark.parametrize("kind", ["zip", "markdown"])
@pytest.mark.parametrize("phase", ["capture", "publication"])
def test_export_rejects_changed_registry_before_opening_payload(
    selected: tuple[MemoryService, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: Literal["zip", "markdown"],
    phase: Literal["capture", "publication"],
) -> None:
    library, registry = selected
    service = PortabilityService(library)
    exports = tmp_path / "exports"
    exports.mkdir()
    suffix = ".here.zip" if kind == "zip" else ".md"
    destination = exports / f"selected{suffix}"
    if phase == "capture":
        destination.write_bytes(_EXISTING)

    def change_at_selected_phase() -> bool:
        if phase == "capture":
            return True
        with os.scandir(exports) as entries:
            ready = any(
                entry.name != destination.name and entry.stat().st_size for entry in entries
            )
        if ready:
            # The already-open export sibling cannot replace an unrelated file.
            destination.write_bytes(_EXISTING)
        return ready

    with monkeypatch.context() as fault:
        observed = _observe_registry(fault, registry, change_at_selected_phase)
        with pytest.raises(LibraryError) as failure:
            if kind == "zip":
                service.export((MEETING_ID,), destination, False, lambda _: None, lambda: False)
            else:
                service.export_markdown(MEETING_ID, destination, lambda: False)

    assert destination.read_bytes() == _EXISTING
    assert set(exports.iterdir()) == {destination}
    assert observed.changes == 1, "The targeted registry enumeration was not exercised"
    expected_opens = [] if phase == "capture" else [(0, name, _ROLES) for name in _FINALS]
    assert observed.opens == expected_opens, "Registry identity changed before payload access"
    assert "Deletion registry unavailable" in str(failure.value)


def test_portable_import_keeps_the_existing_writer_registry_guard(
    selected: tuple[MemoryService, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = selected
    package = tmp_path / "selected.here.zip"
    PortabilityService(source).export((MEETING_ID,), package, False, lambda _: None, lambda: False)
    target = MemoryService(
        tmp_path / "target-library", sessions_root=tmp_path / "target-synthetic-sessions"
    )
    target_registry = _seed_registry(target.data_dir)

    with monkeypatch.context() as fault:
        observed = _observe_registry(fault, target_registry, lambda: True)
        report = PortabilityService(target).import_archive(package, lambda _: None, lambda: False)

    assert [(outcome.meeting_id, outcome.status) for outcome in report.meetings] == [
        (MEETING_ID, "inserted")
    ]
    assert target.get_meeting(MEETING_ID).meeting.id == MEETING_ID
    assert observed.changes > 0
    assert observed.opens == [
        (change, name, _ROLES) for change in range(1, observed.changes + 1) for name in _FINALS
    ]
