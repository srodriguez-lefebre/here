"""M5 B export recovery drives native ownership through real child process death."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import BinaryIO
from zipfile import ZipFile

import pytest
from here.memory import service as service_module
from here.memory.database import Busy, LibraryConnection, LibraryError
from here.memory.models import AudioAsset
from here.memory.service import Conflict, MemoryService
from here.output.ownership import FileLease, LeaseBusy
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="native Windows export ownership")

_CHILD_EXPORT = r"""
import pathlib
import sys
import types

repo, data, sessions, destination, ready = map(pathlib.Path, sys.argv[1:6])
meeting_id, cutover = sys.argv[6:8]
here = types.ModuleType("here")
here.__path__ = [str(repo / "src")]
sys.modules["here"] = here

from here.memory.portability import PortabilityService
from here.memory.service import MemoryService
from here.output.export_binding import WindowsExportSibling

def stop_at_cutover():
    ready.touch()
    if not sys.stdin.buffer.read(1):
        raise RuntimeError("Harness closed before terminating the export")

def progress(written):
    if cutover == "before-rename" and written > 0:
        stop_at_cutover()

if cutover == "after-rename":
    original_replace = WindowsExportSibling.replace_destination

    def replace(self, *, before_rename=None):
        original_replace(self, before_rename=before_rename)
        stop_at_cutover()

    WindowsExportSibling.replace_destination = replace

memory = MemoryService(data, sessions_root=sessions)
PortabilityService(memory).export((meeting_id,), destination, True, progress, lambda: False)
"""


@pytest.fixture
def catalog(tmp_path: Path) -> tuple[MemoryService, Path, Path]:
    sessions = tmp_path / "synthetic-sources"
    source = make_session(sessions / "selected", meeting_id=MEETING_ID)
    memory = MemoryService(tmp_path / "data", sessions_root=sessions)
    assert memory.ingest_session(source.directory).status == "inserted"
    exports = tmp_path / "exports"
    exports.mkdir()
    return memory, sessions, exports / "selected.here.zip"


@contextmanager
def _paused_export(
    memory: MemoryService, sessions: Path, destination: Path, cutover: str = "before-rename"
) -> Iterator[subprocess.Popen[bytes]]:
    ready = destination.parent.parent / "export-ready"
    env = {
        **os.environ,
        "HERE_DATA_DIR": str(memory.data_dir),
        "HERE_ENV_FILE": str(ready.with_suffix(".absent.env")),
        "OPENAI_API_KEY": "synthetic-test-key",
    }
    env.pop("HERE_PROVIDER_SMOKE_KEY", None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CHILD_EXPORT,
            str(Path(__file__).resolve().parents[1]),
            str(memory.data_dir),
            str(sessions),
            str(destination),
            str(ready),
            MEETING_ID,
            cutover,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if process.poll() is not None:
            _, stderr = process.communicate(timeout=5)
            pytest.fail(f"Export child exited before {cutover}: {stderr.decode(errors='replace')}")
        assert ready.is_file(), f"Export child never reached {cutover}"
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def _kill(process: subprocess.Popen[bytes]) -> None:
    assert process.poll() is None
    process.kill()
    assert process.wait(timeout=5) != 0


def _registration(memory: MemoryService) -> Path:
    (registration,) = (memory.data_dir / ".exports").glob("*.json")
    return registration


def _temporary(destination: Path) -> Path:
    (sibling,) = destination.parent.glob(".here-export-*.partial")
    return sibling


@pytest.mark.parametrize("cutover", ["before-rename", "after-rename"])
def test_reconcile_after_real_export_kill_preserves_only_completed_package(
    catalog: tuple[MemoryService, Path, Path], cutover: str
) -> None:
    memory, sessions, destination = catalog
    with _paused_export(memory, sessions, destination, cutover) as process:
        registration = _registration(memory)
        if cutover == "before-rename":
            sibling = _temporary(destination)
            assert not destination.exists()
        else:
            sibling = None
            assert destination in destination.parent.iterdir()
        _kill(process)

    unrelated = destination.with_name("keep.txt")
    unrelated.write_bytes(b"unrelated export neighbor")
    archive = destination.read_bytes() if sibling is None else None
    fresh = MemoryService(memory.data_dir, sessions_root=sessions)
    fresh.reconcile()
    assert not registration.exists()
    if sibling is not None:
        assert not sibling.exists()
        assert not destination.exists()
    else:
        assert destination.read_bytes() == archive
        with ZipFile(destination) as package:
            assert json.loads(package.read("manifest.json"))["meeting_ids"] == [MEETING_ID]
    assert unrelated.read_bytes() == b"unrelated export neighbor"
    assert fresh.get_meeting(MEETING_ID).meeting.id == MEETING_ID


@pytest.mark.parametrize("replaced", ["sibling", "parent"])
def test_unproven_export_sibling_is_retained_and_blocks_pending_deletion(
    catalog: tuple[MemoryService, Path, Path], replaced: str
) -> None:
    memory, sessions, destination = catalog
    with _paused_export(memory, sessions, destination) as process:
        registration = _registration(memory)
        sibling = _temporary(destination)
        _kill(process)

    if replaced == "sibling":
        original = sibling.rename(sibling.with_name("moved-original.partial"))
    else:
        original_parent = sibling.parent.rename(sibling.parent.with_name("moved-original-parent"))
        original = original_parent / sibling.name
        sibling.parent.mkdir()
    before = original.read_bytes()
    foreign = b"unrelated bytes occupying the old export sibling name"
    sibling.write_bytes(foreign)

    fresh = MemoryService(memory.data_dir, sessions_root=sessions)
    report = fresh.reconcile()
    diagnostics = " ".join(map(str, report.warnings))
    assert "export" in diagnostics.lower()
    assert str(sibling) not in diagnostics
    assert sibling.name not in diagnostics
    assert registration.is_file()
    assert sibling.read_bytes() == foreign
    assert original.read_bytes() == before
    assert not destination.exists()

    result = fresh.delete_meeting(MEETING_ID)
    assert result.status == "pending"
    assert fresh.show(MEETING_ID) is None
    assert (fresh.data_dir / "meetings" / MEETING_ID).is_dir()
    assert registration.is_file()
    assert sibling.read_bytes() == foreign
    assert original.read_bytes() == before


def test_live_export_blocks_delete_and_stale_export_is_cleaned_before_delete(
    catalog: tuple[MemoryService, Path, Path],
) -> None:
    memory, sessions, destination = catalog
    with _paused_export(memory, sessions, destination) as process:
        registration = _registration(memory)
        sibling = _temporary(destination)
        fresh = MemoryService(memory.data_dir, sessions_root=sessions)
        fresh.reconcile()
        assert registration.is_file()
        assert sibling in destination.parent.iterdir()
        with pytest.raises(PermissionError):
            sibling.rename(sibling.with_name("moved-while-live.partial"))
        with pytest.raises(PermissionError):
            sibling.parent.rename(sibling.parent.with_name("moved-while-live-parent"))
        with pytest.raises(Busy):
            fresh.delete_meeting(MEETING_ID)
        assert not (memory.data_dir / ".deletions" / f"{MEETING_ID}.json").exists()
        assert fresh.show(MEETING_ID) is not None
        _kill(process)

    assert fresh.delete_meeting(MEETING_ID).status == "completed"
    assert fresh.show(MEETING_ID) is None
    assert not registration.exists()
    assert not sibling.exists()
    assert not destination.exists()
    assert (sessions / "selected" / "audio.wav").is_file()


def test_unreadable_export_registration_keeps_deletion_pending(
    catalog: tuple[MemoryService, Path, Path],
) -> None:
    memory, sessions, destination = catalog
    with _paused_export(memory, sessions, destination) as process:
        registration = _registration(memory)
        sibling = _temporary(destination)
        _kill(process)

    sibling_bytes = sibling.read_bytes()
    malformed = b'{"schema_version":'
    registration.write_bytes(malformed)
    fresh = MemoryService(memory.data_dir, sessions_root=sessions)

    report = fresh.reconcile()
    assert any("export" in str(warning).lower() for warning in report.warnings)
    assert fresh.delete_meeting(MEETING_ID).status == "pending"
    assert fresh.show(MEETING_ID) is None
    assert (fresh.data_dir / "meetings" / MEETING_ID).is_dir()
    assert registration.read_bytes() == malformed
    assert sibling.read_bytes() == sibling_bytes
    assert not destination.exists()


def test_insufficient_export_space_refuses_before_opening_archive_stream(
    catalog: tuple[MemoryService, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from here.memory.portability import InsufficientSpace, PortabilityService
    from here.output.export_binding import WindowsExportSibling

    memory, _, destination = catalog
    asset = memory.get_meeting(MEETING_ID).audio_assets[0]

    def insufficient_space(_: WindowsExportSibling) -> int:
        return asset.size_bytes - 1

    def unexpected_stream(_: WindowsExportSibling) -> None:
        pytest.fail("Opened the archive output before checking destination space")

    monkeypatch.setattr(WindowsExportSibling, "free_space", insufficient_space)
    monkeypatch.setattr(WindowsExportSibling, "open_stream", unexpected_stream)

    with pytest.raises(InsufficientSpace):
        PortabilityService(memory).export(
            (MEETING_ID,), destination, True, lambda _: None, lambda: False
        )

    assert not list(destination.parent.iterdir())
    assert not list((memory.data_dir / ".exports").glob("*.json"))


def test_export_closes_sqlite_for_audio_streaming_and_keeps_rename_under_catalog_lock(
    catalog: tuple[MemoryService, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from here.memory.portability import PortabilityService
    from here.memory.service import _PortableExportView
    from here.output.export_binding import WindowsExportSibling

    memory, _, destination = catalog
    catalog_lock = memory.data_dir / ".memory.lock"
    libraries: list[LibraryConnection] = []
    streaming_connections: list[int] = []
    rename_connections: list[int] = []
    open_library = service_module.open_library
    open_audio = _PortableExportView.open_audio
    replace = WindowsExportSibling.replace_destination

    def tracked_library(data_dir: Path) -> LibraryConnection:
        library = open_library(data_dir)
        libraries.append(library)
        return library

    def assert_connections_closed() -> None:
        assert libraries
        for library in libraries:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                library.connection.execute("SELECT 1")

    @contextmanager
    def stream(view: _PortableExportView, asset: AudioAsset) -> Iterator[BinaryIO]:
        with open_audio(view, asset) as audio:
            streaming_connections.append(len(libraries))
            assert_connections_closed()
            with closing(FileLease(catalog_lock, timeout=0)):
                yield audio

    def rename(
        sibling: WindowsExportSibling, *, before_rename: Callable[[], None] | None = None
    ) -> None:
        rename_connections.append(len(libraries))
        assert_connections_closed()
        with pytest.raises(LeaseBusy):
            FileLease(catalog_lock, timeout=0).close()
        replace(sibling, before_rename=before_rename)

    monkeypatch.setattr(service_module, "open_library", tracked_library)
    monkeypatch.setattr(_PortableExportView, "open_audio", stream)
    monkeypatch.setattr(WindowsExportSibling, "replace_destination", rename)

    PortabilityService(memory).export(
        (MEETING_ID,), destination, True, lambda _: None, lambda: False
    )

    assert len(streaming_connections) == len(rename_connections) == 1
    assert rename_connections[0] > streaming_connections[0] > 0
    assert destination.is_file()


def test_export_refuses_when_resolved_suppression_alias_changes_after_snapshot(
    catalog: tuple[MemoryService, Path, Path],
) -> None:
    from here.memory.portability import PortabilityService

    memory, _, destination = catalog
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
        origin_path, origin_fingerprint = connection.execute(
            "SELECT origin_path, origin_fingerprint FROM ingest_sources WHERE meeting_id=?",
            (MEETING_ID,),
        ).fetchone()
    registry = memory.data_dir / ".deletions"
    registry.mkdir()
    record = registry / f"{OTHER_MEETING_ID}.json"
    write_json(
        record,
        {
            "schema_version": 1,
            "meeting_id": OTHER_MEETING_ID,
            "phase": "pending",
            "origins": [{"origin_path": origin_path, "origin_fingerprint": "f" * 64}],
        },
    )
    initial_record = record.read_bytes()
    assert memory.show(MEETING_ID) is not None
    changed = False

    def resolve_alias(written: int) -> None:
        nonlocal changed
        if written <= 0 or changed:
            return
        # Inject only the exact origin evidence. The tombstone and meeting row stay unchanged.
        with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
            with connection:
                connection.execute(
                    "UPDATE ingest_sources SET origin_fingerprint=? WHERE meeting_id=?",
                    ("f" * 64, MEETING_ID),
                )
        changed = True

    assert origin_fingerprint != "f" * 64
    with pytest.raises(Conflict):
        PortabilityService(memory).export(
            (MEETING_ID,), destination, True, resolve_alias, lambda: False
        )

    assert changed
    assert record.read_bytes() == initial_record
    assert memory.show(MEETING_ID) is None
    assert not list(destination.parent.iterdir())


def test_export_refuses_to_replace_an_existing_archive(
    catalog: tuple[MemoryService, Path, Path],
) -> None:
    from here.memory.portability import PortabilityService

    memory, _, destination = catalog
    with ZipFile(destination, "x") as archive:
        archive.writestr("existing.txt", b"synthetic archive from another operation")
    before = destination.read_bytes()

    with pytest.raises(LibraryError):
        PortabilityService(memory).export(
            (MEETING_ID,), destination, True, lambda _: None, lambda: False
        )

    assert destination.read_bytes() == before
    assert set(destination.parent.iterdir()) == {destination}
    assert not list((memory.data_dir / ".exports").glob("*.json"))


def test_atomic_export_rename_preserves_a_destination_created_at_its_boundary(
    catalog: tuple[MemoryService, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from here.memory.portability import PortabilityService
    from here.output.export_binding import _Windows

    memory, _, destination = catalog
    foreign = b"synthetic bytes created by a competing export"
    at_native_rename = False
    competed = False
    rename = _Windows.rename

    def enter_native_rename(
        windows: _Windows,
        handle: int,
        name: str,
        before_rename: Callable[[], None] | None,
    ) -> None:
        nonlocal at_native_rename
        at_native_rename = True
        try:
            rename(windows, handle, name, before_rename)
        finally:
            at_native_rename = False

    def check_cancellation() -> bool:
        nonlocal competed
        if at_native_rename:
            assert _temporary(destination) in destination.parent.iterdir()
            with destination.open("xb") as stream:
                stream.write(foreign)
            competed = True
        return False

    monkeypatch.setattr(_Windows, "rename", enter_native_rename)

    with pytest.raises(LibraryError):
        PortabilityService(memory).export(
            (MEETING_ID,), destination, True, lambda _: None, check_cancellation
        )

    assert competed
    assert destination.read_bytes() == foreign
    assert set(destination.parent.iterdir()) == {destination}
    assert not list((memory.data_dir / ".exports").glob("*.json"))
