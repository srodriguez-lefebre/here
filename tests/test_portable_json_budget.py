"""The 32 MiB portable JSON budget counts uncompressed UTF-8 bytes."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from datetime import datetime, tzinfo
from pathlib import Path
from random import Random
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest
from here.memory import portability
from here.memory.database import LibraryError
from here.memory.repository import _PortableExportRows
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID, make_session
from portability_fixtures import ArchiveEntry, manifest_bytes, write_archive

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="uses native portable export")

_JSON_BUDGET = 32 * 1024**2
_FIRST = f"meetings/{MEETING_ID}/meeting.json"
_SECOND = "participants.json"
_HALF_BUDGET = _JSON_BUDGET // 2


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return cls(2026, 1, 1, tzinfo=tz)


def _export(service: portability.PortabilityService, destination: Path) -> Path:
    return service.export((MEETING_ID,), destination, True, lambda _: None, lambda: False).path


@pytest.fixture(scope="module")
def origin(tmp_path_factory: pytest.TempPathFactory) -> portability.PortabilityService:
    root = tmp_path_factory.mktemp("portable-json-origin")
    sessions = root / "sessions"
    session = make_session(
        sessions / "first",
        meeting_id=MEETING_ID,
        transcript_text="Synthetic café recording.",
    )
    memory = MemoryService(root / "data", sessions_root=sessions)
    assert memory.ingest_session(session.directory).status == "inserted"
    return portability.PortabilityService(memory)


@pytest.fixture
def target_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Keep the staged meeting and revision paths short on Windows.
    return tmp_path_factory.mktemp("pj")


@pytest.fixture
def service(target_root: Path) -> portability.PortabilityService:
    sessions = target_root / "sessions"
    sessions.mkdir()
    memory = MemoryService(target_root / "data", sessions_root=sessions)
    return portability.PortabilityService(memory)


@pytest.fixture(scope="module")
def packages(
    origin: portability.PortabilityService, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("portable-json-packages")
    seed = _export(origin, root / "seed.here.zip")
    with ZipFile(seed) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {
            name: archive.read(name) for name in archive.namelist() if name != "manifest.json"
        }

    # Periodic noise made of JSON whitespace keeps DEFLATE below the 100:1 limit
    # even with large padding. The decoded meeting and its relations stay intact.
    whitespace = bytes(b" \t\r\n"[value % 4] for value in range(256))
    noise = Random(0).randbytes(32 * 1024).translate(whitespace)
    block = noise.ljust(256 * 1024, b" ")
    for name in (_FIRST, _SECOND):
        target_size = _HALF_BUDGET - 4096
        blocks, tail = divmod(target_size - len(members[name]), len(block))
        members[name] += block * blocks + block[:tail]
        item = next(item for item in manifest["files"] if item["path"] == name)
        item.update(size=target_size, sha256=hashlib.sha256(members[name]).hexdigest())

    payload_json = sum(len(body) for name, body in members.items() if name.endswith(".json"))
    manifest_size = _JSON_BUDGET - payload_json
    document = manifest_bytes(manifest)
    assert len(document) <= manifest_size < 8 * 1024**2
    at_cap_manifest = document.ljust(manifest_size, b" ")
    entries = [
        ArchiveEntry(
            name, body, compression=ZIP_DEFLATED if name in (_FIRST, _SECOND) else ZIP_STORED
        )
        for name, body in members.items()
    ]
    at_cap = write_archive(root / "at-cap.here.zip", at_cap_manifest, entries)
    over_cap = write_archive(root / "over-cap.here.zip", at_cap_manifest + b" ", entries)
    return at_cap, over_cap


def test_inspect_accepts_exact_json_budget_with_audio(
    service: portability.PortabilityService, packages: tuple[Path, Path]
) -> None:
    at_cap, _ = packages
    with ZipFile(at_cap) as archive:
        json_entries = [item for item in archive.infolist() if item.filename.endswith(".json")]
        assert sum(item.file_size for item in json_entries) == _JSON_BUDGET
        assert sum(item.compress_size for item in json_entries) < _JSON_BUDGET
    summary = service.inspect(at_cap)
    assert summary.meeting_ids == (MEETING_ID,)
    assert summary.manifest.audio_included
    assert summary.uncompressed_bytes > _JSON_BUDGET


def test_inspect_rejects_one_byte_over_the_aggregate_json_budget(
    service: portability.PortabilityService, packages: tuple[Path, Path]
) -> None:
    _, over_cap = packages
    with pytest.raises(portability.InvalidPackage):
        service.inspect(over_cap)


def test_import_accepts_exact_json_budget_with_audio(
    service: portability.PortabilityService, packages: tuple[Path, Path]
) -> None:
    at_cap, _ = packages
    report = service.import_archive(at_cap, lambda _: None, lambda: False)
    assert [(item.meeting_id, item.status) for item in report.meetings] == [
        (MEETING_ID, "inserted"),
    ]


def test_import_rejects_over_budget_before_extracting_or_publishing(
    service: portability.PortabilityService, packages: tuple[Path, Path], target_root: Path
) -> None:
    _, over_cap = packages
    before = set(target_root.rglob("*"))
    progress: list[int] = []
    with pytest.raises(portability.InvalidPackage):
        service.import_archive(over_cap, progress.append, lambda: False)
    assert progress == []
    assert set(target_root.rglob("*")) == before


@pytest.fixture
def export_sized_json(
    origin: portability.PortabilityService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[int], None]:
    monkeypatch.setattr(portability, "datetime", _FixedDatetime)
    seed = _export(origin, tmp_path / "seed.here.zip")
    with ZipFile(seed) as archive:
        sizes = {item.filename: item.file_size for item in archive.infolist()}

    original = portability._json_payloads

    def set_budget(delta: int) -> None:
        # Both padded sizes have eight decimal digits in the manifest. Every other
        # serialized manifest field (including the timestamp and hashes) keeps its length.
        changed_digits = sum(8 - len(str(sizes[name])) for name in (_FIRST, _SECOND))
        fixed_bytes = changed_digits + sum(
            size
            for name, size in sizes.items()
            if name.endswith(".json") and name not in (_FIRST, _SECOND)
        )
        targets = {_FIRST: _HALF_BUDGET, _SECOND: _HALF_BUDGET - fixed_bytes + delta}
        assert all(len(str(size)) == 8 for size in targets.values())

        def padded_json(
            rows: _PortableExportRows, include_audio: bool
        ) -> Iterator[tuple[str, bytes]]:
            for name, payload in original(rows, include_audio):
                target = targets.get(name)
                if target is not None:
                    payload = payload.ljust(target, b" ")
                yield name, payload

        monkeypatch.setattr(portability, "_json_payloads", padded_json)

    return set_budget


def test_export_accepts_exact_json_budget_with_audio(
    origin: portability.PortabilityService, export_sized_json: Callable[[int], None], tmp_path: Path
) -> None:
    export_sized_json(0)
    path = _export(origin, tmp_path / "at-cap.here.zip")
    with ZipFile(path) as archive:
        assert (
            sum(item.file_size for item in archive.infolist() if item.filename.endswith(".json"))
            == _JSON_BUDGET
        )
    assert origin.inspect(path).uncompressed_bytes > _JSON_BUDGET


def test_export_refuses_one_byte_over_the_json_budget_without_publishing(
    origin: portability.PortabilityService, export_sized_json: Callable[[int], None], tmp_path: Path
) -> None:
    export_sized_json(1)
    before = set(tmp_path.iterdir())
    with pytest.raises(LibraryError):
        _export(origin, tmp_path / "over-cap.here.zip")
    assert set(tmp_path.iterdir()) == before
