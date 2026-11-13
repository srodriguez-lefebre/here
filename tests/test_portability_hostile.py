"""Hostile ZIP inspection uses synthetic central directories and never extracts them."""

from __future__ import annotations

import hashlib
import stat
import struct
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from here.memory import portability
from here.memory.database import LibraryError, UnsupportedSchema
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID
from portability_fixtures import (
    DUPLICATE_NAME_PAIRS,
    HOSTILE_NAMES,
    ArchiveEntry,
    write_archive,
    write_future_manifest,
)

_PREFIX = f"meetings/{MEETING_ID}"
_MEETING = f"{_PREFIX}/meeting.json"
_AUDIO = f"{_PREFIX}/audio/aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa.wav"
_BASE = (
    ArchiveEntry(_MEETING, b'{"id":"' + MEETING_ID.encode() + b'"}'),
    ArchiveEntry(f"{_PREFIX}/revisions/1/revision.json", b"{}"),
    ArchiveEntry(f"{_PREFIX}/revisions/1/segments.json", b"[]"),
    ArchiveEntry("participants.json", b'{"participants":[],"meeting_participants":[]}'),
    ArchiveEntry("speaker_assignments.json", b"[]"),
)


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> portability.PortabilityService:
    memory = MemoryService(tmp_path / "data")
    open_member = ZipFile.open

    def manifest_only(archive, name, *args, **kwargs):
        if isinstance(name, ZipInfo):
            name_to_check = name.orig_filename
        else:
            name_to_check = name
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode == "r":
            assert name_to_check == "manifest.json", "Inspection opened a package payload"
        return open_member(archive, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", manifest_only)
    return portability.PortabilityService(memory)


def _manifest(
    entries: tuple[ArchiveEntry, ...] = _BASE,
    *,
    audio: bool = False,
    sizes: dict[str, int] | None = None,
) -> dict:
    sizes = sizes or {}
    return {
        "format_version": 1,
        "exporter_version": "0.3.0",
        "exported_at": "2026-01-01T12:00:00Z",
        "meeting_ids": [MEETING_ID],
        "audio_included": audio,
        "participant_ids": [],
        "answer_ids": [],
        "omitted": [],
        "files": [
            {
                "path": entry.name,
                "size": sizes.get(entry.name, len(entry.data)),
                "sha256": hashlib.sha256(entry.data).hexdigest(),
            }
            for entry in entries
        ],
    }


def _assert_refused(service: portability.PortabilityService, source: Path, tmp_path: Path) -> None:
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    with pytest.raises(LibraryError) as failure:
        service.inspect(source)
    assert isinstance(failure.value, portability.InvalidPackage)
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*")} == before


def _declare_sizes(source: Path, members: dict[str, tuple[int, int]]) -> None:
    """Rewrite only synthetic central sizes, without allocating the declared payload."""
    data = bytearray(source.read_bytes())
    end = data.rfind(b"PK\x05\x06")
    position = struct.unpack_from("<I", data, end + 16)[0]
    changed = set()
    while data[position : position + 4] == b"PK\x01\x02":
        name_size, extra_size, comment_size = struct.unpack_from("<HHH", data, position + 28)
        name = bytes(data[position + 46 : position + 46 + name_size]).decode()
        if name in members:
            struct.pack_into("<II", data, position + 20, *members[name])
            changed.add(name)
        position += 46 + name_size + extra_size + comment_size
    assert changed == set(members)
    source.write_bytes(data)


def test_inspect_reads_only_manifest_and_does_not_stage_payloads(
    service: portability.PortabilityService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_archive(tmp_path / "valid.here.zip", _manifest(), _BASE)
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    mkdir = Path.mkdir
    staging = tmp_path / "data" / ".staging"

    def forbid_staging(path, *args, **kwargs):
        assert not path.is_relative_to(staging), "Inspection entered library staging"
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", forbid_staging)
    summary = service.inspect(source)
    assert summary.meeting_ids == (MEETING_ID,)
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*")} == before


def test_inspect_accepts_empty_entry_with_zero_compressed_size(
    service: portability.PortabilityService, tmp_path: Path
) -> None:
    entries = tuple(
        replace(entry, data=b"") if entry.name == _MEETING else entry for entry in _BASE
    )
    source = write_archive(tmp_path / "empty-entry.here.zip", _manifest(entries), entries)
    assert service.inspect(source).meeting_ids == (MEETING_ID,)


@pytest.mark.parametrize("name", HOSTILE_NAMES.values(), ids=HOSTILE_NAMES.keys())
def test_inspect_rejects_hostile_paths(
    service: portability.PortabilityService, tmp_path: Path, name: str
) -> None:
    entries = (*_BASE, ArchiveEntry(name))
    source = write_archive(tmp_path / "hostile-name.here.zip", _manifest(entries), entries)
    _assert_refused(service, source, tmp_path)


@pytest.mark.parametrize("names", DUPLICATE_NAME_PAIRS.values(), ids=DUPLICATE_NAME_PAIRS.keys())
def test_inspect_rejects_duplicate_normalized_names(
    service: portability.PortabilityService, tmp_path: Path, names: tuple[str, str]
) -> None:
    entries = (*_BASE, *(ArchiveEntry(name) for name in names))
    source = write_archive(tmp_path / "normalization.here.zip", _manifest(entries), entries)
    _assert_refused(service, source, tmp_path)


@pytest.mark.parametrize("kind", ["symlink", "directory", "trailing-dot", "trailing-space"])
def test_inspect_rejects_unsafe_entry_roles_and_suffixes(
    service: portability.PortabilityService, tmp_path: Path, kind: str
) -> None:
    match kind:
        case "symlink":
            item = replace(_BASE[0], unix_mode=stat.S_IFLNK | 0o777)
        case "directory":
            item = ArchiveEntry(f"{_PREFIX}/", b"", unix_mode=stat.S_IFDIR | 0o700)
        case "trailing-dot":
            item = replace(_BASE[0], name=_MEETING + ".")
        case _:
            item = replace(_BASE[0], name=_MEETING + " ")
    entries = (item, *_BASE[1:])
    source = write_archive(tmp_path / "unsafe-role.here.zip", _manifest(entries), entries)
    _assert_refused(service, source, tmp_path)


def test_inspect_reports_future_manifest_schema(
    service: portability.PortabilityService, tmp_path: Path
) -> None:
    source = write_future_manifest(tmp_path / "future.here.zip", _manifest(), entries=_BASE)
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    with pytest.raises(UnsupportedSchema):
        service.inspect(source)
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*")} == before


@pytest.mark.parametrize("kind", ["real-deflate", "declared-100000x", "zero-compressed"])
def test_inspect_rejects_excess_ratio_or_nonempty_zero_compressed_size(
    service: portability.PortabilityService, tmp_path: Path, kind: str
) -> None:
    destination = tmp_path / "ratio.here.zip"
    audio = ArchiveEntry(_AUDIO, b"Q" * 65536, compression=ZIP_DEFLATED)
    entries = (*_BASE, audio)
    source = write_archive(destination, _manifest(entries, audio=True), entries)
    if kind == "zero-compressed":
        _declare_sizes(source, {_AUDIO: (0, len(audio.data))})
    if kind == "declared-100000x":
        with ZipFile(source) as archive:
            compressed = archive.getinfo(_AUDIO).compress_size
        declared = 100000 * compressed
        write_archive(
            destination, _manifest(entries, audio=True, sizes={_AUDIO: declared}), entries
        )
        _declare_sizes(source, {_AUDIO: (compressed, declared)})
    _assert_refused(service, source, tmp_path)


def test_inspect_rejects_over_10000_central_entries(
    service: portability.PortabilityService, tmp_path: Path
) -> None:
    audio = tuple(
        ArchiveEntry(f"{_PREFIX}/audio/aaaaaaaa-aaaa-4aaa-8aaa-{index:012x}.wav", b"")
        for index in range(9995)
    )
    entries = (*_BASE, *audio)
    assert len(entries) + 1 == 10001
    source = write_archive(
        tmp_path / "entry-count.here.zip", _manifest(entries, audio=True), entries
    )
    _assert_refused(service, source, tmp_path)


def test_inspect_rejects_total_declared_bytes_over_20_gib_without_reading_payloads(
    service: portability.PortabilityService, tmp_path: Path
) -> None:
    audio = tuple(
        ArchiveEntry(
            f"{_PREFIX}/audio/bbbbbbbb-bbbb-4bbb-8bbb-{index:012x}.wav",
            b"",
            compression=ZIP_DEFLATED,
        )
        for index in range(6)
    )
    entries = (*_BASE, *audio)
    uncompressed = 7 * (1024**3) // 2
    compressed = 64 * 1024**2
    sizes = {entry.name: uncompressed for entry in audio}
    source = write_archive(
        tmp_path / "declared-size.here.zip", _manifest(entries, audio=True, sizes=sizes), entries
    )
    _declare_sizes(source, {name: (compressed, size) for name, size in sizes.items()})
    assert source.stat().st_size < 8192
    _assert_refused(service, source, tmp_path)


@pytest.mark.parametrize(
    "kind",
    ["missing", "malformed", "unlisted", "absent", "size", "duplicate-listing", "duplicate-zip"],
)
def test_inspect_rejects_invalid_manifest_or_inventory(
    service: portability.PortabilityService, tmp_path: Path, kind: str
) -> None:
    destination = tmp_path / "inventory.here.zip"
    entries = _BASE
    manifest = _manifest(entries)
    match kind:
        case "missing":
            with ZipFile(destination, "x") as archive:
                archive.writestr(_MEETING, _BASE[0].data)
        case "malformed":
            write_archive(destination, b'{"format_version":1,', entries)
        case "unlisted":
            write_archive(destination, _manifest(entries[1:]), entries)
        case "absent":
            write_archive(destination, manifest, entries[1:])
        case "size":
            write_archive(
                destination, _manifest(entries, sizes={_MEETING: len(entries[0].data) + 1}), entries
            )
        case "duplicate-listing":
            manifest["files"].append(manifest["files"][0])
            write_archive(destination, manifest, entries)
        case "duplicate-zip":
            with pytest.warns(UserWarning, match="Duplicate name"):
                write_archive(destination, manifest, (*entries, entries[0]))
    _assert_refused(service, destination, tmp_path)
