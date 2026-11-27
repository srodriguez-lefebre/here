"""Hostile ZIP inspection uses synthetic central directories and never extracts them."""

from __future__ import annotations

import hashlib
import stat
import struct
import zlib
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
    ZIP_TIMESTAMP,
    ArchiveEntry,
    manifest_bytes,
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


def test_inspect_accepts_a_small_commented_zip_with_zip64_local_header(
    service: portability.PortabilityService, tmp_path: Path
) -> None:
    source = tmp_path / "zip64.here.zip"
    with ZipFile(source, "x") as archive:
        archive.comment = b"Synthetic portable package"
        with archive.open("manifest.json", "w", force_zip64=True) as manifest:
            manifest.write(manifest_bytes(_manifest()))
        for entry in _BASE:
            archive.writestr(entry.name, entry.data)

    assert service.inspect(source).meeting_ids == (MEETING_ID,)


def test_inspect_caps_compressed_manifest_input_before_opening_it(
    service: portability.PortabilityService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "oversized-manifest.here.zip"
    document = manifest_bytes(_manifest())
    compressed_bytes = 10 * 1024**2
    # A valid DEFLATE stream can consume input without producing any output.
    # Begin with non-final empty stored blocks and finish with the small manifest.
    empty_block = b"\x00\x00\x00\xff\xff"
    final_block = b"\x01" + struct.pack("<HH", len(document), 0xFFFF - len(document))
    raw = empty_block * (compressed_bytes // len(empty_block)) + final_block + document
    with ZipFile(source, "x") as archive:
        manifest = ZipInfo("manifest.json", ZIP_TIMESTAMP)
        manifest.compress_type = ZIP_DEFLATED
        manifest.file_size = len(document)
        manifest.compress_size = len(raw)
        manifest.CRC = zlib.crc32(document)
        manifest.header_offset = 0
        archive.fp.write(manifest.FileHeader())
        archive.fp.write(raw)
        archive.start_dir = archive.fp.tell()
        archive.filelist.append(manifest)
        for entry in _BASE:
            archive.writestr(entry.name, entry.data)

    with ZipFile(source) as archive:
        info = archive.getinfo("manifest.json")
        next_offset = archive.getinfo(_MEETING).header_offset
        assert info.header_offset + 30 + len(info.orig_filename) + info.compress_size == next_offset
        assert info.compress_type == ZIP_DEFLATED
        assert info.compress_size > compressed_bytes
        assert info.file_size == len(document)

    def forbid_member(*args, **kwargs):
        pytest.fail("Opened a manifest exceeding the compressed-byte budget")

    monkeypatch.setattr(ZipFile, "open", forbid_member)
    _assert_refused(service, source, tmp_path)


def _archive_with_meeting_extra(source: Path, extra: bytes) -> None:
    with ZipFile(source, "x") as archive:
        archive.writestr("manifest.json", manifest_bytes(_manifest()))
        for entry in _BASE:
            info = ZipInfo(entry.name, ZIP_TIMESTAMP)
            if entry.name == _MEETING:
                info.extra = extra
            archive.writestr(info, entry.data)


def test_inspect_rejects_excess_extra_fields_before_constructing_zipfile(
    service: portability.PortabilityService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "extra-field-count.here.zip"
    _archive_with_meeting_extra(source, struct.pack("<HH", 0xAFFF, 0) * 65)

    def forbid_zipfile(*args, **kwargs):
        pytest.fail("Constructed ZipFile before bounding central extra-field count")

    monkeypatch.setattr(ZipFile, "__init__", forbid_zipfile)
    _assert_refused(service, source, tmp_path)


def test_inspect_rejects_nul_hidden_in_unicode_path_extra(
    service: portability.PortabilityService, tmp_path: Path
) -> None:
    source = tmp_path / "unicode-path-alias.here.zip"
    raw_name = _MEETING.encode()
    unicode_path = b"\x01" + struct.pack("<I", zlib.crc32(raw_name)) + raw_name + b"\x00alias"
    _archive_with_meeting_extra(
        source, struct.pack("<HH", 0x7075, len(unicode_path)) + unicode_path
    )

    _assert_refused(service, source, tmp_path)


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
