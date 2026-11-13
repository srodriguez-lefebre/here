"""Bounded central-directory inspection for the format-1 meeting package."""

from __future__ import annotations

import json
import stat
import unicodedata
import zlib
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path, PurePosixPath
from struct import Struct
from struct import error as StructError
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from here.memory.database import LibraryError, UnsupportedSchema
from here.output.paths import UnsafeSessionPath, validate_artifact_filename
from pydantic import ValidationError

if TYPE_CHECKING:
    from here.memory.portability import Manifest


_MAX_ENTRIES = 10_000
_MAX_UNCOMPRESSED_BYTES = 20 * 1024**3
_MAX_COMPRESSION_RATIO = 100
# Format-1 canonical paths and normal headers need only a fraction of these
# caps even at 10,000 entries. Bound unused ZIP metadata and JSON padding.
_MAX_CENTRAL_BYTES = 32 * 1024**2
_MAX_MANIFEST_BYTES = 8 * 1024**2
_MANIFEST = "manifest.json"
_GLOBAL_PAYLOADS = frozenset({"participants.json", "speaker_assignments.json"})

# PKWARE APPNOTE sections 4.3.12 and 4.3.14–16, little-endian.
_EOCD = Struct("<4s4H2IH")
_ZIP64_LOCATOR = Struct("<4sIQI")
_ZIP64_EOCD = Struct("<4sQ2H2I4Q")
_CENTRAL_HEADER = Struct("<4s6H3I5H2I")
_LOCAL_HEADER_BYTES = 30
_EXTRA_FIELD = Struct("<HH")
_MAX_ZIP_COMMENT = 0xFFFF
_EOCD_SEARCH_BYTES = _EOCD.size + _MAX_ZIP_COMMENT
_UINT16_MAX = 0xFFFF
_UINT32_MAX = 0xFFFFFFFF
_UNSUPPORTED_MEMBER_FLAGS = (1 << 0) | (1 << 5) | (1 << 6) | (1 << 13)
_NONREGULAR_DOS_ATTRIBUTES = 0x10 | 0x08 | stat.FILE_ATTRIBUTE_REPARSE_POINT


class InvalidPackage(LibraryError):
    """The archive is not a safe, consistent meeting package."""


@dataclass(frozen=True, slots=True)
class PackageSummary:
    manifest: Manifest
    entry_count: int
    uncompressed_bytes: int

    @property
    def meeting_ids(self) -> tuple[str, ...]:
        return self.manifest.meeting_ids


@dataclass(frozen=True, slots=True)
class _CentralDirectory:
    start: int
    size: int
    count: int


@dataclass(frozen=True, slots=True)
class _ZipLayout:
    central: _CentralDirectory
    end: int


class _BoundedZipReader:
    """Reject an unbounded ZipFile read even if a source changes after preflight."""

    def __init__(self, stream: BinaryIO, layout: _ZipLayout) -> None:
        self._stream = stream
        self._layout = layout
        self._opening = True
        self._central_read = False

    def finish_open(self) -> None:
        if not self._central_read:
            raise InvalidPackage("ZIP directory changed during inspection")
        self._opening = False

    def seek(self, offset: int, whence: int = 0) -> int:
        position = self._stream.seek(offset, whence)
        if self._opening and offset == 0 and whence == 2 and position != self._layout.end:
            raise InvalidPackage("ZIP length changed during inspection")
        return position

    def tell(self) -> int:
        return self._stream.tell()

    def read(self, size: int = -1) -> bytes:
        # ZipFile also reads small fixed EOCD/ZIP64 records and may search the
        # standard 64 KiB comment window. Permit the checked directory once.
        if not 0 <= size <= _MAX_CENTRAL_BYTES:
            # CPython 3.12 may use read() for the EOCD comment search.
            end = self._layout.end
            if self._opening and size == -1 and self.tell() == max(0, end - _EOCD_SEARCH_BYTES):
                return self._stream.read(min(end, _EOCD_SEARCH_BYTES))
            raise InvalidPackage("ZIP metadata read exceeds the inspection limit")
        if not self._opening:
            return self._stream.read(size)
        central = self._layout.central
        if size == central.size and self.tell() == central.start:
            self._central_read = True
            data = self._stream.read(size)
            _check_central_records(data, central.count)
            return data
        if size <= _EOCD_SEARCH_BYTES:
            return self._stream.read(size)
        raise InvalidPackage("ZIP directory changed during inspection")


def validate_entry_name(name: str) -> PurePosixPath:
    """Check literal POSIX components before any path normalization or I/O."""
    if not isinstance(name, str) or "\\" in name:
        raise InvalidPackage("Package member has an unsafe path")
    parts = name.split("/")
    try:
        for part in parts:
            validate_artifact_filename(part)
            # Windows device names remain reserved with whitespace before an extension.
            validate_artifact_filename(part.split(".", 1)[0].rstrip(" ") or part)
    except UnsafeSessionPath as error:
        raise InvalidPackage("Package member has an unsafe path") from error
    return PurePosixPath(*parts)


def _name_key(path: PurePosixPath) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", part.rstrip(" .").casefold()) for part in path.parts)


def _zip64_directory(stream: BinaryIO, locator_offset: int, zip64_offset: int) -> _CentralDirectory:
    if zip64_offset + _ZIP64_EOCD.size > locator_offset:
        raise InvalidPackage("ZIP64 directory offset is invalid")
    stream.seek(zip64_offset)
    signature, size, _, _, disk, directory_disk, disk_count, count, length, start = (
        _ZIP64_EOCD.unpack(stream.read(_ZIP64_EOCD.size))
    )
    if (
        signature != b"PK\x06\x06"
        or size + 12 < _ZIP64_EOCD.size
        or zip64_offset + size + 12 != locator_offset
        or disk != 0
        or directory_disk != 0
        or disk_count != count
        or start + length != zip64_offset
    ):
        raise InvalidPackage("ZIP64 central directory is inconsistent")
    return _CentralDirectory(start, length, count)


def _find_zip_layout(stream: BinaryIO) -> _ZipLayout:
    end = stream.seek(0, 2)
    tail_offset = max(0, end - _EOCD_SEARCH_BYTES)
    stream.seek(tail_offset)
    tail = stream.read(_EOCD_SEARCH_BYTES)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or len(tail) - position < _EOCD.size:
        raise InvalidPackage("ZIP end of central directory is missing")
    _, disk, directory_disk, disk_count, count, size, start, comment = _EOCD.unpack_from(
        tail, position
    )
    record_offset = tail_offset + position
    if record_offset + _EOCD.size + comment != end or disk or directory_disk or disk_count != count:
        raise InvalidPackage("ZIP central directory or disk layout is invalid")

    locator_offset = record_offset - _ZIP64_LOCATOR.size
    if locator_offset >= 0:
        stream.seek(locator_offset)
        signature, zip64_disk, zip64_offset, disks = _ZIP64_LOCATOR.unpack(
            stream.read(_ZIP64_LOCATOR.size)
        )
        if signature == b"PK\x06\x07":
            if zip64_disk != 0 or disks != 1:
                raise InvalidPackage("Multi-disk ZIP packages are unsupported")
            central = _zip64_directory(stream, locator_offset, zip64_offset)
            if (
                count not in (central.count, _UINT16_MAX)
                or size not in (central.size, _UINT32_MAX)
                or start not in (central.start, _UINT32_MAX)
            ):
                raise InvalidPackage("ZIP and ZIP64 directory records disagree")
            return _ZipLayout(central, end)

    if start + size != record_offset:
        raise InvalidPackage("ZIP central directory offset is invalid")
    return _ZipLayout(_CentralDirectory(start, size, count), end)


def _check_central_bounds(central: _CentralDirectory) -> None:
    if not 1 <= central.count <= _MAX_ENTRIES:
        raise InvalidPackage("Package has an invalid ZIP entry count")
    if not central.count * _CENTRAL_HEADER.size <= central.size <= _MAX_CENTRAL_BYTES:
        raise InvalidPackage("Package central directory exceeds the metadata limit or is truncated")


def _check_central_records(data: bytes, count: int) -> None:
    """Check the exact buffer given to ZipFile before it allocates per-member state."""
    position = 0
    for _ in range(count):
        if position + _CENTRAL_HEADER.size > len(data):
            raise InvalidPackage("ZIP central directory count does not match its records")
        header = _CENTRAL_HEADER.unpack_from(data, position)
        if header[0] != b"PK\x01\x02" or header[10] == 0 or header[13] != 0:
            raise InvalidPackage("Invalid ZIP central directory member header")
        position += _CENTRAL_HEADER.size + sum(header[10:13])
    if position != len(data):
        raise InvalidPackage("ZIP central directory count does not match its records")


def _check_member_extras(extra: bytes) -> None:
    position = 0
    while position < len(extra):
        tag, length = _EXTRA_FIELD.unpack_from(extra, position)
        position += _EXTRA_FIELD.size + length
        if position > len(extra):
            raise InvalidPackage("ZIP member extra field is truncated")
        # PKWARE UNIX extra stores a link target after its 12-byte metadata
        # prefix. ASi UNIX is a legacy alternate file-type/link representation.
        if (tag == 0x000D and length != 12) or tag == 0x756E:
            raise InvalidPackage("ZIP member has link or special-file metadata")


def _central_members(
    archive: ZipFile, central: _CentralDirectory
) -> tuple[dict[str, ZipInfo], int]:
    infos = archive.infolist()
    if len(infos) != central.count:
        raise InvalidPackage("ZIP central directory changed during inspection")
    members: dict[str, ZipInfo] = {}
    seen: set[tuple[str, ...]] = set()
    uncompressed_bytes = 0
    for info in infos:
        path = validate_entry_name(info.orig_filename)
        key = _name_key(path)
        if info.filename != info.orig_filename or key in seen:
            raise InvalidPackage("Package has duplicate or ambiguous ZIP member paths")
        seen.add(key)
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in (0, stat.S_IFREG) or info.external_attr & _NONREGULAR_DOS_ATTRIBUTES:
            raise InvalidPackage("Package members must be ordinary files")
        _check_member_extras(info.extra)
        if info.compress_type not in (ZIP_STORED, ZIP_DEFLATED) or (
            info.flag_bits & _UNSUPPORTED_MEMBER_FLAGS
        ):
            raise InvalidPackage("ZIP member uses an unsupported encoding")
        if info.compress_type == ZIP_STORED and info.file_size != info.compress_size:
            raise InvalidPackage("Stored ZIP member has inconsistent sizes")
        if info.file_size > info.compress_size * _MAX_COMPRESSION_RATIO:
            raise InvalidPackage("ZIP member exceeds the package compression ratio limit")
        uncompressed_bytes += info.file_size
        if uncompressed_bytes > _MAX_UNCOMPRESSED_BYTES:
            raise InvalidPackage("Package exceeds the total uncompressed size limit")
        members[info.orig_filename] = info
    following_offset = central.start
    for info in sorted(infos, key=lambda member: member.header_offset, reverse=True):
        # Without opening the payload, its compressed bytes and minimum local
        # header must fit before the next member. ZipFile checks the full
        # local header and exact boundary when each member is opened.
        if (
            info.header_offset < 0
            or info.header_offset + _LOCAL_HEADER_BYTES + info.compress_size > following_offset
        ):
            raise InvalidPackage("Package has invalid or overlapping ZIP member offsets")
        following_offset = info.header_offset
    return members, uncompressed_bytes


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidPackage("Package manifest contains duplicate JSON keys")
        result[key] = value
    return result


def _read_manifest(
    archive: ZipFile, info: ZipInfo, manifest_type: type[Manifest]
) -> tuple[Manifest, bytes]:
    if info.file_size > _MAX_MANIFEST_BYTES:
        raise InvalidPackage("Package manifest exceeds the metadata limit")
    with archive.open(info) as member:
        data = member.read(_MAX_MANIFEST_BYTES + 1)
    if len(data) != info.file_size:
        raise InvalidPackage("Package manifest size does not match the ZIP directory")
    document = json.loads(data, object_pairs_hook=_unique_json_object)
    if not isinstance(document, dict):
        raise InvalidPackage("Package manifest must be a JSON object")
    schema = document.get("format_version")
    if type(schema) is not int or schema < 1:
        raise InvalidPackage("Package manifest format version is invalid")
    if schema > 1:
        raise UnsupportedSchema("This package requires a newer portability format")
    if type(document.get("audio_included")) is not bool:
        raise InvalidPackage("Package manifest audio flag is invalid")
    files = document.get("files")
    if not isinstance(files, list) or len(files) >= _MAX_ENTRIES:
        raise InvalidPackage("Package manifest file inventory is invalid")
    if any(not isinstance(entry, dict) or type(entry.get("size")) is not int for entry in files):
        raise InvalidPackage("Package manifest file size must be an integer")
    if type(document.get("exported_at")) is not str:
        raise InvalidPackage("Package manifest export timestamp is invalid")
    return manifest_type.model_validate(document), data


def _is_uuid(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _check_manifest_inventory(
    manifest: Manifest, members: dict[str, ZipInfo], total: int
) -> PackageSummary:
    meetings = set(manifest.meeting_ids)
    answers = set(manifest.answer_ids)
    if (
        not meetings
        or not manifest.exporter_version.strip()
        or manifest.exported_at.utcoffset() != timedelta(0)
        or answers.intersection(manifest.omitted)
    ):
        raise InvalidPackage("Package manifest has invalid format-1 metadata")
    identity_lists = (
        manifest.meeting_ids,
        manifest.participant_ids,
        manifest.answer_ids,
        manifest.omitted,
    )
    for ids in identity_lists:
        if len(set(ids)) != len(ids) or any(not _is_uuid(identity) for identity in ids):
            raise InvalidPackage("Package manifest has an invalid or repeated identity")

    required = (
        _GLOBAL_PAYLOADS
        | {f"meetings/{identity}/meeting.json" for identity in meetings}
        | {f"answers/{identity}.json" for identity in answers}
    )
    seen: set[str] = set()
    revisions: dict[tuple[str, str], set[str]] = {}
    for entry in manifest.files:
        parts = validate_entry_name(entry.path).parts
        member = members.get(entry.path)
        if member is None or entry.path in seen or entry.size != member.file_size:
            raise InvalidPackage("Package manifest does not match the ZIP file inventory")
        seen.add(entry.path)
        if entry.path not in required:
            _check_meeting_role(parts, meetings, manifest.audio_included, revisions)
    if seen != members.keys() - {_MANIFEST} or not required <= seen:
        raise InvalidPackage("Package ZIP and manifest must list the same required payloads")
    if any(kinds != {"revision.json", "segments.json"} for kinds in revisions.values()):
        raise InvalidPackage("Package revision inventory is incomplete")
    return PackageSummary(manifest, len(members), total)


def _check_meeting_role(
    parts: tuple[str, ...],
    meeting_ids: set[str],
    audio_included: bool,
    revisions: dict[tuple[str, str], set[str]],
) -> None:
    match parts:
        case ("meetings", meeting_id, "revisions", number, kind) if (
            meeting_id in meeting_ids
            and number.isascii()
            and number.isdecimal()
            and not number.startswith("0")
            and len(number) <= 19
            and int(number) <= (1 << 63) - 1
            and kind in {"revision.json", "segments.json"}
        ):
            revisions.setdefault((meeting_id, number), set()).add(kind)
        case ("meetings", meeting_id, "audio", name) if (
            meeting_id in meeting_ids
            and audio_included
            and name.endswith(".wav")
            and _is_uuid(name[:-4])
        ):
            return
        case _:
            raise InvalidPackage("Package manifest contains an unknown payload role")


@contextmanager
def open_inspected_package(
    source: Path, manifest_type: type[Manifest]
) -> Iterator[tuple[ZipFile, PackageSummary, bytes]]:
    """Hold one source through preflight, manifest parsing and any later payload reads."""
    try:
        stream = source.open("rb")
    except OSError as error:
        raise InvalidPackage("Could not open the portable package") from error
    with stream, ExitStack() as scope:
        try:
            layout = _find_zip_layout(stream)
            central = layout.central
            _check_central_bounds(central)
            bounded = _BoundedZipReader(stream, layout)
            archive = scope.enter_context(ZipFile(bounded, mode="r"))
            bounded.finish_open()
            members, total = _central_members(archive, central)
            info = members.get(_MANIFEST)
            if info is None:
                raise InvalidPackage("Package manifest is missing")
            manifest, manifest_bytes = _read_manifest(archive, info, manifest_type)
            summary = _check_manifest_inventory(manifest, members, total)
        except (
            BadZipFile,
            EOFError,
            NotImplementedError,
            OSError,
            RecursionError,
            StructError,
            ValidationError,
            ValueError,
            zlib.error,
        ) as error:
            raise InvalidPackage("Could not validate ZIP metadata or manifest") from error
        yield archive, summary, manifest_bytes


def inspect_package(source: Path, manifest_type: type[Manifest]) -> PackageSummary:
    with open_inspected_package(source, manifest_type) as (_, summary, _):
        return summary
