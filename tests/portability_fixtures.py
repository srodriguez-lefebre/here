"""Small synthetic ZIPs for future portability tests; never extract their entries.

The M5 specification does not fix the complete manifest JSON schema. Every builder
therefore takes a manifest supplied by its caller. These helpers author ZIP bytes,
not a claim that any manifest, meeting document or product validator is valid.
"""

from __future__ import annotations

import json
import stat
import struct
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CANARY_BYTES = b"synthetic portability canary\n"
ZIP_TIMESTAMP = (2026, 1, 1, 12, 0, 0)
HOSTILE_NAMES = {
    "posix-absolute": "/synthetic-canary.txt",
    "windows-rooted": "\\synthetic-canary.txt",
    "traversal": "../synthetic-canary.txt",
    "nested-traversal": "meetings/../../synthetic-canary.txt",
    "backslash-traversal": "meetings\\..\\..\\synthetic-canary.txt",
    "unc": "\\\\synthetic-server\\share\\canary.txt",
    "slash-unc": "//synthetic-server/share/canary.txt",
    "drive-absolute": "C:/synthetic-canary.txt",
    "drive-relative": "C:synthetic-canary.txt",
    "alternate-data-stream": "meetings/canary.txt:hidden",
    "device-con": "meetings/CON",
    "device-nul-extension": "meetings/nul.txt",
    "device-com1": "meetings/COM1.wav",
    "device-prn": "meetings/PrN.json",
}
DUPLICATE_NAME_PAIRS = {
    "nfc": ("meetings/caf\u00e9.txt", "meetings/cafe\u0301.txt"),
    "case": ("meetings/Canary.txt", "meetings/canary.TXT"),
    "trailing-dot": ("meetings/canary.txt", "meetings/canary.txt."),
    "trailing-space": ("meetings/canary.txt", "meetings/canary.txt "),
    "backslash": ("meetings/canary.txt", "meetings\\canary.txt"),
}


@dataclass(frozen=True)
class ArchiveEntry:
    name: str
    data: bytes = CANARY_BYTES
    compression: int = zipfile.ZIP_STORED
    unix_mode: int = stat.S_IFREG | 0o600


def manifest_bytes(manifest: Mapping[str, Any] | bytes) -> bytes:
    """Serialize only caller-supplied schema fields; bytes allow malformed JSON cases."""
    if isinstance(manifest, bytes):
        return manifest
    return (json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def write_archive(
    destination: Path,
    manifest: Mapping[str, Any] | bytes,
    entries: Iterable[ArchiveEntry] = (),
) -> Path:
    """Write entries verbatim, including dangerous names, solely inside a ZIP file.

    Entry names never become host filesystem paths. Nothing reads a real library,
    follows a link, launches code, or allocates the declared uncompressed size.
    """
    with zipfile.ZipFile(destination, "w") as archive:
        for entry in (ArchiveEntry("manifest.json", manifest_bytes(manifest)), *entries):
            info = zipfile.ZipInfo(entry.name, date_time=ZIP_TIMESTAMP)
            # ZipInfo normalizes backslashes on Windows. Preserve the authored
            # spelling in both headers so tests can inspect orig_filename too.
            info.filename = entry.name
            info.create_system = 3
            info.external_attr = entry.unix_mode << 16
            info.compress_type = entry.compression
            archive.writestr(info, entry.data, compresslevel=9)
    return destination


def write_hostile_names(root: Path, manifest: Mapping[str, Any] | bytes) -> dict[str, Path]:
    """Return one small archive per absolute/traversal/Windows naming attack."""
    return {
        case: write_archive(root / f"{case}.here.zip", manifest, [ArchiveEntry(name)])
        for case, name in HOSTILE_NAMES.items()
    }


def write_duplicate_names(root: Path, manifest: Mapping[str, Any] | bytes) -> dict[str, Path]:
    """Author distinct central-directory spellings that collide after normalization."""
    return {
        case: write_archive(
            root / f"duplicate-{case}.here.zip",
            manifest,
            [ArchiveEntry(first, b"first canary"), ArchiveEntry(second, b"second canary")],
        )
        for case, (first, second) in DUPLICATE_NAME_PAIRS.items()
    }


def write_symlink_archive(destination: Path, manifest: Mapping[str, Any] | bytes) -> Path:
    """Set the UNIX symlink type in ZIP attributes, without making a host symlink."""
    return write_archive(
        destination,
        manifest,
        [
            ArchiveEntry(
                "meetings/link.wav", b"../../synthetic-outside.wav", unix_mode=stat.S_IFLNK | 0o777
            )
        ],
    )


def write_future_manifest(
    destination: Path,
    manifest: Mapping[str, Any],
    *,
    supported_version: int = 1,
    entries: Iterable[ArchiveEntry] = (),
) -> Path:
    """Change only the specification's known format_version discriminator."""
    return write_archive(
        destination, {**manifest, "format_version": supported_version + 1}, entries
    )


def write_high_ratio_archive(destination: Path, manifest: Mapping[str, Any] | bytes) -> Path:
    """Real DEFLATE payload: only 64 KiB of synthetic data, with a ratio above 100."""
    return write_archive(
        destination,
        manifest,
        [ArchiveEntry("meetings/ratio.txt", b"Q" * 65536, compression=zipfile.ZIP_DEFLATED)],
    )


def write_declared_ratio_archive(
    destination: Path,
    manifest: Mapping[str, Any] | bytes,
    *,
    declared_ratio: int = 100000,
) -> Path:
    """Forge a large declared ratio for central-directory-only inspection tests.

    This is intentionally inconsistent payload metadata, not a real 100,000-fold
    DEFLATE expansion. Both ZIP headers claim the same enlarged size; the actual
    bytes remain the 64 KiB high-ratio fixture. Do not extract this archive.
    """
    if declared_ratio <= 100:
        raise ValueError("The hostile declared ratio must exceed the M5 limit")
    write_high_ratio_archive(destination, manifest)
    with zipfile.ZipFile(destination) as archive:
        info = archive.getinfo("meetings/ratio.txt")
        local_offset = info.header_offset
        declared_size = info.compress_size * declared_ratio
    if declared_size > 0xFFFFFFFF:
        raise ValueError("This small header fixture uses 32-bit ZIP sizes, not ZIP64")
    contents = bytearray(destination.read_bytes())
    end_offset = contents.rfind(b"PK\x05\x06")
    central_offset = struct.unpack_from("<I", contents, end_offset + 16)[0]
    position = central_offset
    while contents[position : position + 4] == b"PK\x01\x02":
        name_size, extra_size, comment_size = struct.unpack_from("<HHH", contents, position + 28)
        name = contents[position + 46 : position + 46 + name_size]
        if name == b"meetings/ratio.txt":
            struct.pack_into("<I", contents, position + 24, declared_size)
            struct.pack_into("<I", contents, local_offset + 22, declared_size)
            destination.write_bytes(contents)
            return destination
        position += 46 + name_size + extra_size + comment_size
    raise ValueError("Synthetic central-directory entry was not found")


def write_excess_entry_count(destination: Path, manifest: Mapping[str, Any] | bytes) -> Path:
    """10,001 total entries including manifest; empty payloads keep it near 1 MiB."""
    return write_archive(
        destination,
        manifest,
        (ArchiveEntry(f"meetings/entry-{index:05d}.txt", b"") for index in range(10000)),
    )
