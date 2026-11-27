"""Private adapter for the optional Windows SQLite FTS frame encoder."""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import sqlite3
import stat
import struct
import sys
import sysconfig
from contextlib import ExitStack, closing
from ctypes import wintypes
from importlib import import_module
from pathlib import Path
from typing import Any, BinaryIO

from here.memory.database import IntegrityFailure, LibraryConnection, LibraryError
from here.output.paths import session_artifact_path

_DLL_NAME = "_native_search_v2.dll"
_MANIFEST_NAME = "here.memory._native_search_manifest"
_ENTRYPOINT = "sqlite3_here_search_v2_init"
_BATCH_ROWS = 256
_MAX_PACKET_BYTES = 8 * 1024 * 1024
_HEADER = struct.Struct(">4sHq")
_LENGTH = struct.Struct(">I")
_UNSUPPORTED = "here search v2: FTS input unsupported"
_FTS_DDL = (
    "CREATE VIRTUAL TABLE segments_fts USING fts5("
    "meeting_id UNINDEXED, revision_id UNINDEXED, segment_id UNINDEXED, text, title, "
    "tokenize='unicode61 remove_diacritics 2')"
)
_FTS_RUNTIME = (
    "SELECT sqlite_source_id(), (SELECT encoding FROM pragma_encoding), "
    "(SELECT sql FROM main.sqlite_schema WHERE type='table' AND name='segments_fts') "
    "WHERE NOT EXISTS (SELECT 1 FROM temp.sqlite_schema "
    "WHERE type IN ('table','view') AND name='segments_fts' COLLATE NOCASE)"
)
_FTS_COLUMNS = "rowid,meeting_id,revision_id,segment_id,text,title"
_BATCH_SQL = f"SELECT here_search_v2_fts_batch({_FTS_COLUMNS}) FROM ("
_BATCH_SELECT = f"SELECT {_FTS_COLUMNS} FROM main.segments_fts"
_BATCH_END = f" ORDER BY rowid LIMIT {_BATCH_ROWS})"
_FIRST_BATCH = _BATCH_SQL + _BATCH_SELECT + _BATCH_END
_NEXT_BATCH = _BATCH_SQL + _BATCH_SELECT + " WHERE rowid > ?" + _BATCH_END
_Key = tuple[bytes, ...]


def _runtime_supported() -> bool:
    return (
        sys.platform == "win32"
        and sys.implementation.name == "cpython"
        and sys.version_info[:3] == (3, 13, 15)
        and sysconfig.get_platform() == "win-amd64"
        and sqlite3.sqlite_version_info == (3, 53, 1)
    )


def _windows_amd64_dll(file: BinaryIO) -> bool:
    dos = file.read(64)
    if len(dos) != 64 or dos[:2] != b"MZ":
        return False
    pe_offset = int.from_bytes(dos[60:64], "little")
    if pe_offset < len(dos):
        return False
    file.seek(pe_offset)
    pe = file.read(26)
    return (
        len(pe) == 26
        and pe[:6] == b"PE\x00\x00\x64\x86"
        and int.from_bytes(pe[22:24], "little") & 0x2000 != 0
        and pe[24:26] == b"\x0b\x02"
    )


def _open_dll_for_load(path: Path) -> BinaryIO:
    # Deny Windows write/delete sharing until SQLite has loaded the verified
    # file. Do not follow a leaf reparse point.
    import msvcrt

    kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True, winmode=0x00000800)
    create = kernel.CreateFileW
    create.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    generic_read, share_read, open_existing, open_reparse = 0x80000000, 1, 3, 0x200000
    handle = create(str(path), generic_read, share_read, None, open_existing, open_reparse, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    try:
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _packaged_encoder(files: ExitStack) -> tuple[Path, str] | None:
    package = Path(__file__).absolute().parent
    try:
        manifest = import_module(_MANIFEST_NAME)
        origin = getattr(manifest, "__file__", None)
        sha = getattr(manifest, "DLL_SHA256", None)
        source_id = getattr(manifest, "SQLITE_SOURCE_ID", None)
        if (
            not isinstance(origin, str)
            or Path(origin).absolute().parent != package
            or not isinstance(sha, str)
            or not re.fullmatch("[0-9a-f]{64}", sha)
            or not isinstance(source_id, str)
            or not source_id
        ):
            return None
        dll = session_artifact_path(package, _DLL_NAME)
        file = files.enter_context(_open_dll_for_load(dll))
        info = os.fstat(file.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink > 1
            or info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or not _windows_amd64_dll(file)
        ):
            return None
        file.seek(0)
        if hashlib.file_digest(file, "sha256").hexdigest() != sha:
            return None
        # Recheck the package path and the still-open leaf before registration.
        current = session_artifact_path(package, _DLL_NAME)
        if not info.st_ino or not os.path.samestat(info, current.stat()):
            return None
        return current, source_id
    except (ImportError, OSError, SyntaxError, ValueError):
        return None


def _load_encoder(connection: sqlite3.Connection, *, catalog: bool) -> bool:
    setting = getattr(sqlite3, "SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION", None)
    if (
        setting is None
        or not hasattr(connection, "load_extension")
        or connection.text_factory is not str
        or not _runtime_supported()
    ):
        return False
    with ExitStack() as files:
        artifact = _packaged_encoder(files)
        if artifact is None:
            return False
        dll, source_id = artifact
        sql = _FTS_RUNTIME if catalog else "SELECT sqlite_source_id()"
        expected = (source_id, "UTF-8", _FTS_DDL) if catalog else (source_id,)
        with closing(connection.execute(sql)) as cursor:
            cursor.row_factory = None
            if cursor.fetchone() != expected:
                return False

        # Enable the C loading API without enabling SQL load_extension().
        connection.setconfig(setting, True)
        try:
            connection.load_extension(str(dll), entrypoint=_ENTRYPOINT)
        except sqlite3.OperationalError:
            # Load refusal leaves the stock encoder available in this read view.
            return False
        finally:
            connection.setconfig(setting, False)
    return True


def load_packaged_encoder(connection: sqlite3.Connection) -> bool:
    """Load the installed encoder for a distribution smoke without a catalog."""
    return _load_encoder(connection, catalog=False)


def load_fts_encoder(library: LibraryConnection) -> bool:
    """Register the verified encoder on this read view's own connection."""
    connection = library.connection
    if not connection.in_transaction:
        raise IntegrityFailure("Native FTS encoding requires a catalog read view")
    return _load_encoder(connection, catalog=True)


def _invalid_packet() -> IntegrityFailure:
    return IntegrityFailure("Native FTS encoder returned an invalid batch")


def _decode_packet(packet: object) -> tuple[int, int, _Key, _Key, memoryview]:
    if type(packet) is not bytes or not _HEADER.size <= len(packet) <= _MAX_PACKET_BYTES:
        raise _invalid_packet()
    magic, count, last_rowid = _HEADER.unpack_from(packet)
    if magic != b"HV2F" or not 1 <= count <= _BATCH_ROWS:
        raise _invalid_packet()

    offset = _HEADER.size
    endpoints: list[_Key] = []
    for _ in range(2):
        cells: list[bytes] = []
        for _ in range(5):
            if offset + _LENGTH.size > len(packet):
                raise _invalid_packet()
            (size,) = _LENGTH.unpack_from(packet, offset)
            offset += _LENGTH.size
            end = offset + size
            if end > len(packet):
                raise _invalid_packet()
            cells.append(packet[offset:end])
            offset = end
        endpoints.append(tuple(cells))
    first, last = endpoints
    body = memoryview(packet)[offset:]
    if len(body) < 17 * count + 1 or body[0] != ord("[") or body[-1] != ord("]") or first > last:
        raise _invalid_packet()
    if count == 1 and first != last:
        raise _invalid_packet()
    return count, last_rowid, first, last, body


def _is_unsupported(error: sqlite3.Error | LibraryError) -> bool:
    original = error if isinstance(error, sqlite3.Error) else error.__cause__
    return (
        isinstance(original, sqlite3.Error)
        and getattr(original, "sqlite_errorcode", None) == sqlite3.SQLITE_MISMATCH
        and original.args == (_UNSUPPORTED,)
    )


def fts_digest(library: LibraryConnection, digest: Any) -> Any:
    """Append proven native frames to a trial digest, or request canonical replay."""
    trial = digest.copy()
    trial.update(b"fts\x00")
    last_rowid: int | None = None
    previous: _Key | None = None
    while True:
        try:
            sql, parameters = (
                (_FIRST_BATCH, ()) if last_rowid is None else (_NEXT_BATCH, (last_rowid,))
            )
            with closing(library.execute(sql, parameters)) as cursor:
                cursor.row_factory = None
                row = cursor.fetchone()
        except (sqlite3.Error, LibraryError) as error:
            if _is_unsupported(error):
                return None
            raise
        if row is None:
            raise _invalid_packet()
        if row[0] is None:
            break
        count, end, first, last, body = _decode_packet(row[0])
        if last_rowid is not None and end <= last_rowid:
            raise _invalid_packet()
        if previous is not None and first < previous:
            return None
        trial.update(len(body).to_bytes(8, "big"))
        trial.update(body)
        last_rowid, previous = end, last
        if count < _BATCH_ROWS or end == (1 << 63) - 1:
            break
    trial.update(b"\x00" * 8)
    return trial
