"""Hold the native NTFS objects used to publish and recover an export sibling."""

from __future__ import annotations

import ctypes
import os
import re
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from ctypes import wintypes
from pathlib import Path
from typing import BinaryIO, TypeVar
from uuid import uuid4

from here.output.paths import UnsafeSessionPath, validate_session_directory

# Volume GUID path, FILE_ID_INFO serial and 128-bit ID, FILE_BASIC_INFO creation time.
type ExportIdentity = tuple[str, int, str, int]

_VOLUME_GUID = re.compile(r"\\\\\?\\volume\{[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\}\\")
_TEMPORARY = re.compile(r"\.here-export-[0-9a-f]{32}\.partial")
_FILE_ID = re.compile(r"[0-9a-f]{32}")
# MS-FSCC 2.1.10 reserves these values for unsupported or unidentified files.
_UNAVAILABLE_FILE_IDS = {"0" * 32, "f" * 32}
_DEVICE = re.compile(r"\\Device\\HarddiskVolume[0-9]+", re.IGNORECASE)
_DRIVE_ANCHOR = re.compile(r"[A-Za-z]:\\")
_UNC_DRIVE = re.compile(r"\\\\([^\\]+)\\([^\\]+)")
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{digit}" for prefix in ("COM", "LPT") for digit in "123456789¹²³"
}

_READ_ATTRIBUTES = 0x80
_SYNCHRONIZE = 0x00100000
_DIRECTORY_ACCESS = _SYNCHRONIZE | _READ_ATTRIBUTES | 0x20  # FILE_TRAVERSE
_DELETE_ACCESS = _SYNCHRONIZE | _READ_ATTRIBUTES | 0x00010000  # DELETE
_STREAM_ACCESS = _DELETE_ACCESS | 0x3  # FILE_READ_DATA | FILE_WRITE_DATA
_SHARE_READ_WRITE = 0x3
_SHARE_ALL = 0x7
_FILE_OPEN = 1
_FILE_CREATE = 2
_FILE_CREATED = 2  # IoStatusBlock.Information after exclusive creation.
_MISSING = {0xC0000034, 0xC000000F}  # OBJECT_NAME_NOT_FOUND, NO_SUCH_FILE
_COLLISION = 0xC0000035  # OBJECT_NAME_COLLISION
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_OPEN_DIRECTORY = 0x02200000  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
_SMB_PROTOCOL = 0x00020000
_REMOTE_OFFLINE = 0x2


class ExportBindingError(OSError):
    """The requested export operation has no verified Windows binding."""


class _NativeError(ExportBindingError):
    def __init__(self, operation: str, status: int) -> None:
        self.status = status & 0xFFFFFFFF
        super().__init__(f"{operation} failed (NTSTATUS 0x{self.status:08X})")


class _UnicodeString(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.USHORT),
        ("MaximumLength", wintypes.USHORT),
        ("Buffer", ctypes.c_void_p),
    ]


class _ObjectAttributes(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.ULONG),
        ("RootDirectory", wintypes.HANDLE),
        ("ObjectName", ctypes.POINTER(_UnicodeString)),
        ("Attributes", wintypes.ULONG),
        ("SecurityDescriptor", ctypes.c_void_p),
        ("SecurityQualityOfService", ctypes.c_void_p),
    ]


class _IoStatusBlock(ctypes.Structure):
    _fields_ = [("StatusOrPointer", ctypes.c_void_p), ("Information", ctypes.c_size_t)]


class _FileIdInfo(ctypes.Structure):
    _fields_ = [
        ("VolumeSerialNumber", ctypes.c_uint64),
        ("FileId", ctypes.c_ubyte * 16),
    ]


class _FileBasicInfo(ctypes.Structure):
    _fields_ = [
        ("CreationTime", ctypes.c_int64),
        ("LastAccessTime", ctypes.c_int64),
        ("LastWriteTime", ctypes.c_int64),
        ("ChangeTime", ctypes.c_int64),
        ("FileAttributes", wintypes.DWORD),
    ]


class _FileAttributeTagInfo(ctypes.Structure):
    _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]


class _FileRemoteProtocolInfo(ctypes.Structure):
    _fields_ = [
        ("StructureVersion", wintypes.USHORT),
        ("StructureSize", wintypes.USHORT),
        ("Protocol", wintypes.ULONG),
        ("ProtocolMajorVersion", wintypes.USHORT),
        ("ProtocolMinorVersion", wintypes.USHORT),
        ("ProtocolRevision", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("Flags", wintypes.ULONG),
        ("GenericReserved", wintypes.ULONG * 8),
        ("ProtocolSpecific", wintypes.ULONG * 16),
    ]


class _FileStandardInfo(ctypes.Structure):
    _fields_ = [
        ("AllocationSize", ctypes.c_int64),
        ("EndOfFile", ctypes.c_int64),
        ("NumberOfLinks", wintypes.DWORD),
        ("DeletePending", ctypes.c_ubyte),
        ("Directory", ctypes.c_ubyte),
    ]


class _FileRenameInformation(ctypes.Structure):
    _fields_ = [
        ("ReplaceIfExists", ctypes.c_ubyte),
        ("RootDirectory", wintypes.HANDLE),
        ("FileNameLength", wintypes.ULONG),
        ("FileName", ctypes.c_wchar * 1),
    ]


class _FileDispositionInfo(ctypes.Structure):
    _fields_ = [("DeleteFile", ctypes.c_ubyte)]


class _FileFsSizeInformation(ctypes.Structure):
    _fields_ = [
        ("TotalAllocationUnits", ctypes.c_int64),
        ("AvailableAllocationUnits", ctypes.c_int64),
        ("SectorsPerAllocationUnit", wintypes.ULONG),
        ("BytesPerSector", wintypes.ULONG),
    ]


_Info = TypeVar("_Info", bound=ctypes.Structure)


def _bind(
    library: ctypes.WinDLL,
    name: str,
    result: type[ctypes._SimpleCData],
    *arguments: type,
) -> ctypes._CFuncPtr:
    entry = getattr(library, name)
    entry.restype = result
    entry.argtypes = arguments
    return entry  # type: ignore[no-any-return]


def _win32_failure(operation: str) -> ExportBindingError:
    return ExportBindingError(f"{operation} failed (Windows error {ctypes.get_last_error()})")


def _canonical_volume(value: str) -> str:
    if _VOLUME_GUID.fullmatch(value.lower()) is None:
        raise ExportBindingError("The export volume GUID is unavailable")
    return value[:4] + "Volume" + value[10:].lower()


def validate_export_identity(value: object) -> ExportIdentity:
    """Validate an identity from the native query or its JSON representation."""
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ExportBindingError("Invalid export identity")
    volume, serial, file_id, created = value
    if (
        not isinstance(volume, str)
        or _VOLUME_GUID.fullmatch(volume.lower()) is None
        or _canonical_volume(volume) != volume
        or type(serial) is not int
        or not 0 < serial < 2**64
        or not isinstance(file_id, str)
        or _FILE_ID.fullmatch(file_id) is None
        or file_id in _UNAVAILABLE_FILE_IDS
        or type(created) is not int
        or not 0 < created < 2**63
    ):
        raise ExportBindingError("Invalid export identity")
    return volume, serial, file_id, created


def _name_bytes(component: str) -> bytes:
    stem = component.partition(".")[0].rstrip(" ").upper()
    if (
        not component
        or component.rstrip(" .") != component
        or stem in _RESERVED_NAMES
        or any(character in '\\/:<>"|?*' or ord(character) < 32 for character in component)
    ):
        raise ExportBindingError("Unsafe export path component")
    try:
        encoded = component.encode("utf-16-le")
    except UnicodeError as exc:
        raise ExportBindingError("Invalid Unicode in export path") from exc
    if len(encoded) > 510:  # NTFS counts up to 255 UTF-16 code units per component.
        raise ExportBindingError("Export path component is too long")
    return encoded


def _absolute_path(path: Path, *, require_absolute: bool) -> Path:
    if not path.is_absolute():
        if require_absolute or path.anchor:
            raise ExportBindingError("An absolute local drive path is required")
        path = Path.cwd() / path
    if _DRIVE_ANCHOR.fullmatch(path.anchor) is None or len(path.parts) < 2:
        raise ExportBindingError("An absolute local drive file path is required")
    for component in path.parts[1:]:
        _name_bytes(component)
    return path


def _configured_root_path(path: Path) -> str:
    if not path.is_absolute():
        if path.anchor:
            raise ExportBindingError("The configured library directory is ambiguous")
        path = Path.cwd() / path
    # Accept regular DOS and UNC path forms, including their extended-length spelling.
    name = str(path)
    if name.startswith("\\\\?\\"):
        name = name[4:]
        if name[:4].upper() == "UNC\\":
            name = "\\\\" + name[4:]
    path = Path(name)
    if _DRIVE_ANCHOR.fullmatch(path.anchor):
        native_name = "\\\\?\\" + name
    else:
        share = _UNC_DRIVE.fullmatch(path.drive)
        if share is None or not path.is_absolute():
            raise ExportBindingError("The configured library directory has no supported path")
        server, share_name = share.groups()
        _name_bytes(server)
        _name_bytes(share_name)
        if share_name.lower() in {"pipe", "mailslot", "ipc$"}:
            raise ExportBindingError("The configured library path must name a file share")
        native_name = "\\\\?\\UNC\\" + name[2:].rstrip("\\")
    for component in path.parts[1:]:
        _name_bytes(component)
    try:
        validate_session_directory(path)
    except (OSError, UnsafeSessionPath) as exc:
        raise ExportBindingError("The configured library directory is unsafe") from exc
    return native_name


class _Windows:
    def __init__(self) -> None:
        if os.name != "nt":
            raise ExportBindingError("Native Windows export is unavailable")
        try:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            native = ctypes.WinDLL("ntdll", use_last_error=True)
            self._bind_functions(kernel, native)
        except (OSError, AttributeError) as exc:
            raise ExportBindingError("Required Windows export APIs are unavailable") from exc

    def _bind_functions(self, kernel: ctypes.WinDLL, native: ctypes.WinDLL) -> None:
        self.drive_type = _bind(kernel, "GetDriveTypeW", wintypes.UINT, wintypes.LPCWSTR)
        self.volume_name = _bind(
            kernel,
            "GetVolumeNameForVolumeMountPointW",
            wintypes.BOOL,
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
        )
        self.dos_device = _bind(
            kernel,
            "QueryDosDeviceW",
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
        )
        self.volume_information = _bind(
            kernel,
            "GetVolumeInformationByHandleW",
            wintypes.BOOL,
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.LPDWORD,
            wintypes.LPDWORD,
            wintypes.LPDWORD,
            wintypes.LPWSTR,
            wintypes.DWORD,
        )
        self.final_path = _bind(
            kernel,
            "GetFinalPathNameByHandleW",
            wintypes.DWORD,
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        self.create_file = _bind(
            kernel,
            "CreateFileW",
            wintypes.HANDLE,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        self.close_handle = _bind(kernel, "CloseHandle", wintypes.BOOL, wintypes.HANDLE)
        self.query_file = _bind(
            kernel,
            "GetFileInformationByHandleEx",
            wintypes.BOOL,
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        self.set_file = _bind(
            kernel,
            "SetFileInformationByHandle",
            wintypes.BOOL,
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        self.process = _bind(kernel, "GetCurrentProcess", wintypes.HANDLE)
        self.duplicate = _bind(
            kernel,
            "DuplicateHandle",
            wintypes.BOOL,
            wintypes.HANDLE,
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        self.nt_create = _bind(
            native,
            "NtCreateFile",
            wintypes.LONG,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.ULONG,
            ctypes.POINTER(_ObjectAttributes),
            ctypes.POINTER(_IoStatusBlock),
            ctypes.c_void_p,
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
        )
        self.nt_set = _bind(
            native,
            "NtSetInformationFile",
            wintypes.LONG,
            wintypes.HANDLE,
            ctypes.POINTER(_IoStatusBlock),
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_int,
        )
        self.nt_volume_size = _bind(
            native,
            "NtQueryVolumeInformationFile",
            wintypes.LONG,
            wintypes.HANDLE,
            ctypes.POINTER(_IoStatusBlock),
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_int,
        )

    def volume_for(self, path: Path) -> str:
        # Reject drive substitutions before resolving the GUID or opening any directory.
        device = ctypes.create_unicode_buffer(1024)
        if not self.dos_device(path.drive, device, len(device)):
            raise _win32_failure("QueryDosDeviceW")
        if _DEVICE.fullmatch(device.value) is None:
            raise ExportBindingError("The export drive is redirected")
        if self.drive_type(path.anchor) not in {2, 3}:  # Local removable or fixed disk.
            raise ExportBindingError("Export requires a local NTFS volume")
        name = ctypes.create_unicode_buffer(50)
        if not self.volume_name(path.anchor, name, len(name)):
            raise _win32_failure("GetVolumeNameForVolumeMountPointW")
        volume = _canonical_volume(name.value)
        volume_device = ctypes.create_unicode_buffer(1024)
        if not self.dos_device(volume[4:-1], volume_device, len(volume_device)):
            raise _win32_failure("QueryDosDeviceW")
        if volume_device.value.casefold() != device.value.casefold():
            raise ExportBindingError("The export drive mapping changed")
        return volume

    @contextmanager
    def _hold_directory_path(self, path: str, access: int) -> Iterator[int]:
        handle = self.create_file(path, access, _SHARE_READ_WRITE, None, 3, _OPEN_DIRECTORY, None)
        if handle is None or handle == _INVALID_HANDLE:
            raise _win32_failure("CreateFileW")
        directory = int(handle)
        try:
            self._verify_type(directory, directory=True)
            yield directory
        finally:
            self.close(directory)

    @contextmanager
    def volume_root(self, volume: str) -> Iterator[int]:
        # The trailing slash opens the root directory of the named volume,
        # as opposed to a direct-access volume handle.
        with self._hold_directory_path(volume, _DIRECTORY_ACCESS) as handle:
            file_system = ctypes.create_unicode_buffer(32)
            if not self.volume_information(
                handle, None, 0, None, None, None, file_system, len(file_system)
            ):
                raise _win32_failure("GetVolumeInformationByHandleW")
            if file_system.value != "NTFS":
                raise ExportBindingError("Export requires a local NTFS volume")
            self.verify_name(handle, volume)
            yield handle

    def close(self, handle: int) -> None:
        if not self.close_handle(handle):
            raise _win32_failure("CloseHandle")

    def open_relative(
        self,
        parent: int,
        name: str,
        *,
        access: int,
        share: int,
        directory: bool = False,
        create: bool = False,
    ) -> int:
        encoded = _name_bytes(name)
        storage = ctypes.create_string_buffer(encoded, len(encoded) + 2)
        string = _UnicodeString(len(encoded), len(storage), ctypes.addressof(storage))
        # OBJ_DONT_REPARSE | OBJ_CASE_INSENSITIVE; name is a single component.
        attributes = _ObjectAttributes(
            ctypes.sizeof(_ObjectAttributes), parent, ctypes.pointer(string), 0x1040, None, None
        )
        handle = wintypes.HANDLE()
        status_block = _IoStatusBlock()
        # OPEN_REPARSE_POINT | SYNCHRONOUS_IO_NONALERT. Directory opens verify
        # the held type before use; DIRECTORY_FILE does not list OPEN_REPARSE_POINT
        # as a compatible option.
        options = 0x00200020 | (0 if directory else 0x40)  # NON_DIRECTORY_FILE
        status = self.nt_create(
            ctypes.byref(handle),
            access,
            ctypes.byref(attributes),
            ctypes.byref(status_block),
            None,
            0,
            share,
            _FILE_CREATE if create else _FILE_OPEN,
            options,
            None,
            0,
        )
        if status != 0:
            raise _NativeError("NtCreateFile", status)
        if handle.value is None or handle.value == _INVALID_HANDLE:
            raise ExportBindingError("NtCreateFile did not return a usable handle")
        if create and status_block.Information != _FILE_CREATED:
            self.close(handle.value)
            raise ExportBindingError("NtCreateFile did not confirm exclusive creation")
        return handle.value

    @contextmanager
    def hold_directory(self, path: Path, volume: str) -> Iterator[tuple[int, ...]]:
        with ExitStack() as stack:
            parent = stack.enter_context(self.volume_root(volume))
            chain = [parent]
            for name in path.parts[1:]:
                parent = self.open_relative(
                    parent,
                    name,
                    access=_DIRECTORY_ACCESS,
                    share=_SHARE_READ_WRITE,
                    directory=True,
                )
                stack.callback(self.close, parent)
                self._verify_type(parent, directory=True)
                chain.append(parent)
            self.verify_name(parent, volume + "\\".join(path.parts[1:]))
            yield tuple(chain)

    @contextmanager
    def hold_export_parent(
        self, parent_path: Path, volume: str, excluded_root: Path | None
    ) -> Iterator[int]:
        with ExitStack() as stack:
            ancestors = stack.enter_context(self.hold_directory(parent_path, volume))
            if excluded_root is not None:
                source = stack.enter_context(
                    self._hold_directory_path(
                        _configured_root_path(excluded_root), _READ_ATTRIBUTES
                    )
                )
                excluded_key = self.source_key(source)
                if any(self.file_key(ancestor) == excluded_key for ancestor in ancestors):
                    raise ExportBindingError("The export destination is inside the library")
            yield ancestors[-1]

    def source_key(self, handle: int) -> tuple[int, str]:
        name = ctypes.create_unicode_buffer(32768)
        # VOLUME_NAME_NT | FILE_NAME_OPENED reports the device of the held object.
        # It avoids normalization queries for each component of an SMB path.
        length = self.final_path(handle, name, len(name), 0xA)
        device = "\\".join(name.value.split("\\", 3)[:3]) if 0 < length < len(name) else ""

        try:
            if _DEVICE.fullmatch(device):
                return self.file_key(handle)

            protocol = self._information(handle, 13, _FileRemoteProtocolInfo)
            if (
                protocol.StructureSize != ctypes.sizeof(protocol)
                or protocol.Protocol != _SMB_PROTOCOL
                or protocol.ProtocolMajorVersion != 3
                or protocol.Flags & _REMOTE_OFFLINE
            ):
                raise ExportBindingError("The configured library has no comparable remote identity")
            # SMB 3 FileIdInformation returns the underlying object's own ID and
            # volume serial. This also catches SMB loopback aliases of local paths.
            return self.file_key(handle)
        except ExportBindingError as exc:
            raise ExportBindingError(
                "The configured library directory identity is unavailable"
            ) from exc

    def verify_name(self, handle: int, expected: str) -> None:
        name = ctypes.create_unicode_buffer(len(expected.encode("utf-16-le")) // 2 + 1)
        # VOLUME_NAME_GUID | FILE_NAME_NORMALIZED; compare the opened long path.
        length = self.final_path(handle, name, len(name), 1)
        if not 0 < length < len(name):
            raise ExportBindingError("The export object has no matching canonical path")
        if name.value.lower() != expected.lower():
            raise ExportBindingError("The export object uses an aliased path")

    def _information(self, handle: int, info_class: int, info_type: type[_Info]) -> _Info:
        info = info_type()
        if not self.query_file(handle, info_class, ctypes.byref(info), ctypes.sizeof(info)):
            raise _win32_failure("GetFileInformationByHandleEx")
        return info

    def _verify_type(self, handle: int, *, directory: bool) -> None:
        tags = self._information(handle, 9, _FileAttributeTagInfo)
        standard = self._information(handle, 1, _FileStandardInfo)
        if (
            tags.FileAttributes & 0x440  # REPARSE_POINT | DEVICE
            or tags.ReparseTag
            or standard.DeletePending
            or bool(standard.Directory) != directory
            or (not directory and standard.NumberOfLinks != 1)
        ):
            raise ExportBindingError("The export object type or link count is unsafe")

    def identity(self, handle: int, volume: str, *, directory: bool = False) -> ExportIdentity:
        self._verify_type(handle, directory=directory)
        serial, file_id = self.file_key(handle)
        basic = self._information(handle, 0, _FileBasicInfo)
        return validate_export_identity((volume, serial, file_id, basic.CreationTime))

    def file_key(self, handle: int) -> tuple[int, str]:
        identity = self._information(handle, 18, _FileIdInfo)
        file_id = bytes(identity.FileId).hex()
        if not identity.VolumeSerialNumber or file_id in _UNAVAILABLE_FILE_IDS:
            raise ExportBindingError("The held file has no usable native identity")
        return identity.VolumeSerialNumber, file_id

    def free_space(self, parent: int) -> int:
        info = _FileFsSizeInformation()
        status_block = _IoStatusBlock()
        status = self.nt_volume_size(
            parent, ctypes.byref(status_block), ctypes.byref(info), ctypes.sizeof(info), 3
        )
        if status != 0:
            raise _NativeError("NtQueryVolumeInformationFile(FileFsSizeInformation)", status)
        if (
            not 0 <= info.AvailableAllocationUnits <= info.TotalAllocationUnits
            or not info.SectorsPerAllocationUnit
            or not info.BytesPerSector
        ):
            raise ExportBindingError("The export volume returned invalid space information")
        return info.AvailableAllocationUnits * info.SectorsPerAllocationUnit * info.BytesPerSector

    def verify_absent(self, parent: int, name: str, *, message: str) -> None:
        try:
            handle = self.open_relative(
                parent, name, access=_READ_ATTRIBUTES | _SYNCHRONIZE, share=_SHARE_ALL
            )
        except _NativeError as exc:
            if exc.status in _MISSING:
                return
            raise
        self.close(handle)
        raise ExportBindingError(message)

    def stream(self, handle: int) -> BinaryIO:
        import msvcrt

        process = self.process()
        duplicate = wintypes.HANDLE()
        if not self.duplicate(process, handle, process, ctypes.byref(duplicate), 0, False, 2):
            raise _win32_failure("DuplicateHandle")
        if duplicate.value is None:
            raise ExportBindingError("DuplicateHandle did not return a usable handle")
        try:
            fd = msvcrt.open_osfhandle(duplicate.value, os.O_BINARY | os.O_RDWR)
        except OSError:
            self.close(duplicate.value)
            raise
        try:
            return os.fdopen(fd, "w+b")
        except BaseException:
            os.close(fd)
            raise

    def rename(
        self, handle: int, destination: str, before_rename: Callable[[], None] | None
    ) -> None:
        name = _name_bytes(destination)
        buffer = ctypes.create_string_buffer(ctypes.sizeof(_FileRenameInformation) + len(name) + 2)
        info = _FileRenameInformation.from_buffer(buffer)
        info.ReplaceIfExists = False
        # Native FileRenameInformation with a simple name and NULL RootDirectory
        # changes only the basename in the source handle's own directory.
        info.RootDirectory = None
        info.FileNameLength = len(name)
        ctypes.memmove(
            ctypes.addressof(buffer) + _FileRenameInformation.FileName.offset, name, len(name)
        )
        status_block = _IoStatusBlock()
        if before_rename is not None:
            before_rename()
        status = self.nt_set(handle, ctypes.byref(status_block), buffer, ctypes.sizeof(buffer), 10)
        if status != 0:
            if (status & 0xFFFFFFFF) == _COLLISION:
                raise ExportBindingError("The export destination already exists")
            raise _NativeError("NtSetInformationFile(FileRenameInformation)", status)

    def dispose(self, handle: int) -> None:
        info = _FileDispositionInfo(True)
        if not self.set_file(handle, 4, ctypes.byref(info), ctypes.sizeof(info)):
            raise _win32_failure("SetFileInformationByHandle(FileDispositionInfo)")


class WindowsExportSibling:
    """An exclusive file and its ancestor chain, retained through a single export."""

    def __init__(
        self,
        windows: _Windows,
        destination: Path,
        path: Path,
        parent: int,
        file: int,
        parent_identity: ExportIdentity,
        file_identity: ExportIdentity,
    ) -> None:
        self._windows = windows
        self._destination = destination
        self._path = path
        self._parent = parent
        self._file: int | None = file
        self._parent_identity = parent_identity
        self._file_identity = file_identity
        self._stream_started = False
        self._stream_open = False
        self._stream_finished = False
        self._published = False
        self._removed = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def parent_identity(self) -> ExportIdentity:
        return self._parent_identity

    @property
    def file_identity(self) -> ExportIdentity:
        return self._file_identity

    def free_space(self) -> int:
        """Query caller-available bytes through the verified destination directory."""
        self._held_file()
        return self._windows.free_space(self._parent)

    def _held_file(self) -> int:
        if self._file is None or self._published or self._removed or self._stream_open:
            raise ExportBindingError("The export sibling is unavailable for this operation")
        volume = self._parent_identity[0]
        if self._windows.identity(self._parent, volume, directory=True) != self._parent_identity:
            raise ExportBindingError("The export parent identity changed")
        if self._windows.identity(self._file, volume) != self._file_identity:
            raise ExportBindingError("The export sibling identity changed")
        return self._file

    @contextmanager
    def open_stream(self) -> Iterator[BinaryIO]:
        if self._stream_started:
            raise ExportBindingError("The export stream can be opened only once")
        handle = self._held_file()
        stream = self._windows.stream(handle)
        self._stream_started = True
        self._stream_open = True
        try:
            with stream:
                yield stream
                stream.flush()
                os.fsync(stream.fileno())
            self._stream_finished = True
        finally:
            self._stream_open = False

    def replace_destination(self, *, before_rename: Callable[[], None] | None = None) -> None:
        if not self._stream_finished:
            raise ExportBindingError("The export stream must finish before publication")
        if self._windows.volume_for(self._destination) != self._parent_identity[0]:
            raise ExportBindingError("The export destination drive changed")
        self._windows.rename(self._held_file(), self._destination.name, before_rename)
        self._published = True

    def remove(self) -> None:
        """Dispose of the still-owned temporary using its exact held handle."""
        handle = self._held_file()
        self._windows.dispose(handle)
        self._removed = True
        self._close()
        self._windows.verify_absent(
            self._parent, self.path.name, message="The removed export sibling is still present"
        )

    def _close(self) -> None:
        if self._file is not None:
            handle, self._file = self._file, None
            self._windows.close(handle)


@contextmanager
def open_export_sibling(
    destination: Path, *, excluded_root: Path | None = None
) -> Iterator[WindowsExportSibling]:
    """Hold an exclusive empty sibling for the caller to register before streaming.

    A process kill before the durable registration can leave only the empty sibling.
    """
    windows = _Windows()
    destination = _absolute_path(destination, require_absolute=False)
    if _TEMPORARY.fullmatch(destination.name.casefold()):
        raise ExportBindingError("The destination name is reserved for export temporaries")
    volume = windows.volume_for(destination)
    with windows.hold_export_parent(destination.parent, volume, excluded_root) as parent:
        parent_identity = windows.identity(parent, volume, directory=True)
        windows.verify_absent(
            parent, destination.name, message="The export destination already exists"
        )
        for _ in range(4):
            path = destination.with_name(f".here-export-{uuid4().hex}.partial")
            try:
                file = windows.open_relative(
                    parent, path.name, access=_STREAM_ACCESS, share=0, create=True
                )
            except _NativeError as exc:
                if exc.status != _COLLISION:
                    raise
            else:
                break
        else:
            raise ExportBindingError("Unable to allocate a unique export sibling")

        try:
            file_identity = windows.identity(file, volume)
            if file_identity[:2] != parent_identity[:2]:
                raise ExportBindingError("The export sibling is not on its parent's volume")
        except BaseException:
            # FILE_CREATE succeeded and this is the original exclusive handle.
            # The sibling has not been handed to a caller or received any payload.
            try:
                windows.dispose(file)
            finally:
                windows.close(file)
            raise

        sibling = WindowsExportSibling(
            windows, destination, path, parent, file, parent_identity, file_identity
        )
        try:
            yield sibling
        finally:
            sibling._close()


def recover_export_sibling(
    path: Path, parent_identity: object, file_identity: object, *, excluded_root: Path | None = None
) -> None:
    """Remove a registered temporary or verify its absence in the registered parent."""
    windows = _Windows()
    path = _absolute_path(path, require_absolute=True)
    if _TEMPORARY.fullmatch(path.name) is None:
        raise ExportBindingError("Invalid registered export sibling name")
    expected_parent = validate_export_identity(parent_identity)
    expected_file = validate_export_identity(file_identity)
    volume = windows.volume_for(path)
    if expected_parent[0] != volume or expected_parent[:2] != expected_file[:2]:
        raise ExportBindingError("The registered export volume differs from its binding")
    with windows.hold_export_parent(path.parent, volume, excluded_root) as parent:
        if windows.identity(parent, volume, directory=True) != expected_parent:
            raise ExportBindingError("The registered export parent identity changed")
        try:
            handle = windows.open_relative(parent, path.name, access=_DELETE_ACCESS, share=0)
        except _NativeError as exc:
            if exc.status in _MISSING:
                return
            raise
        try:
            if windows.identity(handle, volume) != expected_file:
                raise ExportBindingError("The registered export sibling identity changed")
            windows.dispose(handle)
        finally:
            windows.close(handle)
        windows.verify_absent(
            parent, path.name, message="The removed export sibling is still present"
        )
