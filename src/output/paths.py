"""Validate managed session entries and publish through owned temporary files.

These checks reject pre-existing redirects; they do not synchronize concurrent
filesystem attackers or concurrent session writers.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath


class UnsafeSessionPath(ValueError):
    """A managed session entry must not be accessed or recovered through."""


def _is_redirect(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def validate_session_directory(directory: Path) -> None:
    if ".." in directory.parts:
        raise UnsafeSessionPath(f"Unsafe session directory: {directory}")
    absolute = directory.absolute()
    for parent in (*reversed(absolute.parents), absolute):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        if _is_redirect(info) or not stat.S_ISDIR(info.st_mode):
            raise UnsafeSessionPath(f"Unsafe session directory: {parent}")


def session_artifact_path(directory: Path, filename: str) -> Path:
    validate_session_directory(directory)
    windows_path = PureWindowsPath(filename)
    reserved_stems = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in "123456789¹²³"
    }
    if (
        not filename
        or Path(filename).name != filename
        or windows_path.name != filename
        or windows_path.drive
        or windows_path.root
        or filename in {".", ".."}
        or filename.rstrip(" .") != filename
        or any(character in filename for character in ':<>"|?*')
        or any(ord(character) < 32 for character in filename)
        or filename.split(".", 1)[0].upper() in reserved_stems
    ):
        raise UnsafeSessionPath("Artifact must be a filename inside the session")
    path = directory / filename
    try:
        info = path.lstat()
    except FileNotFoundError:
        return path
    if _is_redirect(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
        raise UnsafeSessionPath(f"Unsafe session artifact: {path}")
    if not path.resolve().is_relative_to(directory.resolve()):
        raise UnsafeSessionPath("Artifact must resolve inside the session")
    return path


def reserve_artifact_path(destination: Path, suffix: str) -> Path:
    session_artifact_path(destination.parent, destination.name)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=suffix
    )
    os.close(descriptor)
    return Path(name)


@contextmanager
def staged_artifact_path(destination: Path) -> Iterator[Path]:
    staged = reserve_artifact_path(destination, ".stage")
    try:
        yield staged
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)


def write_artifact_text(destination: Path, content: str, *, encoding: str = "utf-8") -> None:
    with staged_artifact_path(destination) as staged:
        staged.write_text(content, encoding=encoding)
