"""Bounded OS file leases and an explicit application-job lifetime."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path

from here.output.paths import session_artifact_path


class LeaseBusy(RuntimeError):
    """An actual OS lease could not be acquired within its bounded interval."""


class FileLease:
    def __init__(self, path: Path, *, timeout: float = 0.25) -> None:
        self.path = session_artifact_path(path.parent, path.name)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.stream.seek(0)
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError as exc:
                if time.monotonic() >= deadline:
                    self.stream.close()
                    raise LeaseBusy("Local job is busy; retry after its owner completes") from exc
                time.sleep(0.01)

    def close(self) -> None:
        if self.stream.closed:
            return
        try:
            self.stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()


_job: ContextVar[ExitStack | None] = ContextVar("here_owned_job", default=None)


def retain_for_job(cleanup: Callable[[], None]) -> bool:
    """Enroll ownership before publication, including a constructor's error path."""
    scope = _job.get()
    if scope is None:
        return False
    scope.callback(cleanup)
    return True


@contextmanager
def owned_job() -> Iterator[None]:
    with ExitStack() as scope:
        token = _job.set(scope)
        try:
            yield
        finally:
            _job.reset(token)
