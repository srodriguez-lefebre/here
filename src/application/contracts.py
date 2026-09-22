from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from here.application.models import ApplicationEvent, ApplicationSnapshot, StartRequest

EventListener = Callable[[ApplicationEvent], None]
Unsubscribe = Callable[[], None]


class ApplicationController(Protocol):
    """Commands and observations shared by every interface."""

    @property
    def snapshot(self) -> ApplicationSnapshot: ...

    def subscribe(self, listener: EventListener) -> Unsubscribe: ...

    def start(self, request: StartRequest) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    def cancel(self) -> None: ...

    def retry(self, session_dir: Path | None = None) -> None: ...
