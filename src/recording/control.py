from __future__ import annotations

from typing import Protocol

from here.recording.models import RecordingSession


class ControllableRecording(Protocol):
    """A running capture that can be controlled without console input."""

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    def cancel(self) -> None: ...

    def wait(self, timeout: float | None = None) -> RecordingSession: ...
