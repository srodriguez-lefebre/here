from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from here.recording.models import RecordingSession


@dataclass(frozen=True, slots=True)
class OpenedSource:
    label: str
    device_name: str
    device_index: int | None
    sample_rate: int
    channels: int


class ControllableRecording(Protocol):
    """A running capture that can be controlled without console input."""

    @property
    def opened_sources(self) -> tuple[OpenedSource, ...]: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    def cancel(self) -> None: ...

    def wait(self, timeout: float | None = None) -> RecordingSession: ...
