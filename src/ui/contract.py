"""Single adapter boundary between Qt and :mod:`here.application`.

Widgets import application vocabulary only from this module. Keeping command
translation here prevents the visual layer from depending on capture,
transcription, persistence, or concrete controller implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..application import (
    ApplicationController,
    ApplicationEvent,
    ApplicationSnapshot,
    ApplicationState,
    AudioLevel,
    EventKind,
    EventListener,
    SourceMode,
    StartRequest,
    Unsubscribe,
)


class VisualController(Protocol):
    """Command names used by the presentation layer."""

    @property
    def snapshot(self) -> ApplicationSnapshot: ...

    @property
    def output_dir(self) -> Path: ...

    @property
    def source_label(self) -> str: ...

    def subscribe(self, listener: EventListener) -> Unsubscribe: ...

    def start_recording(self) -> None: ...

    def stop_and_process(self) -> None: ...

    def pause_recording(self) -> None: ...

    def resume_recording(self) -> None: ...

    def cancel_recording(self) -> None: ...

    def cancel_processing(self) -> None: ...


class ApplicationUiAdapter:
    """Translate visual intent into the stable application-core contract."""

    def __init__(self, controller: ApplicationController, output_dir: Path) -> None:
        self._controller = controller
        self._output_dir = output_dir

    @property
    def snapshot(self) -> ApplicationSnapshot:
        return self._controller.snapshot

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def source_label(self) -> str:
        return "Micrófono + audio del sistema"

    def subscribe(self, listener: EventListener) -> Unsubscribe:
        return self._controller.subscribe(listener)

    def start_recording(self) -> None:
        self._controller.start(
            StartRequest(output_dir=self._output_dir, source_mode=SourceMode.BOTH)
        )

    def stop_and_process(self) -> None:
        self._controller.stop()

    def pause_recording(self) -> None:
        self._controller.pause()

    def resume_recording(self) -> None:
        self._controller.resume()

    def cancel_recording(self) -> None:
        self._controller.cancel()

    def cancel_processing(self) -> None:
        self._controller.cancel()


__all__ = [
    "ApplicationEvent",
    "ApplicationSnapshot",
    "ApplicationState",
    "ApplicationUiAdapter",
    "AudioLevel",
    "EventKind",
    "VisualController",
]
