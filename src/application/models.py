from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class ApplicationState(StrEnum):
    """Observable lifecycle states for one application-owned job."""

    IDLE = "idle"
    PREPARING = "preparing"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventKind(StrEnum):
    """Events consumed by interfaces without interpreting application logs."""

    STATE_CHANGED = "state_changed"
    AUDIO_LEVEL = "audio_level"
    ERROR_RECORDED = "error_recorded"
    SESSION_PERSISTED = "session_persisted"


class SourceMode(StrEnum):
    BOTH = "both"
    MICROPHONE = "microphone"
    SYSTEM_AUDIO = "system_audio"


@dataclass(frozen=True, slots=True)
class StartRequest:
    output_dir: Path
    source_mode: SourceMode = SourceMode.BOTH
    use_alt_transcription_model: bool = False
    microphone_device_id: int | None = None
    system_device_id: int | None = None


@dataclass(frozen=True, slots=True)
class AudioLevel:
    """Small, aggregated signal measurement; never contains raw audio."""

    source: str
    peak: float
    rms: float
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= self.peak <= 1.0:
            raise ValueError("peak must be between 0 and 1")
        if not 0.0 <= self.rms <= 1.0:
            raise ValueError("rms must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ApplicationError:
    stage: str
    error_type: str
    message: str
    retryable: bool
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class ApplicationSnapshot:
    state: ApplicationState = ApplicationState.IDLE
    session_id: str | None = None
    session_dir: Path | None = None
    started_at: datetime | None = None
    total_paused_seconds: float = 0.0
    recoverable: bool = False
    last_error: ApplicationError | None = None

    @property
    def has_active_work(self) -> bool:
        return self.state in {
            ApplicationState.PREPARING,
            ApplicationState.RECORDING,
            ApplicationState.PAUSED,
            ApplicationState.STOPPING,
            ApplicationState.PROCESSING,
        }


@dataclass(frozen=True, slots=True)
class ApplicationEvent:
    kind: EventKind
    state: ApplicationState
    previous_state: ApplicationState | None = None
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    audio_level: AudioLevel | None = None
    error: ApplicationError | None = None
    session_dir: Path | None = None
    details: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
