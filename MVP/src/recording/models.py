from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RecordedAudioSource:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    label: str

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frames / self.sample_rate


@dataclass(slots=True)
class RecordingSession:
    sources: list[RecordedAudioSource]

    @property
    def duration_seconds(self) -> float:
        return max((source.duration_seconds for source in self.sources), default=0.0)

    def cleanup(self) -> None:
        for source in self.sources:
            source.path.unlink(missing_ok=True)
