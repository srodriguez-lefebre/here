"""Owned capture storage. Checkpoints protect against process death, not power loss."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
import soundfile as sf
from here.output.paths import (
    UnsafeSessionPath,
    session_artifact_path,
    validate_session_directory,
    write_artifact_text,
)
from here.recording.models import RecordedAudioSource, RecordingSession
from pydantic import BaseModel, Field


class JournalSource(BaseModel):
    audio_file: str
    label: str
    device_name: str | None = None
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    frames: int = Field(default=0, ge=0)


class JournalDocument(BaseModel):
    schema_version: Literal[1] = 1
    capture_id: str
    destination: str
    started_at: datetime
    updated_at: datetime
    state: Literal["capturing", "interrupted", "closed"] = "capturing"
    sources: list[JournalSource] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)


class CaptureJournal:
    def __init__(self, root: Path, document: JournalDocument) -> None:
        self.root = root.absolute()
        self.document = document
        self._lock = threading.RLock()
        self.validate()

    @property
    def directory(self) -> Path:
        return self.root / ".captures" / self.document.capture_id

    @property
    def destination(self) -> Path:
        self.validate()
        return self.root / self.document.destination

    def validate(self) -> None:
        from here.output.metadata import SessionEventMetadata

        if str(UUID(self.document.capture_id)) != self.document.capture_id:
            raise UnsafeSessionPath("Invalid capture identity")
        validate_session_directory(self.directory)
        # Validate a directory name using the filename rules before directory checks.
        name = self.document.destination
        expected_name = f"{self.document.started_at:%Y%m%d_%H%M%S}_{self.document.capture_id}"
        if name != expected_name:
            raise UnsafeSessionPath("Capture destination disagrees with reserved identity")
        session_artifact_path(self.root / ".captures", name)
        if name.startswith("."):
            raise UnsafeSessionPath("Invalid capture destination")
        validate_session_directory(self.root / name)
        session_artifact_path(self.directory, "journal.json")
        names = [item.audio_file for item in self.document.sources]
        if len(names) != len(set(names)):
            raise UnsafeSessionPath("Duplicate capture source")
        for name in names:
            session_artifact_path(self.directory, name)
        for event in self.document.events:
            SessionEventMetadata.model_validate(event)

    @classmethod
    def create(cls, root: Path) -> CaptureJournal:
        now = datetime.now().astimezone()
        identity = str(uuid4())
        journal = cls(
            root,
            JournalDocument(
                capture_id=identity,
                destination=f"{now:%Y%m%d_%H%M%S}_{identity}",
                started_at=now,
                updated_at=now,
            ),
        )
        journal.directory.mkdir(parents=True)
        journal._publish()
        return journal

    @classmethod
    def load(cls, root: Path, capture_id: str) -> CaptureJournal:
        if str(UUID(capture_id)) != capture_id:
            raise UnsafeSessionPath("Invalid capture identity")
        path = session_artifact_path(root / ".captures" / capture_id, "journal.json")
        document = JournalDocument.model_validate_json(path.read_text(encoding="utf-8"))
        if document.capture_id != capture_id:
            raise UnsafeSessionPath("Capture identity disagrees with directory")
        return cls(root, document)

    def _publish(self) -> None:
        self.validate()
        self.document.updated_at = datetime.now().astimezone()
        write_artifact_text(
            self.directory / "journal.json", self.document.model_dump_json(indent=2)
        )

    def open_writer(
        self, *, label: str, sample_rate: int, channels: int, device_name: str | None = None
    ) -> CaptureWriter:
        with self._lock:
            item = JournalSource(
                audio_file=f"source_{len(self.document.sources) + 1:02d}.wav",
                label=label,
                sample_rate=sample_rate,
                channels=channels,
                device_name=device_name,
            )
            self.document.sources.append(item)
            self._publish()
            return CaptureWriter(self, item)

    def event(self, kind: str, **details: object) -> None:
        with self._lock:
            self.document.events.append(
                {
                    "kind": kind,
                    "occurred_at": datetime.now().astimezone().isoformat(),
                    "recorded_duration_seconds": max(
                        (item.frames / item.sample_rate for item in self.document.sources),
                        default=0.0,
                    ),
                    "details": details,
                }
            )
            self._publish()

    def finish(self, error: BaseException | None = None) -> None:
        with self._lock:
            self.document.state = "interrupted" if error else "closed"
            self.event(
                "capture_error" if error else "capture_stopped",
                error_type=type(error).__name__ if error else None,
                error_message=str(error) if error else None,
            )

    def recording_session(self) -> RecordingSession:
        current = self.load(self.root, self.document.capture_id)
        sources = []
        for item in current.document.sources:
            path = session_artifact_path(current.directory, item.audio_file)
            if not path.exists() and item.frames == 0:
                continue
            info = sf.info(path)
            if (info.samplerate, info.channels) != (
                item.sample_rate,
                item.channels,
            ) or info.frames < item.frames:
                raise ValueError("Capture WAV geometry disagrees with journal")
            # A kill after the header update but before journal replace can leave more
            # valid frames than advertised. The actual closed/readable WAV is evidence.
            sources.append(
                RecordedAudioSource(
                    path, info.samplerate, info.channels, info.frames, item.label, item.device_name
                )
            )
        return RecordingSession(sources, journal=current, meeting_id=current.document.capture_id)

    def discard(self) -> None:
        if not self.directory.exists():
            return
        current = self.load(self.root, self.document.capture_id)
        # Check all entries before deleting any; unknown files are preserved.
        current.recording_session()
        files = [
            session_artifact_path(current.directory, source.audio_file)
            for source in current.document.sources
        ]
        files.append(session_artifact_path(current.directory, "journal.json"))
        for path in files:
            path.unlink(missing_ok=True)
        try:
            current.directory.rmdir()
        except OSError:
            pass


class CaptureWriter:
    """write/checkpoint/close are serialized on the source's capture thread."""

    def __init__(self, journal: CaptureJournal, source: JournalSource) -> None:
        self.journal = journal
        self.source = source
        self.path = session_artifact_path(journal.directory, source.audio_file)
        self._writer = sf.SoundFile(
            self.path,
            mode="w",
            samplerate=source.sample_rate,
            channels=source.channels,
            subtype="PCM_16",
        )
        self._frames = 0
        self._checkpoint_at = time.monotonic()

    def write(self, block: np.ndarray) -> None:
        session_artifact_path(self.journal.directory, self.source.audio_file)
        self._writer.write(block)
        self._frames += len(block)
        if time.monotonic() - self._checkpoint_at >= 0.5:
            self.checkpoint()

    def checkpoint(self) -> None:
        self.journal.validate()
        self._writer.flush()
        # libsndfile 1.2.2 sndfile.h: SFC_UPDATE_HEADER_NOW = 0x1060.
        # SoundFile exposes sf_command via its FFI but not this enum constant.
        # The documented success return is zero (not SF_TRUE).
        result = sf._snd.sf_command(self._writer._file, 0x1060, sf._ffi.NULL, 0)
        if result != 0 or sf._snd.sf_error(self._writer._file):
            raise OSError("WAV header checkpoint failed")
        self._writer.flush()
        with self.journal._lock:
            self.source.frames = self._frames
            self.journal._publish()
        self._checkpoint_at = time.monotonic()

    def close(self) -> None:
        if self._writer.closed:
            return
        try:
            self.checkpoint()
        finally:
            self._writer.close()
