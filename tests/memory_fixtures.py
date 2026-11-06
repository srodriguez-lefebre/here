"""Synthetic session corpus for memory tests; no recordings or provider access.

The WAV has two one-second PCM marks on its recorded timeline. A five-minute
wall-clock pause separates their capture events without adding frames. Callers
own the temporary root and can mutate these files for fault-injection tests.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SAMPLE_RATE = 8000
FRAME_COUNT = SAMPLE_RATE * 2
FIRST_MARK = 1000
SECOND_MARK = 2000
MEETING_ID = "11111111-1111-4111-8111-111111111111"
OTHER_MEETING_ID = "22222222-2222-4222-8222-222222222222"
LEGACY_SESSION_ID = "20260101_120000"
SEGMENT_TEXTS = ("Synthetic copper canary begins.", "Synthetic violet canary follows.")
TRANSCRIPT_TEXT = "\n".join(SEGMENT_TEXTS)

# These are metadata values only. Fixture creation never dereferences them.
HOSTILE_ARTIFACT_PATHS = (
    "../outside.wav",
    "..\\outside.wav",
    "/outside.wav",
    "C:\\synthetic-outside\\outside.wav",
    "\\\\synthetic-server\\share\\outside.wav",
    "audio.wav:alternate",
    "NUL.wav",
)


@dataclass(frozen=True)
class SyntheticSession:
    directory: Path
    meeting_id: str | None
    audio_sha256: str

    @property
    def audio(self) -> Path:
        return self.directory / "audio.wav"

    @property
    def metadata(self) -> Path:
        return self.directory / "session.json"


@dataclass(frozen=True)
class SyntheticCorpus:
    legacy: SyntheticSession
    identified: SyntheticSession
    copied_legacy: SyntheticSession
    same_audio_other_identity: SyntheticSession
    conflicting_identity: SyntheticSession

    @property
    def duplicate_origin(self) -> Path:
        """Import this same origin twice to test idempotence."""
        return self.legacy.directory


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_marked_wav(path: Path) -> str:
    """Write reproducible mono PCM16 data; return its whole-file SHA-256."""
    samples = (
        (FIRST_MARK if frame < SAMPLE_RATE else SECOND_MARK) * (1 if (frame // 20) % 2 == 0 else -1)
        for frame in range(FRAME_COUNT)
    )
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(struct.pack(f"<{FRAME_COUNT}h", *samples))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_session(
    directory: Path,
    *,
    meeting_id: str | None = None,
    transcript_text: str = TRANSCRIPT_TEXT,
    segments: list[dict[str, Any]] | None = None,
    paused_seconds: float = 300.0,
) -> SyntheticSession:
    """Create schema-v1 artifacts, omitting meeting_id for a historical session.

    JSON is authored independently of production models to expose reader contract
    regressions. Text is UTF-8 with BOM, matching M1's transcript artifact format.
    Both chunks deliberately use speaker_0 in different request scopes.
    """
    directory.mkdir(parents=True, exist_ok=False)
    audio_hash = write_marked_wav(directory / "audio.wav")
    started_at = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    completed_at = started_at + timedelta(seconds=2 + paused_seconds)
    source = {
        "label": "synthetic",
        "device_name": None,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "frames": FRAME_COUNT,
        "duration_seconds": 2.0,
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "session_id": LEGACY_SESSION_ID,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": 2.0,
        "status": "completed",
        "failure_stage": None,
        "recoverable_audio": "audio.wav",
        "sources": [source],
        "capture_sources": [{**source, "audio_file": "audio.wav"}],
        "transcription_model": "synthetic-transcription-model",
        "cleanup_model": "synthetic-cleanup-model",
        "cleanup_enabled": False,
        "alt_model_used": False,
        "live_pipeline_attempted": False,
        "live_pipeline_used": False,
        "fallback_used": False,
        "output_files": [
            "audio.wav",
            "transcript.txt",
            "transcript.md",
            "session.json",
            "segments.json",
            "chunks.json",
            "events.json",
        ],
        "total_paused_seconds": paused_seconds,
    }
    if meeting_id is not None:
        metadata["meeting_id"] = meeting_id
    write_json(directory / "session.json", metadata)
    (directory / "transcript.txt").write_text(transcript_text, encoding="utf-8-sig")
    (directory / "transcript.md").write_text(
        f"# Synthetic recording\n\n{transcript_text}\n", encoding="utf-8"
    )
    if segments is None:
        segments = [
            {
                "text": text,
                "start": float(index),
                "end": float(index + 1),
                "speaker": "speaker_0",
                "chunk_index": index,
                "speaker_scope": f"synthetic-request-{index}",
            }
            for index, text in enumerate(SEGMENT_TEXTS)
        ]
    write_json(
        directory / "segments.json",
        {
            "schema_version": 1,
            "semantics": "provider_evidence",
            "timing_reference": "recording_seconds",
            "speaker_identity_scope": "request",
            "segments": segments,
        },
    )
    write_json(directory / "chunks.json", {"schema_version": 1, "chunks": []})
    write_json(
        directory / "events.json",
        {
            "schema_version": 1,
            "events": [
                {
                    "kind": "paused",
                    "occurred_at": (started_at + timedelta(seconds=1)).isoformat(),
                    "recorded_duration_seconds": 1.0,
                    "total_paused_seconds": 0.0,
                    "details": {},
                },
                {
                    "kind": "resumed",
                    "occurred_at": (started_at + timedelta(seconds=1 + paused_seconds)).isoformat(),
                    "recorded_duration_seconds": 1.0,
                    "total_paused_seconds": paused_seconds,
                    "details": {},
                },
            ],
        },
    )
    return SyntheticSession(directory, meeting_id, audio_hash)


def make_corpus(root: Path) -> SyntheticCorpus:
    """Independent UUIDs share PCM; a conflicting UUID has changed text."""
    legacy = make_session(root / "legacy")
    identified = make_session(root / "identified", meeting_id=MEETING_ID)
    copied_directory = root / "copied-legacy"
    shutil.copytree(legacy.directory, copied_directory)
    copied_legacy = SyntheticSession(copied_directory, None, legacy.audio_sha256)
    other = make_session(root / "other-identity", meeting_id=OTHER_MEETING_ID)
    conflict_text = "Synthetic changed canary."
    conflict = make_session(
        root / "conflicting-identity",
        meeting_id=MEETING_ID,
        transcript_text=conflict_text,
        segments=[{"text": conflict_text, "start": 0.0, "end": 1.0}],
    )
    return SyntheticCorpus(legacy, identified, copied_legacy, other, conflict)


def make_hostile_session(directory: Path, artifact_path: str) -> SyntheticSession:
    """Replace only manifest references; never create or open the hostile target."""
    session = make_session(directory, meeting_id=MEETING_ID)
    metadata = json.loads(session.metadata.read_text(encoding="utf-8"))
    metadata["recoverable_audio"] = artifact_path
    metadata["output_files"].append(artifact_path)
    metadata["capture_sources"][0]["audio_file"] = artifact_path
    write_json(session.metadata, metadata)
    return session
