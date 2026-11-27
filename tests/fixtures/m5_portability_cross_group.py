"""Synthetic v1 source sessions for portable search and citation tests."""

from __future__ import annotations

import json
from pathlib import Path

from memory_fixtures import MEETING_ID, SEGMENT_TEXTS, make_session, write_json

LEGACY_AUDIO_NAME = "take.wav"
SEARCH_TERM = "violet"
SEARCH_START_MS = 1000


def make_cross_group_sessions(root: Path) -> tuple[Path, Path]:
    """Return original and retry session directories for MEETING_ID.

    Both revisions reference the same marked PCM bytes by a declared v1 basename.
    The search term occurs in the retry's second segment, at the second mark.
    """
    historical_text = SEGMENT_TEXTS[0]
    original = make_session(
        root / "original",
        meeting_id=MEETING_ID,
        transcript_text=historical_text,
        segments=[{"text": historical_text, "start": 0.0, "end": 1.0}],
    )
    retry = make_session(root / "retry", meeting_id=MEETING_ID)

    for session in (original, retry):
        metadata = json.loads(session.metadata.read_text(encoding="utf-8"))
        metadata["recoverable_audio"] = LEGACY_AUDIO_NAME
        metadata["capture_sources"][0]["audio_file"] = LEGACY_AUDIO_NAME
        metadata["output_files"] = [
            LEGACY_AUDIO_NAME if name == session.audio.name else name
            for name in metadata["output_files"]
        ]
        session.audio.rename(session.directory / LEGACY_AUDIO_NAME)
        write_json(session.metadata, metadata)

    return original.directory, retry.directory
