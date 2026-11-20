"""Citation resolution uses exact guarded catalog identities and recording time."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from here.memory.database import LibraryError
from here.memory.models import Citation, CitationTarget, Segment
from here.memory.service import InvalidInput, MemoryService, NotFound
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json

_TEXT = "A😀 café fin."
_SEGMENTS = [
    {"text": _TEXT, "start": 0.0, "end": 1.0, "speaker": "speaker_0", "speaker_scope": "first"},
    {"text": "After the pause.", "start": 1.0, "end": 2.0},
    {"text": "End bound only.", "end": 2.0},
    {"text": "Untimed evidence."},
]


@pytest.fixture
def memory(tmp_path: Path) -> MemoryService:
    sessions = tmp_path / "synthetic-sessions"
    origin = make_session(
        sessions / "original", meeting_id=MEETING_ID, transcript_text=_TEXT, segments=_SEGMENTS
    )
    # Existing M1 evidence can omit the timezone, leaving the catalog date unknown.
    import json

    metadata = json.loads(origin.metadata.read_text(encoding="utf-8"))
    metadata["started_at"] = "2026-01-01T12:00:00"
    metadata["completed_at"] = "2026-01-01T12:05:02"
    write_json(origin.metadata, metadata)
    memory = MemoryService(tmp_path / "data", sessions_root=sessions)
    assert memory.ingest_session(origin.directory).status == "inserted"
    assert memory.get_meeting(MEETING_ID).meeting.started_at is None
    return memory


def _reference(segment: Segment, **evidence: object) -> Citation:
    return replace(Citation(MEETING_ID, segment.revision_id, segment.id), **evidence)


def _asset_path(memory: MemoryService, revision_id: str) -> Path:
    detail = memory.get_meeting(MEETING_ID)
    (link,) = (item for item in detail.revision_audio_assets if item.revision_id == revision_id)
    asset = next(item for item in detail.audio_assets if item.id == link.asset_id)
    return memory.data_dir / asset.relative_path


def _revise(memory: MemoryService, name: str, *, changed_audio: bool) -> str:
    previous = memory.get_meeting(MEETING_ID).meeting.active_revision_id
    assert previous is not None
    directory = memory.data_dir.parent / "synthetic-sessions" / name
    source = make_session(
        directory, meeting_id=MEETING_ID, transcript_text=f"Synthetic {name}.", segments=_SEGMENTS
    )
    if changed_audio:
        audio = source.audio.read_bytes()
        source.audio.write_bytes(audio[:-1] + bytes((audio[-1] ^ 1,)))
    assert memory.ingest_session(source.directory, revision_of=previous).status == "revised"
    current = memory.get_meeting(MEETING_ID).meeting.active_revision_id
    assert current is not None and current != previous
    return current


def test_resolver_uses_exact_historical_and_current_audio_links(
    memory: MemoryService,
) -> None:
    initial = memory.get_meeting(MEETING_ID)
    first = initial.revisions[0].id
    middle = _revise(memory, "second", changed_audio=True)
    last = _revise(memory, "third", changed_audio=False)
    first_path = _asset_path(memory, first)
    middle_path = _asset_path(memory, middle)
    assert first_path != middle_path and _asset_path(memory, last) == first_path

    participant = memory.add_participant("Synthetic reviewer")
    memory.link_participant(MEETING_ID, participant.id)
    first_segment = initial.segments[0]
    assert first_segment.speaker_key is not None
    memory.assign_speaker(MEETING_ID, first_segment.speaker_key, participant.id, scope=first)
    detail = memory.get_meeting(MEETING_ID)

    for revision_id, path, status, speaker in (
        (first, first_path, "historical", participant.display_name),
        (middle, middle_path, "historical", None),
        (last, first_path, "current", None),
    ):
        segment = next(item for item in detail.segments if item.revision_id == revision_id)
        result = memory.resolve_citation(
            _reference(segment, quote="😀 café", start_char=1, end_char=7)
        )
        assert result == CitationTarget(
            meeting_id=MEETING_ID,
            revision_id=revision_id,
            segment_id=segment.id,
            title=detail.meeting.title,
            started_at=detail.meeting.started_at,
            text=_TEXT,
            speaker=speaker,
            start_ms=0,
            end_ms=1000,
            timing_source="provider",
            audio_path=path,
            status=status,
        )
        assert path.is_file()


@pytest.mark.parametrize(
    ("ordinal", "start", "end", "timing"),
    [(1, 1000, 2000, "provider"), (2, None, 2000, "provider"), (3, None, None, "absent")],
)
def test_resolver_preserves_recording_offsets_and_nullable_bounds(
    memory: MemoryService, ordinal: int, start: int | None, end: int | None, timing: str
) -> None:
    detail = memory.get_meeting(MEETING_ID)
    assert detail.meeting.paused_ms == 300_000
    segment = detail.segments[ordinal]
    result = memory.resolve_citation(_reference(segment))
    assert result.text == segment.text
    assert (result.start_ms, result.end_ms, result.timing_source) == (start, end, timing)
    assert result.started_at is None


@pytest.mark.parametrize(
    "evidence",
    [
        pytest.param({"quote": "wrong"}, id="quote-not-in-segment"),
        pytest.param({"quote": "café", "start_char": 1, "end_char": 7}, id="wrong-at-span"),
        pytest.param({"quote": "😀 café", "start_char": 1, "end_char": 11}, id="byte-offset"),
        pytest.param({"start_char": 1}, id="one-sided-span"),
        pytest.param({"start_char": -1, "end_char": 3}, id="negative-span"),
        pytest.param({"start_char": 3, "end_char": 2}, id="reversed-span"),
        pytest.param({"start_char": 3, "end_char": 100}, id="oversized-span"),
        pytest.param({"start_char": True, "end_char": 2}, id="bool-span"),
        pytest.param({"meeting_id": "not-a-uuid"}, id="malformed-meeting-id"),
        pytest.param({"revision_id": "{" + OTHER_MEETING_ID + "}"}, id="noncanonical-revision-id"),
        pytest.param({"segment_id": "not-a-uuid"}, id="malformed-segment-id"),
    ],
)
def test_resolver_rejects_invalid_citation_fields(
    memory: MemoryService, evidence: dict[str, object]
) -> None:
    segment = memory.get_meeting(MEETING_ID).segments[0]
    with pytest.raises(InvalidInput):
        memory.resolve_citation(_reference(segment, **evidence))


def test_resolver_rejects_revision_or_segment_from_another_meeting(memory: MemoryService) -> None:
    origin = make_session(
        memory.data_dir.parent / "synthetic-sessions" / "other", meeting_id=OTHER_MEETING_ID
    )
    assert memory.ingest_session(origin.directory).status == "inserted"
    first = memory.get_meeting(MEETING_ID).segments[0]
    other = memory.get_meeting(OTHER_MEETING_ID).segments[0]
    for invalid in (_reference(other), _reference(first, segment_id=other.id)):
        with pytest.raises(InvalidInput):
            memory.resolve_citation(invalid)


@pytest.mark.parametrize("fault", ["missing", "digest", "missing-link", "ambiguous-link"])
def test_resolver_never_substitutes_another_playable_asset(
    memory: MemoryService, fault: str
) -> None:
    current = _revise(memory, "second", changed_audio=True)
    detail = memory.get_meeting(MEETING_ID)
    old = detail.segments[0]
    old_audio = _asset_path(memory, old.revision_id)
    current_audio = _asset_path(memory, current)
    assert old_audio != current_audio and current_audio.is_file()
    if fault == "missing":
        old_audio.rename(old_audio.with_suffix(".unavailable"))
    elif fault == "digest":
        old = detail.segments[0]
        audio = old_audio.read_bytes()
        old_audio.write_bytes(audio[:-1] + bytes((audio[-1] ^ 1,)))
    else:
        with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
            with connection:
                if fault == "missing-link":
                    connection.execute(
                        "DELETE FROM revision_audio_assets WHERE revision_id=?", (old.revision_id,)
                    )
                else:
                    (alternate,) = (
                        link.asset_id
                        for link in detail.revision_audio_assets
                        if link.revision_id == current
                    )
                    connection.execute(
                        "INSERT INTO revision_audio_assets VALUES (?,?,?,?,?)",
                        (MEETING_ID, old.revision_id, alternate, 1, "alternate.wav"),
                    )

    result = memory.resolve_citation(_reference(old))
    assert (result.status, result.audio_path, result.text) == ("missing", None, old.text)


def _reject_audio_access(monkeypatch: pytest.MonkeyPatch) -> None:
    open_path = Path.open
    open_file = open

    def forbid_wav(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).suffix.lower() == ".wav":
            pytest.fail("Resolver accessed WAV content before establishing authority")
        return open_file(file, *args, **kwargs)

    def forbid_wav_path(path, *args, **kwargs):
        if path.suffix.lower() == ".wav":
            pytest.fail("Resolver accessed WAV content before establishing authority")
        return open_path(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", forbid_wav)
    monkeypatch.setattr(Path, "open", forbid_wav_path)


@pytest.mark.parametrize("redirect", ["outside-managed-root", "hardlink"])
def test_resolver_rejects_redirects_before_opening_audio(
    memory: MemoryService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, redirect: str
) -> None:
    detail = memory.get_meeting(MEETING_ID)
    segment = detail.segments[0]
    path = _asset_path(memory, segment.revision_id)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(path.read_bytes())
    if redirect == "hardlink":
        path.rename(path.with_suffix(".original"))
        os.link(outside, path)
    else:
        with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
            with connection:
                connection.execute(
                    "UPDATE audio_assets SET relative_path=? WHERE id=?",
                    (str(outside), detail.audio_assets[0].id),
                )

    _reject_audio_access(monkeypatch)
    result = memory.resolve_citation(_reference(segment))
    assert result.status == "missing"
    assert result.audio_path is None


@pytest.mark.parametrize("scope", ["pending-uuid", "complete-alias", "invalid-registry"])
def test_resolver_applies_r42_before_any_audio_access(
    memory: MemoryService, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    initial = memory.get_meeting(MEETING_ID).segments[0]
    _revise(memory, "second", changed_audio=True)
    registry = memory.data_dir / ".deletions"
    registry.mkdir()
    if scope == "pending-uuid":
        identity, phase, origins = MEETING_ID, "pending", []
    else:
        identity, phase = OTHER_MEETING_ID, "complete"
        with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
            origin, fingerprint = connection.execute(
                "SELECT origin_path,origin_fingerprint FROM ingest_sources "
                "WHERE meeting_id=? AND revision_id=?",
                (MEETING_ID, initial.revision_id),
            ).fetchone()
        origins = [{"origin_path": origin, "origin_fingerprint": fingerprint}]

    record = registry / f"{identity}.json"
    error = NotFound
    if scope == "invalid-registry":
        record.write_text('{"schema_version":', encoding="utf-8")
        error = LibraryError
    else:
        write_json(
            record,
            {"schema_version": 1, "meeting_id": identity, "phase": phase, "origins": origins},
        )
    _reject_audio_access(monkeypatch)
    with pytest.raises(error):
        memory.resolve_citation(_reference(initial))
