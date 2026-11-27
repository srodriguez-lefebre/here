"""Exercise playback authority for portable imports with a standard WAV source name."""

from __future__ import annotations

import hashlib
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from here.memory.artifacts import (
    PORTABLE_GROUP_NAME,
    PORTABLE_RECEIPT_NAME,
    read_portable_completion,
    read_portable_group,
    read_portable_receipt,
    validate_portable_group_members,
)
from here.memory.database import open_library
from here.memory.models import Citation
from here.memory.portability import PortabilityService
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID, SEGMENT_TEXTS, make_session

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="uses native portable export")


@dataclass(frozen=True)
class ImportedAudio:
    memory: MemoryService
    reference: Citation
    asset_id: str
    wav: Path


@pytest.fixture
def imported_audio(tmp_path: Path) -> ImportedAudio:
    source_sessions = tmp_path / "source" / "sessions"
    source = MemoryService(source_sessions.parent / "data", sessions_root=source_sessions)
    native = make_session(source_sessions / "original", meeting_id=MEETING_ID)
    assert source.ingest_session(native.directory).status == "inserted"
    package = (
        PortabilityService(source)
        .export((MEETING_ID,), tmp_path / "citation.here.zip", True, lambda _: None, lambda: False)
        .path
    )
    target_sessions = tmp_path / "target" / "sessions"
    target_sessions.mkdir(parents=True)
    target = MemoryService(target_sessions.parent / "data", sessions_root=target_sessions)
    report = PortabilityService(target).import_archive(package, lambda _: None, lambda: False)
    assert [(row.meeting_id, row.status) for row in report.meetings] == [(MEETING_ID, "inserted")]

    detail = target.get_meeting(MEETING_ID)
    (asset,) = detail.audio_assets
    (revision,) = detail.revisions
    (link,) = detail.revision_audio_assets
    segment = detail.segments[0]
    assert segment.revision_id == revision.id == detail.meeting.active_revision_id
    assert (link.revision_id, link.asset_id, link.ordinal, link.source_file) == (
        revision.id,
        asset.id,
        0,
        "audio.wav",
    )
    root = target.data_dir / "meetings" / MEETING_ID
    revision_dir = root / "revisions" / revision.id
    assert not (revision_dir / "session.json").exists()
    group = read_portable_group(root, MEETING_ID)
    receipt = read_portable_receipt(revision_dir, group)
    completion = read_portable_completion(root / "states", group)
    validate_portable_group_members(group, completion, (receipt,))

    wav = revision_dir / f"audio_{UUID(asset.id).hex}.wav"
    assert target.data_dir / asset.relative_path == wav
    data = wav.read_bytes()
    assert len(data) == asset.size_bytes
    assert hashlib.sha256(data).hexdigest() == asset.sha256 == native.audio_sha256
    return ImportedAudio(target, Citation(MEETING_ID, revision.id, segment.id), asset.id, wav)


def test_imported_standard_audio_resolves_with_complete_group(
    imported_audio: ImportedAudio,
) -> None:
    result = imported_audio.memory.resolve_citation(imported_audio.reference)
    assert (result.status, result.audio_path, result.text, result.start_ms) == (
        "current",
        imported_audio.wav,
        SEGMENT_TEXTS[0],
        0,
    )


@pytest.mark.parametrize(
    "fault", ["malformed-receipt", "missing-group-and-receipt", "wrong-basename"]
)
def test_imported_standard_audio_requires_local_authority_before_wav_open(
    imported_audio: ImportedAudio, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    memory = imported_audio.memory
    wav = imported_audio.wav
    receipt_path = wav.with_name(PORTABLE_RECEIPT_NAME)
    root = memory.data_dir / "meetings" / MEETING_ID
    if fault == "malformed-receipt":
        receipt_path.write_text("[]", encoding="utf-8")
        group = read_portable_group(root, MEETING_ID)
        read_portable_completion(root / "states", group)
    elif fault == "missing-group-and-receipt":
        receipt_path.rename(tmp_path / "retired-receipt.json")
        (root / PORTABLE_GROUP_NAME).rename(tmp_path / "retired-group.json")
    else:
        # A standard native name with identical bytes cannot replace a portable role.
        substituted = wav.with_name("audio.wav")
        substituted.write_bytes(wav.read_bytes())
        with closing(open_library(memory.data_dir)) as library:
            changed = library.execute(
                "UPDATE audio_assets SET relative_path=? WHERE id=?",
                (substituted.relative_to(memory.data_dir).as_posix(), imported_audio.asset_id),
            )
            assert changed.rowcount == 1

    open_path = Path.open
    attempted_wav_opens: list[Path] = []

    def spy_open(path: Path, *args, **kwargs):
        if path.suffix.lower() == ".wav" and path.parent == wav.parent:
            attempted_wav_opens.append(path)
        return open_path(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)
    result = memory.resolve_citation(imported_audio.reference)
    assert not attempted_wav_opens, "Resolver opened a managed WAV before proving its portable role"
    assert (result.status, result.audio_path, result.text, result.start_ms) == (
        "missing",
        None,
        SEGMENT_TEXTS[0],
        0,
    )
