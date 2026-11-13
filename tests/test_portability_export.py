"""M5 B core export exercises the real catalog with synthetic recordings."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from zipfile import ZipFile

import pytest
from here.memory.models import MeetingPatch
from here.memory.service import Conflict, MemoryService
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session


@pytest.fixture
def memory(tmp_path: Path) -> MemoryService:
    sessions_root = tmp_path / "synthetic-sources"
    sessions_root.mkdir()
    return MemoryService(tmp_path / "data", sessions_root=sessions_root)


@pytest.fixture
def destination(tmp_path: Path) -> Path:
    directory = tmp_path / "exports"
    directory.mkdir()
    return directory / "selected.here.zip"


def _ingest(memory: MemoryService, source: Path, *, meeting_id: str | None = None) -> str:
    session = make_session(source, meeting_id=meeting_id)
    result = memory.ingest_session(session.directory)
    assert result.status == "inserted"
    assert result.meeting_id is not None
    return result.meeting_id


@pytest.mark.parametrize("include_audio", [True, False], ids=["with-audio", "without-audio"])
def test_export_manifest_matches_payloads_and_revision_audio_links(
    memory: MemoryService, destination: Path, tmp_path: Path, include_audio: bool
) -> None:
    from here.memory.portability import PortabilityService

    sources = tmp_path / "synthetic-sources"
    ids = (
        _ingest(memory, sources / "identified", meeting_id=MEETING_ID),
        _ingest(memory, sources / "another", meeting_id=OTHER_MEETING_ID),
        _ingest(memory, sources / "legacy"),
    )
    first_revision = memory.get_meeting(MEETING_ID).meeting.active_revision_id
    assert first_revision is not None
    revised = make_session(
        sources / "identified-retry",
        meeting_id=MEETING_ID,
        transcript_text="Synthetic retry evidence.",
        segments=[{"text": "Synthetic retry evidence.", "start": 0.0, "end": 1.0}],
    )
    pcm_wav = revised.audio.read_bytes()
    revised.audio.write_bytes(pcm_wav[:-1] + bytes((pcm_wav[-1] ^ 1,)))
    assert memory.ingest_session(revised.directory, revision_of=first_revision).status == "revised"
    meetings = [memory.get_meeting(meeting_id) for meeting_id in ids]
    assert len(meetings[0].audio_assets) == 2
    staged_progress: list[int] = []

    def progress(written: int) -> None:
        if not destination.exists() and written > 0:
            assert any(destination.parent.iterdir())
            staged_progress.append(written)

    result = PortabilityService(memory).export(
        ids, destination, include_audio, progress=progress, cancellation=lambda: False
    )

    assert result.path == destination
    assert staged_progress and staged_progress == sorted(staged_progress)
    assert set(destination.parent.iterdir()) == {destination}
    expected_payloads = {"participants.json", "speaker_assignments.json"}
    with ZipFile(destination) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format_version"] == 1
        assert manifest["exporter_version"]
        assert datetime.fromisoformat(manifest["exported_at"]).utcoffset() == timedelta(0)
        assert len(manifest["meeting_ids"]) == len(ids)
        assert set(manifest["meeting_ids"]) == set(ids)
        assert manifest["audio_included"] is include_audio
        entries = {entry["path"]: entry for entry in manifest["files"]}
        assert len(entries) == len(manifest["files"])

        for detail in meetings:
            prefix = f"meetings/{detail.meeting.id}"
            meeting_path = f"{prefix}/meeting.json"
            expected_payloads.add(meeting_path)
            meeting = json.loads(archive.read(meeting_path))
            assert meeting["id"] == detail.meeting.id
            assert meeting["legacy_session_id"] == detail.meeting.legacy_session_id
            assert meeting["active_revision_id"] == detail.meeting.active_revision_id
            assert "row_version" not in meeting
            assets = {asset["id"]: asset for asset in meeting["audio_assets"]}
            assert set(assets) == {asset.id for asset in detail.audio_assets}
            for asset in detail.audio_assets:
                payload = assets[asset.id]
                assert payload["size_bytes"] == asset.size_bytes
                assert payload["sha256"] == asset.sha256
                assert payload["sample_rate"] == asset.sample_rate
                assert payload["channels"] == asset.channels
                assert payload["duration_ms"] == asset.duration_ms
                if include_audio:
                    asset_path = f"{prefix}/audio/{asset.id}.wav"
                    expected_payloads.add(asset_path)
                    assert payload["path"] == asset_path
                    managed_wav = memory.data_dir / asset.relative_path
                    assert archive.read(asset_path) == managed_wav.read_bytes()
                else:
                    assert payload["path"] is None
                    assert payload["status"] == "missing"

            for revision in detail.revisions:
                revision_path = f"{prefix}/revisions/{revision.revision_number}/revision.json"
                segments_path = f"{prefix}/revisions/{revision.revision_number}/segments.json"
                expected_payloads.update((revision_path, segments_path))
                document = json.loads(archive.read(revision_path))
                assert document["id"] == revision.id
                assert document["revision_number"] == revision.revision_number
                assert "origin" not in document
                keys = ("meeting_id", "revision_id", "asset_id", "ordinal", "source_file")
                assert [tuple(link[key] for key in keys) for link in document["audio_links"]] == [
                    (
                        link.meeting_id,
                        link.revision_id,
                        link.asset_id,
                        link.ordinal,
                        link.source_file,
                    )
                    for link in detail.revision_audio_assets
                    if link.revision_id == revision.id
                ]

        assert set(entries) == expected_payloads
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert set(names) == expected_payloads | {"manifest.json"}
        for path, entry in entries.items():
            payload = archive.read(path)
            assert entry["size"] == len(payload)
            assert entry["sha256"] == hashlib.sha256(payload).hexdigest()


def test_export_excludes_private_library_material_and_unselected_meetings(
    memory: MemoryService, destination: Path, tmp_path: Path
) -> None:
    from here.memory.portability import PortabilityService

    source = tmp_path / "synthetic-sources" / "synthetic-local-origin"
    meeting_id = _ingest(memory, source, meeting_id=MEETING_ID)
    private_meeting = make_session(
        source.parent / "unselected",
        meeting_id=OTHER_MEETING_ID,
        transcript_text="synthetic-unselected-transcript-canary",
        segments=[{"text": "synthetic-unselected-transcript-canary"}],
    )
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as reader:
        reader.execute("SELECT id FROM meetings").fetchall()
        assert memory.ingest_session(private_meeting.directory).status == "inserted"
        canaries = {
            ".env": b"synthetic-env-secret-canary",
            "backups/library-canary.sqlite3": b"synthetic-backup-canary",
            "logs/here.log": b"synthetic-log-canary",
            "settings.json": b"synthetic-settings-canary",
            ".staging/private-canary": b"synthetic-staging-canary",
        }
        for name, marker in canaries.items():
            path = memory.data_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(marker)

        PortabilityService(memory).export(
            (meeting_id,), destination, False, progress=lambda _: None, cancellation=lambda: False
        )

    with ZipFile(destination) as archive:
        names = archive.namelist()
        assert all(
            "sqlite" not in name.lower() and not name.startswith(f"meetings/{OTHER_MEETING_ID}/")
            for name in names
        )
        assert not {Path(name).name for name in canaries} & {Path(name).name for name in names}
        payloads = [archive.read(name) for name in names]
        for marker in (*canaries.values(), b"synthetic-unselected-transcript-canary"):
            assert all(marker not in payload for payload in payloads)
        assert any(b"Synthetic copper canary begins." in payload for payload in payloads)
        detail = memory.get_meeting(meeting_id)
        local_values = (
            *(revision.origin for revision in detail.revisions),
            *(asset.relative_path for asset in detail.audio_assets),
        )
        for value in local_values:
            encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
            assert all(encoded not in payload for payload in payloads)


def test_cancelled_export_removes_its_sibling_and_never_publishes(
    memory: MemoryService, destination: Path, tmp_path: Path
) -> None:
    from here.memory.portability import ExportCancelled, PortabilityService

    meeting_id = _ingest(memory, tmp_path / "synthetic-sources" / "selected", meeting_id=MEETING_ID)
    unrelated = destination.with_name("keep.txt")
    unrelated.write_bytes(b"unrelated destination file")
    cancellation = Event()

    def cancel_after_writing(written: int) -> None:
        if written > 0:
            assert not destination.exists()
            assert set(destination.parent.iterdir()) - {unrelated}
            cancellation.set()

    with pytest.raises(ExportCancelled):
        PortabilityService(memory).export(
            (meeting_id,),
            destination,
            True,
            progress=cancel_after_writing,
            cancellation=cancellation.is_set,
        )

    assert cancellation.is_set()
    assert set(destination.parent.iterdir()) == {unrelated}
    assert unrelated.read_bytes() == b"unrelated destination file"


@pytest.mark.parametrize("change", ["row-version", "participant-relationship"])
def test_export_refuses_changed_snapshot_before_publication(
    memory: MemoryService, destination: Path, tmp_path: Path, change: str
) -> None:
    from here.memory.portability import PortabilityService

    meeting_id = _ingest(memory, tmp_path / "synthetic-sources" / "selected", meeting_id=MEETING_ID)
    participant = memory.add_participant("Synthetic Reviewer")
    before = memory.link_participant(meeting_id, participant.id, "observer").meeting
    changed = False

    def edit_after_writing(written: int) -> None:
        nonlocal changed
        if written <= 0 or changed:
            return
        assert not destination.exists()
        if change == "row-version":
            memory.update_meeting(
                meeting_id, MeetingPatch(title="Synthetic edit"), before.row_version
            )
        else:
            # The public lifecycle edit increments row_version; isolate relationship comparison.
            with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
                with connection:
                    connection.execute(
                        "UPDATE meeting_participants SET role=? WHERE meeting_id=? "
                        "AND participant_id=?",
                        ("reviewer", meeting_id, participant.id),
                    )
        changed = True

    with pytest.raises(Conflict):
        PortabilityService(memory).export(
            (meeting_id,),
            destination,
            True,
            progress=edit_after_writing,
            cancellation=lambda: False,
        )

    assert changed
    current = memory.get_meeting(meeting_id)
    if change == "row-version":
        assert current.meeting.row_version == before.row_version + 1
        assert current.meeting.title == "Synthetic edit"
    else:
        assert current.meeting.row_version == before.row_version
        assert current.participant_links[0].role == "reviewer"
    assert not list(destination.parent.iterdir())
