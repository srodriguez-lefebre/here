"""Single-meeting Markdown export through the real portability and catalog services."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from here.memory.models import MeetingPatch
from here.memory.portability import ExportCancelled, PortabilityService
from here.memory.service import (
    Conflict,
    MemoryService,
    NotFound,
    _ExportCleanupPending,
    _PortableExportView,
)
from here.output.export_binding import WindowsExportSibling
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json

_TITLE = "Revisión sintética"
_NOTES = "Notas para el próximo encuentro."
_ACTIVE_TEXT = "Se aprobó el diseño después de la revisión."
_OLD_TEXT = "Synthetic historical transcript canary."
_LOCAL_ORIGIN = "synthetic-private-origin-canary"
_SOURCE_MARKUP_CANARY = "synthetic-source-markup-canary"
_EXISTING = b"preexisting destination contents"


def _set_timezone_free_dates(metadata_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # As in historical M1 recordings, timezone-free dates are unknown to the catalog.
    metadata["started_at"] = "2026-01-01T12:00:00"
    metadata["completed_at"] = "2026-01-01T12:05:02"
    write_json(metadata_path, metadata)


@pytest.fixture
def memory(tmp_path: Path) -> MemoryService:
    sources = tmp_path / "synthetic-sources"
    source = make_session(
        sources / _LOCAL_ORIGIN,
        meeting_id=MEETING_ID,
        transcript_text=_OLD_TEXT,
        segments=[{"text": _OLD_TEXT}],
    )
    _set_timezone_free_dates(source.metadata)
    memory = MemoryService(tmp_path / "configured-library", sessions_root=sources)
    assert memory.ingest_session(source.directory).status == "inserted"
    return memory


@pytest.fixture
def destination(tmp_path: Path) -> Path:
    directory = tmp_path / "exports"
    directory.mkdir()
    return directory / "meeting.md"


def test_markdown_exports_active_transcript_and_allowed_metadata(
    memory: MemoryService, destination: Path, tmp_path: Path
) -> None:
    detail = memory.get_meeting(MEETING_ID)
    previous_revision = detail.meeting.active_revision_id
    assert previous_revision is not None
    revised = make_session(
        tmp_path / "synthetic-sources" / "synthetic-retry",
        meeting_id=MEETING_ID,
        transcript_text=_ACTIVE_TEXT,
        segments=[{"text": _ACTIVE_TEXT}],
    )
    _set_timezone_free_dates(revised.metadata)
    (revised.directory / "transcript.md").write_text(
        f"<script>{_SOURCE_MARKUP_CANARY}</script>", encoding="utf-8"
    )
    assert (
        memory.ingest_session(revised.directory, revision_of=previous_revision).status == "revised"
    )
    detail = memory.get_meeting(MEETING_ID)
    memory.update_meeting(
        MEETING_ID, MeetingPatch(title=_TITLE, notes=_NOTES), detail.meeting.row_version
    )
    detail = memory.get_meeting(MEETING_ID)
    assert detail.meeting.active_revision_id != previous_revision
    assert detail.meeting.started_at is detail.meeting.completed_at is None
    assert detail.audio_assets
    secret = "synthetic-private-secret-canary"
    (memory.data_dir / ".env").write_text(f"OPENAI_API_KEY={secret}", encoding="utf-8")

    PortabilityService(memory).export_markdown(MEETING_ID, destination, cancellation=lambda: False)

    document = destination.read_text(encoding="utf-8")
    assert _TITLE in document
    assert MEETING_ID in document
    assert _NOTES in document
    assert _ACTIVE_TEXT in document
    assert not re.search(r"(?im)^.*(?:started|completed).*\d{4}[-/]\d{2}[-/]\d{2}", document)
    for private_value in (
        "2026-01-01",
        _OLD_TEXT,
        _SOURCE_MARKUP_CANARY,
        _LOCAL_ORIGIN,
        str(memory.data_dir),
        memory.data_dir.as_posix(),
        str(revised.directory),
        revised.directory.as_posix(),
        secret,
        "audio.wav",
        *(asset.relative_path for asset in detail.audio_assets),
    ):
        assert private_value not in document
    assert set(destination.parent.iterdir()) == {destination}


def test_cancelled_markdown_removes_written_temporary(
    memory: MemoryService, destination: Path
) -> None:
    saw_written_sibling = False

    def cancellation() -> bool:
        nonlocal saw_written_sibling
        assert not destination.exists()
        with os.scandir(destination.parent) as entries:
            saw_written_sibling = any(entry.stat().st_size > 0 for entry in entries)
        return saw_written_sibling

    with pytest.raises(ExportCancelled):
        PortabilityService(memory).export_markdown(
            MEETING_ID, destination, cancellation=cancellation
        )

    assert saw_written_sibling
    assert not list(destination.parent.iterdir())


@pytest.mark.parametrize("deleted", [False, True], ids=["missing", "deleted"])
def test_unavailable_markdown_leaves_existing_destination_intact(
    memory: MemoryService, destination: Path, deleted: bool
) -> None:
    meeting_id = OTHER_MEETING_ID
    if deleted:
        meeting_id = MEETING_ID
        memory.delete_meeting(meeting_id)
    destination.write_bytes(_EXISTING)

    with pytest.raises(NotFound):
        PortabilityService(memory).export_markdown(
            meeting_id, destination, cancellation=lambda: False
        )

    assert destination.read_bytes() == _EXISTING
    assert set(destination.parent.iterdir()) == {destination}


def test_markdown_revalidates_catalog_through_guarded_publication(
    memory: MemoryService, destination: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = memory.get_meeting(MEETING_ID).meeting
    publish = _PortableExportView.publish
    published_title = "Synthetic edit during export"

    def edit_before_publish(
        view: _PortableExportView, temporary: WindowsExportSibling, *, cancelled: Callable[[], bool]
    ) -> None:
        destination.write_bytes(_EXISTING)
        memory.update_meeting(MEETING_ID, MeetingPatch(title=published_title), before.row_version)
        publish(view, temporary, cancelled=cancelled)

    monkeypatch.setattr(_PortableExportView, "publish", edit_before_publish)
    with pytest.raises(Conflict):
        PortabilityService(memory).export_markdown(
            MEETING_ID, destination, cancellation=lambda: False
        )

    assert memory.get_meeting(MEETING_ID).meeting.title == published_title
    assert destination.read_bytes() == _EXISTING
    assert set(destination.parent.iterdir()) == {destination}


def test_markdown_reports_cleanup_pending_after_publishing(
    memory: MemoryService, destination: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_retirement(service: MemoryService, registration: Path) -> None:
        assert service is memory
        assert registration.is_file()
        assert _OLD_TEXT in destination.read_text(encoding="utf-8")
        raise _ExportCleanupPending()

    with monkeypatch.context() as failure:
        failure.setattr(MemoryService, "_retire_export", fail_retirement)
        result = PortabilityService(memory).export_markdown(
            MEETING_ID, destination, cancellation=lambda: False
        )

    [registration] = (memory.data_dir / ".exports").glob("*.json")
    published = destination.read_bytes()
    assert _OLD_TEXT.encode("utf-8") in published
    assert set(destination.parent.iterdir()) == {destination}

    MemoryService(
        memory.data_dir, sessions_root=memory.data_dir.parent / "synthetic-sources"
    ).reconcile()
    assert not registration.exists()
    assert destination.read_bytes() == published

    assert result.path == destination
    assert result.warnings
    for warning in result.warnings:
        assert warning.strip()
        assert destination.name not in warning
        assert registration.name not in warning
        assert str(destination.parent) not in warning
        assert str(memory.data_dir) not in warning
