"""Exercise imported history through the M3 search and citation services."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from uuid import UUID

import pytest
from fixtures.m5_portability_cross_group import (
    LEGACY_AUDIO_NAME,
    SEARCH_START_MS,
    SEARCH_TERM,
    make_cross_group_sessions,
)
from here.memory.artifacts import (
    portable_completion_name,
    read_portable_completion,
    read_portable_group,
    read_portable_receipt,
    validate_portable_group_members,
)
from here.memory.database import open_library
from here.memory.models import Citation, MeetingQuery, SearchPage, SearchQuery
from here.memory.portability import PortabilityService
from here.memory.search import SearchService
from here.memory.service import InvalidInput, MemoryService, NotFound
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, SEGMENT_TEXTS, make_session, write_json

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="uses native portable export")

_THIRD_MEETING_ID = "33333333-3333-4333-8333-333333333333"


@pytest.fixture
def source(tmp_path: Path) -> MemoryService:
    sessions = tmp_path / "source" / "sessions"
    original, retry = make_cross_group_sessions(sessions)
    memory = MemoryService(sessions.parent / "data", sessions_root=sessions)
    result = memory.ingest_session(original)
    assert (result.status, result.meeting_id) == ("inserted", MEETING_ID)
    parent = memory.get_meeting(MEETING_ID).meeting.active_revision_id
    assert parent is not None
    result = memory.ingest_session(retry, revision_of=parent)
    assert (result.status, result.meeting_id) == ("revised", MEETING_ID)
    return memory


def _search_page(search: SearchService, query: SearchQuery) -> SearchPage:
    page = search.search(query)
    assert isinstance(page, SearchPage)
    assert not page.degraded
    return page


def test_portable_import_updates_ranked_search_and_preserves_list_keyset(
    tmp_path: Path, source: MemoryService
) -> None:
    sessions = tmp_path / "destination" / "sessions"
    target = MemoryService(sessions.parent / "data", sessions_root=sessions)
    for identity in (OTHER_MEETING_ID, _THIRD_MEETING_ID):
        native = make_session(
            sessions / identity,
            meeting_id=identity,
            transcript_text=SEGMENT_TEXTS[0],
            segments=[{"text": SEGMENT_TEXTS[0], "start": 0.0, "end": 1.0}],
        )
        assert target.ingest_session(native.directory).status == "inserted"

    search = SearchService(partial(open_library, target.data_dir))
    term_query = SearchQuery(text=SEARCH_TERM, limit=1)
    ranked_query = SearchQuery(text="copper", limit=1)
    list_query = MeetingQuery(limit=1)
    assert _search_page(search, term_query).items == ()
    ranked_before = _search_page(search, ranked_query)
    listed_before = search.list_meetings(list_query)
    assert ranked_before.next_cursor is not None
    assert listed_before.next_cursor is not None
    assert listed_before.items[0].meeting.id == OTHER_MEETING_ID

    package = (
        PortabilityService(source)
        .export((MEETING_ID,), tmp_path / "search.here.zip", False, lambda _: None, lambda: False)
        .path
    )
    portability = PortabilityService(target)
    report = portability.import_archive(package, lambda _: None, lambda: False)
    assert [(row.meeting_id, row.status) for row in report.meetings] == [(MEETING_ID, "inserted")]

    source_detail = source.get_meeting(MEETING_ID)
    assert len(source_detail.revisions) == 2
    (segment,) = (item for item in source_detail.segments if item.text == SEGMENT_TEXTS[1])
    assert segment.revision_id == source_detail.meeting.active_revision_id
    term_page = _search_page(search, term_query)
    (hit,) = term_page.items
    assert term_page.next_cursor is None
    assert (hit.meeting_id, hit.revision_id, hit.segment_id) == (
        MEETING_ID,
        segment.revision_id,
        segment.id,
    )
    assert (hit.start_ms, hit.end_ms, hit.timing_source) == (SEARCH_START_MS, 2000, "provider")
    assert SEARCH_TERM in hit.snippet

    with pytest.raises(InvalidInput) as stale:
        search.search(replace(ranked_query, cursor=ranked_before.next_cursor))
    assert stale.value.reason == "corpus_changed"
    (next_meeting,) = search.list_meetings(
        replace(list_query, cursor=listed_before.next_cursor)
    ).items
    assert next_meeting.meeting.id == _THIRD_MEETING_ID

    fresh = _search_page(search, ranked_query)
    assert fresh.next_cursor is not None
    continuation = replace(ranked_query, cursor=fresh.next_cursor)
    next_ranked = _search_page(search, continuation)
    assert len(next_ranked.items) == 1 and next_ranked.items != fresh.items
    report = portability.import_archive(package, lambda _: None, lambda: False)
    assert [(row.meeting_id, row.status) for row in report.meetings] == [(MEETING_ID, "unchanged")]
    assert _search_page(search, continuation) == next_ranked


@pytest.fixture
def imported(tmp_path: Path, source: MemoryService) -> MemoryService:
    package = (
        PortabilityService(source)
        .export((MEETING_ID,), tmp_path / "citation.here.zip", True, lambda _: None, lambda: False)
        .path
    )
    sessions = tmp_path / "destination" / "sessions"
    sessions.mkdir(parents=True)
    target = MemoryService(sessions.parent / "data", sessions_root=sessions)
    report = PortabilityService(target).import_archive(package, lambda _: None, lambda: False)
    assert [(row.meeting_id, row.status) for row in report.meetings] == [(MEETING_ID, "inserted")]

    (original_asset,) = source.get_meeting(MEETING_ID).audio_assets
    (local_asset,) = target.get_meeting(MEETING_ID).audio_assets
    assert replace(local_asset, relative_path=original_asset.relative_path) == original_asset

    source_root = tmp_path / "source"
    source_root.rename(tmp_path / "retired-source")
    package.rename(tmp_path / "retired-package")
    assert not source_root.exists() and not package.exists()
    return MemoryService(target.data_dir, sessions_root=sessions)


def test_imported_legacy_citations_use_complete_local_group_for_both_revisions(
    imported: MemoryService, tmp_path: Path
) -> None:
    meeting = imported.get_meeting(MEETING_ID)
    historical, active = meeting.revisions
    assert meeting.meeting.active_revision_id == active.id
    (asset,) = meeting.audio_assets
    first_link, second_link = meeting.revision_audio_assets
    assert (first_link.revision_id, second_link.revision_id) == (historical.id, active.id)
    for link in (first_link, second_link):
        assert (link.asset_id, link.ordinal, link.source_file) == (asset.id, 0, LEGACY_AUDIO_NAME)

    root = imported.data_dir / "meetings" / MEETING_ID
    group = read_portable_group(root, MEETING_ID)
    assert group.revision_ids == (historical.id, active.id)
    completion = read_portable_completion(root / "states", group)
    receipts = []
    for link in (first_link, second_link):
        directory = root / "revisions" / link.revision_id
        assert not (directory / "session.json").exists()
        receipt = read_portable_receipt(directory, group)
        assert receipt.snapshot["revision_audio_assets"] == [asdict(link)]
        receipts.append(receipt)
    validate_portable_group_members(group, completion, receipts)

    wav = root / "revisions" / historical.id / f"audio_{UUID(asset.id).hex}.wav"
    assert imported.data_dir / asset.relative_path == wav
    data = wav.read_bytes()
    assert (len(data), hashlib.sha256(data).hexdigest()) == (asset.size_bytes, asset.sha256)

    (first_segment,) = (part for part in meeting.segments if part.revision_id == historical.id)
    (second_mark,) = (
        part
        for part in meeting.segments
        if part.revision_id == active.id and part.text == SEGMENT_TEXTS[1]
    )
    old = imported.resolve_citation(Citation(MEETING_ID, historical.id, first_segment.id))
    current = imported.resolve_citation(Citation(MEETING_ID, active.id, second_mark.id))
    assert [
        (
            item.meeting_id,
            item.revision_id,
            item.segment_id,
            item.status,
            item.audio_path,
            item.text,
            item.start_ms,
            item.end_ms,
            item.timing_source,
        )
        for item in (old, current)
    ] == [
        (
            MEETING_ID,
            historical.id,
            first_segment.id,
            "historical",
            wav,
            SEGMENT_TEXTS[0],
            0,
            1000,
            "provider",
        ),
        (
            MEETING_ID,
            active.id,
            second_mark.id,
            "current",
            wav,
            SEGMENT_TEXTS[1],
            SEARCH_START_MS,
            2000,
            "provider",
        ),
    ]


@pytest.mark.parametrize("fault", ["incomplete-group", "same-size-tampering", "suppressed"])
def test_imported_citation_fails_closed_without_valid_local_authority(
    imported: MemoryService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    meeting = imported.get_meeting(MEETING_ID)
    (asset,) = meeting.audio_assets
    wav = imported.data_dir / asset.relative_path
    root = imported.data_dir / "meetings" / MEETING_ID
    group = read_portable_group(root, MEETING_ID)
    proof = root / "states" / portable_completion_name(group.operation_id)
    assert wav.is_file() and proof.is_file()
    (segment,) = (part for part in meeting.segments if part.text == SEGMENT_TEXTS[1])
    reference = Citation(MEETING_ID, segment.revision_id, segment.id)

    if fault == "same-size-tampering":
        data = wav.read_bytes()
        wav.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    else:
        if fault == "incomplete-group":
            proof.rename(tmp_path / "retired-completion")
        else:
            registry = imported.data_dir / ".deletions"
            registry.mkdir(exist_ok=True)
            write_json(
                registry / f"{MEETING_ID}.json",
                {"schema_version": 1, "meeting_id": MEETING_ID, "phase": "pending", "origins": []},
            )

        open_path = Path.open

        def reject_wav_open(path, *args, **kwargs):
            if path.suffix.lower() == ".wav":
                pytest.fail("Citation opened WAV bytes before establishing local authority")
            return open_path(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", reject_wav_open)

    if fault == "suppressed":
        with pytest.raises(NotFound):
            imported.resolve_citation(reference)
    else:
        result = imported.resolve_citation(reference)
        assert (result.status, result.audio_path, result.text, result.start_ms) == (
            "missing",
            None,
            segment.text,
            SEARCH_START_MS,
        )
