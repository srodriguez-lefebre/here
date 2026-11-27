"""Public search regressions over a synthetic catalog and its real FTS index."""

from contextlib import closing
from dataclasses import replace
from functools import partial
from uuid import UUID

import pytest
from here.memory.database import open_library
from here.memory.models import MeetingPatch, MeetingQuery, SearchQuery
from here.memory.search import SearchService
from here.memory.service import InvalidInput, MemoryService
from memory_fixtures import MEETING_ID, make_session
from test_memory_search import _search_all


# Frozen v1 listing token for the short filter below, after the first meeting.
LEGACY_LIST_CURSOR = (
    "WyJoZXJlLXNlYXJjaCIsMSwibGlzdCIsWyJhYWFhIixudWxsLG51bGwsW10sbnVsbCxmYWxzZV0sImRhdGUt"
    "ZGVzYy1udWxsLWxhc3QvaWQtYXNjIixbMCwiMjAyNi0wMS0wMVQxMjowMDowMCswMDowMCIsIjAwMDAwMDAw"
    "LTAwMDAtNDAwMC04MDAwLTAwMDAwMDAwMDAwMSJdXQ"
)


@pytest.mark.parametrize(
    "repeated_text",
    ["needle needle", "needle needle needle"],
    ids=["equal-rank", "different-rank"],
)
def test_repeated_fts_matches_collapse_before_limit_and_keyset(tmp_path, repeated_text):
    root = tmp_path / "library"
    texts = ("needle needle", "needle background", "needle background background")
    source = make_session(
        tmp_path / "origin",
        meeting_id=MEETING_ID,
        transcript_text="\n".join(texts),
        segments=[
            {"text": text, "start": index / 2, "end": (index + 1) / 2}
            for index, text in enumerate(texts)
        ],
    )
    memory = MemoryService(root, sessions_root=tmp_path / "sessions")
    assert memory.ingest_session(source.directory).status == "inserted"
    detail = memory.get_meeting(MEETING_ID)
    assert tuple(segment.text for segment in detail.segments) == texts
    repeated, second, third = (segment.id for segment in detail.segments)

    search = SearchService(partial(open_library, root))
    query = SearchQuery(text="needle", limit=2)
    previous = search.search(query)
    assert [hit.segment_id for hit in previous.items] == [repeated, second]
    assert previous.next_cursor

    # Model a duplicate already present in the real derived index.
    with closing(open_library(root)) as library:
        library.execute(
            "INSERT INTO segments_fts(meeting_id,revision_id,segment_id,text,title) "
            "SELECT meeting_id,revision_id,segment_id,?,title FROM segments_fts "
            "WHERE segment_id=?",
            (repeated_text, repeated),
        )
        ranked = library.execute(
            "SELECT segment_id,bm25(segments_fts) AS rank FROM segments_fts "
            "WHERE segments_fts MATCH ? ORDER BY rank,segment_id",
            ('"needle"',),
        ).fetchall()
        # Both physical matches rank ahead of the other segments. Either can
        # represent the repeated segment without changing the expected page order.
        assert [row[0] for row in ranked] == [repeated, repeated, second, third]
        assert (ranked[0][1] == ranked[1][1]) is (repeated_text == "needle needle")

    # Duplicates still contribute to the corpus generation and invalidate old ranks.
    with pytest.raises(InvalidInput) as stale:
        search.search(replace(query, cursor=previous.next_cursor))
    assert stale.value.reason == "corpus_changed"

    page = search.search(query)
    assert [hit.segment_id for hit in page.items] == [repeated, second]
    assert page.next_cursor
    final = search.search(replace(query, cursor=page.next_cursor))
    assert [hit.segment_id for hit in final.items] == [third]
    assert final.next_cursor is None

    # With the page boundary inside the duplicate group, a different-rank
    # copy cannot reappear on a continuation.
    hits = _search_all(search, replace(query, limit=1))
    assert [hit.segment_id for hit in hits] == [repeated, second, third]
    with closing(open_library(root)) as library:
        copies = library.execute(
            "SELECT count(*) FROM segments_fts WHERE segment_id=?", (repeated,)
        ).fetchone()[0]
    assert copies == 2


def test_listing_round_trips_cursor_for_long_title_filter(tmp_path):
    root = tmp_path / "library"
    term = "a" * (13 * 1024)
    identities = tuple(str(UUID(int=index, version=4)) for index in range(1, 5))
    first, second, third, _ = identities
    titles = (f"{term} first", f"{term} second", f"{term} third", "Outside the filter")
    memory = MemoryService(root, sessions_root=tmp_path / "sessions")
    # The fixture gives all four meetings the same date, leaving the ID as the tiebreaker.
    for index, (meeting_id, title) in enumerate(zip(identities, titles), 1):
        source = make_session(tmp_path / f"origin-{index}", meeting_id=meeting_id)
        assert memory.ingest_session(source.directory).status == "inserted"
        meeting = memory.get_meeting(meeting_id).meeting
        updated = memory.update_meeting(meeting_id, MeetingPatch(title=title), meeting.row_version)
        assert updated.meeting.title == title

    search = SearchService(partial(open_library, root))
    legacy_query = MeetingQuery(text="aaaa", limit=1, cursor=LEGACY_LIST_CURSOR)
    legacy_page = search.list_meetings(legacy_query)
    assert [item.meeting.id for item in legacy_page.items] == [second]

    query = MeetingQuery(text=term, limit=1)
    page = search.list_meetings(query)
    assert [item.meeting.id for item in page.items] == [first]
    assert page.next_cursor

    continuation = replace(query, cursor=page.next_cursor)
    next_page = search.list_meetings(replace(continuation, text=f" \t{term}\n"))
    assert [item.meeting.id for item in next_page.items] == [second]
    assert next_page.next_cursor
    last_page = search.list_meetings(replace(query, cursor=next_page.next_cursor))
    assert [item.meeting.id for item in last_page.items] == [third]
    assert last_page.next_cursor is None

    with pytest.raises(InvalidInput):
        search.list_meetings(replace(continuation, text=term[:-1] + "b"))
    with pytest.raises(InvalidInput):
        search.list_meetings(replace(query, cursor="!"))
