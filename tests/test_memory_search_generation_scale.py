"""Public generation controls across a modest multi-batch real catalog.

Row positions select fixture stimuli only. No digest/codec reconstruction, private
generation helper, batch-size assertion or SQL fetch-strategy contract is used.
"""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest
from test_memory_search import _api, _Factory, _open_catalog, _search_all


@pytest.fixture
def generation_catalog(tmp_path, monkeypatch):
    from search_fixtures import SearchCorpus

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    corpus = SearchCorpus(meeting_count=64, segments_per_meeting=14)
    root = tmp_path / "generation-catalog"
    library, repo = _open_catalog(root, corpus)
    try:
        assert library.execute("SELECT count(*) FROM meetings").fetchone()[0] == 64
        assert library.execute("SELECT count(*) FROM segments").fetchone()[0] == 896
        assert library.execute("SELECT count(*) FROM segments_fts").fetchone()[0] == 819
        yield root, library, repo, corpus
    finally:
        library.close()


def _ordered_cohort_segments(corpus, suppressed=()):
    # Authored cohort0..49 has equal BM25 inputs and dates tied in groups of4.
    indices = sorted(range(50), key=lambda index: (-(index // 4), index))
    return [
        corpus.segment_id(index, 0)
        for index in indices
        if corpus.meeting_id(index) not in suppressed
    ]


@pytest.mark.parametrize(
    "position", [255, 256, 257, -1], ids=["row255", "row256", "row257", "last"]
)
@pytest.mark.parametrize("hidden", [False, True], ids=["nonmatching", "suppressed"])
def test_nonmatching_or_suppressed_index_change_beyond_early_rows_invalidates_cursor(
    generation_catalog, position, hidden
):
    from search_fixtures import COHORT_TOKEN
    from test_memory_delete import _record, _view

    search, models = _api()
    errors = importlib.import_module("here.memory.service")
    root, library, repo, corpus = generation_catalog
    rows = library.execute(
        "SELECT meeting_id,segment_id,text FROM segments_fts "
        "ORDER BY meeting_id,revision_id,segment_id"
    ).fetchall()
    target = rows[-1 if position == -1 else position - 1]
    assert COHORT_TOKEN not in target["text"]
    assert (
        library.execute(
            "SELECT status FROM meetings WHERE id=?", (target["meeting_id"],)
        ).fetchone()[0]
        == "completed"
    )
    if hidden:
        _record(root, identity=target["meeting_id"], phase="pending")
    snapshot = _view(root)
    assert snapshot.suppressed_meeting_ids == ((target["meeting_id"],) if hidden else ())
    # Real B suppression leaves all source rows and the target document alive.
    assert library.execute("SELECT count(*) FROM segments_fts").fetchone()[0] == 819
    assert (
        library.execute(
            "SELECT text FROM segments_fts WHERE segment_id=?", (target["segment_id"],)
        ).fetchone()[0]
        == target["text"]
    )
    query = models.SearchQuery(text=COHORT_TOKEN, include_partial=True, limit=7)
    first = search.SearchService(_Factory(root)).search(query)
    assert first.next_cursor
    expected = _ordered_cohort_segments(corpus, snapshot.suppressed_meeting_ids)
    assert [hit.segment_id for hit in first.items] == expected[:7]
    # Controlled FTS-only data, including delimiters/unicode/long fields. It is
    # deliberately unrelated to the query and leaves catalog revision IDs intact.
    text = 'independent [] , "quote" line\n café 雪 ' + "longfield " * 512 + str(position)
    with repo.transaction():
        changed = library.execute(
            "UPDATE segments_fts SET text=? WHERE segment_id=?", (text, target["segment_id"])
        )
        assert changed.rowcount == 1
    assert _view(root) == snapshot
    observer = _Factory(root)
    with pytest.raises(errors.InvalidInput) as caught:
        search.SearchService(observer).search(replace(query, cursor=first.next_cursor))
    assert caught.value.reason == "corpus_changed"
    assert not any("MATCH" in sql.upper() for sql, _, _ in observer.calls)
    assert observer.connections and all(connection.closed for connection in observer.connections)
    fresh = _search_all(search.SearchService(_Factory(root)), query)
    assert [hit.segment_id for hit in fresh] == expected
    assert len({hit.segment_id for hit in fresh}) == len(expected)
    assert all(hit.meeting_id not in snapshot.suppressed_meeting_ids for hit in fresh)


@pytest.mark.parametrize("include_partial", [False, True])
def test_large_real_reindex_noop_preserves_fresh_instance_cursor_and_all_continuations(
    generation_catalog, include_partial
):
    from search_fixtures import COHORT_TOKEN
    from test_memory_delete import _record, _view

    search, models = _api()
    backend = importlib.import_module("here.memory.service")
    root, library, _, corpus = generation_catalog
    suppressed = corpus.meeting_id(19)
    _record(root, identity=suppressed, phase="complete")
    snapshot = _view(root)
    assert snapshot.suppressed_meeting_ids == (suppressed,)
    assert (
        library.execute(
            "SELECT count(*) FROM segments_fts WHERE meeting_id=?", (suppressed,)
        ).fetchone()[0]
        == 13
    )
    memory = backend.MemoryService(root)
    # The original fixture indexes a complete deleting row. The FIRST real B
    # rebuild normalizes deleting/suppressed visibility; it is NOT called a no-op.
    normalized = memory.reindex()
    assert (normalized.indexed_meetings, normalized.indexed_segments) == (61, 793)
    assert library.execute("SELECT count(*) FROM segments_fts").fetchone()[0] == 793
    query = models.SearchQuery(text=COHORT_TOKEN, include_partial=include_partial, limit=7)
    expected = _ordered_cohort_segments(corpus, (suppressed,))
    first = search.SearchService(_Factory(root)).search(query)
    same = search.SearchService(_Factory(root)).search(query)
    assert first.next_cursor and first.next_cursor == same.next_cursor
    assert first.items == same.items
    before = _search_all(search.SearchService(_Factory(root)), query)
    assert [hit.segment_id for hit in before] == expected
    # The SECOND real producer rebuild preserves identical logical inputs.
    assert memory.reindex() == normalized
    assert _view(root) == snapshot
    after = search.SearchService(_Factory(root)).search(query)
    assert after.next_cursor == first.next_cursor
    assert after.items == first.items
    continued = search.SearchService(_Factory(root)).search(
        replace(query, cursor=first.next_cursor)
    )
    assert [hit.segment_id for hit in continued.items] == expected[7:14]
    all_after = _search_all(search.SearchService(_Factory(root)), query)
    assert [hit.segment_id for hit in all_after] == expected
    assert len({hit.segment_id for hit in all_after}) == len(expected)
