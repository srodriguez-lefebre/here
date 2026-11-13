"""Real literal query path with the existing A FTS runtime probe disabled.

No mock search pages or shadow maintenance API. B5 reindex/suppression tests await
the accepted real adapter and remain explicitly outstanding in the RED report.
"""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest
from test_memory_search import (
    _api,
    _identity,
    _invalid_input,
    _mutate_corpus,
    _open_catalog,
    _publish,
    _search_all,
    _service,
)
from test_memory_search import (
    catalog as catalog,  # Explicit fixture re-export; shared setup has one author.
)


def _force_literal(monkeypatch):
    """Patch A's real probe, including an already imported reference if present."""
    module = importlib.import_module("here.memory.service")
    original = module.fts_available
    monkeypatch.setattr(module, "fts_available", lambda connection: False)
    search, _ = _api()
    if getattr(search, "fts_available", None) is original:
        monkeypatch.setattr(search, "fts_available", module.fts_available)


def test_missing_fts_keeps_listing_filters_and_degraded_search_usable(catalog, monkeypatch):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    _force_literal(monkeypatch)
    root, _, _, corpus = catalog
    service = _service(root)
    assert service.capabilities().fts_available is False
    listed = service.list_meetings(models.MeetingQuery(limit=10))
    assert len(listed.items) == 10
    case = corpus.filter_case
    page = service.search(
        models.SearchQuery(
            text=COHORT_TOKEN,
            from_utc=case["from_utc"],
            to_utc=case["to_utc"],
            statuses=case["statuses"],
            participant_id=case["participant_id"],
            limit=10,
        )
    )
    assert page.degraded is True
    assert {hit.segment_id for hit in page.items} == set(case["filtered_segment_ids"])
    assert page.next_cursor is None


@pytest.mark.parametrize("literal", ["%", "_", "\\", "NEAR", "title:x", "*", '"'])
@pytest.mark.parametrize("missing_fts", [False, True])
def test_literal_routes_escape_wildcards_and_match_tokens_across_title_and_text(
    tmp_path, monkeypatch, literal, missing_fts
):
    _, models = _api()
    library, repo = _open_catalog(tmp_path)
    try:
        # Partial route is literal even with FTS; a title token AND text token
        # must match the same segment, not require both in one field.
        actual = _publish(
            repo, 1, title=f"titlemarker {literal}", texts=("bodymarker",), partial=not missing_fts
        )
        _publish(
            repo, 2, title="titlemarker ordinary", texts=("bodymarker",), partial=not missing_fts
        )
        _publish(
            repo, 3, title=f"titlemarker {literal}", texts=("unrelated",), partial=not missing_fts
        )
        if missing_fts:
            _force_literal(monkeypatch)
        page = _service(tmp_path).search(
            models.SearchQuery(text=f"titlemarker {literal} bodymarker", include_partial=True)
        )
        assert [hit.segment_id for hit in page.items] == list(actual[2])
        assert page.items[0].snippet == "bodymarker"
        assert page.items[0].partial is (not missing_fts)
    finally:
        library.close()


def test_literal_title_only_snippet_is_actual_start_of_text_bounded_to_160(tmp_path, monkeypatch):
    _, models = _api()
    library, repo = _open_catalog(tmp_path)
    try:
        text = "α" * 200 + " actual source"
        _publish(repo, 1, title="titleonlyneedle", texts=(text,))
        _force_literal(monkeypatch)
        page = _service(tmp_path).search(models.SearchQuery(text="titleonlyneedle"))
        assert len(page.items) == 1
        assert page.items[0].snippet == text[:160]
        assert page.items[0].start_ms is None
        assert page.items[0].end_ms is None
        assert page.items[0].timing_source == "absent"
    finally:
        library.close()


def test_fallback_global_rank_zero_orders_by_date_id_segment_across_null_boundary(
    tmp_path, monkeypatch
):
    _, models = _api()
    library, repo = _open_catalog(tmp_path)
    try:
        older = _publish(repo, 1, date="2026-01-01T00:00:00+00:00")
        newest = _publish(
            repo, 2, date="2026-01-03T00:00:00+00:00", partial=True, texts=("needle", "needle")
        )
        tied = _publish(repo, 3, date="2026-01-03T00:00:00+00:00")
        unknown = _publish(repo, 4, partial=True, texts=("needle", "needle"))
        _force_literal(monkeypatch)
        hits = _search_all(
            _service(tmp_path), models.SearchQuery(text="needle", include_partial=True, limit=2)
        )
        assert [hit.segment_id for hit in hits] == [*newest[2], *tied[2], *older[2], *unknown[2]]
        assert [hit.partial for hit in hits] == [True, True, False, False, True, True]
    finally:
        library.close()


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ({"from_utc": "2026-01-02T00:00:00+00:00"}, [4, 3]),
        ({"to_utc": "2026-01-03T00:00:00+00:00"}, [3, 2]),
        ({"from_utc": "2026-01-02T00:00:00+00:00", "to_utc": "2026-01-03T00:00:00+00:00"}, [3]),
    ],
)
def test_literal_date_bounds_match_fts_rules_and_exclude_nulls(
    tmp_path, monkeypatch, bounds, expected
):
    _, models = _api()
    library, repo = _open_catalog(tmp_path)
    try:
        dates = (
            None,
            "2026-01-01T00:00:00+00:00",
            "2026-01-02T00:00:00+00:00",
            "2026-01-03T00:00:00+00:00",
        )
        for number, date in enumerate(dates, 1):
            _publish(repo, number, date=date, partial=number % 2 == 0)
        _force_literal(monkeypatch)
        page = _service(tmp_path).search(
            models.SearchQuery(text="needle", include_partial=True, **bounds)
        )
        assert [hit.meeting_id for hit in page.items] == [_identity(i) for i in expected]
    finally:
        library.close()


def test_literal_visibility_excludes_historical_deleting_and_unselected_partial(
    catalog, monkeypatch
):
    _, models = _api()
    _force_literal(monkeypatch)
    root, _, _, corpus = catalog
    case = corpus.visibility_case
    service = _service(root)
    ordinary = service.search(models.SearchQuery(text=case["text"]))
    assert [hit.segment_id for hit in ordinary.items] == [case["active_complete_segment_id"]]
    opted = service.search(models.SearchQuery(text=case["text"], include_partial=True))
    assert {hit.segment_id for hit in opted.items} == {
        case["active_complete_segment_id"],
        case["partial_segment_id"],
    }
    assert {hit.segment_id for hit in opted.items if hit.partial} == {case["partial_segment_id"]}


@pytest.mark.parametrize(
    "mutation",
    [
        "nonmatching_insert",
        "nonmatching_delete",
        "indexed_title",
        "date",
        "status",
        "filter_relation",
        "active_revision",
    ],
)
@pytest.mark.parametrize("include_partial", [False, True])
def test_literal_search_also_rejects_whole_corpus_generation_changes(
    catalog, monkeypatch, mutation, include_partial
):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    _force_literal(monkeypatch)
    root, _, _, corpus = catalog
    query = models.SearchQuery(text=COHORT_TOKEN, limit=2, include_partial=include_partial)
    first = _service(root).search(query)
    assert first.degraded and first.next_cursor
    _mutate_corpus(root, corpus, mutation)
    with pytest.raises(_invalid_input()) as caught:
        _service(root).search(replace(query, cursor=first.next_cursor))
    assert caught.value.reason == "corpus_changed"
    fresh = _search_all(_service(root), query)
    assert len(fresh) == 50
    assert len({hit.segment_id for hit in fresh}) == 50


def test_literal_generation_token_is_stable_across_fresh_services(catalog, monkeypatch):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    _force_literal(monkeypatch)
    query = models.SearchQuery(text=COHORT_TOKEN, limit=2, include_partial=True)
    first = _service(catalog[0]).search(query)
    second = _service(catalog[0]).search(query)
    assert first.next_cursor == second.next_cursor
    continuation = _service(catalog[0]).search(replace(query, cursor=first.next_cursor))
    assert len(continuation.items) == 2
    assert {hit.segment_id for hit in first.items}.isdisjoint(
        hit.segment_id for hit in continuation.items
    )


def test_cursor_cannot_switch_from_fts_to_literal_backend(catalog, monkeypatch):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    query = models.SearchQuery(text=COHORT_TOKEN, limit=2)
    fts_page = _service(catalog[0]).search(query)
    assert fts_page.next_cursor
    _force_literal(monkeypatch)
    with pytest.raises(_invalid_input()):
        _service(catalog[0]).search(replace(query, cursor=fts_page.next_cursor))
