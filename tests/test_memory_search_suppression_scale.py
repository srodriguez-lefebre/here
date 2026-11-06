"""Functional large-suppression control, not a benchmark or a SQL-strategy pin."""

from __future__ import annotations

import sqlite3

from test_memory_search import _api, _Factory, _identity, _open_catalog, _publish, _search_all


def test_more_than_two_hundred_suppressed_ids_work_with_small_sql_variable_limit(
    tmp_path, monkeypatch
):
    from test_memory_delete import _record, _view
    from test_memory_search_guarded import _list_all

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    search, models = _api()
    root = tmp_path / "small-catalog-large-suppression"
    library, repo = _open_catalog(root)
    try:
        entities = {}
        for number, day, status in (
            (201, "08", "completed"),
            (202, "07", "completed"),
            (203, "06", "completed"),
            (204, "05", "completed"),
            (205, "04", "completed"),
            (206, "06", "failed"),
        ):
            entities[number] = _publish(
                repo,
                number,
                date=f"2026-01-{day}T00:00:00+00:00",
                title="Synthetic scale meeting",
                texts=("scaleneedle",),
                status=status,
            )
        participant = _identity(800)
        with repo.transaction():
            library.execute(
                "INSERT INTO participants(id,display_name) VALUES (?,?)",
                (participant, "Synthetic scale participant"),
            )
            for number in entities:
                library.execute(
                    "INSERT INTO meeting_participants(meeting_id,participant_id) VALUES (?,?)",
                    (_identity(number), participant),
                )
        suppressed = [_identity(201), _identity(202)] + [
            _identity(5000 + number) for number in range(218)
        ]
        for number, identity in enumerate(suppressed):
            _record(root, identity=identity, phase="pending" if number % 2 else "complete")
        # Actual B producer prerequisite, with both SQL and FTS hide still undone.
        assert _view(root).suppressed_meeting_ids == tuple(sorted(suppressed))
        assert (
            library.execute(
                "SELECT count(*) FROM segments_fts WHERE meeting_id IN (?,?)",
                (_identity(201), _identity(202)),
            ).fetchone()[0]
            == 2
        )

        factory = _Factory(root)
        limits = []

        def limited_factory():
            owned = factory()
            owned.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 64)
            limits.append(owned.connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
            return owned

        service = search.SearchService(limited_factory)
        filters = {
            "from_utc": "2026-01-01T00:00:00+00:00",
            "to_utc": "2026-01-10T00:00:00+00:00",
            "statuses": ("completed",),
            "participant_id": participant,
            "limit": 2,
        }
        listing = models.MeetingQuery(text="Synthetic scale", **filters)
        first_list = service.list_meetings(listing)
        assert [item.meeting.id for item in first_list.items] == [_identity(203), _identity(204)]
        assert first_list.next_cursor
        all_listed = _list_all(service, listing)
        assert [item.meeting.id for item in all_listed] == [
            _identity(number) for number in (203, 204, 205)
        ]
        query = models.SearchQuery(text="scaleneedle", include_partial=True, **filters)
        first_search = service.search(query)
        assert [hit.meeting_id for hit in first_search.items] == [_identity(203), _identity(204)]
        assert first_search.next_cursor
        hits = _search_all(service, query)
        assert [hit.segment_id for hit in hits] == [
            entities[number][2][0] for number in (203, 204, 205)
        ]
        assert len({hit.segment_id for hit in hits}) == 3
        assert limits and all(value == 64 for value in limits)
        assert all(connection.closed for connection in factory.connections)
    finally:
        library.close()
