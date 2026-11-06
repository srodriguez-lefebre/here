"""Independent real-B5/M3 consumer contracts, prepared without running runtime.

Run only after root accepts and integrates the exact B5 producer handoff. Registry
fixtures reuse B's existing helper; SQL below is controlled synthetic catalog data,
never a second parser, deletion implementation, native adapter or index rebuild.
"""

from __future__ import annotations

import importlib
import os
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from test_memory_search import _Factory, _api, _identity, _open_catalog, _publish, _search_all
from test_memory_search_fallback import _force_literal

TOKEN = "guardneedle"
DATES = {
    1: "2026-01-08T00:00:00+00:00",
    2: "2026-01-07T00:00:00+00:00",
    3: "2026-01-06T00:00:00+00:00",
    4: "2026-01-05T00:00:00+00:00",
    5: "2026-01-05T00:00:00+00:00",
    6: "2026-01-04T00:00:00+00:00",
}
ROUTES = [(False, False), (False, True), (True, False), (True, True)]


def _producer_view(root):
    # Runtime access only: absent/unaccepted B5 must never prevent collection or
    # turn into a skip. Parent runs these after its real producer preconditions.
    from test_memory_delete import _view

    return _view(root)


def _record(root, identity, *, phase="pending", origins=()):
    from test_memory_delete import _record as producer_record

    return producer_record(root, identity=identity, phase=phase, origins=origins)


@pytest.fixture
def guarded_catalog(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = tmp_path / "library"
    library, repo = _open_catalog(root)
    _, models = _api()
    entities = {}
    try:
        for number in range(1, 10):
            entities[number] = _publish(
                repo, number, date=DATES.get(number), partial=number in (2, 6, 9),
                texts=(TOKEN, TOKEN) if number == 6 else (TOKEN,),
            )
        origin = os.path.normcase(os.path.normpath(str(tmp_path / "offline-origin")))
        different_origin = os.path.normcase(os.path.normpath(str(tmp_path / "distinct-origin")))
        pair = {"origin_path": origin, "origin_fingerprint": "a" * 64}
        with repo.transaction():
            for number, path, fingerprint in (
                (3, origin, "a" * 64),
                (4, origin, "b" * 64),
                (5, different_origin, "a" * 64),
            ):
                repo.insert_ingest_source(models.IngestSource(
                    id=_identity(900 + number), origin_path=path,
                    origin_fingerprint=fingerprint, meeting_id=_identity(number),
                    revision_id=entities[number][1],
                ))
        yield {
            "root": root, "library": library, "repo": repo, "entities": entities,
            "pair": pair,
        }
    finally:
        library.close()


def _barriers(data, phase="pending"):
    root = data["root"]
    paths = [
        _record(root, _identity(1), phase=phase, origins=(data["pair"],)),
        _record(root, _identity(2), phase=phase),
        _record(root, _identity(7), phase=phase),
    ]
    snapshot = _producer_view(root)
    # Producer prerequisite: raw UUID 1/2/7 and reconstructed exact-pair alias 3.
    assert snapshot.suppressed_meeting_ids == tuple(sorted(_identity(i) for i in (1, 2, 3, 7)))
    assert _identity(4) not in snapshot.suppressed_meeting_ids  # origin alone
    assert _identity(5) not in snapshot.suppressed_meeting_ids  # fingerprint alone
    assert isinstance(snapshot.fingerprint, str) and snapshot.fingerprint
    # The SQL hide transaction has NOT run. The guard alone must prevent exposure.
    assert data["library"].execute(
        "SELECT count(*) FROM meetings WHERE id IN (?,?,?,?) AND status!='deleting'",
        tuple(_identity(i) for i in (1, 2, 3, 7)),
    ).fetchone()[0] == 4
    assert data["library"].execute(
        "SELECT count(*) FROM segments_fts WHERE meeting_id IN (?,?,?)",
        tuple(_identity(i) for i in (1, 3, 7)),
    ).fetchone()[0] == 3
    return paths, snapshot


def _service(root):
    search, _ = _api()
    return search.SearchService(_Factory(root))


def _list_all(service, query):
    items = []
    seen = set()
    for _ in range(20):
        page = service.list_meetings(query)
        items.extend(page.items)
        if page.next_cursor is None:
            return items
        assert page.next_cursor not in seen, "Guarded listing cursor must advance"
        seen.add(page.next_cursor)
        query = replace(query, cursor=page.next_cursor)
    pytest.fail("Minimal guarded listing failed to exhaust")


def _expected_segments(data, numbers):
    return [segment for number in numbers for segment in data["entities"][number][2]]


def _catalog_inputs(data):
    """Read controlled SQL/FTS fixture state to isolate registry-only mutations."""
    return tuple(
        tuple(tuple(row) for row in data["library"].execute(sql))
        for sql in (
            "SELECT * FROM meetings ORDER BY id",
            "SELECT * FROM ingest_sources ORDER BY id",
            "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id",
        )
    )


def _assert_visible_search(data, *, missing_fts, include_partial):
    _, models = _api()
    service = _service(data["root"])
    query = models.SearchQuery(text=TOKEN, include_partial=include_partial, limit=2)
    first = service.search(query)
    assert first.degraded is missing_fts
    assert [hit.meeting_id for hit in first.items] == [_identity(4), _identity(5)]
    assert first.next_cursor, "Suppression must fill LIMIT before returning a cursor"
    hits = _search_all(service, query)
    numbers = (
        [4, 5, 6, 8, 9] if missing_fts and include_partial
        else [4, 5, 8, 6, 9] if include_partial else [4, 5, 8]
    )
    assert [hit.segment_id for hit in hits] == _expected_segments(data, numbers)
    assert len({hit.segment_id for hit in hits}) == len(hits)
    assert all(hit.meeting_id not in {_identity(i) for i in (1, 2, 3, 7)} for hit in hits)
    assert {hit.meeting_id for hit in hits if hit.partial} == (
        {_identity(6), _identity(9)} if include_partial else set()
    )
    return hits


@pytest.mark.parametrize("phase", ["pending", "complete"])
@pytest.mark.parametrize(("missing_fts", "include_partial"), ROUTES)
def test_real_barriers_and_exact_aliases_are_filtered_before_global_search_limit(
    guarded_catalog, monkeypatch, phase, missing_fts, include_partial
):
    data = guarded_catalog
    _barriers(data, phase)
    if missing_fts:
        _force_literal(monkeypatch)
    _assert_visible_search(data, missing_fts=missing_fts, include_partial=include_partial)


@pytest.mark.parametrize("phase", ["pending", "complete"])
@pytest.mark.parametrize("include_partial", [False, True])
def test_real_barriers_filter_listing_and_empty_search_before_null_keyset_limit(
    guarded_catalog, phase, include_partial
):
    _, models = _api()
    data = guarded_catalog
    _barriers(data, phase)
    service = _service(data["root"])
    query = models.MeetingQuery(include_partial=include_partial, limit=2)
    first = service.list_meetings(query)
    assert [item.meeting.id for item in first.items] == [_identity(4), _identity(5)]
    expected = [4, 5, 6, 8, 9] if include_partial else [4, 5, 8]
    rows = _list_all(service, query)
    assert [item.meeting.id for item in rows] == [_identity(i) for i in expected]
    assert rows[-1].meeting.started_at is None
    empty = service.search(models.SearchQuery(
        text=" \t ", include_partial=include_partial, limit=2
    ))
    assert empty == first
    empty_next = service.search(models.SearchQuery(
        text="\n", include_partial=include_partial, limit=2, cursor=empty.next_cursor
    ))
    listed_next = service.list_meetings(replace(query, cursor=first.next_cursor))
    assert empty_next == listed_next


@pytest.mark.parametrize(("missing_fts", "include_partial"), ROUTES)
def test_guarded_routes_keep_inclusive_exclusive_dates_and_exclude_nulls(
    guarded_catalog, monkeypatch, missing_fts, include_partial
):
    _, models = _api()
    data = guarded_catalog
    _barriers(data)
    if missing_fts:
        _force_literal(monkeypatch)
    query = models.SearchQuery(
        text=TOKEN, include_partial=include_partial, from_utc=DATES[6],
        to_utc=DATES[2], limit=1,
    )
    hits = _search_all(_service(data["root"]), query)
    numbers = [4, 5, 6] if include_partial else [4, 5]
    assert [hit.segment_id for hit in hits] == _expected_segments(data, numbers)
    assert all(hit.started_at is not None for hit in hits)


@pytest.mark.parametrize(("missing_fts", "include_partial"), ROUTES)
def test_new_registry_barrier_invalidates_search_but_not_listing_continuation(
    guarded_catalog, monkeypatch, missing_fts, include_partial
):
    _, models = _api()
    errors = importlib.import_module("here.memory.service")
    data = guarded_catalog
    root = data["root"]
    before = _producer_view(root)
    assert before.suppressed_meeting_ids == ()
    if missing_fts:
        _force_literal(monkeypatch)
    query = models.SearchQuery(text=TOKEN, include_partial=include_partial, limit=2)
    listing = models.MeetingQuery(include_partial=include_partial, limit=2)
    old_search = _service(root).search(query)
    old_list = _service(root).list_meetings(listing)
    assert old_search.next_cursor and old_list.next_cursor
    sql_before = _catalog_inputs(data)
    _, after = _barriers(data)
    assert _catalog_inputs(data) == sql_before
    assert after.fingerprint != before.fingerprint
    with pytest.raises(errors.InvalidInput) as caught:
        _service(root).search(replace(query, cursor=old_search.next_cursor))
    assert caught.value.reason == "corpus_changed"
    _assert_visible_search(data, missing_fts=missing_fts, include_partial=include_partial)
    remaining = _list_all(_service(root), replace(listing, cursor=old_list.next_cursor))
    expected = [4, 5, 6, 8, 9] if include_partial else [4, 5, 8]
    assert [item.meeting.id for item in remaining] == [_identity(i) for i in expected]


@pytest.mark.parametrize(("missing_fts", "include_partial"), ROUTES)
def test_alias_visibility_change_with_identical_registry_invalidates_old_search_cursor(
    guarded_catalog, monkeypatch, missing_fts, include_partial
):
    _, models = _api()
    errors = importlib.import_module("here.memory.service")
    data = guarded_catalog
    paths, before = _barriers(data)
    raw_registry = [path.read_bytes() for path in paths]
    if missing_fts:
        _force_literal(monkeypatch)
    query = models.SearchQuery(text=TOKEN, include_partial=include_partial, limit=2)
    first = _service(data["root"]).search(query)
    assert first.next_cursor
    listing = models.MeetingQuery(include_partial=include_partial, limit=2)
    listed = _service(data["root"]).list_meetings(listing)
    assert listed.next_cursor
    # No meeting/revision/index metadata changes: only actual origin-pair linkage.
    with data["repo"].transaction():
        data["library"].execute(
            "UPDATE ingest_sources SET origin_fingerprint=? WHERE meeting_id=?",
            ("c" * 64, _identity(3)),
        )
    after = _producer_view(data["root"])
    assert after.suppressed_meeting_ids == tuple(sorted(_identity(i) for i in (1, 2, 7)))
    assert after.fingerprint != before.fingerprint
    assert [path.read_bytes() for path in paths] == raw_registry
    with pytest.raises(errors.InvalidInput) as caught:
        _service(data["root"]).search(replace(query, cursor=first.next_cursor))
    assert caught.value.reason == "corpus_changed"
    hits = _search_all(_service(data["root"]), query)
    numbers = (
        [3, 4, 5, 6, 8, 9] if missing_fts and include_partial
        else [3, 4, 5, 8, 6, 9] if include_partial else [3, 4, 5, 8]
    )
    assert [hit.segment_id for hit in hits] == _expected_segments(data, numbers)
    remaining = _list_all(
        _service(data["root"]), replace(listing, cursor=listed.next_cursor)
    )
    expected_list = [6, 8, 9] if include_partial else [8]
    assert [item.meeting.id for item in remaining] == [_identity(i) for i in expected_list]


@pytest.mark.parametrize("target", ["suppressed", "nonmatching"])
@pytest.mark.parametrize("include_partial", [False, True])
def test_suppression_fingerprint_does_not_replace_actual_whole_fts_generation(
    guarded_catalog, target, include_partial
):
    _, models = _api()
    errors = importlib.import_module("here.memory.service")
    data = guarded_catalog
    _publish(data["repo"], 11, texts=("unrelated corpus document",))
    _, snapshot = _barriers(data)
    query = models.SearchQuery(text=TOKEN, include_partial=include_partial, limit=2)
    first = _service(data["root"]).search(query)
    assert first.next_cursor
    identity = _identity(1 if target == "suppressed" else 11)
    with data["repo"].transaction():
        data["library"].execute(
            "UPDATE segments_fts SET text=? WHERE meeting_id=?",
            ("unrelated changed index-only corpus length words words words", identity),
        )
    assert _producer_view(data["root"]) == snapshot
    with pytest.raises(errors.InvalidInput) as caught:
        _service(data["root"]).search(replace(query, cursor=first.next_cursor))
    assert caught.value.reason == "corpus_changed"
    _assert_visible_search(data, missing_fts=False, include_partial=include_partial)


@pytest.mark.parametrize(("missing_fts", "include_partial"), ROUTES)
def test_equivalent_real_suppression_inputs_keep_cursors_stable_across_connections(
    guarded_catalog, monkeypatch, missing_fts, include_partial
):
    _, models = _api()
    data = guarded_catalog
    _, snapshot = _barriers(data)
    assert _producer_view(data["root"]) == snapshot
    if missing_fts:
        _force_literal(monkeypatch)
    query = models.SearchQuery(text=TOKEN, include_partial=include_partial, limit=2)
    first = _service(data["root"]).search(query)
    again = _service(data["root"]).search(query)
    assert [hit.meeting_id for hit in first.items] == [_identity(4), _identity(5)]
    assert first.next_cursor == again.next_cursor
    assert first.items == again.items
    continued = _service(data["root"]).search(replace(query, cursor=first.next_cursor))
    assert continued.items
    assert {hit.segment_id for hit in first.items}.isdisjoint(
        hit.segment_id for hit in continued.items
    )


@pytest.mark.parametrize("missing_fts", [False, True])
def test_real_sole_b_reindex_preserves_complete_partial_pages_and_noop_generation(
    guarded_catalog, monkeypatch, missing_fts
):
    _, models = _api()
    backend = importlib.import_module("here.memory.service")
    data = guarded_catalog
    _barriers(data)
    if missing_fts:
        _force_literal(monkeypatch)
    memory = backend.MemoryService(data["root"])
    queries = [models.SearchQuery(text=TOKEN, include_partial=value, limit=2)
               for value in (False, True)]
    before = [_assert_visible_search(
        data, missing_fts=missing_fts, include_partial=value
    ) for value in (False, True)]
    index_sql = "SELECT * FROM segments_fts ORDER BY meeting_id,segment_id"
    index_before = [tuple(row) for row in data["library"].execute(index_sql)]
    partial_before = [tuple(row) for row in data["library"].execute(
        "SELECT r.id,s.id,s.text FROM transcript_revisions r "
        "JOIN segments s ON s.revision_id=r.id WHERE r.status='partial' ORDER BY r.id,s.id"
    )]
    # The REAL producer operation is the sole writer. No search fake/reindex stub.
    report = memory.reindex()
    assert (report.indexed_meetings, report.indexed_segments) == (
        (0, 0) if missing_fts else (3, 3)
    )
    if missing_fts:
        assert any(isinstance(warning, backend.FtsUnavailable) for warning in report.warnings)
        assert [tuple(row) for row in data["library"].execute(index_sql)] == index_before
    else:
        assert data["library"].execute(
            "SELECT count(DISTINCT meeting_id),count(*) FROM segments_fts"
        ).fetchone()[:] == (3, 3)
        assert {row[0] for row in data["library"].execute(
            "SELECT meeting_id FROM segments_fts"
        )} == {_identity(i) for i in (4, 5, 8)}
    after = [_search_all(_service(data["root"]), query) for query in queries]
    assert after == before
    assert [tuple(row) for row in data["library"].execute(
        "SELECT r.id,s.id,s.text FROM transcript_revisions r "
        "JOIN segments s ON s.revision_id=r.id WHERE r.status='partial' ORDER BY r.id,s.id"
    )] == partial_before
    cursors = [_service(data["root"]).search(query).next_cursor for query in queries]
    assert memory.reindex() == report  # Actual no-op producer rebuild, not a fake.
    for query, cursor in zip(queries, cursors, strict=True):
        fresh = _service(data["root"]).search(query)
        assert fresh.next_cursor == cursor
        continued = _service(data["root"]).search(replace(query, cursor=cursor))
        assert continued.items


def _consumer(service, models, operation):
    if operation == "capabilities":
        return service.capabilities()
    if operation == "list":
        return service.list_meetings(models.MeetingQuery(include_partial=True, limit=2))
    return service.search(models.SearchQuery(
        text=" \t" if operation == "empty" else TOKEN, include_partial=True, limit=2
    ))


@pytest.mark.parametrize("fault", ["malformed", "inaccessible", "unsafe-role"])
@pytest.mark.parametrize("operation", ["capabilities", "list", "search", "empty"])
def test_real_bad_registry_fails_m3_exposure_closed_and_closes_owned_connection(
    guarded_catalog, monkeypatch, fault, operation
):
    search, models = _api()
    database = importlib.import_module("here.memory.database")
    data = guarded_catalog
    path = _record(data["root"], _identity(1))
    if fault == "malformed":
        path.write_text("[]", encoding="utf-8")
    elif fault == "unsafe-role":
        # A directory cannot fill a final regular-file registry role. This needs
        # no symlink privilege and must not be treated as a genuinely absent file.
        path.unlink()
        path.mkdir()
    else:
        original_open = Path.open

        def denied_open(candidate, *args, **kwargs):
            if candidate == path:
                raise PermissionError("Synthetic registry file access denied")
            return original_open(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied_open)
    with pytest.raises(database.LibraryError):
        _producer_view(data["root"])
    factory = _Factory(data["root"])
    with pytest.raises(database.LibraryError):
        _consumer(search.SearchService(factory), models, operation)
    assert len(factory.connections) == 1
    assert factory.connections[0].closed
    assert not any("MATCH" in sql.upper() for sql, _, _ in factory.calls)
    assert data["library"].execute("SELECT count(*) FROM meetings").fetchone()[0] == 9


@pytest.mark.parametrize("operation", ["capabilities", "list", "search", "empty"])
def test_genuine_absence_remains_empty_and_m3_uses_one_real_read_view_and_close(
    guarded_catalog, monkeypatch, operation
):
    search, models = _api()
    backend = importlib.import_module("here.memory.service")
    database = importlib.import_module("here.memory.database")
    data = guarded_catalog
    assert not (data["root"] / ".deletions").exists()
    snapshot = _producer_view(data["root"])
    assert snapshot.suppressed_meeting_ids == ()
    assert not (data["root"] / ".deletions").exists()
    original_view = backend.catalog_read_view
    original_close = database.LibraryConnection.close
    observed = []
    closes = []

    @contextmanager
    def observed_view(library):
        with original_view(library) as real_snapshot:
            observed.append((library, real_snapshot))
            yield real_snapshot

    def observed_close(library):
        closes.append(library)
        return original_close(library)

    monkeypatch.setattr(backend, "catalog_read_view", observed_view)
    for name, binding in tuple(vars(search).items()):
        if binding is original_view:
            monkeypatch.setattr(search, name, observed_view)
    monkeypatch.setattr(database.LibraryConnection, "close", observed_close)
    factory = _Factory(data["root"])
    result = _consumer(search.SearchService(factory), models, operation)
    assert result is not None
    assert len(observed) == 1, "M3 must consume the real B-owned view, not its own BEGIN"
    assert observed[0][1] == snapshot
    assert len(factory.connections) == 1
    assert closes == [observed[0][0]]
    assert factory.connections[0].closed
    begins = [sql for sql in factory.trace if sql.strip().upper().startswith("BEGIN")]
    assert len(begins) == 1, "The B context owns the single read transaction"
    assert all(active for sql, _, active in factory.calls
               if sql.lstrip().upper().startswith(("SELECT", "WITH")))
    assert not (data["root"] / ".deletions").exists()


def test_real_guard_lock_busy_reaches_m3_as_typed_error_and_closes_factory_connection(
    guarded_catalog
):
    search, models = _api()
    database = importlib.import_module("here.memory.database")
    ownership = importlib.import_module("here.output.ownership")
    root = guarded_catalog["root"]
    assert _producer_view(root).suppressed_meeting_ids == ()
    factory = _Factory(root)
    with closing(ownership.FileLease(root / ".memory.lock")):
        with pytest.raises(database.Busy):
            search.SearchService(factory).search(models.SearchQuery(text=TOKEN, limit=2))
    assert len(factory.connections) == 1 and factory.connections[0].closed
