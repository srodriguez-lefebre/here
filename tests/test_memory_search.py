"""Independent M3 contracts over the real A catalog, never private recordings.

B4 title mutation and B5 suppression/reindex acceptance require their real handoff;
the SQL mutations here are controlled corpus fixtures, not lifecycle substitutes.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
from dataclasses import replace
from uuid import UUID

import pytest


def _api():
    try:
        search = importlib.import_module("here.memory.search")
    except ModuleNotFoundError as error:
        if error.name != "here.memory.search":
            raise
        pytest.fail("M3 feature missing: here.memory.search", pytrace=False)
    models = importlib.import_module("here.memory.models")
    assert hasattr(models, "MeetingQuery"), "Shared MeetingQuery DTO handoff is required"
    assert hasattr(models, "SearchQuery"), "Shared SearchQuery DTO handoff is required"
    return search, models


def _invalid_input():
    # Pending B4 handoff: one shared error, never a test-only competing DTO.
    return importlib.import_module("here.memory.service").InvalidInput


def _identity(number):
    return str(UUID(int=number + 10000, version=4))


class _ObservedConnection:
    """Observe unexpanded SQL; SQLite trace callbacks expand bound values."""

    def __init__(self, connection, calls, trace, after_read=None):
        self.connection = connection
        self.calls = calls
        self.closed = False
        self.after_read = after_read
        connection.set_trace_callback(trace.append)

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, sql, parameters=()):
        self.calls.append((sql, parameters, self.connection.in_transaction))
        cursor = self.connection.execute(sql, parameters)
        if self.after_read and sql.lstrip().upper().startswith(("SELECT", "WITH")):
            callback, self.after_read = self.after_read, None
            callback()
        return cursor

    def close(self):
        self.closed = True
        return self.connection.close()


class _Factory:
    """PLAN seam: a fresh operation-owned LibraryConnection per invocation."""

    def __init__(self, root, after_read=None):
        self.root = root
        self.calls = []
        self.trace = []
        self.connections = []
        self.after_read = after_read

    def __call__(self):
        database = importlib.import_module("here.memory.database")
        library = database.open_library(self.root)
        observed = _ObservedConnection(library.connection, self.calls, self.trace, self.after_read)
        self.after_read = None
        library.connection = observed
        self.connections.append(observed)
        return library


def _open_catalog(root, corpus=None):
    """Use A transactions/DTO insertion and A's actual FTS schema helper."""
    database = importlib.import_module("here.memory.database")
    repository = importlib.import_module("here.memory.repository")
    models = importlib.import_module("here.memory.models")
    service = importlib.import_module("here.memory.service").MemoryService(root)
    library = database.open_library(root)
    database.migrate(library)
    repo = repository.MemoryRepository(library)
    assert service._fts(repo), "These contracts require actual SQLite FTS5"
    if corpus is not None:
        with repo.transaction():
            for row in corpus.rows("participants"):
                library.execute(
                    "INSERT INTO participants(id,display_name) VALUES (?,?)",
                    (row["id"], row["display_name"]),
                )
            active = []
            for row in corpus.rows("meetings"):
                active.append((row["id"], row["active_revision_id"]))
                repo.insert_meeting(models.Meeting(**{**row, "active_revision_id": None}))
            for row in corpus.rows("transcript_revisions"):
                repo.insert_revision(models.TranscriptRevision(**row))
            for row in corpus.rows("segments"):
                repo.insert_segments((models.Segment(**row),))
            for table in ("meeting_participants", "speaker_assignments"):
                for row in corpus.rows(table):
                    columns = ",".join(row)
                    placeholders = ",".join("?" for _ in row)
                    library.execute(
                        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                        tuple(row.values()),
                    )
            for meeting_id, revision_id in active:
                repo.activate_revision(meeting_id, revision_id)
            # Fixture construction of the accepted derived index, not a reindex API.
            library.execute(
                "INSERT INTO segments_fts(meeting_id,revision_id,segment_id,text,title) "
                "SELECT m.id,r.id,s.id,s.text,m.title FROM meetings m "
                "JOIN transcript_revisions r ON r.id=m.active_revision_id "
                "JOIN segments s ON s.revision_id=r.id WHERE r.status='complete'"
            )
    return library, repo


def _publish(
    repo,
    number,
    *,
    date=None,
    title="Synthetic",
    texts=("needle",),
    partial=False,
    status="completed",
    timing=None,
):
    models = importlib.import_module("here.memory.models")
    meeting_id = _identity(number)
    revision_id = _identity(number + 100000)
    raw = "\n".join(texts)
    meeting = models.Meeting(
        id=meeting_id, title=title, started_at=date, status=models.MeetingStatus(status)
    )
    revision = models.TranscriptRevision(
        id=revision_id,
        meeting_id=meeting_id,
        revision_number=1,
        raw_text=raw,
        provider="synthetic",
        model="authored",
        origin="import",
        content_hash=hashlib.sha256(raw.encode()).hexdigest(),
        status="partial" if partial else "complete",
    )
    segments = tuple(
        models.Segment(
            id=_identity(number * 1000 + ordinal + 1000000),
            revision_id=revision_id,
            ordinal=ordinal,
            text=text,
            **(timing or {}),
        )
        for ordinal, text in enumerate(texts)
    )
    with repo.transaction():
        repo.insert_meeting(meeting)
        repo.insert_revision(revision)
        repo.insert_segments(segments)
        repo.activate_revision(meeting_id, revision_id)
        if not partial:
            for segment in segments:
                repo.library.execute(
                    "INSERT INTO segments_fts VALUES (?,?,?,?,?)",
                    (meeting_id, revision_id, segment.id, segment.text, title),
                )
    return meeting_id, revision_id, tuple(segment.id for segment in segments)


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    from search_fixtures import small_corpus

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    corpus = small_corpus()
    root = tmp_path / "synthetic-library"
    library, repo = _open_catalog(root, corpus)
    yield root, library, repo, corpus
    library.close()


def _service(root):
    search, _ = _api()
    return search.SearchService(_Factory(root))


def _search_all(service, query):
    items = []
    seen_cursors = set()
    for _ in range(1000):
        page = service.search(query)
        items.extend(page.items)
        if page.next_cursor is None:
            return items
        assert page.next_cursor not in seen_cursors, "Continuation must advance"
        seen_cursors.add(page.next_cursor)
        query = replace(query, cursor=page.next_cursor)
    pytest.fail("Search never exhausted its bounded synthetic corpus")


@pytest.mark.parametrize(
    ("raw", "normalized", "expression"),
    [
        ("  Café\tÁRBOL  ", "Café ÁRBOL", '"Café" "ÁRBOL"'),
        ("AI", "AI", '"AI"'),
        ('a"b', 'a"b', '"a""b"'),
        ('"quote', '"quote', '"""quote"'),
        ("NEAR OR AND NOT", "NEAR OR AND NOT", '"NEAR" "OR" "AND" "NOT"'),
        ("* ^ -x title:x col:", "* ^ -x title:x col:", '"*" "^" "-x" "title:x" "col:"'),
        ("\x00\x01  needle\n", "needle", '"needle"'),
        (" \n\t", "", ""),
    ],
)
def test_user_tokens_are_normalized_and_quoted_without_fts_operators(raw, normalized, expression):
    search, _ = _api()
    assert search.normalize_query_text(raw) == normalized
    assert search.build_fts_expression(raw) == expression


@pytest.mark.parametrize("text", ["cafe", "ÁRBOL", "AI", "NEAR", "OR", "-x", "title:x", '"quote'])
def test_actual_fts_accepts_accents_case_short_and_operator_looking_tokens(catalog, text):
    _, models = _api()
    root, _, _, corpus = catalog
    hits = _service(root).search(models.SearchQuery(text=text, limit=100)).items
    index = 54 if text in {"cafe", "ÁRBOL"} else 56 if text == '"quote' else 55
    expected = corpus.segment_id(index, 0)
    assert expected in {hit.segment_id for hit in hits}


@pytest.mark.parametrize("text", ["*", "^", "col:", '"'])
def test_punctuation_tokens_never_become_fts_syntax(catalog, text):
    # unicode61 cannot index punctuation-only words; no fictitious hit is required.
    _, models = _api()
    page = _service(catalog[0]).search(models.SearchQuery(text=text))
    assert isinstance(page.items, tuple)


def test_nonexistent_term_is_empty_not_an_fts_syntax_error(catalog):
    from search_fixtures import MISSING_TOKEN

    _, models = _api()
    assert not _service(catalog[0]).search(models.SearchQuery(text=MISSING_TOKEN)).items


def test_match_and_filters_are_bound_in_unexpanded_sql(catalog):
    search, models = _api()
    root, _, _, corpus = catalog
    factory = _Factory(root)
    service = search.SearchService(factory)
    raw = "rawsentinel' OR 1=1 -- title:x"
    service.search(
        models.SearchQuery(text=raw, participant_id=corpus.filter_case["participant_id"])
    )
    assert factory.calls, "Observe actual SQL, not the expanded SQLite trace"
    assert any("MATCH" in sql.upper() and parameters for sql, parameters, _ in factory.calls)
    for sql, _, _ in factory.calls:
        assert raw not in sql
        assert "rawsentinel" not in sql
        assert "participant' OR" not in sql
        assert " OFFSET " not in sql.upper()


def test_empty_search_uses_listing_without_empty_match(catalog):
    search, models = _api()
    factory = _Factory(catalog[0])
    service = search.SearchService(factory)
    page = service.search(models.SearchQuery(text=" \t ", limit=10))
    listed = service.list_meetings(models.MeetingQuery(limit=10))
    assert [item.meeting.id for item in page.items] == [item.meeting.id for item in listed.items]
    assert not any("MATCH" in sql.upper() for sql, _, _ in factory.calls)


def test_filters_apply_before_global_limit_for_fifty_matches_and_five_survivors(catalog):
    _, models = _api()
    root, _, _, corpus = catalog
    case = corpus.filter_case
    query = models.SearchQuery(
        **{
            key: case[key]
            for key in ("text", "from_utc", "to_utc", "statuses", "participant_id", "limit")
        }
    )
    page = _service(root).search(query)
    assert {hit.segment_id for hit in page.items} == set(case["filtered_segment_ids"])
    assert page.next_cursor is None
    listed = _service(root).list_meetings(
        models.MeetingQuery(
            from_utc=case["from_utc"],
            to_utc=case["to_utc"],
            statuses=case["statuses"],
            participant_id=case["participant_id"],
            limit=10,
        )
    )
    assert {item.meeting.id for item in listed.items} == {
        corpus.meeting_id(i) for i in range(45, 50)
    }


def test_search_never_exposes_historical_deleting_or_default_partial_hits(catalog):
    _, models = _api()
    root, _, _, corpus = catalog
    case = corpus.visibility_case
    service = _service(root)
    page = service.search(models.SearchQuery(text=case["text"]))
    assert [hit.segment_id for hit in page.items] == [case["active_complete_segment_id"]]
    partial = service.search(models.SearchQuery(text=case["text"], include_partial=True))
    assert {hit.segment_id for hit in partial.items} == {
        case["active_complete_segment_id"],
        case["partial_segment_id"],
    }
    assert {hit.segment_id for hit in partial.items if hit.partial} == {case["partial_segment_id"]}
    historical = models.SearchQuery(text=case["historical_only_text"], include_partial=True)
    deleting = models.SearchQuery(text=case["deleting_only_text"], include_partial=True)
    assert not service.search(historical).items
    assert not service.search(deleting).items


def test_listing_keyset_handles_ties_nulls_insert_delete_and_reservation(tmp_path):
    _, models = _api()
    root = tmp_path / "library"
    library, repo = _open_catalog(root)
    try:
        _publish(repo, 1, date="2026-01-03T00:00:00+00:00")
        _publish(repo, 2, date="2026-01-03T00:00:00+00:00")
        _publish(repo, 3, date="2026-01-02T00:00:00+00:00")
        _publish(repo, 4)
        with repo.transaction():
            repo.insert_meeting(
                models.Meeting(id=_identity(5), status=models.MeetingStatus.PROCESSING)
            )
            repo.insert_ingest_source(
                models.IngestSource(
                    id=_identity(900),
                    origin_path="synthetic/audio-only",
                    origin_fingerprint="a" * 64,
                    meeting_id=_identity(5),
                    revision_id=None,
                )
            )
        service = _service(root)
        first = service.list_meetings(models.MeetingQuery(limit=2))
        assert [row.meeting.id for row in first.items] == [_identity(1), _identity(2)]
        assert first.next_cursor
        # Before the boundary: must not shift later pages. After it: must be included.
        _publish(repo, 6, date="2026-01-04T00:00:00+00:00")
        _publish(repo, 7, date="2026-01-01T00:00:00+00:00")
        with repo.transaction():
            library.execute("UPDATE meetings SET status='deleting' WHERE id=?", (_identity(3),))
        second = service.list_meetings(models.MeetingQuery(limit=2, cursor=first.next_cursor))
        assert [row.meeting.id for row in second.items] == [_identity(7), _identity(4)]
        third = service.list_meetings(models.MeetingQuery(limit=2, cursor=second.next_cursor))
        assert [row.meeting.id for row in third.items] == [_identity(5)]
        assert third.next_cursor is None
        reservation = third.items[0].meeting
        assert reservation.started_at is None
        assert reservation.active_revision_id is None
        assert reservation.completed_at is None
        assert (
            library.execute(
                "SELECT count(*) FROM transcript_revisions WHERE meeting_id=?", (reservation.id,)
            ).fetchone()[0]
            == 0
        )
    finally:
        library.close()


@pytest.mark.parametrize("bound", ["from_utc", "to_utc", "both"])
def test_date_bounds_are_inclusive_exclusive_and_exclude_unknown_dates(tmp_path, bound):
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
            _publish(repo, number, date=date)
        filters = {}
        if bound in {"from_utc", "both"}:
            filters["from_utc"] = "2026-01-02T00:00:00+00:00"
        if bound in {"to_utc", "both"}:
            filters["to_utc"] = "2026-01-03T00:00:00+00:00"
        expected = {"from_utc": [4, 3], "to_utc": [3, 2], "both": [3]}[bound]
        service = _service(tmp_path)
        listed = service.list_meetings(models.MeetingQuery(**filters))
        found = service.search(models.SearchQuery(text="needle", **filters))
        assert [row.meeting.id for row in listed.items] == [_identity(i) for i in expected]
        assert [hit.meeting_id for hit in found.items] == [_identity(i) for i in expected]
    finally:
        library.close()


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5])
def test_invalid_limits_are_typed_input_errors(catalog, limit):
    _, models = _api()
    for query_type, operation, extra in (
        (models.MeetingQuery, "list_meetings", {}),
        (models.SearchQuery, "search", {"text": "needle"}),
    ):
        with pytest.raises(_invalid_input()):
            getattr(_service(catalog[0]), operation)(query_type(limit=limit, **extra))


@pytest.mark.parametrize("cursor", ["not-a-cursor", "", "e30=", "null", "[]", "\x00"])
def test_malformed_cursor_is_rejected_before_page_sql(catalog, cursor):
    search, models = _api()
    factory = _Factory(catalog[0])
    service = search.SearchService(factory)
    with pytest.raises(_invalid_input()):
        service.search(models.SearchQuery(text="synthetic", cursor=cursor))
    assert not any("MATCH" in sql.upper() for sql, _, _ in factory.calls)


def _malformed_envelope(cursor, path, value):
    """Mutate the actual private array codec, preserving unrelated real keys.

    Semantic field names keep rejection scenarios readable; positions follow the
    source-only ec39fe5 codec rather than the provisional object-envelope design.
    """
    padded = cursor + "=" * (-len(cursor) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    field_positions = {"v": 1, "kind": 2, "order": 4, "keys": 5}
    if len(path) == 1:
        payload[field_positions[path[0]]] = value
    else:
        key_positions = (
            {"rank": 0, "null": 1, "date": 2, "meeting_id": 3, "segment_id": 4}
            if payload[2] == "search"
            else {"null": 0, "date": 1, "meeting_id": 2}
        )
        payload[field_positions[path[0]]][key_positions[path[1]]] = value
    # Deliberately permit NaN/Infinity in malformed JSON numeric payloads; the
    # production decoder must reject them rather than issue SQLite page SQL.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=True).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


@pytest.mark.parametrize("operation", ["list_meetings", "search"])
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("v",), 2),
        (("v",), True),
        (("v",), "1"),
        (("v",), None),
        (("kind",), "unrecognized-kind"),
        (("kind",), 1),
        (("order",), "unrecognized-order"),
        (("order",), None),
        (("keys",), []),
        (("keys",), "not-a-key-object"),
        (("keys", "null"), True),
        (("keys", "null"), False),
        (("keys", "null"), "0"),
        (("keys", "null"), 0.0),
        (("keys", "null"), 2),
        (("keys", "null"), None),
        # Real first-page fixture cursors have non-NULL dates and null=0.
        (("keys", "null"), 1),
        (("keys", "date"), None),
        (("keys", "date"), True),
        (("keys", "date"), 20260101),
        (("keys", "date"), "not-a-date"),
        (("keys", "date"), "2026-02-30T00:00:00+00:00"),
        (("keys", "date"), "2026-01-01T00:00:00"),
        (("keys", "date"), "2026-01-01"),
        (("keys", "meeting_id"), None),
        (("keys", "meeting_id"), True),
        (("keys", "meeting_id"), []),
    ],
)
def test_private_cursor_payload_rejects_invalid_types_null_date_version_and_order(
    catalog, operation, path, value
):
    from search_fixtures import COHORT_TOKEN

    search, models = _api()
    factory = _Factory(catalog[0])
    service = search.SearchService(factory)
    query = (
        models.SearchQuery(text=COHORT_TOKEN, limit=2)
        if operation == "search"
        else models.MeetingQuery(limit=2)
    )
    first = getattr(service, operation)(query)
    assert first.next_cursor
    malformed = _malformed_envelope(first.next_cursor, path, value)
    factory.calls.clear()
    with pytest.raises(_invalid_input()):
        getattr(service, operation)(replace(query, cursor=malformed))
    assert not any(
        "MATCH" in sql.upper() or sql.lstrip().upper().startswith(("SELECT", "WITH"))
        for sql, _, _ in factory.calls
    ), "Malformed key payload must be rejected before generation/page SQL"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rank", float("nan")),
        ("rank", float("inf")),
        ("rank", float("-inf")),
        ("rank", True),
        ("rank", False),
        ("rank", "0"),
        ("rank", None),
        ("rank", []),
        ("segment_id", None),
        ("segment_id", True),
        ("segment_id", 1),
        ("segment_id", []),
    ],
)
def test_private_search_cursor_rejects_nonfinite_rank_and_wrong_key_types_before_sql(
    catalog, key, value
):
    from search_fixtures import COHORT_TOKEN

    search, models = _api()
    factory = _Factory(catalog[0])
    service = search.SearchService(factory)
    query = models.SearchQuery(text=COHORT_TOKEN, limit=2)
    first = service.search(query)
    assert first.next_cursor
    malformed = _malformed_envelope(first.next_cursor, ("keys", key), value)
    factory.calls.clear()
    with pytest.raises(_invalid_input()):
        service.search(replace(query, cursor=malformed))
    assert not any(
        "MATCH" in sql.upper() or sql.lstrip().upper().startswith(("SELECT", "WITH"))
        for sql, _, _ in factory.calls
    ), "Invalid search keys must be rejected before they reach SQLite"


@pytest.mark.parametrize(
    "changed", ["text", "statuses", "participant_id", "from_utc", "to_utc", "include_partial"]
)
def test_search_cursor_is_bound_to_normalized_query_filters_and_route(catalog, changed):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    service = _service(catalog[0])
    query = models.SearchQuery(text=COHORT_TOKEN, limit=2)
    first = service.search(query)
    assert first.next_cursor
    change = {
        "text": "another",
        "statuses": ("failed",),
        "participant_id": "other",
        "from_utc": "2025-01-01T00:00:00+00:00",
        "to_utc": "2027-01-01T00:00:00+00:00",
        "include_partial": True,
    }[changed]
    with pytest.raises(_invalid_input()):
        service.search(replace(query, cursor=first.next_cursor, **{changed: change}))


def test_fts_pages_use_actual_bm25_then_date_id_segment_order(catalog):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    root, library, _, _ = catalog
    expected = library.execute(
        "SELECT f.segment_id,bm25(segments_fts) AS rank FROM segments_fts f "
        "JOIN meetings m ON m.id=f.meeting_id "
        "WHERE segments_fts MATCH ? AND m.status != 'deleting' "
        "ORDER BY rank ASC,(m.started_at IS NULL) ASC,m.started_at DESC,m.id ASC,f.segment_id ASC",
        ('"syntheticcohortneedle"',),
    ).fetchall()
    assert len(expected) == 50
    assert all(row["rank"] < 0 for row in expected), "Actual FTS BM25, not literal rank zero"
    hits = _search_all(_service(root), models.SearchQuery(text=COHORT_TOKEN, limit=7))
    assert [hit.segment_id for hit in hits] == [row["segment_id"] for row in expected]
    assert len({hit.segment_id for hit in hits}) == 50


def test_equal_fts_ranks_continue_through_date_id_segment_ties_and_null_dates(tmp_path):
    _, models = _api()
    library, repo = _open_catalog(tmp_path)
    try:
        newest = _publish(repo, 1, date="2026-01-03T00:00:00+00:00", texts=("needle", "needle"))
        tied = _publish(repo, 2, date="2026-01-03T00:00:00+00:00")
        older = _publish(repo, 3, date="2026-01-02T00:00:00+00:00")
        null_first = _publish(repo, 4, texts=("needle", "needle"))
        null_last = _publish(repo, 5)
        ranks = library.execute(
            "SELECT bm25(segments_fts) FROM segments_fts WHERE segments_fts MATCH ?",
            ('"needle"',),
        ).fetchall()
        assert len({row[0] for row in ranks}) == 1
        hits = _search_all(_service(tmp_path), models.SearchQuery(text="needle", limit=1))
        assert [hit.segment_id for hit in hits] == [
            *newest[2],
            *tied[2],
            *older[2],
            *null_first[2],
            *null_last[2],
        ]
    finally:
        library.close()


def test_listing_cursor_is_bound_to_filters_and_cannot_be_used_for_search(catalog):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    service = _service(catalog[0])
    first = service.list_meetings(models.MeetingQuery(limit=2))
    assert first.next_cursor
    with pytest.raises(_invalid_input()):
        service.list_meetings(
            models.MeetingQuery(limit=2, statuses=("failed",), cursor=first.next_cursor)
        )
    with pytest.raises(_invalid_input()):
        service.search(models.SearchQuery(text=COHORT_TOKEN, cursor=first.next_cursor))


def test_equivalent_normalized_query_can_continue_the_same_search(catalog):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    service = _service(catalog[0])
    first = service.search(models.SearchQuery(text=COHORT_TOKEN, limit=2))
    second = service.search(
        models.SearchQuery(text=f" \t{COHORT_TOKEN}\n", limit=2, cursor=first.next_cursor)
    )
    assert len(second.items) == 2
    assert {hit.segment_id for hit in first.items}.isdisjoint(
        hit.segment_id for hit in second.items
    )


def test_hybrid_has_one_global_bm25_then_literal_null_keyset_without_index_mutation(tmp_path):
    _, models = _api()
    library, repo = _open_catalog(tmp_path)
    try:
        complete = _publish(
            repo, 1, date="2026-01-01T00:00:00+00:00", texts=("needle", "needle extra")
        )
        partial_new = _publish(
            repo,
            2,
            date="2026-01-03T00:00:00+00:00",
            partial=True,
            texts=("needle", "needle"),
        )
        partial_old = _publish(repo, 3, date="2026-01-02T00:00:00+00:00", partial=True)
        partial_null = _publish(repo, 4, partial=True)
        index_sql = "SELECT * FROM segments_fts ORDER BY segment_id"
        before = [tuple(row) for row in library.execute(index_sql)]
        hits = _search_all(
            _service(tmp_path), models.SearchQuery(text="needle", include_partial=True, limit=2)
        )
        assert {hit.segment_id for hit in hits[:2]} == set(complete[2])
        assert [hit.segment_id for hit in hits[2:]] == [
            *partial_new[2],
            *partial_old[2],
            *partial_null[2],
        ]
        assert [hit.partial for hit in hits] == [False, False, True, True, True, True]
        assert [tuple(row) for row in library.execute(index_sql)] == before
    finally:
        library.close()


def _mutate_corpus(root, corpus, mutation, target_index=60):
    """Another actual connection; nonmatching rows still affect corpus statistics."""
    database = importlib.import_module("here.memory.database")
    repository = importlib.import_module("here.memory.repository")
    library = database.open_library(root)
    repo = repository.MemoryRepository(library)
    target = corpus.meeting_id(target_index)
    try:
        if mutation == "nonmatching_insert":
            _publish(repo, 800, texts=("entirely unrelated new document",))
            return
        with repo.transaction():
            if mutation == "nonmatching_delete":
                library.execute("DELETE FROM segments_fts WHERE meeting_id=?", (target,))
                library.execute("UPDATE meetings SET active_revision_id=NULL WHERE id=?", (target,))
                library.execute("DELETE FROM meetings WHERE id=?", (target,))
            elif mutation == "indexed_title":
                library.execute(
                    "UPDATE meetings SET title='new unrelated title',"
                    "row_version=row_version+1 WHERE id=?",
                    (target,),
                )
                library.execute(
                    "UPDATE segments_fts SET title='new unrelated title' WHERE meeting_id=?",
                    (target,),
                )
            elif mutation == "date":
                library.execute("UPDATE meetings SET started_at=NULL WHERE id=?", (target,))
            elif mutation == "status":
                library.execute("UPDATE meetings SET status='deleting' WHERE id=?", (target,))
            elif mutation == "filter_relation":
                library.execute("DELETE FROM meeting_participants WHERE meeting_id=?", (target,))
            elif mutation == "active_revision":
                repo.activate_revision(target, corpus.revision_id(target_index, historical=True))
                library.execute("DELETE FROM segments_fts WHERE meeting_id=?", (target,))
                library.execute(
                    "INSERT INTO segments_fts SELECT m.id,r.id,s.id,s.text,m.title FROM meetings m "
                    "JOIN transcript_revisions r ON r.id=m.active_revision_id "
                    "JOIN segments s ON s.revision_id=r.id WHERE m.id=?",
                    (target,),
                )
            elif mutation == "index_only":
                # Deliberately change rank inputs without changing catalog metadata.
                library.execute(
                    "UPDATE segments_fts SET text='unrelated changed index length "
                    "words words words' WHERE meeting_id=?",
                    (target,),
                )
            else:
                raise AssertionError(f"Unknown controlled mutation: {mutation}")
    finally:
        library.close()


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
        "index_only",
    ],
)
@pytest.mark.parametrize("include_partial", [False, True])
def test_whole_corpus_change_rejects_old_rank_before_returning_a_page(
    catalog, mutation, include_partial
):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    root, _, _, corpus = catalog
    query = models.SearchQuery(text=COHORT_TOKEN, limit=3, include_partial=include_partial)
    service = _service(root)
    first = service.search(query)
    assert first.next_cursor
    _mutate_corpus(root, corpus, mutation)
    # A brand-new service/connection defeats process counters and data_version.
    with pytest.raises(_invalid_input()) as caught:
        _service(root).search(replace(query, cursor=first.next_cursor))
    assert caught.value.reason == "corpus_changed"
    fresh = _search_all(_service(root), query)
    assert len(fresh) == 50
    assert len({hit.segment_id for hit in fresh}) == 50


def test_filtered_out_fts_document_still_invalidates_rank_generation(catalog):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    root, _, _, corpus = catalog
    query = models.SearchQuery(text=COHORT_TOKEN, statuses=("completed",), limit=2)
    first = _service(root).search(query)
    assert first.next_cursor
    # Meeting 40 is failed and excluded by this query, but remains in FTS BM25.
    _mutate_corpus(root, corpus, "index_only", target_index=40)
    with pytest.raises(_invalid_input()) as caught:
        _service(root).search(replace(query, cursor=first.next_cursor))
    assert caught.value.reason == "corpus_changed"


def test_selected_partial_branch_mutation_invalidates_the_hybrid_generation(catalog):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    root, _, _, corpus = catalog
    query = models.SearchQuery(text=COHORT_TOKEN, include_partial=True, limit=2)
    first = _service(root).search(query)
    # Partial 51 is not in persistent FTS and does not match this query.
    _mutate_corpus(root, corpus, "indexed_title", target_index=51)
    with pytest.raises(_invalid_input()) as caught:
        _service(root).search(replace(query, cursor=first.next_cursor))
    assert caught.value.reason == "corpus_changed"


def test_identical_corpus_generation_is_reproducible_across_fresh_connections(catalog):
    from search_fixtures import COHORT_TOKEN

    _, models = _api()
    query = models.SearchQuery(text=COHORT_TOKEN, limit=3)
    first = _service(catalog[0]).search(query)
    again = _service(catalog[0]).search(query)
    assert first.next_cursor == again.next_cursor
    continuation = _service(catalog[0]).search(replace(query, cursor=first.next_cursor))
    assert len(continuation.items) == 3
    assert {hit.segment_id for hit in first.items}.isdisjoint(
        hit.segment_id for hit in continuation.items
    )


def test_generation_and_page_share_one_explicit_short_read_snapshot(catalog):
    search, models = _api()
    factory = _Factory(catalog[0])
    page = search.SearchService(factory).search(models.SearchQuery(text="synthetic", limit=2))
    assert page.items
    selects = [
        (sql, active)
        for sql, _, active in factory.calls
        if sql.lstrip().upper().startswith(("SELECT", "WITH"))
    ]
    assert selects
    assert all(active for _, active in selects), "Generation/page reads require explicit snapshot"
    assert any(statement.strip().upper().startswith("BEGIN") for statement in factory.trace)
    assert all(connection.closed for connection in factory.connections)


def test_concurrent_commit_between_generation_and_fetch_cannot_mix_snapshots(catalog):
    from search_fixtures import COHORT_TOKEN

    search, models = _api()
    root, _, _, corpus = catalog
    query = models.SearchQuery(text=COHORT_TOKEN, limit=3)
    before = _service(root).search(query)
    committed = []

    def writer_after_first_read():
        _mutate_corpus(root, corpus, "nonmatching_insert")
        committed.append(True)

    factory = _Factory(root, after_read=writer_after_first_read)
    during = search.SearchService(factory).search(query)
    assert committed == [True]
    assert [hit.segment_id for hit in during.items] == [hit.segment_id for hit in before.items]
    assert during.next_cursor == before.next_cursor
    with pytest.raises(_invalid_input()) as caught:
        _service(root).search(replace(query, cursor=during.next_cursor))
    assert caught.value.reason == "corpus_changed"


def test_snippets_preserve_plain_text_and_only_measured_or_absent_timing(tmp_path):
    _, models = _api()
    library, repo = _open_catalog(tmp_path)
    try:
        measured = _publish(
            repo,
            1,
            texts=("<script>needle</script> literal evidence",),
            timing={"start_ms": 0, "end_ms": 1234, "timing_source": "provider"},
        )
        absent = _publish(repo, 2, partial=True, texts=("x" * 200 + " needle " + "y" * 200,))
        hits = _search_all(
            _service(tmp_path), models.SearchQuery(text="needle", include_partial=True)
        )
        by_id = {hit.segment_id: hit for hit in hits}
        timed = by_id[measured[2][0]]
        assert (timed.start_ms, timed.end_ms, timed.timing_source) == (0, 1234, "provider")
        untimed = by_id[absent[2][0]]
        assert (untimed.start_ms, untimed.end_ms, untimed.timing_source) == (None, None, "absent")
        assert len(untimed.snippet) <= 160
        assert "needle" in untimed.snippet
        assert untimed.snippet in "x" * 200 + " needle " + "y" * 200
        assert "<script>" in timed.snippet, "Plain source text must not become markup"
    finally:
        library.close()


def test_fts_snippet_is_character_bounded_even_for_one_very_long_token(tmp_path):
    _, models = _api()
    library, repo = _open_catalog(tmp_path)
    try:
        _publish(repo, 1, texts=("z" * 3000 + " needle original",))
        page = _service(tmp_path).search(models.SearchQuery(text="needle"))
        assert len(page.items) == 1
        assert 0 < len(page.items[0].snippet) <= 160
        assert "needle" in page.items[0].snippet
    finally:
        library.close()
