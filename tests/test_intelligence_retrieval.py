"""R79 retrieval controls over a real, temporary SQLite catalog."""

from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import replace
from importlib import import_module
from uuid import UUID

import pytest
from fixtures.intelligence_corpus import (
    MATCH_COUNT,
    MEETING_COUNT,
    OR_TERMS,
    SELECTED_MATCH_COUNT,
    SPARSE_MATCH_COUNT,
    make_evidence_window_corpus,
    make_intelligence_corpus,
)
from test_memory_search import _Factory, _identity, _open_catalog, _publish
from test_memory_search_fallback import _force_literal
from test_memory_search_guarded import _record

from here.memory import models
from here.memory.database import LibraryError, open_library
from here.memory.search import SearchService
from here.memory.service import InvalidInput


def _query(**changes):
    assert hasattr(models, "EvidenceQuery"), "M4 evidence query boundary is missing"
    return models.EvidenceQuery(
        **{
            "purpose": "match",
            "terms": OR_TERMS,
            "meeting_ids": (),
            "whole_library": True,
            "per_meeting": 6,
            "total": 24,
            "evidence_chars": 24_000,
            **changes,
        }
    )


def _scoped(*meeting_ids, **changes):
    return _query(meeting_ids=meeting_ids, whole_library=False, **changes)


def _segment_id(meeting_id, ordinal):
    # Ascending IDs agree with transcript order for the neutral-rank window oracle.
    return str(UUID(int=UUID(meeting_id).int + ((ordinal + 1) << 48)))


def _load(root, meetings):
    library, repo = _open_catalog(root)
    with closing(library), repo.transaction():
        for source in meetings:
            revision_id = str(UUID(int=UUID(source.meeting_id).int + (1 << 112)))
            repo.insert_meeting(
                models.Meeting(
                    id=source.meeting_id,
                    title=source.title,
                    status=models.MeetingStatus.COMPLETED,
                )
            )
            repo.insert_revision(
                models.TranscriptRevision(
                    id=revision_id,
                    meeting_id=source.meeting_id,
                    revision_number=1,
                    raw_text=source.transcript_text,
                    provider="synthetic",
                    model="authored",
                    origin="import",
                    content_hash=hashlib.sha256(source.transcript_text.encode()).hexdigest(),
                )
            )
            segments = tuple(
                models.Segment(
                    id=_segment_id(source.meeting_id, ordinal),
                    revision_id=revision_id,
                    ordinal=ordinal,
                    text=text,
                )
                for ordinal, text in enumerate(source.passages)
            )
            repo.insert_segments(segments)
            repo.activate_revision(source.meeting_id, revision_id)
            for segment in segments:
                library.execute(
                    "INSERT INTO segments_fts VALUES (?,?,?,?,?)",
                    (source.meeting_id, revision_id, segment.id, segment.text, source.title),
                )


def _ids(snapshot):
    return [block.segment_id for block in snapshot.blocks]


def _global_bm25_order(root, members, per_meeting):
    # Compute BM25 with every fixture row still indexed. Filtering in Python
    # here is an oracle for relative ranking, never a product candidate query.
    with closing(open_library(root)) as library:
        ranked = list(
            library.execute(
                "SELECT meeting_id,segment_id,bm25(segments_fts) AS score "
                "FROM segments_fts WHERE segments_fts MATCH ? "
                "ORDER BY score,meeting_id,segment_id",
                ('"cobalto" OR "azafrán"',),
            )
        )
    assert len(ranked) == MATCH_COUNT
    meeting_ranks = defaultdict(int)
    rounds = [[] for _ in range(per_meeting)]
    for meeting_id, segment_id, _score in ranked:
        if meeting_id in members:
            index = meeting_ranks[meeting_id]
            meeting_ranks[meeting_id] += 1
            if index < per_meeting:
                rounds[index].append(segment_id)
    return [segment for current_round in rounds for segment in current_round]


@pytest.fixture(scope="module")
def loaded_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("grounded-budget")
    meetings = make_intelligence_corpus()
    _load(root, meetings)
    return root, meetings


@pytest.mark.parametrize(
    ("offset", "match_count"),
    [(0, MATCH_COUNT), (1, SELECTED_MATCH_COUNT), (2, SPARSE_MATCH_COUNT)],
)
def test_corpus_scope_and_ranked_rounds(loaded_corpus, offset, match_count):
    root, meetings = loaded_corpus
    assert len(meetings) == MEETING_COUNT
    members = tuple(meeting.meeting_id for meeting in meetings[offset:])
    assert sum(
        any(term in text for term in OR_TERMS)
        for meeting in meetings[offset:]
        for text in meeting.passages
    ) == match_count
    expected = _global_bm25_order(root, members, per_meeting=6)
    factory = _Factory(root)
    search = SearchService(factory)
    query = _scoped(*members, evidence_chars=72_000) if offset else _query(evidence_chars=72_000)

    first = search.evidence_snapshot(query)
    second = search.evidence_snapshot(query)

    assert _ids(first) == _ids(second) == expected[:24]
    assert len(set(_ids(first))) == 24
    counts = Counter(block.meeting_id for block in first.blocks)
    assert len(counts) == 24 and set(counts) <= set(members)
    assert max(counts.values()) <= 6
    assert sum(len(block.text) for block in first.blocks) <= query.evidence_chars
    assert first.truncated is True
    assert first.degraded is False
    assert len(factory.connections) == 2 and all(conn.closed for conn in factory.connections)
    match_calls = [
        (sql, parameters, in_transaction)
        for sql, parameters, in_transaction in factory.calls
        if "MATCH" in sql.upper()
    ]
    assert match_calls
    assert any(
        any(
            isinstance(value, str) and '"cobalto" OR "azafrán"' in value
            for value in parameters
        )
        and "?" in sql
        and in_transaction
        for sql, parameters, in_transaction in match_calls
    )


@pytest.mark.parametrize("fitting_ordinal", [3, 4])
def test_metadata_window_and_nul_preflight(tmp_path, monkeypatch, fitting_ordinal):
    window, _ = make_evidence_window_corpus(oversized_before=fitting_ordinal)
    _load(tmp_path, (window,))
    _force_literal(monkeypatch)
    observed = []

    def factory():
        library = open_library(tmp_path)
        original = library.connection.row_factory

        def row_factory(cursor, row):
            observed.append(row)
            return original(cursor, row)

        library.connection.row_factory = row_factory
        return library

    query = _scoped(
        window.meeting_id, terms=("cobalto",), per_meeting=1, total=1, evidence_chars=32
    )
    result = SearchService(factory).evidence_snapshot(query)

    fitting = window.passages[fitting_ordinal]
    assert [block.text for block in result.blocks] == ([fitting] if fitting_ordinal == 3 else [])
    assert result.truncated is True
    assert result.degraded is True
    oversized = {text for text in window.passages if len(text) > query.evidence_chars}
    assert not any(cell in oversized for row in observed for cell in row if isinstance(cell, str))
    segment_ids = {
        _segment_id(window.meeting_id, ordinal) for ordinal in range(len(window.passages))
    }
    seen_ids = {cell for row in observed for cell in row if isinstance(cell, str)}
    assert _segment_id(window.meeting_id, 0) in seen_ids
    assert len(segment_ids & seen_ids) <= 4 * query.total + 1
    assert sum(cell == fitting for row in observed for cell in row) == int(fitting_ordinal == 3)


@pytest.mark.parametrize("limit", ["characters", "per_meeting", "total"])
def test_adjacent_source_is_deduplicated_and_charged(tmp_path, limit):
    _, source = make_evidence_window_corpus(oversized_before=0)
    _load(tmp_path, (source,))
    totals = {
        "characters": (3, 3),
        "per_meeting": (2, 3),
        "total": (3, 2),
    }
    per_meeting, total = totals[limit]
    full_chars = sum(map(len, source.passages))
    query = _scoped(
        source.meeting_id, per_meeting=per_meeting, total=total, evidence_chars=full_chars
    )
    search = SearchService(_Factory(tmp_path))
    included = search.evidence_snapshot(query)
    expected = set(source.passages if limit == "characters" else source.passages[::2])
    assert len(included.blocks) == len(expected)
    assert {block.text for block in included.blocks} == expected
    assert included.truncated is (limit != "characters")

    tight = search.evidence_snapshot(replace(query, evidence_chars=full_chars - 1))
    assert len(tight.blocks) == 2
    assert {block.text for block in tight.blocks} == set(source.passages[::2])
    assert tight.truncated is True


@pytest.mark.parametrize("missing_fts", [False, True])
def test_partial_opt_in_literal_or_and_neutral_ties(tmp_path, monkeypatch, missing_fts):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        complete = _publish(repo, 20, texts=("cobalto complete",))
        partial = _publish(
            repo, 21, date="2026-02-02T00:00:00+00:00", texts=("azafrán parcial 🧭",), partial=True
        )
        same_date = _publish(
            repo, 22, date="2026-02-02T00:00:00+00:00", texts=("literal % azafrán",), partial=True
        )
        _publish(repo, 23, partial=True, texts=("unrelated",))
    if missing_fts:
        _force_literal(monkeypatch)
    search = SearchService(_Factory(tmp_path))
    query = _query()
    assert _ids(search.evidence_snapshot(query)) == list(complete[2])

    result = search.evidence_snapshot(replace(query, include_partial=True))
    ranked = (partial, same_date, complete) if missing_fts else (complete, partial, same_date)
    assert _ids(result) == [record[2][0] for record in ranked]
    assert result.degraded is missing_fts
    ranks = {block.meeting_id: block.rank for block in result.blocks}
    assert ranks[partial[0]] == ranks[same_date[0]] == 0
    if missing_fts:
        assert ranks[complete[0]] == 0
    else:
        assert ranks[complete[0]] < 0
    assert [block.partial for block in result.blocks] == [item != complete for item in ranked]
    actual = {block.meeting_id: block.text for block in result.blocks}
    assert actual[partial[0]] == "azafrán parcial 🧭"

    escaped = search.evidence_snapshot(replace(query, terms=("%", "_"), include_partial=True))
    assert _ids(escaped) == list(same_date[2])


def test_empty_match_never_reads_segments_or_expands_scope(tmp_path):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        meeting, _, _ = _publish(repo, 31, texts=("mere selection is not evidence",))
    factory = _Factory(tmp_path)
    search = SearchService(factory)
    result = search.evidence_snapshot(_scoped(meeting, terms=()))
    assert not result.blocks
    assert result.truncated is False
    assert not any("FROM SEGMENTS" in sql.upper() for sql, _, _ in factory.calls)


def _change_source(repo, case, meeting, revision, segment):
    library = repo.library
    with repo.transaction():
        if case == "text":
            library.execute("UPDATE segments SET text='changed source' WHERE id=?", (segment,))
        elif case == "timing":
            library.execute("UPDATE segments SET end_ms=11 WHERE id=?", (segment,))
        elif case == "hash":
            library.execute(
                "UPDATE transcript_revisions SET content_hash=? WHERE id=?", ("b" * 64, revision)
            )
        elif case == "absent":
            library.execute("DELETE FROM segments WHERE id=?", (segment,))
        else:
            raw = "cobalto nueva revisión"
            new_revision = _identity(777)
            previous = repo.get_revision(revision)
            assert previous is not None
            repo.insert_revision(
                replace(
                    previous,
                    id=new_revision,
                    revision_number=2,
                    raw_text=raw,
                    content_hash=hashlib.sha256(raw.encode()).hexdigest(),
                )
            )
            repo.activate_revision(meeting, new_revision)


@pytest.mark.parametrize(
    "case", ["text", "timing", "hash", "absent", "revision", "pending", "complete"]
)
def test_revalidation_uses_a_new_guarded_view(tmp_path, case):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        meeting, revision, segments = _publish(
            repo,
            32,
            texts=("cobalto 🌱",),
            timing={"start_ms": 0, "end_ms": 10, "timing_source": "imported_explicit"},
        )
        factory = _Factory(tmp_path)
        search = SearchService(factory)
        snapshot = search.evidence_snapshot(_scoped(meeting))
        assert _ids(snapshot) == list(segments)
        assert snapshot.blocks[0].start_ms == 0 and snapshot.blocks[0].end_ms == 10
        assert search.revalidate_evidence(snapshot) is True
        if case in ("pending", "complete"):
            _record(tmp_path, meeting, phase=case)
            assert not search.evidence_snapshot(_scoped(meeting)).blocks
        else:
            _change_source(repo, case, meeting, revision, segments[0])
        assert search.revalidate_evidence(snapshot) is False
        assert len(factory.connections) >= 3
        assert all(connection.closed for connection in factory.connections)


def test_summary_keyset_is_bound_to_the_selected_revision(tmp_path):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        meeting, revision, segments = _publish(repo, 33, texts=("uno", "dos", "tres"))
        search = SearchService(_Factory(tmp_path))
        query = _scoped(meeting, purpose="summary", terms=(), total=1, per_meeting=1)
        first = search.evidence_snapshot(query)
        assert _ids(first) == [segments[0]]
        assert first.revision_id == revision
        assert first.next_ordinal == 0
        continuation = replace(query, after_ordinal=first.next_ordinal, revision_id=revision)
        second = search.evidence_snapshot(continuation)
        assert _ids(second) == [segments[1]]
        assert second.next_ordinal == 1
        third = search.evidence_snapshot(replace(continuation, after_ordinal=second.next_ordinal))
        assert _ids(third) == [segments[2]]
        assert third.next_ordinal is None
        _change_source(repo, "revision", meeting, revision, segments[0])
        with pytest.raises(InvalidInput) as error:
            search.evidence_snapshot(continuation)
        assert error.value.reason == "corpus_changed"

def test_invalid_registry_fails_closed_for_both_reads(tmp_path):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        meeting, _, _ = _publish(repo, 34, texts=("cobalto", "contexto"))
        search = SearchService(_Factory(tmp_path))
        query = _scoped(meeting)
        snapshot = search.evidence_snapshot(query)
        broken = _record(tmp_path, meeting)
        broken.write_text("{broken json", encoding="utf-8")
        with pytest.raises(LibraryError):
            search.evidence_snapshot(query)
        with pytest.raises(LibraryError):
            search.revalidate_evidence(snapshot)


def _retriever_api(search):
    try:
        requests = import_module("here.intelligence.models")
        retrieval = import_module("here.intelligence.retrieval")
    except ModuleNotFoundError as error:
        if error.name not in (
            "here.intelligence",
            "here.intelligence.models",
            "here.intelligence.retrieval",
        ):
            raise
        pytest.fail("M4 intelligence retrieval boundary is missing", pytrace=False)
    return retrieval.Retriever(search), requests.QuestionRequest


def test_retriever_accounts_for_actual_default_budget_evidence(loaded_corpus):
    root, meetings = loaded_corpus
    sparse = meetings[2:]
    members = tuple(meeting.meeting_id for meeting in sparse)
    search = SearchService(_Factory(root))
    retriever, question = _retriever_api(search)

    result = retriever.retrieve(
        question(question="¿Qué hay de cobalto o azafrán?", meeting_ids=members)
    )

    counts = Counter(block.meeting_id for block in result.blocks)
    assert 1 < len(counts) <= len(members) and set(counts) <= set(members)
    assert max(counts.values()) <= 6
    assert len(result.blocks) <= 24
    assert sum(len(block.text) for block in result.blocks) <= 24_000
    assert result.coverage.segments == len(result.blocks)
    assert result.coverage.meetings == len(counts)
    assert result.coverage.truncated is True
    assert result.coverage.degraded is False


def test_retriever_drops_stopwords_and_short_terms(tmp_path):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        selected = _publish(repo, 41, texts=("cobalto",))
        noise = _publish(repo, 42, texts=("the and que una xy",))
        factory = _Factory(tmp_path)
        retriever, question = _retriever_api(SearchService(factory))
        scope = {"meeting_ids": (selected[0], noise[0])}

        assert _ids(retriever.retrieve(question(question="the and que una xy", **scope))) == []
        assert _ids(
            retriever.retrieve(question(question="the and que una xy cobalto", **scope))
        ) == list(selected[2])


def test_r50_origin_alias_is_excluded_before_candidate_and_neighbor_reads(tmp_path):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        hidden = _publish(repo, 51, texts=("azafrán", "contexto reservado"))
        visible = _publish(repo, 52, texts=("cobalto",))
        origin = os.path.normcase(os.path.normpath(str(tmp_path / "synthetic-origin")))
        fingerprint = "a" * 64
        with repo.transaction():
            repo.insert_ingest_source(
                models.IngestSource(
                    id=_identity(53),
                    origin_path=origin,
                    origin_fingerprint=fingerprint,
                    meeting_id=hidden[0],
                    revision_id=hidden[1],
                )
            )
        search = SearchService(_Factory(tmp_path))
        original = search.evidence_snapshot(_scoped(hidden[0]))
        assert {block.text for block in original.blocks} == {"azafrán", "contexto reservado"}

        _record(
            tmp_path,
            _identity(54),
            origins=({"origin_path": origin, "origin_fingerprint": fingerprint},),
        )

        assert search.revalidate_evidence(original) is False
        query = _scoped(hidden[0], visible[0], per_meeting=1, total=1)
        assert _ids(search.evidence_snapshot(query)) == list(visible[2])