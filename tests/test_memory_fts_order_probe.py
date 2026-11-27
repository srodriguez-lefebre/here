"""Diagnostic RED for computing the exact v2 generation without an FTS sort."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest
from test_memory_search import _api, _Factory, _open_catalog, _publish
from test_memory_search_row_representation import _oracles, _producer_bytes, _vector

_CELLS = "meeting_id,revision_id,segment_id,text,title"
_FTS = f"SELECT {_CELLS} FROM segments_fts"
_CANONICAL = re.compile(
    r"\bfrom segments_fts order by meeting_id,revision_id,segment_id,text,title\b"
)
_NUM_ROWS = 260


def _canonical_sorts(trace):
    return [
        sql
        for sql in trace
        if _CANONICAL.search(re.sub(r"\s*,\s*", ",", " ".join(sql.lower().split())))
    ]


def _rows(library, order):
    return [tuple(row) for row in library.execute(f"{_FTS} ORDER BY {order}")]


def _add(library, row, rowid=None):
    library.execute(
        f"INSERT INTO segments_fts(rowid,{_CELLS}) VALUES (?,?,?,?,?,?)",
        (rowid, *row),
    )


def _move(library, old, new):
    row = tuple(library.execute(f"{_FTS} WHERE rowid=?", (old,)).fetchone())
    library.execute("DELETE FROM segments_fts WHERE rowid=?", (old,))
    _add(library, row, new)


def _add_malformed_text(library, rowid):
    _add(library, ("z", "rev", "seg", "zzz", "t"), rowid)
    library.execute(
        "UPDATE segments_fts_content SET c3=CAST(X'7A7AFF' AS TEXT) WHERE id=?",
        (rowid,),
    )


def test_monotone_text_uses_no_canonical_fts_sort_and_reproduces_v2_bytes(tmp_path, monkeypatch):
    library = _vector(tmp_path / "r61", "rows257")
    try:
        assert _rows(library, "rowid") == _rows(library, _CELLS)
        trace = []
        library.connection.set_trace_callback(trace.append)
        assert _producer_bytes(library, monkeypatch) == _oracles()["cases"]["rows257"]
        assert not _canonical_sorts(trace), "Monotone FTS content still used the canonical SQL sort"
    finally:
        library.close()


@pytest.fixture
def ordered_catalog(tmp_path):
    root = tmp_path / "catalog"
    library, repo = _open_catalog(root)
    try:
        _publish(repo, 82, title="Café 雪 🙂", texts=("needle 🦊 café",) * _NUM_ROWS)
        # Keep every orphan cell in the digest, including same-key title/text ties
        # and a byte-for-byte duplicate. These documents do not become page hits.
        for text, title in (("é", "a"), ("雪", "z"), ("雪", "é"), ("雪", "é"), ("🙂", "a")):
            _add(library, ("z", "rev", "seg", text, title))
        assert _rows(library, "rowid") == _rows(library, _CELLS)
        assert library.execute("SELECT count(*) FROM segments_fts").fetchone()[0] == _NUM_ROWS + 5
        yield root, library, repo
    finally:
        library.close()


@pytest.mark.parametrize("old,new", [(2, -10), (_NUM_ROWS + 2, _NUM_ROWS + 6)])
def test_rowid_reassignment_preserves_bytes_pages_and_old_cursor(
    ordered_catalog, monkeypatch, old, new
):
    root, library, repo = ordered_catalog
    search, models = _api()
    query = models.SearchQuery(text="needle", limit=3, include_partial=True)
    before = _producer_bytes(library, monkeypatch)
    first = search.SearchService(_Factory(root)).search(query)
    assert first.next_cursor
    continuation = replace(query, cursor=first.next_cursor)
    second = search.SearchService(_Factory(root)).search(continuation)
    assert second.items
    with repo.transaction():
        _move(library, old, new)
    assert _rows(library, "rowid") != _rows(library, _CELLS)
    assert _producer_bytes(library, monkeypatch) == before
    factory = _Factory(root)
    svc = search.SearchService(factory)
    assert svc.search(query) == first
    assert svc.search(continuation) == second
    assert _canonical_sorts(factory.trace)


def test_interior_inversion_after_one_batch_replays_canonical_sql(tmp_path, monkeypatch):
    from here.memory import _native_search

    # This trace assertion targets the stock Python scan even in a compiled test profile.
    monkeypatch.setattr(_native_search, "load_fts_encoder", lambda _library: False)
    library, repo = _open_catalog(tmp_path / "interior")
    try:
        _publish(repo, 83, texts=("needle 🦊 café",) * 520)
        ordered = _rows(library, "rowid")
        assert len(ordered) == 520 and ordered == _rows(library, _CELLS)
        before = _producer_bytes(library, monkeypatch)
        assert before["digest"]

        # Swap positions 258 and 259, beyond the first 256-row frame and
        # before the last 256 rows. Keep both the contents and rowid set.
        with repo.transaction():
            _move(library, 258, 521)
            _move(library, 259, 258)
            _move(library, 521, 259)
        reordered = _rows(library, "rowid")
        assert reordered == ordered[:257] + [ordered[258], ordered[257]] + ordered[259:]
        assert _rows(library, _CELLS) == ordered

        trace = []
        library.connection.set_trace_callback(trace.append)
        assert _producer_bytes(library, monkeypatch) == before
        (canonical,) = _canonical_sorts(trace)
        assert any(
            re.search(
                r"\bfrom segments_fts order by rowid(?: asc)?;?$",
                " ".join(sql.lower().split()),
            )
            for sql in trace[: trace.index(canonical)]
        ), "Canonical sort must follow the attempted ascending FTS rowid scan"
    finally:
        library.close()


@pytest.mark.parametrize("value,storage", [(2, "integer"), (1.25, "real"), (None, "null")])
def test_runtime_storage_class_uses_canonical_order_and_preserves_bytes(
    ordered_catalog, monkeypatch, value, storage
):
    _, library, repo = ordered_catalog
    # Earlier cells tie. SQLite storage order, rather than comparing Python
    # objects or turning numeric/NULL values into text, decides the fourth cell.
    _add(library, ("z", "rev", "seg", value, "é"), _NUM_ROWS + 6)
    assert library.execute(
        "SELECT typeof(text) FROM segments_fts WHERE rowid=?", (_NUM_ROWS + 6,)
    ).fetchone()[0] == storage
    trace = []
    library.connection.set_trace_callback(trace.append)
    after_text = _producer_bytes(library, monkeypatch)
    assert after_text["digest"]
    assert _canonical_sorts(trace)
    with repo.transaction():
        _move(library, _NUM_ROWS + 6, -10)
    trace.clear()
    assert _producer_bytes(library, monkeypatch) == after_text
    assert _canonical_sorts(trace)


def test_late_blob_precedes_early_rowid_decode_error_in_canonical_order(
    ordered_catalog, monkeypatch
):
    _, library, _ = ordered_catalog
    # An empty meeting ID sorts before the fixture UUIDs. The BLOB reaches
    # JSON encoding in the first 256-row frame; the malformed TEXT under "z"
    # belongs in a later frame and must not take precedence even at rowid -10.
    _add(library, ("", "rev", "seg", sqlite3.Binary(b"blob"), "t"), _NUM_ROWS + 6)
    _add_malformed_text(library, -10)
    trace = []
    library.connection.set_trace_callback(trace.append)
    packet = _producer_bytes(library, monkeypatch)
    assert packet["error"] == {
        "type": "builtins.TypeError",
        "args": ["Object of type bytes is not JSON serializable"],
    }
    assert _canonical_sorts(trace)


def test_malformed_text_without_earlier_encoding_error_replays_canonical_sql(
    ordered_catalog, monkeypatch
):
    from here.memory.database import IntegrityFailure

    _, library, _ = ordered_catalog
    _add_malformed_text(library, _NUM_ROWS + 6)
    trace = []
    library.connection.set_trace_callback(trace.append)
    with pytest.raises(IntegrityFailure) as failure:
        _producer_bytes(library, monkeypatch)
    cause = failure.value.__cause__
    assert type(cause) is sqlite3.OperationalError
    assert "Could not decode to UTF-8" in str(cause)
    assert not library.connection.in_transaction
    assert _canonical_sorts(trace)


def test_utf16_binary_order_comes_from_sqlite(tmp_path, monkeypatch):
    root = tmp_path / "utf16"
    root.mkdir()
    with closing(sqlite3.connect(root / "library.sqlite3")) as database:
        database.execute("PRAGMA encoding='UTF-16le'")
        database.execute("CREATE TABLE encoding_seed(value TEXT)")
        database.execute("DROP TABLE encoding_seed")
    library, repo = _open_catalog(root)
    try:
        assert library.execute("PRAGMA encoding").fetchone()[0] == "UTF-16le"
        for text in ("a", "Ā", "雪", "🙂"):
            _add(library, ("z", "rev", "seg", text, "t"))
        assert _rows(library, "rowid") == sorted(_rows(library, "rowid"))
        assert _rows(library, "rowid") != _rows(library, _CELLS)
        trace = []
        library.connection.set_trace_callback(trace.append)
        before = _producer_bytes(library, monkeypatch)
        assert _canonical_sorts(trace)
        with repo.transaction():
            _move(library, 1, 10)
        trace.clear()
        assert _producer_bytes(library, monkeypatch) == before
        assert _canonical_sorts(trace)
    finally:
        library.close()
