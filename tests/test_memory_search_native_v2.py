"""R75 native FTS framing, replay, public cursor and real R50 controls."""

from __future__ import annotations

import base64
import importlib
import json
import os
import sqlite3
import struct
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from test_memory_delete import _record
from test_memory_fts_order_probe import _add, _add_malformed_text, _canonical_sorts, _move, _rows
from test_memory_search import _api, _Factory, _identity, _open_catalog, _publish
from test_memory_search_row_representation import _oracles, _producer_bytes, _vector

_AGGREGATE = "here_search_v2_fts_batch"
_ENTRYPOINT = "sqlite3_here_search_v2_init"
_UNSUPPORTED = "here search v2: FTS input unsupported"
_COLUMNS = "meeting_id,revision_id,segment_id,text,title"


@pytest.fixture
def loader_module():
    try:
        return importlib.import_module("here.memory._native_search")
    except ModuleNotFoundError as error:
        if error.name != "here.memory._native_search":
            raise
        pytest.fail("R75 native FTS loader/dispatch is missing", pytrace=False)


@pytest.fixture
def native_dll():
    value = os.environ.get("HERE_TEST_SEARCH_V2_DLL")
    if value is None:
        pytest.skip("compiled R75 test DLL requires HERE_TEST_SEARCH_V2_DLL")
    dll = Path(value)
    assert dll.is_absolute() and dll.is_file()
    return dll


def _load(connection, dll):
    connection.enable_load_extension(True)
    try:
        connection.load_extension(str(dll), entrypoint=_ENTRYPOINT)
    finally:
        connection.enable_load_extension(False)


def _install_real_loader(monkeypatch, module, dll):
    calls = []

    def load(library):
        assert library.connection.in_transaction
        _load(library.connection, dll)
        calls.append(library)
        return True

    monkeypatch.setattr(module, "load_fts_encoder", load)
    return calls


def _native_statements(trace):
    return [sql for sql in trace if _AGGREGATE in sql.lower()]


def _expected_result(rows, last_rowid):
    def endpoint(row):
        cells = [value.encode("utf-8") for value in row]
        return b"".join(struct.pack(">I", len(cell)) + cell for cell in cells)

    frame = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (
        b"HV2F"
        + struct.pack(">Hq", len(rows), last_rowid)
        + endpoint(rows[0])
        + endpoint(rows[-1])
        + frame.encode("utf-8")
    )


def _aggregate(connection):
    return connection.execute(
        f"SELECT {_AGGREGATE}(rowid,a,b,c,d,e) FROM ("
        "SELECT rowid,a,b,c,d,e FROM batch ORDER BY rowid LIMIT -1)"
    ).fetchone()[0]


@pytest.fixture
def aggregate_connection(native_dll):
    with closing(sqlite3.connect(":memory:")) as connection:
        _load(connection, native_dll)
        connection.execute("CREATE TABLE batch(a, b, c, d, e)")
        yield connection


def test_empty_aggregate_returns_sql_null(aggregate_connection):
    assert _aggregate(aggregate_connection) is None


@pytest.mark.parametrize("count", [1, 255, 256])
def test_envelope_uses_exact_utf8_and_keeps_duplicate_rows(aggregate_connection, count):
    anchor = ("", "é", "雪", 'a\x00\x01\b\t\n\f\r"\\', "🙂\u2028/\x7f")
    rows = [anchor] * count
    rows[-1] = (*anchor[:-1], anchor[-1] + " suffix")
    # Exercise signed encoding, sparse rowids and the positive endpoint.
    last = 2**63 - 1 if count == 256 else -1
    rowids = [-(2**63) + index * 2 for index in range(count - 1)] + [last]
    aggregate_connection.executemany(
        "INSERT INTO batch(rowid,a,b,c,d,e) VALUES (?,?,?,?,?,?)", zip(rowids, rows)
    )
    assert _aggregate(aggregate_connection) == _expected_result(rows, last)


def _assert_unsupported(connection):
    with pytest.raises(sqlite3.DatabaseError) as rejected:
        _aggregate(connection)
    assert rejected.value.sqlite_errorcode == sqlite3.SQLITE_MISMATCH
    assert str(rejected.value) == _UNSUPPORTED


@pytest.mark.parametrize("cell", [None, 1, 1.0, -0.0, float("inf"), b"binary"])
@pytest.mark.parametrize("field", range(1, 6))
def test_aggregate_declines_non_text_in_every_field(aggregate_connection, cell, field):
    values = ["a"] * 5
    values[field - 1] = cell
    aggregate_connection.execute("INSERT INTO batch VALUES (?,?,?,?,?)", values)
    _assert_unsupported(aggregate_connection)


@pytest.mark.parametrize("column", "abcde")
@pytest.mark.parametrize("invalid", ["80", "c080", "eda080", "f4908080", "e282"])
def test_aggregate_declines_strict_utf8_violations(aggregate_connection, column, invalid):
    aggregate_connection.execute("INSERT INTO batch VALUES ('a','a','a','a','a')")
    aggregate_connection.execute(f"UPDATE batch SET {column}=CAST(X'{invalid}' AS TEXT)")
    _assert_unsupported(aggregate_connection)


@pytest.mark.parametrize("inversion", [1, 255, None])
def test_aggregate_declines_inversion_or_length_overflow(aggregate_connection, inversion):
    count = 257 if inversion is None else 256
    rows = [("a", "b", "c", f"{index:03}", "e") for index in range(count)]
    if inversion is not None:
        rows[inversion - 1], rows[inversion] = rows[inversion], rows[inversion - 1]
    aggregate_connection.executemany("INSERT INTO batch VALUES (?,?,?,?,?)", rows)
    _assert_unsupported(aggregate_connection)


@pytest.mark.parametrize("case", ["empty", "one-empty", "rows255", "rows256", "rows257"])
def test_real_producer_keeps_the_frozen_v2_stream(
    tmp_path, monkeypatch, native_dll, loader_module, case
):
    calls = _install_real_loader(monkeypatch, loader_module, native_dll)
    with closing(_vector(tmp_path / "vector", case)) as library:
        trace = []
        library.connection.set_trace_callback(trace.append)
        assert _producer_bytes(library, monkeypatch) == _oracles()["cases"][case]
        assert calls == [library]
        assert _native_statements(trace), "Native function did not run"
        assert not _canonical_sorts(trace)


def test_unavailable_loader_selects_stock_before_native_sql(tmp_path, monkeypatch):
    try:
        loader_module = importlib.import_module("here.memory._native_search")
    except ModuleNotFoundError as error:
        if error.name != "here.memory._native_search":
            raise
        pytest.fail("R75 native FTS loader/dispatch is missing", pytrace=False)
    with closing(_vector(tmp_path / "fallback", "rows257")) as library:
        trace = []
        seen = []
        library.connection.set_trace_callback(trace.append)

        def unavailable(candidate):
            assert candidate is library and candidate.connection.in_transaction
            seen.append(candidate)
            return False

        monkeypatch.setattr(loader_module, "load_fts_encoder", unavailable)
        assert _producer_bytes(library, monkeypatch) == _oracles()["cases"]["rows257"]
        assert seen == [library], "Search never made the native availability decision"
        assert not _native_statements(trace)


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "catalog"
    library, repo = _open_catalog(root)
    with closing(library):
        _publish(repo, 82, texts=("needle café 🦊",) * 520)
        _record(root, identity=_identity(9000), phase="complete")
        assert _rows(library, "rowid") == _rows(library, _COLUMNS)
        yield root, library, repo


@pytest.mark.parametrize("first", [1, 256, 258])
def test_replay_restarts_canonical_stream_in_same_snapshot(
    catalog, native_dll, loader_module, monkeypatch, first
):
    _, library, repo = catalog
    stock = _producer_bytes(library, monkeypatch)
    with repo.transaction():
        _move(library, first, -1)
        _move(library, first + 1, first)
        _move(library, -1, first + 1)
    calls = _install_real_loader(monkeypatch, loader_module, native_dll)
    trace = []
    library.connection.set_trace_callback(trace.append)
    assert _producer_bytes(library, monkeypatch) == stock
    assert calls == [library]
    (sort,) = _canonical_sorts(trace)
    attempts = _native_statements(trace)
    assert len(attempts) >= (1 if first == 1 else 2)
    assert trace.index(attempts[-1]) < trace.index(sort)
    between = trace[trace.index(attempts[0]) : trace.index(sort) + 1]
    assert not any(sql.split()[0].upper() in {"BEGIN", "COMMIT", "ROLLBACK"} for sql in between)


def test_typed_ties_keep_sqlite_order_and_canonical_frame(
    catalog, native_dll, loader_module, monkeypatch
):
    _, library, _ = catalog
    for value in [1.0, 1, -0.0, 0.0, None, "1"]:
        _add(library, ("z", "rev", "seg", value, "same"))
    with monkeypatch.context() as stock:
        stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
        expected = _producer_bytes(library, stock)
    _install_real_loader(monkeypatch, loader_module, native_dll)
    trace = []
    library.connection.set_trace_callback(trace.append)
    assert _producer_bytes(library, monkeypatch) == expected
    assert expected["digest"]
    assert _native_statements(trace) and _canonical_sorts(trace)


def test_canonical_blob_error_precedes_physically_earlier_utf8_error(
    catalog, native_dll, loader_module, monkeypatch
):
    _, library, _ = catalog
    _add_malformed_text(library, -10)
    _add(library, ("", "rev", "seg", b"blob", "title"))
    with monkeypatch.context() as stock:
        stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
        expected = _producer_bytes(library, stock)
    assert expected["error"] == {
        "type": "builtins.TypeError", "args": ["Object of type bytes is not JSON serializable"]
    }
    _install_real_loader(monkeypatch, loader_module, native_dll)
    trace = []
    library.connection.set_trace_callback(trace.append)
    assert _producer_bytes(library, monkeypatch) == expected
    assert _native_statements(trace) and _canonical_sorts(trace)


def _legacy_cursor(search, query, token):
    payload = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    assert payload[1] == 2
    payload[1] = 1
    payload[3] = search._scope(search._validated(query))
    return search._encode(payload)


def test_public_pages_and_legacy_current_cursors_round_trip_between_encoders(
    catalog, native_dll, loader_module, monkeypatch
):
    root, _, repo = catalog
    search, models = _api()
    hidden = _identity(82)
    visible = _publish(repo, 83, texts=("needle visible",) * 4)
    _record(root, identity=hidden)
    with monkeypatch.context() as stock:
        stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
        service = search.SearchService(_Factory(root))
        query = models.SearchQuery(text="needle", limit=2)
        stock_first = service.search(query)
        assert stock_first.next_cursor
        legacy = _legacy_cursor(search, query, stock_first.next_cursor)
        stock_second = service.search(replace(query, cursor=stock_first.next_cursor))
        assert stock_second.items

    calls = _install_real_loader(monkeypatch, loader_module, native_dll)
    factory = _Factory(root)
    service = search.SearchService(factory)
    native_first = service.search(query)
    assert native_first == stock_first
    assert {item.meeting_id for item in native_first.items} == {visible[0]}
    assert service.search(replace(query, cursor=stock_first.next_cursor)) == stock_second
    assert service.search(replace(query, cursor=legacy)) == stock_second
    assert len(calls) == 3
    assert _native_statements(factory.trace)
    assert not _canonical_sorts(factory.trace)
    assert all(connection.closed for connection in factory.connections)
    with monkeypatch.context() as stock:
        stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
        assert service.search(replace(query, cursor=native_first.next_cursor)) == stock_second

    # This unmatched, unattached document is outside the visible page, but
    # belongs to the FTS corpus and must still invalidate its cursor.
    _, library, _ = catalog
    _add(library, ("z", "revision", "segment", "irrelevant", "title"))
    with pytest.raises(search.InvalidInput) as changed:
        service.search(replace(query, cursor=native_first.next_cursor))
    assert changed.value.reason == "corpus_changed"


@pytest.mark.parametrize("fault", ["malformed", "inaccessible", "unsafe-role"])
def test_r50_rejects_before_loading_or_reading_search(catalog, loader_module, monkeypatch, fault):
    from here.memory.database import LibraryError

    root, _, _ = catalog
    search, models = _api()
    path = _record(root, identity=_identity(82))
    if fault == "malformed":
        path.write_text("[]", encoding="utf-8")
    elif fault == "unsafe-role":
        path.unlink()
        path.mkdir()
    else:
        open_path = Path.open

        def deny(candidate, *args, **kwargs):
            if candidate == path:
                raise PermissionError("Synthetic registry open denied")
            return open_path(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "open", deny)

    def premature_load(_library):
        pytest.fail("Native code reached before R50 validated the registry")

    monkeypatch.setattr(loader_module, "load_fts_encoder", premature_load)
    factory = _Factory(root)
    service = search.SearchService(factory)
    with pytest.raises(LibraryError):
        service.search(models.SearchQuery(text="needle"))
    assert len(factory.connections) == 1 and factory.connections[0].closed
    assert not _native_statements(factory.trace)
    assert not _canonical_sorts(factory.trace)
    assert not any(" MATCH " in sql.upper() for sql, _, _ in factory.calls)


@pytest.mark.parametrize("failure", ["invalid-envelope", "sql-error"])
def test_native_non_domain_fault_never_replays_canonical_sql(
    catalog, loader_module, monkeypatch, failure
):
    from here.memory.database import LibraryError

    root, _, _ = catalog
    search, models = _api()

    class BrokenAggregate:
        def step(self, *_cells):
            if failure == "sql-error":
                raise RuntimeError("Synthetic unrelated aggregate failure")

        def finalize(self):
            return b"HV2F"  # No count, rowid, endpoints or JSON.

    def load(library):
        assert library.connection.in_transaction
        library.connection.create_aggregate(_AGGREGATE, 6, BrokenAggregate)
        return True

    monkeypatch.setattr(loader_module, "load_fts_encoder", load)
    factory = _Factory(root)
    with pytest.raises((LibraryError, sqlite3.Error)):
        search.SearchService(factory).search(models.SearchQuery(text="needle"))
    assert _native_statements(factory.trace)
    assert not _canonical_sorts(factory.trace)
    assert len(factory.connections) == 1 and factory.connections[0].closed

