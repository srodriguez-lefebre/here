"""R75 ACTIVE dispatch, v2 bytes and same-view canonical replay."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest
from test_memory_delete import _record
from test_memory_fts_order_probe import _canonical_sorts, _move
from test_memory_search import _api, _Factory, _identity, _open_catalog, _publish
from test_memory_search_native_v2 import (
    _AGGREGATE,
    _install_real_loader,
    _legacy_cursor,
    _load,
    loader_module,
    native_dll,
)
from test_memory_search_row_representation import (
    _oracles,
    _producer_bytes,
    _public_fixture,
    _vector,
)

_ACTIVE = "here_search_v2_active_row"
_UNSUPPORTED = "here search v2: ACTIVE input unsupported"


def _row_json(*cells):
    return json.dumps(cells, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _active_statements(trace):
    return [
        sql
        for statement in trace
        if "left join segments" in (sql := " ".join(statement.lower().split()))
        and "order by m.id,r.id,s.ordinal,s.id" in re.sub(r"\s*,\s*", ",", sql)
    ]


def _assert_replay(trace):
    native, canonical = _active_statements(trace)
    assert _ACTIVE in native and _ACTIVE not in canonical
    native_index = next(
        index for index, sql in enumerate(trace) if _active_statements([sql]) == [native]
    )
    canonical_index = next(
        index for index, sql in enumerate(trace) if _active_statements([sql]) == [canonical]
    )
    assert native_index < canonical_index
    assert not any(
        sql.strip().split()[0].upper() in {"BEGIN", "COMMIT", "ROLLBACK"}
        for sql in trace[native_index:canonical_index]
    )


def test_successful_native_load_routes_active_rows_through_sqlite(
    tmp_path, loader_module, monkeypatch
):
    from here.memory import _native_search

    loaded = []
    rows = []

    def encode(*cells):
        assert library.connection.in_transaction
        rows.append(cells)
        return _row_json(*cells)

    def load(candidate):
        assert candidate.connection.in_transaction
        loaded.append(candidate)
        candidate.connection.create_function(_ACTIVE, 9, encode)
        return True

    monkeypatch.setattr(loader_module, "load_fts_encoder", load)
    # This test targets ACTIVE only. Separate real DLL controls cover FTS plus ACTIVE.
    search, _ = _api()
    monkeypatch.setattr(
        _native_search, "fts_digest", lambda candidate, digest: search._ordered_fts_digest(
            candidate, digest
        )
    )
    with closing(_vector(tmp_path, "rows257")) as library:
        assert _producer_bytes(library, monkeypatch) == _oracles()["cases"]["rows257"]
        assert loaded == [library]
        assert len(rows) == 257, "Successful native dispatch never encoded the ACTIVE rows"
        assert [row[4] for row in rows] == list(range(257))


@pytest.fixture
def active_connection(native_dll):
    with closing(sqlite3.connect(":memory:")) as connection:
        _load(connection, native_dll)
        yield connection


def test_active_row_bytes_preserve_types_and_strict_utf8(active_connection):
    cells = ("", None, "café雪🦊\u2028", -(2**63), 2**63 - 1, 0, -1, 'a\x00\x01\b\t\n\f\r"\\/', None)
    (row,) = active_connection.execute(
        f"SELECT {_ACTIVE}(?,?,?,?,?,?,?,?,?)", cells
    ).fetchone()
    assert type(row) is bytes
    assert row == _row_json(*cells)


@pytest.mark.parametrize("cell", ["1.0", "X'61'", "CAST(X'FF' AS TEXT)", "CAST(X'EDA080' AS TEXT)"])
def test_active_row_reports_only_the_documented_domain_error(active_connection, cell):
    with pytest.raises(sqlite3.DatabaseError) as rejected:
        active_connection.execute(f"SELECT {_ACTIVE}(1,2,3,4,5,{cell},7,8,9)").fetchone()
    assert rejected.value.sqlite_errorcode == sqlite3.SQLITE_MISMATCH
    assert str(rejected.value) == _UNSUPPORTED


def test_active_row_cap_counts_encoded_bytes(active_connection):
    cap = 8 * 1024 * 1024
    sql = f"SELECT {_ACTIVE}(1,2,3,4,5,?,7,8,9)"
    overhead = len(_row_json(1, 2, 3, 4, 5, "", 7, 8, 9))
    text = "a" * (cap - overhead)
    (row,) = active_connection.execute(sql, (text,)).fetchone()
    assert row == _row_json(1, 2, 3, 4, 5, text, 7, 8, 9)
    with pytest.raises(sqlite3.DatabaseError) as rejected:
        active_connection.execute(sql, (text[:-1] + "é",)).fetchone()
    assert rejected.value.sqlite_errorcode == sqlite3.SQLITE_MISMATCH
    assert str(rejected.value) == _UNSUPPORTED


def test_active_is_direct_only(active_connection):
    active_connection.execute(f"CREATE VIEW active AS SELECT {_ACTIVE}(1,2,3,4,5,6,7,8,9)")
    with pytest.raises(sqlite3.OperationalError, match="unsafe use"):
        active_connection.execute("SELECT * FROM active").fetchone()


def test_refusal_to_register_while_busy_leaves_no_partial_functions(native_dll):
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("CREATE TABLE entries(x)")
        connection.executemany("INSERT INTO entries VALUES (?)", [(1,), (2,)])
        with closing(connection.execute("SELECT x FROM entries")) as busy:
            assert busy.fetchone() == (1,)
            with pytest.raises(sqlite3.OperationalError):
                _load(connection, native_dll)
            assert busy.fetchall() == [(2,)]
        for function, arguments in [(_AGGREGATE, "1,2,3,4,5,6"), (_ACTIVE, "1,2,3,4,5,6,7,8,9")]:
            with pytest.raises(sqlite3.OperationalError, match="no such function"):
                connection.execute(f"SELECT {function}({arguments})").fetchone()
        _load(connection, native_dll)
        assert connection.execute(f"SELECT {_ACTIVE}(1,2,3,4,5,6,7,8,9)").fetchone() == (
            b"[1,2,3,4,5,6,7,8,9]",
        )


@pytest.mark.parametrize("case", ["rows255", "rows256", "rows257"])
def test_real_active_preserves_frozen_v2_frames(
    tmp_path, monkeypatch, loader_module, native_dll, case
):
    calls = _install_real_loader(monkeypatch, loader_module, native_dll)
    with closing(_vector(tmp_path, case)) as library:
        trace = []
        library.connection.set_trace_callback(trace.append)
        assert _producer_bytes(library, monkeypatch) == _oracles()["cases"][case]
        assert calls == [library]
        (active,) = _active_statements(trace)
        assert _ACTIVE in active


def test_real_active_preserves_null_extended_row(tmp_path, monkeypatch, loader_module, native_dll):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        _publish(repo, 62, texts=())
        with monkeypatch.context() as stock:
            stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
            expected = _producer_bytes(library, stock)
        calls = _install_real_loader(monkeypatch, loader_module, native_dll)
        trace = []
        library.connection.set_trace_callback(trace.append)
        assert _producer_bytes(library, monkeypatch) == expected
        assert calls == [library]
        (active,) = _active_statements(trace)
        assert _ACTIVE in active


@pytest.mark.parametrize("ordinal", [1, 256])
def test_active_stays_native_after_canonical_fts_replay(
    tmp_path, monkeypatch, loader_module, native_dll, ordinal
):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        _publish(repo, 67, texts=("same text",) * 260)
        with monkeypatch.context() as stock:
            stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
            expected = _producer_bytes(library, stock)
        with repo.transaction():
            _move(library, ordinal, -1)
            _move(library, ordinal + 1, ordinal)
            _move(library, -1, ordinal + 1)
        calls = _install_real_loader(monkeypatch, loader_module, native_dll)
        trace = []
        library.connection.set_trace_callback(trace.append)
        assert _producer_bytes(library, monkeypatch) == expected
        assert calls == [library]
        assert _canonical_sorts(trace)
        (active,) = _active_statements(trace)
        assert _ACTIVE in active


@pytest.mark.parametrize("ordinal", [0, 300])
def test_active_real_after_any_frame_replays_the_whole_section(
    tmp_path, monkeypatch, loader_module, native_dll, ordinal
):
    library, repo = _open_catalog(tmp_path)
    with closing(library):
        _publish(repo, 68, texts=("fixture segment",) * 302)
        library.execute("UPDATE segments SET start_ms=1.25 WHERE ordinal=?", (ordinal,))
        library.connection.commit()
        with monkeypatch.context() as stock:
            stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
            expected = _producer_bytes(library, stock)
        assert expected["digest"]
        calls = _install_real_loader(monkeypatch, loader_module, native_dll)
        trace = []
        library.connection.set_trace_callback(trace.append)
        assert _producer_bytes(library, monkeypatch) == expected
        assert calls == [library]
        _assert_replay(trace)


def test_active_replay_preserves_blob_precedence_over_later_invalid_utf8(
    tmp_path, monkeypatch, loader_module, native_dll
):
    from here.memory.database import IntegrityFailure

    library, repo = _open_catalog(tmp_path)
    with closing(library):
        _publish(repo, 66, texts=("fixture segment",) * 302)
        library.execute("UPDATE segments SET text=CAST(X'FF' AS TEXT) WHERE ordinal=300")
        library.connection.commit()
        with monkeypatch.context() as stock:
            stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
            with pytest.raises(IntegrityFailure):
                _producer_bytes(library, stock)
            library.execute("UPDATE segments SET text=X'01' WHERE ordinal=0")
            library.connection.commit()
            expected = _producer_bytes(library, stock)
        assert expected["error"] == _oracles()["cases"]["blob-error"]["error"]
        _install_real_loader(monkeypatch, loader_module, native_dll)
        trace = []
        library.connection.set_trace_callback(trace.append)
        assert _producer_bytes(library, monkeypatch) == expected
        _assert_replay(trace)


def test_active_unrelated_sql_error_propagates_without_replay(
    tmp_path, monkeypatch, loader_module, native_dll
):
    from here.memory.database import LibraryError

    install = _install_real_loader(monkeypatch, loader_module, native_dll)
    real_load = loader_module.load_fts_encoder

    def fail(*_cells):
        raise ValueError("Synthetic unrelated scalar error")

    def load(library):
        real_load(library)
        library.connection.create_function(_ACTIVE, 9, fail)
        return True

    monkeypatch.setattr(loader_module, "load_fts_encoder", load)
    with closing(_vector(tmp_path, "rows257")) as library:
        trace = []
        library.connection.set_trace_callback(trace.append)
        with pytest.raises((LibraryError, sqlite3.Error)):
            _producer_bytes(library, monkeypatch)
        assert install == [library]
        (active,) = _active_statements(trace)
        assert _ACTIVE in active


def test_active_public_cursors_match_stock_and_survive_noop(
    tmp_path, monkeypatch, loader_module, native_dll
):
    root = tmp_path / "public"
    memory, _, normalized = _public_fixture(root)
    search, models = _api()
    query = models.SearchQuery(text="r61needle", include_partial=True, limit=2)
    with monkeypatch.context() as stock:
        stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
        service = search.SearchService(_Factory(root))
        first = service.search(query)
        second = service.search(replace(query, cursor=first.next_cursor))
    assert first.next_cursor and second.items
    _install_real_loader(monkeypatch, loader_module, native_dll)
    factory = _Factory(root)
    service = search.SearchService(factory)
    native_first = service.search(query)
    assert native_first == first
    assert service.search(replace(query, cursor=first.next_cursor)) == second
    legacy = _legacy_cursor(search, query, first.next_cursor)
    assert service.search(replace(query, cursor=legacy)) == second
    assert memory.reindex() == normalized
    assert service.search(query) == native_first
    assert any(_ACTIVE in sql for sql in _active_statements(factory.trace))
    with monkeypatch.context() as stock:
        stock.setattr(loader_module, "load_fts_encoder", lambda _library: False)
        assert service.search(replace(query, cursor=native_first.next_cursor)) == second


def test_active_loader_refusal_and_invalid_r50_stay_outside_native_sql(
    tmp_path, monkeypatch, loader_module
):
    from here.memory.database import LibraryError

    root = tmp_path / "catalog"
    with closing(_vector(root, "rows257")):
        pass
    search, models = _api()
    calls = []

    def unavailable(library):
        assert library.connection.in_transaction
        calls.append(library)
        return False

    monkeypatch.setattr(loader_module, "load_fts_encoder", unavailable)
    factory = _Factory(root)
    service = search.SearchService(factory)
    query = models.SearchQuery(text="r61cell")
    assert service.search(query).items
    assert len(calls) == 1
    (active,) = _active_statements(factory.trace)
    assert _ACTIVE not in active

    path = _record(root, identity=_identity(9001))
    path.write_text("[]", encoding="utf-8")
    rejected = _Factory(root)
    with pytest.raises(LibraryError):
        search.SearchService(rejected).search(query)
    assert len(calls) == 1
    assert not _active_statements(rejected.trace)
