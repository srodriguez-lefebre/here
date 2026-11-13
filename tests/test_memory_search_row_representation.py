"""R61 producer-byte oracles and public controls, never a copied generation codec.

Goldens must first be measured from frozen v2 under a separate explicit grant.
Their absence is a setup prerequisite, not semantic RED. CI reads durable JSON;
it never needs a historical Git checkout or regenerates its expected values.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import platform
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from test_memory_search import _api, _Factory, _identity, _open_catalog, _publish, _search_all

CASES = {
    "empty": 0,
    "one-empty": 1,
    "rows255": 255,
    "rows256": 256,
    "rows257": 257,
    "finite-float": 1,
    "blob-error": 1,
}
BASELINE_QUERY_BLOB = "f3033a16f83c093c2179bb3c3fb7d9cb0389fb37"


def _oracle_path():
    return Path(
        os.environ.get(
            "HERE_R61_ORACLE_PATH",
            str(Path(__file__).with_name("search_generation_v2_oracles.json")),
        )
    )


def _oracles():
    path = _oracle_path()
    if not path.is_file():
        pytest.fail(
            "R61 baseline oracle NOT_CAPTURED; root must measure/freeze it first", pytrace=False
        )
    packet = json.loads(path.read_text(encoding="utf-8"))
    assert packet["schema_version"] == 1
    assert packet["baseline_query_blob"] == BASELINE_QUERY_BLOB
    assert set(packet["cases"]) == set(CASES)
    return packet


def _vector(root, case):
    from test_memory_delete import _record, _view

    library, repo = _open_catalog(root)
    count = CASES[case]
    if count:
        text = "" if case == "one-empty" else 'r61cell [] , "quote" café 雪\x00 line\n'
        entities = _publish(repo, 61, title="", texts=(text,) * count)
        with repo.transaction():
            participant = _identity(9100)
            library.execute("INSERT INTO participants VALUES (?,?)", (participant, "R61 雪"))
            library.execute(
                "INSERT INTO meeting_participants VALUES (?,?,?)", (entities[0], participant, "")
            )
            if case == "finite-float":
                # Scalar-encoding vector, not a claim of measured audio timing.
                library.execute(
                    "UPDATE segments SET start_ms=?,end_ms=?,timing_source=? WHERE id=?",
                    (1.25, 9.5, "imported_explicit", entities[2][0]),
                )
            if case == "blob-error":
                library.execute(
                    "UPDATE segments SET text=? WHERE id=?",
                    (sqlite3.Binary(b"R61 unsupported BLOB"), entities[2][0]),
                )
        assert repo.get_meeting(entities[0]).title == ""
    # Nonempty REAL guard in every vector, without a temporary origin in its hash.
    raw_id = _identity(9000)
    _record(root, identity=raw_id, phase="complete")
    assert _view(root).suppressed_meeting_ids == (raw_id,)
    assert library.execute("SELECT count(*) FROM segments_fts").fetchone()[0] == count
    return library


def _producer_bytes(library, monkeypatch):
    """Forward the actual SHA256; record bytes, never construct source frames."""
    search, _ = _api()
    backend = importlib.import_module("here.memory.service")
    original_sha256 = hashlib.sha256
    observed = []

    class CapturedDigest:
        def __init__(self, data=b"", *args, **kwargs):
            self.actual = original_sha256(data, *args, **kwargs)
            self.parts = [bytes(data)]

        def update(self, data):
            self.parts.append(bytes(data))
            return self.actual.update(data)

        def hexdigest(self):
            return self.actual.hexdigest()

    def capture(data=b"", *args, **kwargs):
        digest = CapturedDigest(data, *args, **kwargs)
        observed.append(digest)
        return digest

    error = None
    result = None
    assert library.connection.row_factory is sqlite3.Row
    with backend.catalog_read_view(library) as suppression:
        # The B fingerprint is already REAL and complete before the spy exists.
        with monkeypatch.context() as recording:
            recording.setattr(search.hashlib, "sha256", capture)
            try:
                result = search.SearchService._generation(
                    library, suppression, indexed=True, partial=True
                )
            except TypeError as caught:
                error = {
                    "type": type(caught).__module__ + "." + type(caught).__qualname__,
                    "args": list(caught.args),
                }
        assert library.connection.row_factory is sqlite3.Row
        named = library.execute("SELECT 42 AS named_value").fetchone()
        assert isinstance(named, sqlite3.Row) and named["named_value"] == 42
    assert len(observed) == 1
    return {
        "stream_b64": base64.b64encode(b"".join(observed[0].parts)).decode("ascii"),
        "digest": result,
        "error": error,
        "actual_hash_hexdigest": observed[0].hexdigest(),
    }


def _public_fixture(root):
    from test_memory_delete import _record, _view

    backend = importlib.import_module("here.memory.service")
    library, repo = _open_catalog(root)
    try:
        visible = _publish(repo, 70, title="R61 public", texts=("r61needle source",) * 5)
        hidden = _publish(repo, 71, title="R61 hidden", texts=("r61needle hidden",))
        _record(root, identity=hidden[0], phase="pending")
        assert _view(root).suppressed_meeting_ids == (hidden[0],)
        assert library.execute("SELECT count(*) FROM segments_fts").fetchone()[0] == 6
    finally:
        library.close()
    memory = backend.MemoryService(root)
    normalized = memory.reindex()  # First removal of hidden FTS is not a no-op.
    assert (normalized.indexed_meetings, normalized.indexed_segments) == (1, 5)
    return memory, visible, normalized


@pytest.mark.skipif(
    os.environ.get("HERE_R61_CAPTURE_BASELINE") != "1",
    reason="One-off frozen-v2 oracle capture requires root's separate grant",
)
def test_capture_frozen_v2_oracles(tmp_path, monkeypatch, record_property):
    search, models = _api()
    source_hash = hashlib.sha256(Path(search.__file__).read_bytes()).hexdigest()
    assert source_hash == os.environ["HERE_R61_BASELINE_SOURCE_SHA256"]
    output = _oracle_path()
    assert output.is_absolute() and not output.exists(), "Never overwrite a measured oracle"
    packet = {
        "schema_version": 1,
        "baseline_query_blob": BASELINE_QUERY_BLOB,
        "source_raw_sha256": source_hash,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "cases": {},
    }
    for case in CASES:
        library = _vector(tmp_path / case, case)
        try:
            value = _producer_bytes(library, monkeypatch)
            assert (value["error"] is not None) is (case == "blob-error")
            packet["cases"][case] = value
        finally:
            library.close()
    root = tmp_path / "public"
    _, visible, _ = _public_fixture(root)
    query = models.SearchQuery(text="r61needle", include_partial=True, limit=2)
    first = search.SearchService(_Factory(root)).search(query)
    assert [item.segment_id for item in first.items] == list(visible[2][:2])
    assert first.next_cursor
    packet["public"] = {"cursor": first.next_cursor, "all_segment_ids": list(visible[2])}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(packet, destination, ensure_ascii=False, indent=2)
    record_property("oracle_path", str(output))
    record_property("oracle_raw_sha256", hashlib.sha256(output.read_bytes()).hexdigest())


@pytest.mark.parametrize("case", CASES)
def test_actual_generation_bytes_digest_errors_and_factory_equal_measured_v2(
    tmp_path, monkeypatch, case
):
    expected = _oracles()["cases"][case]
    library = _vector(tmp_path / case, case)
    try:
        assert _producer_bytes(library, monkeypatch) == expected
    finally:
        library.close()


def test_real_old_v2_cursor_fresh_connections_noop_and_named_page_detail_are_preserved(tmp_path):
    packet = _oracles()
    search, models = _api()
    root = tmp_path / "public"
    memory, visible, normalized = _public_fixture(root)
    query = models.SearchQuery(text="r61needle", include_partial=True, limit=2)
    first = search.SearchService(_Factory(root)).search(query)
    old_cursor = packet["public"]["cursor"]  # Real measured token, never reconstructed.
    assert first.next_cursor == old_cursor
    continued = search.SearchService(_Factory(root)).search(replace(query, cursor=old_cursor))
    assert [hit.segment_id for hit in continued.items] == list(visible[2][2:4])
    assert memory.reindex() == normalized  # SECOND real operation is the no-op.
    after = search.SearchService(_Factory(root)).search(query)
    assert after.next_cursor == old_cursor
    hits = _search_all(search.SearchService(_Factory(root)), query)
    assert [hit.segment_id for hit in hits] == packet["public"]["all_segment_ids"]
    assert len({hit.segment_id for hit in hits}) == 5
    listed = search.SearchService(_Factory(root)).list_meetings(models.MeetingQuery())
    assert [item.meeting.id for item in listed.items] == [visible[0]]
    detail = memory.get_meeting(visible[0])
    assert detail.meeting.title == "R61 public"
    assert [segment.id for segment in detail.segments] == list(visible[2])
