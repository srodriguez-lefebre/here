"""Catalog dates represent the measured instant using UTC offsets."""

from __future__ import annotations

import importlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest
from memory_fixtures import MEETING_ID, make_session, write_json


def _memory(root):
    return importlib.import_module("here.memory.service").MemoryService(root)


def _instant_in_utc(value, expected):
    parsed = datetime.fromisoformat(value)
    assert parsed == expected
    assert parsed.utcoffset() == timedelta(0)


def test_ingest_normalizes_aware_session_dates_to_utc_without_changing_source(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Ingest used network"))
    memory = _memory(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    metadata = json.loads(source.metadata.read_text(encoding="utf-8"))
    metadata["started_at"] = "2026-01-01T12:00:00-03:00"
    metadata["completed_at"] = "2026-01-01T12:05:02-03:00"
    write_json(source.metadata, metadata)
    original = source.metadata.read_bytes()
    assert memory.ingest_session(source.directory).status == "inserted"
    assert source.metadata.read_bytes() == original
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        started, completed = connection.execute(
            "SELECT started_at,completed_at FROM meetings WHERE id=?", (MEETING_ID,)
        ).fetchone()
    _instant_in_utc(started, datetime(2026, 1, 1, 15, 0, 0, tzinfo=timezone.utc))
    _instant_in_utc(completed, datetime(2026, 1, 1, 15, 5, 2, tzinfo=timezone.utc))


def test_capture_start_normalizes_explicit_aware_date_to_utc(tmp_path):
    root = tmp_path / "data"
    memory = _memory(root)
    try:
        memory.begin_capture(MEETING_ID, started_at="2026-01-01T12:00:00-03:00")
        with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
            started = connection.execute(
                "SELECT started_at FROM meetings WHERE id=?", (MEETING_ID,)
            ).fetchone()[0]
        _instant_in_utc(started, datetime(2026, 1, 1, 15, 0, 0, tzinfo=timezone.utc))
    finally:
        memory.finish_capture(MEETING_ID)
