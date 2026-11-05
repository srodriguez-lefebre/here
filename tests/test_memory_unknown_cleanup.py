"""Reconciliation preserves unrecognized bytes while finishing valid sibling work."""

from __future__ import annotations

import importlib
import sqlite3
from contextlib import closing

from memory_fixtures import MEETING_ID, make_session


def test_unknown_file_in_abandoned_uuid_stage_survives_valid_sibling_reconciliation(tmp_path):
    root = tmp_path / "data"
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        operation = connection.execute(
            "SELECT id FROM pending_operations WHERE phase='complete'"
        ).fetchone()[0]
        active_revision = connection.execute(
            "SELECT active_revision_id FROM meetings WHERE id=?", (MEETING_ID,)
        ).fetchone()[0]
        # An interrupted completion marker after commit is genuine reconcilable work.
        connection.execute(
            "UPDATE pending_operations SET phase='prepared' WHERE id=?", (operation,)
        )
        connection.commit()
    abandoned = root / ".staging" / "33333333-3333-4333-8333-333333333333"
    abandoned.mkdir(parents=True)
    unknown = abandoned / "unrecognized-user-material.bin"
    preserved = b"synthetic unrecognized bytes: directory ownership does not authorize deletion"
    unknown.write_bytes(preserved)

    report = memory.reconcile()

    assert unknown.is_file(), "Reconciliation deleted unrecognized material in an abandoned stage"
    assert unknown.read_bytes() == preserved
    assert report.warnings, "Unknown staged material needs a reported/deferred cleanup outcome"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute(
            "SELECT phase FROM pending_operations WHERE id=?", (operation,)
        ).fetchone() == ("complete",)
        assert connection.execute("SELECT id,active_revision_id FROM meetings").fetchall() == [
            (MEETING_ID, active_revision)
        ]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    again = memory.reconcile()
    assert unknown.read_bytes() == preserved
    assert again.warnings
    assert source.audio.is_file() and source.metadata.is_file()
