"""Recovery must reject an ambiguous committed publication token."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

from here.application.processing import SessionProcessor
from here.application.recovery import RecoveryService
from here.memory.artifacts import normalized_origin
from memory_fixtures import OTHER_MEETING_ID, make_session
from test_memory_delete import _files, _rows
from test_memory_recovery_publication import MARKER, _observe_source
from test_memory_recovery_publication import sandbox as sandbox
from test_memory_recovery_publication_review import (
    RETRY_TEXT,
    _fail_marker_cleanup,
    _processor,
    _receipt,
    _revisions,
    _seal_fingerprint,
    _sealed_intent,
)


def test_recovery_denies_ambiguous_committed_token_before_source_effects(sandbox, monkeypatch):
    memory, sessions = sandbox
    root = memory.data_dir
    directory = make_session(sessions / "legacy").directory
    identity, parent = memory.prepare_retry(directory, None)
    marker = directory / MARKER
    with _fail_marker_cleanup(monkeypatch, marker):
        _processor(memory, RETRY_TEXT).retry(directory)

    intent = _sealed_intent(directory, None)
    assert intent.revision_of == parent
    active = memory.active_revision(identity)
    assert active is not None and active != parent
    exact_receipt = _receipt(root, identity, active)
    assert exact_receipt["publication"] == {
        "publication_id": intent.publication_id,
        "revision_id": active,
        "revision_of": parent,
        "origin_path": normalized_origin(directory),
        "origin_fingerprint": _seal_fingerprint(intent),
        "source_kind": intent.source_kind,
    }

    other = make_session(sessions / "other", meeting_id=OTHER_MEETING_ID)
    result = SessionProcessor(memory_service=memory).publish_memory(other.directory)
    assert (result.status, result.meeting_id) == ("inserted", OTHER_MEETING_ID)
    other_active = memory.active_revision(OTHER_MEETING_ID)
    assert other_active is not None
    other_receipt = _receipt(root, OTHER_MEETING_ID, other_active)
    assert other_receipt["publication"]["publication_id"] != intent.publication_id
    journal = "SELECT id,meeting_id,phase,payload_json FROM pending_operations ORDER BY rowid"
    rows = _rows(root, journal)
    first_journal = next(row for row in rows if row[0] == exact_receipt["operation_id"])
    other_journal = next(row for row in rows if row[0] == other_receipt["operation_id"])
    assert first_journal[1:3] == (identity, "complete")
    assert other_journal[1:3] == (OTHER_MEETING_ID, "complete")
    assert rows.index(first_journal) < rows.index(other_journal)
    assert json.loads(first_journal[3])["publication"] == exact_receipt["publication"]
    other_payload = json.loads(other_journal[3])
    assert other_payload["publication"] == other_receipt["publication"]

    # The schema does not enforce uniqueness on the token within a journal's JSON.
    # Simulate a duplicate by changing just the token in a second committed proof.
    other_payload["publication"]["publication_id"] = intent.publication_id
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection, connection:
        changed = connection.execute(
            "UPDATE pending_operations SET payload_json=? WHERE id=?",
            (json.dumps(other_payload), other_journal[0]),
        )
        assert changed.rowcount == 1

    journals = _rows(root, journal)
    assert journals[rows.index(first_journal)] == first_journal
    changed_journal = journals[rows.index(other_journal)]
    assert changed_journal[:3] == other_journal[:3]
    assert json.loads(changed_journal[3]) == other_payload
    revisions = _revisions(root)
    meetings_sql = "SELECT id,status,active_revision_id FROM meetings ORDER BY id"
    meetings = _rows(root, meetings_sql)
    preserved = _files(directory)
    other_preserved = _files(other.directory)
    managed = _files(root / "meetings")

    with _observe_source(monkeypatch, directory) as effects:
        RecoveryService(sessions, memory_service=memory).discover()

    assert _revisions(root) == revisions
    assert _rows(root, meetings_sql) == meetings
    assert _rows(root, journal) == journals
    assert _files(root / "meetings") == managed
    assert _files(other.directory) == other_preserved
    assert (effects, _files(directory) == preserved) == ([], True)
