"""Deserialized recovery hashes never grant credentials an artifact role."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json

CANARY = b"syntheticrecoverycredentialcanary"
SECRET_BYTES = b"SYNTHETIC_CREDENTIAL=" + CANARY + b"\n"


def _memory(root):
    return importlib.import_module("here.memory.service").MemoryService(root)


def _seed_siblings(tmp_path):
    root = tmp_path / "data"
    memory = _memory(root)
    bad = make_session(tmp_path / "source-bad", meeting_id=MEETING_ID)
    good = make_session(tmp_path / "source-good", meeting_id=OTHER_MEETING_ID)
    assert memory.ingest_session(bad.directory).status == "inserted"
    assert memory.ingest_session(good.directory).status == "inserted"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        operations = {
            row["meeting_id"]: dict(row)
            for row in connection.execute("SELECT * FROM pending_operations")
        }
    return root, memory, operations


def _guard_secret(monkeypatch, secret):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.absolute() == secret.absolute():
            pytest.fail("Recovery attempted to read credentials through a contained hash entry")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Recovery used network"))


def _assert_no_leak(root, secret):
    assert secret.read_bytes() == SECRET_BYTES
    for path in root.rglob("*"):
        if path.is_file() and path != secret:
            assert CANARY not in path.read_bytes(), f"Credentials leaked into {path.name}"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='segments_fts'").fetchone():
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM segments_fts WHERE segments_fts MATCH ?",
                    (CANARY.decode(),),
                ).fetchone()[0]
                == 0
            )


def test_prepared_operation_rejects_credential_hash_before_read_and_completes_sibling(
    tmp_path,
    monkeypatch,
):
    root, memory, operations = _seed_siblings(tmp_path)
    operation = operations[MEETING_ID]
    secret = root / operation["destination_path"] / ".env"
    secret.write_bytes(SECRET_BYTES)
    payload = json.loads(operation["payload_json"])
    payload["hashes"][".env"] = hashlib.sha256(SECRET_BYTES).hexdigest()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute("UPDATE pending_operations SET phase='prepared'")
        connection.execute(
            "UPDATE pending_operations SET payload_json=? WHERE id=?",
            (json.dumps(payload), operation["id"]),
        )
        connection.commit()
    with monkeypatch.context() as patch:
        _guard_secret(patch, secret)
        report = memory.reconcile()
    assert report.warnings
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute(
            "SELECT phase FROM pending_operations WHERE id=?", (operations[OTHER_MEETING_ID]["id"],)
        ).fetchone() == ("complete",)
        assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 2
    _assert_no_leak(root, secret)


def test_receipt_rebuild_rejects_credential_hash_before_read_and_restores_valid_sibling(
    tmp_path,
    monkeypatch,
):
    root, memory, operations = _seed_siblings(tmp_path)
    bad_directory = root / operations[MEETING_ID]["destination_path"]
    receipt = bad_directory / "catalog.json"
    snapshot = json.loads(receipt.read_text(encoding="utf-8"))
    bad_revision = snapshot["revision"]["id"]
    good_payload = json.loads(operations[OTHER_MEETING_ID]["payload_json"])
    good_revision = good_payload["revision_id"]
    secret = bad_directory / ".env"
    secret.write_bytes(SECRET_BYTES)
    snapshot["hashes"][".env"] = hashlib.sha256(SECRET_BYTES).hexdigest()
    write_json(receipt, snapshot)
    del memory
    gc.collect()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    assert root.resolve().is_relative_to(tmp_path.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    with monkeypatch.context() as patch:
        _guard_secret(patch, secret)
        report = _memory(root).reconcile()
    assert report.warnings
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute(
            "SELECT active_revision_id FROM meetings WHERE id=?", (OTHER_MEETING_ID,)
        ).fetchone() == (good_revision,)
        assert (
            connection.execute(
                "SELECT id FROM transcript_revisions WHERE id=?", (bad_revision,)
            ).fetchone()
            is None
        )
    _assert_no_leak(root, secret)


def test_forged_staging_plan_cannot_authorize_deleting_unknown_credential_file(
    tmp_path,
    monkeypatch,
):
    root, memory, operations = _seed_siblings(tmp_path)
    operation_id = "33333333-3333-4333-8333-333333333333"
    stage = root / ".staging" / operation_id
    stage.mkdir(parents=True)
    # An extension alone is not an explicit session/audio-role declaration.
    secret = stage / "unregistered-credential.wav"
    secret.write_bytes(SECRET_BYTES)
    write_json(
        stage / ".publication.json",
        {
            "schema_version": 1,
            "operation_id": operation_id,
            "hashes": {secret.name: hashlib.sha256(SECRET_BYTES).hexdigest()},
        },
    )
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute(
            "UPDATE pending_operations SET phase='prepared' WHERE id=?",
            (operations[OTHER_MEETING_ID]["id"],),
        )
        connection.commit()
    with monkeypatch.context() as patch:
        _guard_secret(patch, secret)
        report = memory.reconcile()
    assert secret.is_file(), "An untrusted registration plan authorized credential deletion"
    assert report.warnings
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute(
            "SELECT phase FROM pending_operations WHERE id=?", (operations[OTHER_MEETING_ID]["id"],)
        ).fetchone() == ("complete",)
    _assert_no_leak(root, secret)


def test_rebuild_skips_unknown_state_file_before_read_and_restores_generated_snapshots(
    tmp_path,
    monkeypatch,
):
    root, memory, operations = _seed_siblings(tmp_path)
    expected = sorted(
        (identity, json.loads(operation["payload_json"])["revision_id"])
        for identity, operation in operations.items()
    )
    states = root / "meetings" / MEETING_ID / "states"
    assert any(states.iterdir()), "Setup must retain genuine service-generated state snapshots"
    secret = states / ".env"
    secret.write_bytes(SECRET_BYTES)
    del memory
    gc.collect()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    assert root.resolve().is_relative_to(tmp_path.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    with monkeypatch.context() as patch:
        _guard_secret(patch, secret)
        _memory(root).reconcile()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert (
            connection.execute("SELECT id,active_revision_id FROM meetings ORDER BY id").fetchall()
            == expected
        )
    _assert_no_leak(root, secret)
