"""R54 journal aliases must not grant private pre-journal cleanup authority."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from uuid import uuid4

import pytest
from here.memory.database import Busy
from here.memory.service import MemoryService
from here.output.ownership import FileLease
from memory_fixtures import MEETING_ID, make_session, write_json


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


@pytest.fixture(autouse=True)
def synthetic_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("Local staging reconciliation attempted provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)


@pytest.mark.parametrize("reference_column", ["staging_path", "destination_path"])
@pytest.mark.parametrize("spelling", ["dot", "windows-case"])
def test_lexical_journal_reference_preserves_live_owned_stage_and_cleans_true_orphan(
    tmp_path, reference_column, spelling
):
    if spelling == "windows-case" and os.name != "nt":
        pytest.skip("Windows case-equivalent staging path")

    root = tmp_path / "data"
    memory = MemoryService(root, sessions_root=tmp_path / "sessions")
    source = make_session(tmp_path / "origin", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    operation_id, stage_id, orphan_id, revision_id = (str(uuid4()) for _ in range(4))

    def private_stage(identity, text):
        stage = root / ".staging" / identity
        stage.mkdir(parents=True)
        copies = {"transcript.txt": text, "raw_transcript.txt": text}
        hashes = {name: hashlib.sha256(content).hexdigest() for name, content in copies.items()}
        for name, content in copies.items():
            (stage / name).write_bytes(content)
        write_json(
            stage / ".publication.json",
            {
                "schema_version": 1,
                "operation_id": identity,
                "hashes": hashes,
            },
        )
        return stage, hashes

    stage, hashes = private_stage(stage_id, b"Synthetic referenced private publication copy")
    orphan, _ = private_stage(orphan_id, b"Synthetic unreferenced private publication copy")
    before = {path.name: path.read_bytes() for path in stage.iterdir()}
    alias = f".staging/./{stage_id}" if spelling == "dot" else f".STAGING/{stage_id}"
    assert (root / alias).samefile(stage)
    staging_path = alias if reference_column == "staging_path" else f".staging/{operation_id}"
    destination_path = (
        alias
        if reference_column == "destination_path"
        else f"meetings/{MEETING_ID}/revisions/{revision_id}"
    )
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute(
            "INSERT INTO pending_operations "
            "(id,kind,phase,meeting_id,staging_path,destination_path,payload_json) "
            "VALUES (?,'publish_revision','prepared',?,?,?,?)",
            (
                operation_id,
                MEETING_ID,
                staging_path,
                destination_path,
                json.dumps({"revision_id": revision_id, "hashes": hashes}),
            ),
        )
        connection.commit()
    owner_path = root / ".owners" / f"{MEETING_ID}.lock"
    owner_path.parent.mkdir(exist_ok=True)
    # Identified-stage guard with a real canonical OS owner, not a fabricated
    # native producer or authority inferred from the stage operation UUID.
    with closing(FileLease(owner_path, timeout=0)):
        with pytest.raises(Busy):
            MemoryService.check_owner(root, MEETING_ID)
        report = memory.reconcile()
        with pytest.raises(Busy):
            MemoryService.check_owner(root, MEETING_ID)
        assert not orphan.exists(), "Genuine private pre-journal sibling must still be cleaned"
        assert _rows(
            root,
            "SELECT phase,staging_path,destination_path FROM pending_operations WHERE id=?",
            (operation_id,),
        ) == [("prepared", staging_path, destination_path)]
        skipped = [item for item in report.skipped if item.identity == operation_id]
        assert len(skipped) == 1 and isinstance(skipped[0].reason, str) and skipped[0].reason
        assert all(item.identity != operation_id for item in (*report.completed, *report.reverted))
        assert stage.is_dir(), "Lexical journal alias bypassed the live owner's stage protection"
        assert {path.name: path.read_bytes() for path in stage.iterdir()} == before
