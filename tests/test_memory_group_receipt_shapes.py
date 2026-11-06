"""Whole-group GR2: malformed receipt shapes cannot abort healthy sibling rebuild."""

from __future__ import annotations

import gc
import importlib
import json
import sqlite3
from contextlib import closing

import pytest
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json
from typer.testing import CliRunner


def _tree(directory):
    return {
        str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("shape", ["array", "null", "string", "number", "hashes-array"])
def test_malformed_receipt_is_warned_and_retained_while_good_sibling_rebuilds(
    tmp_path, monkeypatch, shape
):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Receipt maintenance allocated provider/network transport")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    module = importlib.import_module("here.memory.service")
    memory = module.MemoryService(root)
    bad = make_session(tmp_path / "sources" / "bad", meeting_id=MEETING_ID)
    good = make_session(tmp_path / "sources" / "good", meeting_id=OTHER_MEETING_ID)
    assert memory.ingest_session(bad.directory).status == "inserted"
    assert memory.ingest_session(good.directory).status == "inserted"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        good_revisions = connection.execute(
            "SELECT id,meeting_id,revision_number,raw_text,content_hash "
            "FROM transcript_revisions WHERE meeting_id=?",
            (OTHER_MEETING_ID,),
        ).fetchall()
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    bad_directory = root / "meetings" / MEETING_ID
    (receipt,) = (bad_directory / "revisions").glob("*/catalog.json")
    document = json.loads(receipt.read_text(encoding="utf-8"))
    if shape == "hashes-array":
        document["hashes"] = []
    else:
        document = {"array": [], "null": None, "string": "synthetic invalid receipt", "number": 37}[
            shape
        ]
    write_json(receipt, document)
    unknown = bad_directory / "unexpected-owned-synthetic.bin"
    unknown.write_bytes(b"Synthetic unknown file must survive malformed receipt isolation.")
    before_bad = _tree(bad_directory)
    del memory
    gc.collect()
    assert root.resolve().is_relative_to(tmp_path.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    # Rebuild may use managed evidence only, never substitute an external scan.
    bad.directory.rename(tmp_path / "bad-source-offline")
    good.directory.rename(tmp_path / "good-source-offline")
    memory = module.MemoryService(root)
    report = memory.reconcile()
    assert report.warnings and all(str(w).strip() for w in report.warnings)
    assert _tree(bad_directory) == before_bad
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert (
            connection.execute(
                "SELECT id,meeting_id,revision_number,raw_text,content_hash "
                "FROM transcript_revisions WHERE meeting_id=?",
                (OTHER_MEETING_ID,),
            ).fetchall()
            == good_revisions
        )
        assert connection.execute(
            "SELECT status FROM meetings WHERE id=?", (OTHER_MEETING_ID,)
        ).fetchone() == ("completed",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert memory.reconcile().warnings
    assert _tree(bad_directory) == before_bad
    recovery = importlib.import_module("here.application.recovery").RecoveryService(
        tmp_path / "sessions"
    )
    assert isinstance(recovery.discover(), list)
    result = CliRunner().invoke(
        importlib.import_module("here.cli").app, ["memory", "reconcile", "--json"]
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict) and payload["warnings"]
    assert _tree(bad_directory) == before_bad
