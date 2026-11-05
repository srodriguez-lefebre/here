"""Round-one review regressions for distinct input, conflict and storage exits."""

from __future__ import annotations

import importlib
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from memory_fixtures import make_corpus, make_session
from typer.testing import CliRunner


def _cli(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return root, importlib.import_module("here.cli").app


@pytest.mark.parametrize("input_kind", ["missing-manifest", "malformed-manifest"])
def test_invalid_session_directory_or_manifest_is_input_exit_two(tmp_path, monkeypatch, input_kind):
    root, app = _cli(tmp_path, monkeypatch)
    source = tmp_path / "not-a-session"
    source.mkdir()
    sentinel = source / "keep.txt"
    sentinel.write_bytes(b"synthetic unrelated directory")
    if input_kind == "malformed-manifest":
        (source / "session.json").write_text("{invalid synthetic JSON", encoding="utf-8")
    result = CliRunner().invoke(app, ["memory", "ingest", str(source), "--json"])
    assert result.exit_code == 2, result.output + repr(result.exception)
    payload = json.loads(result.stdout)
    assert payload.get("error") or payload.get("warnings")
    assert sentinel.read_bytes() == b"synthetic unrelated directory"
    if (root / "library.sqlite3").exists():
        with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
            assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 0


def test_real_staging_copy_permission_failure_is_storage_exit_four(tmp_path, monkeypatch):
    root, app = _cli(tmp_path, monkeypatch)
    source = make_session(tmp_path / "source")
    before = (source.audio.read_bytes(), source.metadata.read_bytes())
    original = shutil.copy2

    def denied_copy(src, dst, *args, **kwargs):
        if Path(dst).absolute().is_relative_to((root / ".staging").absolute()):
            raise PermissionError("Synthetic staging copy permission denied")
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", denied_copy)
    result = CliRunner().invoke(app, ["memory", "ingest", str(source.directory), "--json"])
    assert result.exit_code == 4, result.output + repr(result.exception)
    payload = json.loads(result.stdout)
    assert payload.get("error") or payload.get("warnings")
    assert (source.audio.read_bytes(), source.metadata.read_bytes()) == before
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 0
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_uuid_content_conflict_remains_conflict_exit_three(tmp_path, monkeypatch):
    root, app = _cli(tmp_path, monkeypatch)
    corpus = make_corpus(tmp_path / "sources")
    runner = CliRunner()
    first = runner.invoke(app, ["memory", "ingest", str(corpus.identified.directory), "--json"])
    assert first.exit_code == 0, first.output
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        prior = connection.execute(
            "SELECT id,raw_text,content_hash FROM transcript_revisions"
        ).fetchall()
    conflict = runner.invoke(
        app, ["memory", "ingest", str(corpus.conflicting_identity.directory), "--json"]
    )
    assert conflict.exit_code == 3, conflict.output + repr(conflict.exception)
    assert json.loads(conflict.stdout)["status"] == "rejected"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert (
            connection.execute(
                "SELECT id,raw_text,content_hash FROM transcript_revisions"
            ).fetchall()
            == prior
        )


def test_postcommit_one_shot_housekeeping_failure_is_cli_success_with_warning(
    tmp_path, monkeypatch
):
    root, app = _cli(tmp_path, monkeypatch)
    source = make_session(tmp_path / "source")
    module = importlib.import_module("here.memory.service")
    original = module.MemoryService._write_receipt
    failures = []

    def receipt(self, *args, **kwargs):
        if not failures:
            with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
                assert (
                    connection.execute("SELECT COUNT(*) FROM transcript_revisions").fetchone()[0]
                    == 1
                )
                assert connection.execute("SELECT active_revision_id FROM meetings").fetchone()[0]
            failures.append("postcommit")
            raise OSError("Synthetic receipt failure after the real commit")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(module.MemoryService, "_write_receipt", receipt)
    result = CliRunner().invoke(app, ["memory", "ingest", str(source.directory), "--json"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    payload = json.loads(result.stdout)
    assert payload["status"] == "inserted" and payload["warnings"]
    assert failures == ["postcommit"]
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM transcript_revisions").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
