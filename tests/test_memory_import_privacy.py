"""Contained source paths are still untrusted: imports never read credentials."""

from __future__ import annotations

import builtins
import importlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import soundfile as sf
from memory_fixtures import MEETING_ID, make_session, write_json

CANARY = "synthetic-import-secret-must-never-enter-managed-memory"


@pytest.mark.parametrize("reference", ["output_files", "recoverable_audio"])
def test_import_rejects_credential_file_reference_before_read_or_publication(
    tmp_path,
    monkeypatch,
    reference,
):
    service_module = importlib.import_module("here.memory.service")
    root = tmp_path / "managed"
    service = service_module.MemoryService(root)
    source = make_session(tmp_path / "external-source", meeting_id=MEETING_ID)
    secret = source.directory / ".env"
    secret.write_text("SYNTHETIC_CREDENTIAL=" + CANARY + "\n", encoding="utf-8")
    secret_before = secret.read_bytes()
    metadata = json.loads(source.metadata.read_text(encoding="utf-8"))
    if reference == "output_files":
        metadata["output_files"].append(".env")
    else:
        metadata["recoverable_audio"] = ".env"
    write_json(source.metadata, metadata)
    manifest_before = source.metadata.read_bytes()
    original_path_open = Path.open
    original_builtin_open = builtins.open
    original_info = sf.info
    attempted_reads = []

    def check_read(path):
        if isinstance(path, (str, Path)) and Path(path).absolute() == secret.absolute():
            attempted_reads.append(str(path))
            pytest.fail("Import attempted to read a contained credential file before rejection")

    def guarded_path_open(path, *args, **kwargs):
        check_read(path)
        return original_path_open(path, *args, **kwargs)

    def guarded_builtin_open(path, *args, **kwargs):
        check_read(path)
        return original_builtin_open(path, *args, **kwargs)

    def guarded_audio_probe(path, *args, **kwargs):
        # libsndfile opens paths natively, bypassing Python's open functions.
        check_read(path)
        return original_info(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded_path_open)
        patch.setattr(builtins, "open", guarded_builtin_open)
        patch.setattr(sf, "info", guarded_audio_probe)
        patch.setattr("socket.socket.connect", lambda *args: pytest.fail("Import used network"))
        result = service.ingest_session(source.directory)
    assert attempted_reads == []
    assert result.status == "rejected"
    assert result.warnings
    assert secret.read_bytes() == secret_before
    assert source.metadata.read_bytes() == manifest_before
    assert source.audio.is_file()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM audio_assets").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM transcript_revisions").fetchone()[0] == 0
    for managed_file in root.rglob("*"):
        if managed_file.is_file():
            assert CANARY.encode() not in managed_file.read_bytes()
