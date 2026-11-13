"""Interrupted v1 audio deletion must retain enough role proof for local recovery."""

from __future__ import annotations

import ctypes
import json
import os
import sqlite3
from contextlib import closing, contextmanager

import pytest
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID, make_session, write_json


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _files(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture(autouse=True)
def synthetic_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("HERE_PROVIDER_SMOKE_KEY", raising=False)
    monkeypatch.delenv("HERE_WHEEL_PYTHON", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Local v1 deletion attempted provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)


@pytest.fixture
def v1_catalog(tmp_path):
    source = make_session(tmp_path / "origin", meeting_id=MEETING_ID)
    source.audio.rename(source.directory / "take.wav")
    document = json.loads(source.metadata.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    document["recoverable_audio"] = "take.wav"
    document["capture_sources"][0]["audio_file"] = "take.wav"
    document["output_files"] = [
        "take.wav" if name == "audio.wav" else name for name in document["output_files"]
    ]
    write_json(source.metadata, document)
    memory = MemoryService(tmp_path / "data", sessions_root=tmp_path / "sessions")
    result = memory.ingest_session(source.directory)
    assert result.status == "inserted" and result.meeting_id == MEETING_ID
    detail = memory.show(MEETING_ID)
    assert detail is not None and len(detail["audio_assets"]) == 1
    audio = memory.data_dir / detail["audio_assets"][0]["relative_path"]
    assert audio.name == "take.wav"
    assert audio.read_bytes() == (source.directory / "take.wav").read_bytes()
    return memory, source.directory, audio


@contextmanager
def _deny_windows_delete(path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    # Real read handle sharing reads only: deletion is denied by Windows.
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    try:
        yield
    finally:
        assert kernel.CloseHandle(handle), ctypes.get_last_error()


@pytest.mark.skipif(os.name != "nt", reason="Windows FILE_SHARE_DELETE semantics")
@pytest.mark.parametrize("resume", ["delete", "reconcile", "db-loss-reconcile"])
def test_declared_v1_audio_delete_resumes_after_real_handle_release(v1_catalog, tmp_path, resume):
    memory, origin, audio = v1_catalog
    root = memory.data_dir
    original = _files(origin)
    audio_bytes = audio.read_bytes()
    registry = root / ".deletions" / f"{MEETING_ID}.json"
    with _deny_windows_delete(audio):
        result = memory.delete_meeting(MEETING_ID)
        assert (result.status, result.meeting_id, result.error_category) == (
            "pending",
            MEETING_ID,
            "io",
        )
        assert result.reason
        record = json.loads(registry.read_text(encoding="utf-8"))
        assert record["phase"] == "pending" and record["meeting_id"] == MEETING_ID
        assert record["origins"]
        assert memory.show(MEETING_ID) is None
        assert audio.read_bytes() == audio_bytes
        assert _rows(root, "SELECT status FROM meetings WHERE id=?", (MEETING_ID,)) == [
            ("deleting",)
        ]
        assert _rows(root, "SELECT phase FROM pending_operations WHERE kind='delete'") == [
            ("prepared",)
        ]
        assert _files(origin) == original
        # Diagnostic only: requiring its absence would freeze the ordering bug.
        manifest_present_after_denial = (audio.parent / "session.json").is_file()

    if resume == "db-loss-reconcile":
        assert root.parent == tmp_path and not root.is_symlink() and not root.is_junction()
        for suffix in ("", "-wal", "-shm", "-journal"):
            database = root / ("library.sqlite3" + suffix)
            assert database.parent == root
            assert not database.is_symlink() and not database.is_junction()
            database.unlink(missing_ok=True)
    restarted = MemoryService(root, sessions_root=tmp_path / "sessions")
    diagnostic = f"{resume=}; {manifest_present_after_denial=}"
    if resume == "delete":
        result = restarted.delete_meeting(MEETING_ID)
        assert (result.status, result.meeting_id, result.error_category) == (
            "completed",
            MEETING_ID,
            None,
        ), diagnostic
    else:
        report = restarted.reconcile()
        assert any(item.identity == MEETING_ID for item in report.completed), diagnostic
        assert all(item.identity != MEETING_ID for item in report.skipped), diagnostic
    assert not audio.exists(), diagnostic
    assert not (root / "meetings" / MEETING_ID).exists(), diagnostic
    assert json.loads(registry.read_text(encoding="utf-8"))["phase"] == "complete"
    assert _rows(root, "SELECT id FROM meetings WHERE id=?", (MEETING_ID,)) == []
    assert _rows(root, "SELECT phase FROM pending_operations WHERE kind='delete'") == [
        ("complete",)
    ]
    assert restarted.show(MEETING_ID) is None
    assert restarted.ingest_session(origin).status == "rejected"
    repeated = restarted.reconcile()
    assert all(item.identity != MEETING_ID for item in repeated.skipped)
    assert json.loads(registry.read_text(encoding="utf-8"))["phase"] == "complete"
    assert not (root / "meetings" / MEETING_ID).exists()
    assert _files(origin) == original


def test_undeclared_wav_is_retained_instead_of_suffix_authorized(v1_catalog, tmp_path):
    memory, origin, audio = v1_catalog
    original = _files(origin)
    unknown = audio.parent / "unregistered.wav"
    unknown_bytes = b"Synthetic unknown child, not a declared audio role"
    unknown.write_bytes(unknown_bytes)
    result = memory.delete_meeting(MEETING_ID)
    assert (result.status, result.meeting_id, result.error_category) == (
        "pending",
        MEETING_ID,
        "io",
    )
    assert result.reason
    assert unknown.read_bytes() == unknown_bytes
    assert memory.show(MEETING_ID) is None
    restarted = MemoryService(memory.data_dir, sessions_root=tmp_path / "sessions")
    report = restarted.reconcile()
    assert all(item.identity != MEETING_ID for item in report.completed)
    assert any(item.identity == MEETING_ID for item in report.skipped)
    assert unknown.read_bytes() == unknown_bytes
    registry = memory.data_dir / ".deletions" / f"{MEETING_ID}.json"
    assert json.loads(registry.read_text(encoding="utf-8"))["phase"] == "pending"
    assert _files(origin) == original
