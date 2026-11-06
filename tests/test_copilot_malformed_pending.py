"""Copilot C4: invalid pending envelopes retain a visible, nonretryable session."""

from __future__ import annotations

import importlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import soundfile as sf
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json

CANARY = "synthetic-malformed-marker-private-canary-do-not-display"


def _setup(tmp_path, monkeypatch):
    root, sessions = tmp_path / "data", tmp_path / "sessions"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Marker discovery allocated provider/network transport")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    monkeypatch.setattr("here.application.processing.transcribe_recording_session", forbidden)
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    recovery = importlib.import_module("here.application.recovery").RecoveryService(sessions)
    return root, sessions, memory, recovery


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def _set_failed(source):
    metadata = json.loads(source.metadata.read_text(encoding="utf-8"))
    metadata["status"] = "failed"
    metadata["failure_stage"] = "offline_transcription"
    write_json(source.metadata, metadata)


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("shape", ["array", "invalid-json", "non-string-parent"])
def test_invalid_marker_keeps_valid_session_visible_without_audio_or_provider_retry(
    tmp_path, monkeypatch, status, shape
):
    root, sessions, memory, recovery = _setup(tmp_path, monkeypatch)
    source = make_session(sessions / "a-malformed", meeting_id=MEETING_ID)
    if status == "failed":
        _set_failed(source)
    assert memory.ingest_session(source.directory).status == "inserted"
    previous = _rows(root, "SELECT id,meeting_id,raw_text,content_hash FROM transcript_revisions")
    old_meeting = _rows(root, "SELECT id,status,active_revision_id FROM meetings")
    private = tmp_path / "synthetic-unrelated.txt"
    private.write_text(CANARY, encoding="utf-8")
    marker = source.directory / "memory_pending.json"
    if shape == "array":
        marker.write_bytes(b"[]")
    elif shape == "invalid-json":
        marker.write_text('{"diagnostic":"' + CANARY, encoding="utf-8")
    else:
        write_json(
            marker,
            {
                "schema_version": 1,
                "meeting_id": MEETING_ID,
                "revision_of": 42,
                "diagnostic": CANARY,
                "diagnostic_path": str(private),
            },
        )
    before = _snapshot(source.directory)
    good = make_session(sessions / "z-valid-sibling", meeting_id=OTHER_MEETING_ID)
    good_marker = good.directory / "memory_pending.json"
    write_json(
        good_marker,
        {
            "schema_version": 1,
            "meeting_id": OTHER_MEETING_ID,
            "revision_of": None,
        },
    )
    reads = []
    original_info, original_open = sf.info, Path.open

    def protected(path):
        try:
            actual = Path(path).absolute()
        except TypeError:
            return False
        return actual == private.absolute() or (
            actual.parent == source.directory.absolute() and actual.suffix.lower() == ".wav"
        )

    def tracked_info(path, *args, **kwargs):
        if protected(path):
            reads.append("audio inspection")
        return original_info(path, *args, **kwargs)

    def tracked_open(path, *args, **kwargs):
        if protected(path):
            reads.append("protected material open")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as watch:
        watch.setattr(sf, "info", tracked_info)
        watch.setattr(Path, "open", tracked_open)
        first = recovery.discover()
        second = recovery.discover()
    # Own byte oracles run after tracking ends; the readers above always delegate.
    assert reads == [], "Invalid marker must not continue into WAV/private material inspection"
    assert _snapshot(source.directory) == before
    assert private.read_text(encoding="utf-8") == CANARY
    assert not good_marker.exists()
    assert _rows(root, "SELECT id,status FROM meetings WHERE id=?", (OTHER_MEETING_ID,)) == [
        (OTHER_MEETING_ID, "completed")
    ]
    assert (
        _rows(root, "SELECT id,status,active_revision_id FROM meetings WHERE id=?", (MEETING_ID,))
        == old_meeting
    )
    assert (
        _rows(
            root,
            "SELECT id,meeting_id,raw_text,content_hash FROM transcript_revisions "
            "WHERE meeting_id=?",
            (MEETING_ID,),
        )
        == previous
    )
    first_warning = [item for item in first if item.session_dir == source.directory]
    second_warning = [item for item in second if item.session_dir == source.directory]
    assert len(first_warning) == len(second_warning) == 1, (
        "Valid session disappeared because its pending marker was malformed"
    )
    candidate = second_warning[0]
    assert candidate.capture_id == MEETING_ID and candidate.status == status
    assert candidate.can_retry is False
    assert isinstance(candidate.error_summary, str) and 0 < len(candidate.error_summary) <= 512
    assert CANARY not in candidate.error_summary and str(private) not in candidate.error_summary
    assert [item for item in second if item.session_dir == good.directory] == []
    contract = importlib.import_module("here.ui.contract")
    core = importlib.import_module("here.ui.fake_controller").FakeApplicationController()
    retried = []
    monkeypatch.setattr(core, "retry", lambda *args: retried.append(args))

    def forbidden_credentials(*args, **kwargs):
        pytest.fail("Nonretryable marker warning requested provider credentials")

    monkeypatch.setattr(contract, "require_provider_key", forbidden_credentials)
    adapter = contract.ApplicationUiAdapter(core, sessions, recovery_service=recovery)
    with pytest.raises(RuntimeError):
        adapter.retry_candidate(candidate)
    assert retried == [] and _snapshot(source.directory) == before
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_control_valid_failed_marker_replays_and_preserves_explicit_provider_retry(
    tmp_path, monkeypatch
):
    root, sessions, _memory, recovery = _setup(tmp_path, monkeypatch)
    source = make_session(sessions / "normal-failed", meeting_id=MEETING_ID)
    _set_failed(source)
    before = _snapshot(source.directory)
    marker = source.directory / "memory_pending.json"
    write_json(marker, {"schema_version": 1, "meeting_id": MEETING_ID, "revision_of": None})
    (candidate,) = recovery.discover()
    assert candidate.status == "failed" and candidate.can_retry is True
    assert candidate.capture_id == MEETING_ID and not marker.exists()
    assert _snapshot(source.directory) == before
    assert recovery.discover() == [candidate]
    contract = importlib.import_module("here.ui.contract")
    core = importlib.import_module("here.ui.fake_controller").FakeApplicationController()
    credentials, retried = [], []
    monkeypatch.setattr(contract, "require_provider_key", lambda: credentials.append("explicit"))
    monkeypatch.setattr(core, "retry", lambda directory: retried.append(directory))
    adapter = contract.ApplicationUiAdapter(core, sessions, recovery_service=recovery)
    adapter.retry_candidate(candidate)
    assert credentials == ["explicit"] and retried == [source.directory]
    assert _rows(root, "SELECT id,status FROM meetings") == [(MEETING_ID, "failed")]


def test_control_completed_session_without_marker_does_not_gain_a_false_warning(
    tmp_path, monkeypatch
):
    root, sessions, memory, recovery = _setup(tmp_path, monkeypatch)
    source = make_session(sessions / "normal-completed", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    before = _snapshot(source.directory)
    assert recovery.discover() == recovery.discover() == []
    assert _snapshot(source.directory) == before
    assert _rows(root, "SELECT id,status FROM meetings") == [(MEETING_ID, "completed")]
