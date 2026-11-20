"""Exercise the meetings CLI through the application and catalog boundaries."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import here.cli as cli
import pytest
from here.application import HereApplicationController
from here.application.processing import SessionProcessor
from here.application.recovery import RecoveryService
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json
from typer.testing import CliRunner


@pytest.fixture
def configured_library(tmp_path, monkeypatch):
    data = tmp_path / "data"
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setenv("HERE_DATA_DIR", str(data))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(sessions))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    return MemoryService(data, sessions_root=sessions), sessions


@pytest.mark.parametrize("change", ["replacement", "completed"])
def test_retry_revalidates_materialized_session_before_replay_or_provider(
    configured_library, monkeypatch, tmp_path, change
):
    memory, sessions = configured_library
    selected = make_session(sessions / "selected", meeting_id=MEETING_ID)
    failed = json.loads(selected.metadata.read_text(encoding="utf-8"))
    failed.update(status="failed", failure_stage="transcription")
    write_json(selected.metadata, failed)
    assert memory.ingest_session(selected.directory).meeting_id == MEETING_ID

    if change == "replacement":
        # Give the replacement its own canonical parent. Its material otherwise
        # looks like an ordinary retryable session to the processing worker.
        replacement = make_session(tmp_path / "replacement", meeting_id=OTHER_MEETING_ID)
        write_json(replacement.metadata, {**failed, "meeting_id": OTHER_MEETING_ID})
        assert memory.ingest_session(replacement.directory).meeting_id == OTHER_MEETING_ID

    materialized = []
    materialize = RecoveryService.materialize

    def change_after_materialize(recovery, candidate):
        path = materialize(recovery, candidate)
        assert candidate.expected_meeting_id == MEETING_ID
        assert path == selected.directory
        materialized.append(path)
        if change == "replacement":
            path.rename(tmp_path / "original")
            replacement.directory.rename(path)
        else:
            write_json(selected.metadata, {**failed, "status": "completed", "failure_stage": None})
        return path

    calls = []
    replay = SessionProcessor._replay_pending

    def observe_replay(processor, directory):
        calls.append(("replay", directory))
        return replay(processor, directory)

    def forbidden_provider(session, **options):
        calls.append(("provider", session.meeting_id))
        raise AssertionError("Changed retry material reached the provider boundary")

    def forbidden_capture(*args, **kwargs):
        pytest.fail("The retry command started capture or live transcription")

    def controller(*, memory_service):
        processor = SessionProcessor(
            transcribe=forbidden_provider, retry_delays=(), memory_service=memory_service
        )
        return HereApplicationController(
            memory_service=memory_service,
            processor=processor,
            capture_factory=forbidden_capture,
            live_factory=forbidden_capture,
        )

    monkeypatch.setattr(RecoveryService, "materialize", change_after_materialize)
    monkeypatch.setattr(SessionProcessor, "_replay_pending", observe_replay)
    monkeypatch.setattr(cli, "create_default_controller", controller)

    result = CliRunner().invoke(cli.app, ["meetings", "retry", MEETING_ID[:8], "--json"])

    assert materialized == [selected.directory], result.output + repr(result.exception)
    assert calls == []
    assert result.exit_code in (3, 4), result.output + repr(result.exception)
    assert json.loads(result.stdout)["error_category"] in ("conflict", "io")


def test_catalog_connection_failure_returns_io_json(configured_library, monkeypatch, tmp_path):
    memory, _ = configured_library
    selected = make_session(tmp_path / "catalog-source", meeting_id=MEETING_ID)
    assert memory.ingest_session(selected.directory).meeting_id == MEETING_ID
    catalog = memory.data_dir / "library.sqlite3"
    connect = sqlite3.connect
    failed_opens = []

    def fail_catalog(database, *args, **kwargs):
        if database == catalog or (isinstance(database, str) and Path(database) == catalog):
            failed_opens.append(database)
            raise sqlite3.OperationalError("unable to open database file")
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", fail_catalog)

    result = CliRunner().invoke(cli.app, ["meetings", "list", "--json"])

    assert failed_opens, result.output + repr(result.exception)
    assert result.exit_code == 4, result.output + repr(result.exception)
    payload = json.loads(result.stdout)
    assert payload["error_category"] == "io"
    assert isinstance(payload["error"], str) and payload["error"]
