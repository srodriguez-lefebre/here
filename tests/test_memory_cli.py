"""CLI memory commands and file transcription use real local persistence."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from uuid import UUID

import here.cli as cli_module
import pytest
from here.transcription.client import TranscriptionResult
from here.transcription.segments import TranscriptSegment
from loguru import logger
from memory_fixtures import MEETING_ID, TRANSCRIPT_TEXT, make_corpus, make_session, write_marked_wav
from typer.testing import CliRunner

runner = CliRunner()


def _root(tmp_path, monkeypatch):
    root = tmp_path / "catalog"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    return root


def _json(result, expected_exit=0):
    assert result.exit_code == expected_exit, result.output + repr(result.exception)
    assert result.stdout.strip(), result.output
    return json.loads(result.stdout)


def _provider(monkeypatch, texts):
    outputs = iter(texts)

    def transcribe(session, **kwargs):
        text = next(outputs)
        assert session.sources and session.sources[0].path.is_file()
        return TranscriptionResult(
            raw_text=text,
            final_text=text,
            segments=[
                TranscriptSegment(
                    text, start=0.0, end=1.0, speaker="speaker_0", speaker_scope="synthetic-request"
                )
            ],
        )

    # The provider/transcriber is the only replaced production operation.
    monkeypatch.setattr(cli_module, "transcribe_recording_session", transcribe)


def test_memory_cli_ingest_reconcile_show_json_round_trip_without_provider(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "socket.socket.connect", lambda *args: pytest.fail("Memory CLI used network")
    )
    first = _json(
        runner.invoke(cli_module.app, ["memory", "ingest", str(source.directory), "--json"])
    )
    assert first["status"] == "inserted"
    assert first["meeting_id"] == MEETING_ID
    assert isinstance(first["warnings"], list)
    again = _json(
        runner.invoke(cli_module.app, ["memory", "ingest", str(source.directory), "--json"])
    )
    assert (again["status"], again["meeting_id"]) == ("unchanged", MEETING_ID)
    recovery = _json(runner.invoke(cli_module.app, ["memory", "reconcile", "--json"]))
    assert isinstance(recovery["warnings"], list)
    shown = _json(runner.invoke(cli_module.app, ["memory", "show", MEETING_ID, "--json"]))
    assert shown["meeting"]["id"] == MEETING_ID
    assert shown["meeting"]["status"] == "completed"
    assert len(shown["audio_assets"]) == len(shown["revisions"]) == 1
    assert shown["audio_assets"][0]["sha256"] == source.audio_sha256
    assert shown["revisions"][0]["raw_text"] == TRANSCRIPT_TEXT
    assert [segment["ordinal"] for segment in shown["segments"]] == [0, 1]
    with sqlite3.connect(root / "library.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 1


@pytest.mark.parametrize("command", ["invalid-id", "absent-source"])
def test_memory_cli_invalid_input_is_json_error_exit_two(tmp_path, monkeypatch, command):
    _root(tmp_path, monkeypatch)
    arguments = (
        ["show", "not-a-uuid"]
        if command == "invalid-id"
        else ["ingest", str(tmp_path / "absent-source")]
    )
    error = _json(runner.invoke(cli_module.app, ["memory", *arguments, "--json"]), 2)
    assert error["error"]


def test_memory_cli_unknown_meeting_is_json_error_exit_three(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    error = _json(runner.invoke(cli_module.app, ["memory", "show", MEETING_ID, "--json"]), 3)
    assert error["error"]


def test_memory_cli_same_uuid_conflict_is_json_rejection_exit_three(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    corpus = make_corpus(tmp_path)
    _json(
        runner.invoke(
            cli_module.app, ["memory", "ingest", str(corpus.identified.directory), "--json"]
        )
    )
    conflict = _json(
        runner.invoke(
            cli_module.app,
            [
                "memory",
                "ingest",
                str(corpus.conflicting_identity.directory),
                "--json",
            ],
        ),
        3,
    )
    assert conflict["status"] == "rejected"
    assert conflict["warnings"]
    with sqlite3.connect(root / "library.sqlite3") as connection:
        assert connection.execute("SELECT raw_text FROM transcript_revisions").fetchall() == [
            (TRANSCRIPT_TEXT,)
        ]


def test_memory_cli_corrupt_database_is_json_error_exit_four_without_overwrite(
    tmp_path, monkeypatch
):
    root = _root(tmp_path, monkeypatch)
    root.mkdir()
    path = root / "library.sqlite3"
    path.write_bytes(b"synthetic corrupt SQLite data")
    error = _json(runner.invoke(cli_module.app, ["memory", "reconcile", "--json"]), 4)
    assert error["error"]
    assert path.read_bytes() == b"synthetic corrupt SQLite data"


def test_trans_cli_and_existing_session_retry_publish_one_identity_with_revision_segments(
    tmp_path, monkeypatch
):
    root = _root(tmp_path, monkeypatch)
    audio = tmp_path / "external.wav"
    write_marked_wav(audio)
    external_before = audio.read_bytes()
    output = tmp_path / "sessions"
    _provider(monkeypatch, ["First CLI evidence.", "Revised CLI evidence."])
    first = runner.invoke(cli_module.app, ["trans", str(audio), "--output-dir", str(output)])
    assert first.exit_code == 0, first.output + repr(first.exception)
    assert (root / "library.sqlite3").is_file(), "here trans did not publish through memory"
    sessions = list(output.iterdir())
    assert len(sessions) == 1
    saved = sessions[0]
    metadata = json.loads((saved / "session.json").read_text(encoding="utf-8"))
    meeting_id = metadata["meeting_id"]
    UUID(meeting_id)
    with sqlite3.connect(root / "library.sqlite3") as connection:
        previous = connection.execute("SELECT id, raw_text FROM transcript_revisions").fetchone()
        assert previous[1] == "First CLI evidence."
        assert connection.execute("SELECT COUNT(*) FROM segments").fetchone()[0] == 1
    retry = runner.invoke(cli_module.app, ["trans", str(saved / "audio.wav")])
    assert retry.exit_code == 0, retry.output + repr(retry.exception)
    assert audio.read_bytes() == external_before
    assert (
        json.loads((saved / "session.json").read_text(encoding="utf-8"))["meeting_id"] == meeting_id
    )
    with sqlite3.connect(root / "library.sqlite3") as connection:
        assert connection.execute("SELECT id FROM meetings").fetchall() == [(meeting_id,)]
        assert connection.execute("SELECT COUNT(*) FROM audio_assets").fetchone()[0] == 1
        assert connection.execute(
            "SELECT revision_number, raw_text FROM transcript_revisions ORDER BY revision_number"
        ).fetchall() == [(1, "First CLI evidence."), (2, "Revised CLI evidence.")]
        assert (
            connection.execute(
                "SELECT raw_text FROM transcript_revisions WHERE id=?", (previous[0],)
            ).fetchone()[0]
            == previous[1]
        )
        active = connection.execute("SELECT active_revision_id FROM meetings").fetchone()[0]
        assert active != previous[0]
        assert connection.execute(
            "SELECT text, start_ms, end_ms FROM segments WHERE revision_id=?", (active,)
        ).fetchall() == [("Revised CLI evidence.", 0, 1000)]
        asset = connection.execute("SELECT relative_path, sha256 FROM audio_assets").fetchone()
        assert hashlib.sha256((root / asset[0]).read_bytes()).hexdigest() == asset[1]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_trans_cli_catalog_fault_preserves_published_audio_and_pending_recovery_message(
    tmp_path, monkeypatch
):
    root = _root(tmp_path, monkeypatch)
    root.mkdir()
    (root / "library.sqlite3").write_bytes(b"synthetic corrupt catalog")
    audio = tmp_path / "external.wav"
    write_marked_wav(audio)
    before = audio.read_bytes()
    _provider(monkeypatch, ["Evidence survives a catalog fault."])
    logs = io.StringIO()
    sink = logger.add(logs, format="{message}")
    output = tmp_path / "sessions"
    try:
        result = runner.invoke(cli_module.app, ["trans", str(audio), "--output-dir", str(output)])
    finally:
        logger.remove(sink)
    assert result.exit_code == 0, result.output + repr(result.exception)
    sessions = list(output.iterdir())
    assert len(sessions) == 1
    assert (sessions[0] / "audio.wav").is_file()
    metadata = json.loads((sessions[0] / "session.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert (sessions[0] / "transcript.txt").read_text(
        encoding="utf-8-sig"
    ) == "Evidence survives a catalog fault."
    assert audio.read_bytes() == before
    message = (result.output + logs.getvalue()).lower()
    assert "audio" in message and "biblioteca" in message and "recuper" in message
