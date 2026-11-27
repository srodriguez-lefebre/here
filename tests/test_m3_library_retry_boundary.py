"""Exercise the meetings CLI through the application and catalog boundaries."""

from __future__ import annotations

import json
import shutil
import sqlite3
import wave
from pathlib import Path

import here.application.processing as processing
import here.cli as cli
import pytest
from here.application import HereApplicationController
from here.application.processing import SessionProcessor
from here.application.recovery import RecoveryService
from here.config.settings import ProviderConfigurationError
from here.memory.service import MemoryService
from here.transcription.client import TranscriptionResult
from here.transcription.segments import TranscriptSegment
from memory_fixtures import FRAME_COUNT, MEETING_ID, OTHER_MEETING_ID, make_session, write_json
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


@pytest.fixture
def failed_session(configured_library):
    memory, sessions = configured_library
    selected = make_session(sessions / "selected", meeting_id=MEETING_ID)
    failed = json.loads(selected.metadata.read_text(encoding="utf-8"))
    failed.update(status="failed", failure_stage="transcription")
    write_json(selected.metadata, failed)
    assert memory.ingest_session(selected.directory).meeting_id == MEETING_ID
    return selected, failed


@pytest.fixture
def install_transcriber(monkeypatch):
    def forbidden_capture(*args, **kwargs):
        pytest.fail("The retry command started capture or live transcription")

    def install(transcribe):
        def controller(*, memory_service):
            processor = SessionProcessor(
                transcribe=transcribe, retry_delays=(), memory_service=memory_service
            )
            return HereApplicationController(
                memory_service=memory_service,
                processor=processor,
                capture_factory=forbidden_capture,
                live_factory=forbidden_capture,
            )

        monkeypatch.setattr(cli, "create_default_controller", controller)

    return install


@pytest.mark.parametrize(
    "change", ["replacement", "completed", "empty-audio", "empty-normalized-with-raw"]
)
def test_retry_revalidates_materialized_session_before_replay_or_provider(
    configured_library, failed_session, install_transcriber, monkeypatch, tmp_path, change
):
    memory, _ = configured_library
    selected, failed = failed_session

    if change == "replacement":
        # Give the replacement its own canonical parent. Its material otherwise
        # looks like an ordinary retryable session to the processing worker.
        replacement = make_session(tmp_path / "replacement", meeting_id=OTHER_MEETING_ID)
        write_json(replacement.metadata, {**failed, "meeting_id": OTHER_MEETING_ID})
        assert memory.ingest_session(replacement.directory).meeting_id == OTHER_MEETING_ID

    if change == "empty-normalized-with-raw":
        raw = selected.directory / "raw.wav"
        shutil.copyfile(selected.audio, raw)
        raw_bytes = raw.read_bytes()
        failed["capture_sources"] = [{**failed["capture_sources"][0], "audio_file": raw.name}]
        failed["output_files"].append(raw.name)
        write_json(selected.metadata, failed)

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
        elif change == "completed":
            write_json(selected.metadata, {**failed, "status": "completed", "failure_stage": None})
        else:
            # The normalized WAV matches its manifest, even when a raw source still has frames.
            source = {**failed["sources"][0], "frames": 0, "duration_seconds": 0.0}
            with wave.open(str(selected.audio), "wb") as audio:
                audio.setnchannels(source["channels"])
                audio.setsampwidth(2)
                audio.setframerate(source["sample_rate"])
            write_json(
                selected.metadata,
                {
                    **failed,
                    "duration_seconds": 0.0
                    if change == "empty-audio"
                    else failed["duration_seconds"],
                    "sources": [source],
                    "capture_sources": (
                        failed["capture_sources"]
                        if change == "empty-normalized-with-raw"
                        else [{**source, "audio_file": selected.audio.name}]
                    ),
                },
            )
        return path

    calls = []
    replay = SessionProcessor._replay_pending

    def observe_replay(processor, directory):
        calls.append(("replay", directory))
        return replay(processor, directory)

    def forbidden_provider(session, **options):
        calls.append(("provider", session.meeting_id, tuple(s.frames for s in session.sources)))
        raise AssertionError("Changed retry material reached the provider boundary")

    monkeypatch.setattr(RecoveryService, "materialize", change_after_materialize)
    monkeypatch.setattr(SessionProcessor, "_replay_pending", observe_replay)
    install_transcriber(forbidden_provider)

    result = CliRunner().invoke(cli.app, ["meetings", "retry", MEETING_ID[:8], "--json"])

    assert materialized == [selected.directory], result.output + repr(result.exception)
    if change == "empty-normalized-with-raw":
        assert raw.read_bytes() == raw_bytes
    assert calls == []
    assert result.exit_code in (3, 4), result.output + repr(result.exception)
    assert json.loads(result.stdout)["error_category"] in ("conflict", "io")


def test_missing_retry_audio_is_rejected_before_provider_key(
    failed_session, install_transcriber, monkeypatch, tmp_path
):
    selected, _ = failed_session
    materialize = RecoveryService.materialize
    removed = tmp_path / "retained-original.wav"

    def move_audio_after_materialize(recovery, candidate):
        path = materialize(recovery, candidate)
        assert path == selected.directory
        selected.audio.rename(removed)
        return path

    calls = []

    def missing_key():
        calls.append("provider key")
        raise ProviderConfigurationError("Synthetic missing provider key")

    def forbidden_provider(*args, **kwargs):
        calls.append("transcription")
        raise AssertionError("Missing retry audio reached transcription")

    def forbidden_replay(*args, **kwargs):
        calls.append("replay")
        raise AssertionError("Missing retry audio reached replay")

    monkeypatch.setattr(RecoveryService, "materialize", move_audio_after_materialize)
    monkeypatch.setattr(processing, "require_provider_key", missing_key)
    monkeypatch.setattr(processing, "transcribe_recording_session", forbidden_provider)
    monkeypatch.setattr(SessionProcessor, "_replay_pending", forbidden_replay)
    install_transcriber(forbidden_provider)

    result = CliRunner().invoke(cli.app, ["meetings", "retry", MEETING_ID[:8], "--json"])

    assert removed.is_file(), result.output + repr(result.exception)
    assert calls == []
    assert result.exit_code in (3, 4), result.output + repr(result.exception)
    assert json.loads(result.stdout)["error_category"] in ("conflict", "io")


@pytest.mark.parametrize(
    "omit_recoverable_audio", [False, True], ids=["recoverable-audio", "declared-capture-source"]
)
def test_retry_publishes_new_revision_for_unchanged_selected_session(
    configured_library, failed_session, install_transcriber, omit_recoverable_audio
):
    memory, _ = configured_library
    selected, failed = failed_session
    if omit_recoverable_audio:
        del failed["recoverable_audio"]
        failed["output_files"].remove(selected.audio.name)
        write_json(selected.metadata, failed)
    old_revision = memory.active_revision(MEETING_ID)
    assert old_revision is not None
    text = "Synthetic retry produced fresh evidence."
    calls = []

    def transcribe(session, **options):
        (source,) = session.sources
        calls.append((session.meeting_id, source.path, source.frames))
        return TranscriptionResult(text, text, segments=[TranscriptSegment(text, start=0, end=1)])

    install_transcriber(transcribe)

    result = CliRunner().invoke(cli.app, ["meetings", "retry", MEETING_ID[:8], "--json"])

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert json.loads(result.stdout) == {"meeting_id": MEETING_ID, "status": "completed"}
    assert calls == [(MEETING_ID, selected.audio, FRAME_COUNT)]
    detail = memory.show(MEETING_ID)
    assert detail is not None
    assert detail["meeting"]["status"] == "completed"
    new_revision = detail["meeting"]["active_revision_id"]
    assert new_revision is not None and new_revision != old_revision
    assert {revision["id"] for revision in detail["revisions"]} == {old_revision, new_revision}
    assert any(
        segment["revision_id"] == new_revision and segment["text"] == text
        for segment in detail["segments"]
    )


def test_retry_rejects_undeclared_default_audio_beside_declared_raw_source(
    configured_library, failed_session, install_transcriber
):
    memory, _ = configured_library
    selected, failed = failed_session
    old_revision = memory.active_revision(MEETING_ID)
    assert old_revision is not None
    raw = selected.directory / "raw.wav"
    shutil.copyfile(selected.audio, raw)
    original_audio = selected.audio.read_bytes()
    del failed["recoverable_audio"]
    failed["output_files"].remove(selected.audio.name)
    failed["capture_sources"][0]["audio_file"] = raw.name
    write_json(selected.metadata, failed)

    calls = []

    def forbidden_provider(session, **options):
        calls.extend(source.path for source in session.sources)
        raise AssertionError("An undeclared default WAV reached the provider boundary")

    install_transcriber(forbidden_provider)

    result = CliRunner().invoke(cli.app, ["meetings", "retry", MEETING_ID[:8], "--json"])

    assert calls == []
    assert result.exit_code in (3, 4), result.output + repr(result.exception)
    assert json.loads(result.stdout)["error_category"] in ("conflict", "io")
    assert selected.audio.read_bytes() == raw.read_bytes() == original_audio
    assert memory.active_revision(MEETING_ID) == old_revision


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
