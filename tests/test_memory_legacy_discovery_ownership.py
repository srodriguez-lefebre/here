"""Legacy source IDs cannot bypass the resolved canonical processing owner."""

from __future__ import annotations

import importlib
import json
import shutil
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from uuid import UUID

import pytest
import soundfile as sf
from memory_fixtures import TRANSCRIPT_TEXT, make_session


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _result(text):
    client = importlib.import_module("here.transcription.client")
    segments = importlib.import_module("here.transcription.segments")
    return client.TranscriptionResult(
        raw_text=text, final_text=text, segments=[segments.TranscriptSegment(text)]
    )


@pytest.mark.parametrize("source_id", [None, "synthetic-m1-opaque"], ids=["none", "opaque"])
@pytest.mark.parametrize("operation", ["discover", "materialize", "second-producer"])
def test_resolved_legacy_owner_guards_recovery_and_second_producer_before_audio_inspection(
    tmp_path, monkeypatch, source_id, operation
):
    root, sessions = tmp_path / "data", tmp_path / "sessions"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-legacy-ownership-key")
    monkeypatch.setenv("CLEANUP_ENABLED", "false")

    def forbidden(*args, **kwargs):
        pytest.fail("Legacy owner checks allocated real provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    service = importlib.import_module("here.memory.service")
    processing = importlib.import_module("here.application.processing")
    recovery_module = importlib.import_module("here.application.recovery")
    database = importlib.import_module("here.memory.database")
    memory = service.MemoryService(root)
    source = make_session(sessions / "legacy", meeting_id=source_id)
    original_audio = source.audio.read_bytes()
    prior_metadata = json.loads(source.metadata.read_text(encoding="utf-8"))
    # A caller's old selection contains the source identity exactly as recorded,
    # not a fabricated canonical UUID. Materialize must resolve/recheck the owner
    # from this real origin after that selection has become stale during work.
    stale = recovery_module.RecoveryCandidate(
        source.directory,
        prior_metadata["session_id"],
        prior_metadata["duration_seconds"],
        prior_metadata["status"],
        None,
        True,
        prior_metadata.get("meeting_id"),
    )
    entered, release = threading.Event(), threading.Event()
    boundary, outcome, provider_calls = {}, {}, []

    def first_provider(session, **kwargs):
        provider_calls.append("first")
        boundary["source_id"] = session.meeting_id
        boundary["meetings"] = _rows(root, "SELECT id,status,active_revision_id FROM meetings")
        boundary["revisions"] = _rows(
            root,
            "SELECT id,meeting_id,revision_number,raw_text,content_hash "
            "FROM transcript_revisions ORDER BY revision_number",
        )
        entered.set()
        assert release.wait(20), "Test did not release the synthetic first provider"
        return _result("Synthetic first owner completed evidence.")

    def second_provider(session, **kwargs):
        provider_calls.append("second")
        assert session.meeting_id == source_id
        return _result("Synthetic owner released evidence.")

    first = processing.SessionProcessor(
        transcribe=first_provider, retry_delays=(), memory_service=memory
    )
    second = processing.SessionProcessor(
        transcribe=second_provider,
        retry_delays=(),
        memory_service=service.MemoryService(root),
    )

    def run_first():
        try:
            outcome["artifacts"] = first.transcribe_audio_path(source.audio, sessions)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run_first, name="synthetic-canonical-legacy-owner")
    worker.start()
    try:
        assert entered.wait(15), repr(outcome.get("error"))
        assert boundary["source_id"] == source_id
        assert len(boundary["meetings"]) == 1
        identity, status, parent = boundary["meetings"][0]
        assert str(UUID(identity)) == identity and status == "processing"
        assert len(boundary["revisions"]) == 1
        assert boundary["revisions"][0][:4] == (parent, identity, 1, TRANSCRIPT_TEXT)
        independent = service.MemoryService(root)
        try:
            with pytest.raises(database.Busy):
                independent.begin_processing(identity)
        finally:
            independent.finish_capture(identity)
        before = _snapshot(source.directory)
        revisions_before = _rows(
            root, "SELECT id,meeting_id,revision_number,raw_text FROM transcript_revisions"
        )
        accesses = []
        original_info = sf.info
        original_read_text = Path.read_text
        original_copy = shutil.copy2

        def is_source_material(path):
            try:
                actual = Path(path).absolute()
            except TypeError:
                return False
            return actual.parent == source.directory.absolute() and (
                actual.suffix.lower() == ".wav" or actual.name == "memory_pending.json"
            )

        def tracked_info(path, *args, **kwargs):
            if is_source_material(path):
                accesses.append("audio inspection")
            return original_info(path, *args, **kwargs)

        def tracked_read(path, *args, **kwargs):
            if is_source_material(path):
                accesses.append("marker read")
            return original_read_text(path, *args, **kwargs)

        def tracked_copy(src, dst, *args, **kwargs):
            if is_source_material(src) or is_source_material(dst):
                accesses.append("audio copy")
            return original_copy(src, dst, *args, **kwargs)

        recovery = recovery_module.RecoveryService(sessions)
        with monkeypatch.context() as watch:
            watch.setattr(sf, "info", tracked_info)
            watch.setattr(Path, "read_text", tracked_read)
            watch.setattr(shutil, "copy2", tracked_copy)
            if operation == "discover":
                assert not any(
                    candidate.session_dir == source.directory for candidate in recovery.discover()
                ), "Discovery offered a legacy origin whose canonical producer is live"
            elif operation == "materialize":
                with pytest.raises(database.Busy):
                    recovery.materialize(stale)
            else:
                with pytest.raises(database.Busy):
                    second.transcribe_audio_path(source.audio, sessions)
            assert accesses == [], "Resolved canonical owner must guard before audio/marker reads"
        assert provider_calls == ["first"]
        assert _snapshot(source.directory) == before
        assert (
            _rows(root, "SELECT id,meeting_id,revision_number,raw_text FROM transcript_revisions")
            == revisions_before
        )
        assert _rows(root, "SELECT id,status,active_revision_id FROM meetings") == [
            (identity, "processing", parent)
        ]
    finally:
        release.set()
        worker.join(15)
        assert not worker.is_alive(), "Real producer did not finish after provider release"

    assert "error" not in outcome, repr(outcome.get("error"))
    assert outcome["artifacts"].metadata.status == "completed"
    assert outcome["artifacts"].metadata.meeting_id == source_id
    recovery = recovery_module.RecoveryService(sessions)
    assert recovery.materialize(stale) == source.directory
    assert not any(c.session_dir == source.directory for c in recovery.discover())
    # The first publication may choose a new normalized filename. Follow actual
    # committed metadata for the post-release producer instead of stale audio.wav.
    current = json.loads(source.metadata.read_text(encoding="utf-8"))
    active_audio = source.directory / current["recoverable_audio"]
    assert active_audio.is_file()
    assert second.transcribe_audio_path(active_audio, sessions).metadata.status == "completed"
    assert provider_calls == ["first", "second"]
    assert _rows(root, "SELECT id,status FROM meetings") == [(identity, "completed")]
    revisions = _rows(
        root,
        "SELECT id,meeting_id,revision_number,raw_text,content_hash "
        "FROM transcript_revisions ORDER BY revision_number",
    )
    assert [r[2] for r in revisions] == [1, 2, 3]
    assert revisions[0] == boundary["revisions"][0]
    assert len({r[0] for r in revisions}) == 3
    assert _rows(root, "SELECT active_revision_id FROM meetings") == [(revisions[-1][0],)]
    assert json.loads(source.metadata.read_text(encoding="utf-8")).get("meeting_id") == source_id
    # The initial input was the declared normalized file and remains valid source
    # evidence; publication can select a different current filename on retry.
    assert source.audio.read_bytes() == original_audio
    assert not (source.directory / "memory_pending.json").exists()
