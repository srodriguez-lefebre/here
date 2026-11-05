"""Materializing a pre-lease journal must acquire ownership before publication."""

from __future__ import annotations

import importlib
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


@pytest.mark.parametrize("operation", ["discover", "materialize"])
def test_legacy_journal_without_prior_lease_is_owned_during_real_materialization(
    tmp_path, monkeypatch, operation
):
    root, sessions = tmp_path / "data", tmp_path / "sessions"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Legacy journal materialization allocated provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    journal_type = importlib.import_module("here.recording.journal").CaptureJournal
    recovery_module = importlib.import_module("here.application.recovery")
    processing = importlib.import_module("here.application.processing")
    database = importlib.import_module("here.memory.database")
    journal = journal_type.create(sessions, owned=False)
    writer = journal.open_writer(label="synthetic", sample_rate=8000, channels=1)
    writer.write(np.full((1024, 1), 1234, dtype=np.int16))
    writer.close()
    journal.finish(RuntimeError("Synthetic legacy process interruption"))
    source_path = journal.directory / "source_01.wav"
    source_bytes = source_path.read_bytes()
    external = tmp_path / "external-unrelated.wav"
    external.write_bytes(source_bytes)
    first_service = recovery_module.RecoveryService(sessions)
    (candidate,) = first_service.discover()
    assert candidate.capture_id == journal.document.capture_id and candidate.can_retry
    assert not journal_type.owner_path(sessions, candidate.capture_id).exists()
    assert not (root / ".owners" / f"{candidate.capture_id}.lock").exists()
    assert not (root / "library.sqlite3").exists()
    entered, release = threading.Event(), threading.Event()
    outcome = {}
    real_normalize = processing.materialize_normalized_session

    def normalize(session, directory, **kwargs):
        if threading.current_thread() is worker:
            outcome["destination"] = Path(directory)
            entered.set()
            assert release.wait(20), "Test did not release real legacy materialization"
        return real_normalize(session, directory, **kwargs)

    monkeypatch.setattr(processing, "materialize_normalized_session", normalize)

    def materialize_first():
        try:
            outcome["result"] = first_service.materialize(candidate)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=materialize_first, name="synthetic-legacy-materializer")
    worker.start()
    try:
        assert entered.wait(10), repr(outcome.get("error"))
        assert outcome["destination"] == candidate.session_dir
        assert worker.is_alive()
        before_journal = _snapshot(journal.directory)
        before_destination = _snapshot(candidate.session_dir)
        assert not (candidate.session_dir / "session.json").exists()
        accesses = []
        original_info, original_copy = sf.info, shutil.copy2

        def material(path):
            try:
                actual = Path(path).absolute()
            except TypeError:
                return False
            return actual.suffix.lower() == ".wav" and actual.parent in {
                journal.directory.absolute(),
                candidate.session_dir.absolute(),
            }

        def tracked_info(path, *args, **kwargs):
            if material(path):
                accesses.append("audio inspection")
            return original_info(path, *args, **kwargs)

        def tracked_copy(src, dst, *args, **kwargs):
            if material(src) or material(dst):
                accesses.append("audio copy")
            return original_copy(src, dst, *args, **kwargs)

        second_service = recovery_module.RecoveryService(sessions)
        with monkeypatch.context() as watch:
            watch.setattr(sf, "info", tracked_info)
            watch.setattr(shutil, "copy2", tracked_copy)
            if operation == "discover":
                assert not any(
                    item.capture_id == candidate.capture_id for item in second_service.discover()
                ), "Discovery offered a legacy journal whose materializer is still publishing"
            else:
                with pytest.raises(database.Busy):
                    second_service.materialize(candidate)
            assert accesses == [], "A competing recovery inspected material before owner check"
        assert _snapshot(journal.directory) == before_journal
        assert _snapshot(candidate.session_dir) == before_destination
        assert external.read_bytes() == source_bytes
    finally:
        release.set()
        worker.join(15)
        assert not worker.is_alive(), "Real materializer did not finish after explicit release"
    assert "error" not in outcome, repr(outcome.get("error"))
    assert outcome["result"] == candidate.session_dir
    assert (candidate.session_dir / "source_01.wav").read_bytes() == source_bytes
    assert external.read_bytes() == source_bytes
    metadata = json.loads((candidate.session_dir / "session.json").read_text(encoding="utf-8"))
    assert metadata["meeting_id"] == candidate.capture_id and metadata["status"] == "failed"
    normalized = sf.info(candidate.session_dir / metadata["recoverable_audio"])
    assert normalized.channels == 1
    assert normalized.frames * 8000 == 1024 * normalized.samplerate
    # Owned journal material can be consumed only after persistence; its exact raw
    # bytes remain in the published failed session, unrelated origins stay intact.
    after = _snapshot(candidate.session_dir)
    second_service = recovery_module.RecoveryService(sessions)
    assert second_service.materialize(candidate) == candidate.session_dir
    assert second_service.materialize(candidate) == candidate.session_dir
    assert _snapshot(candidate.session_dir) == after
    assert any(item.session_dir == candidate.session_dir for item in second_service.discover())
    assert external.read_bytes() == source_bytes
