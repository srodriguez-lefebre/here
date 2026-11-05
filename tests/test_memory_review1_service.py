"""Round-one review regressions for committed publication outcomes."""

from __future__ import annotations

import gc
import hashlib
import importlib
import sqlite3
from contextlib import closing

import numpy as np
import pytest
import soundfile as sf
from memory_fixtures import MEETING_ID, make_corpus, make_session


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


@pytest.mark.parametrize("step", ["receipt", "state", "completion"])
@pytest.mark.parametrize("failure_mode", ["one-shot", "persistent"])
def test_postcommit_housekeeping_failure_reports_committed_retry_and_recovers(
    tmp_path,
    monkeypatch,
    step,
    failure_mode,
):
    module = importlib.import_module("here.memory.service")
    database = importlib.import_module("here.memory.database")
    repository = importlib.import_module("here.memory.repository")
    root = tmp_path / "data"
    memory = module.MemoryService(root)
    corpus = make_corpus(tmp_path / "origins")
    assert memory.ingest_session(corpus.identified.directory).status == "inserted"
    previous = _rows(root, "SELECT id,raw_text,content_hash FROM transcript_revisions")[0]
    old_files = {
        path: path.read_bytes()
        for path in (root / "meetings" / MEETING_ID).rglob("*")
        if path.is_file()
    }
    failures = []
    if step == "receipt":
        target, name = module.MemoryService, "_write_receipt"
    elif step == "state":
        target, name = module.MemoryService, "_write_state"
    else:
        target, name = repository.MemoryRepository, "transition_pending_operation"
    original = getattr(target, name)

    def housekeeping(self, *args, **kwargs):
        if step == "completion" and args[1] != "complete":
            return original(self, *args, **kwargs)
        # Fault only after the actual revision/activation transaction is visible.
        published = _rows(root, "SELECT id FROM transcript_revisions WHERE revision_number=2")
        if published and (failure_mode == "persistent" or not failures):
            assert _rows(root, "SELECT active_revision_id FROM meetings") == published
            failures.append(step)
            if failure_mode == "persistent":
                raise database.Busy("Synthetic postcommit housekeeping remains busy")
            raise OSError("Synthetic one-shot postcommit housekeeping failure")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(target, name, housekeeping)
        result = memory.ingest_session(
            corpus.conflicting_identity.directory, revision_of=previous[0]
        )
    assert failures, "The injected fault must occur after the real commit"
    assert (result.status, result.meeting_id) == ("revised", MEETING_ID)
    assert result.warnings
    revisions = _rows(
        root,
        "SELECT id,revision_number,raw_text FROM transcript_revisions ORDER BY revision_number",
    )
    assert [row[1:] for row in revisions] == [(1, previous[1]), (2, "Synthetic changed canary.")]
    assert _rows(root, "SELECT active_revision_id FROM meetings") == [(revisions[1][0],)]
    assert _rows(
        root, "SELECT id,raw_text,content_hash FROM transcript_revisions WHERE id=?", (previous[0],)
    ) == [previous]
    assert all(path.read_bytes() == contents for path, contents in old_files.items())
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Recovery used network"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    memory.reconcile()
    memory.reconcile()
    assert (
        _rows(
            root,
            "SELECT id,revision_number,raw_text FROM transcript_revisions ORDER BY revision_number",
        )
        == revisions
    )
    assert _rows(root, "SELECT COUNT(*) FROM pending_operations WHERE phase='prepared'") == [(0,)]
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_audio_a_b_a_revisions_resolve_exact_reused_asset_before_and_after_rebuild(
    tmp_path,
    monkeypatch,
):
    module = importlib.import_module("here.memory.service")
    root = tmp_path / "data"
    memory = module.MemoryService(root)
    origins = tmp_path / "origins"
    sources = []
    for name, text in (
        ("first", "First evidence on A"),
        ("second", "Evidence on B"),
        ("third", "New evidence on A again"),
    ):
        sources.append(
            make_session(
                origins / name,
                meeting_id=MEETING_ID,
                transcript_text=text,
                segments=[{"text": text, "start": 0.0, "end": 1.0}],
            )
        )
    sf.write(sources[1].audio, np.full(16000, 2345, dtype=np.int16), 8000, subtype="PCM_16")
    expected_hashes = [hashlib.sha256(source.audio.read_bytes()).hexdigest() for source in sources]
    assert expected_hashes[0] == expected_hashes[2] != expected_hashes[1]
    previous = None
    for index, source in enumerate(sources):
        result = memory.ingest_session(source.directory, revision_of=previous)
        assert result.status == ("inserted" if index == 0 else "revised")
        previous = memory.show(MEETING_ID)["meeting"]["active_revision_id"]

    def resolved(show):
        revisions = sorted(show["revisions"], key=lambda value: value["revision_number"])
        links = show["revision_audio_assets"]
        assets = {asset["id"]: asset for asset in show["audio_assets"]}
        assert len(assets) == 2 and len(links) == 3
        selected = []
        for revision, expected_hash in zip(revisions, expected_hashes, strict=True):
            (link,) = [link for link in links if link["revision_id"] == revision["id"]]
            assert link["meeting_id"] == revision["meeting_id"] == MEETING_ID
            assert (link["ordinal"], link["source_file"]) == (0, "audio.wav")
            asset = assets[link["asset_id"]]
            assert asset["sha256"] == expected_hash
            assert (
                hashlib.sha256((root / asset["relative_path"]).read_bytes()).hexdigest()
                == expected_hash
            )
            selected.append(asset["id"])
        assert selected[0] == selected[2] != selected[1]
        return sorted(links, key=lambda value: (value["revision_id"], value["ordinal"]))

    before = memory.show(MEETING_ID)
    links_before = resolved(before)
    del memory
    gc.collect()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    origins.rename(tmp_path / "unavailable-origins")
    assert root.resolve().is_relative_to(tmp_path.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Rebuild used network"))
    restored = module.MemoryService(root)
    restored.reconcile()
    after = restored.show(MEETING_ID)
    assert resolved(after) == links_before
    assert after["meeting"]["active_revision_id"] == before["meeting"]["active_revision_id"]
    assert sorted(after["revisions"], key=lambda value: value["id"]) == sorted(
        before["revisions"], key=lambda value: value["id"]
    )
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_explicit_retry_of_exact_known_origin_creates_revisions_without_duplicate_source_key(
    tmp_path,
    monkeypatch,
):
    module = importlib.import_module("here.memory.service")
    root = tmp_path / "data"
    memory = module.MemoryService(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    original_files = {path.name: path.read_bytes() for path in source.directory.iterdir()}
    assert memory.ingest_session(source.directory).status == "inserted"
    previous = _rows(root, "SELECT id,raw_text,content_hash FROM transcript_revisions")[0]
    outcomes = []
    for _attempt in range(2):
        parent = _rows(root, "SELECT active_revision_id FROM meetings")[0][0]
        memory.begin_processing(MEETING_ID)
        try:
            outcomes.append(memory.ingest_session(source.directory, revision_of=parent).status)
        finally:
            memory.finish_capture(MEETING_ID)
    assert {path.name: path.read_bytes() for path in source.directory.iterdir()} == original_files
    memory.reconcile()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    del memory
    gc.collect()
    assert root.resolve().is_relative_to(tmp_path.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Rebuild used network"))
    restored = module.MemoryService(root)
    restored.reconcile()
    assert outcomes == ["revised", "revised"]
    assert _rows(root, "SELECT id,status FROM meetings") == [(MEETING_ID, "completed")]
    revisions = _rows(
        root,
        "SELECT id,revision_number,raw_text,content_hash FROM transcript_revisions "
        "ORDER BY revision_number",
    )
    assert [row[1] for row in revisions] == [1, 2, 3]
    assert revisions[0][0] == previous[0]
    assert all(row[2:] == previous[1:] for row in revisions)
    assert _rows(root, "SELECT active_revision_id FROM meetings") == [(revisions[-1][0],)]
    assert _rows(root, "SELECT COUNT(*) FROM ingest_sources") == [(1,)]
    assert restored.ingest_session(source.directory).status == "unchanged"


def test_control_ordinary_identical_import_is_still_idempotent(tmp_path):
    root = tmp_path / "data"
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    previous = _rows(root, "SELECT id,raw_text,content_hash FROM transcript_revisions")
    assert memory.ingest_session(source.directory).status == "unchanged"
    assert _rows(root, "SELECT id,raw_text,content_hash FROM transcript_revisions") == previous
    assert _rows(root, "SELECT id,status FROM meetings") == [(MEETING_ID, "completed")]
