"""R62 recovery publication boundaries over real local catalog state."""

from __future__ import annotations

import json
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import soundfile as sf
from here.application.processing import SessionProcessor
from here.application.recovery import RecoveryService
from here.memory import service as service_module
from here.memory.artifacts import publication_hashes, read_pending_intent, write_durable_json
from here.memory.database import LibraryConnection
from here.memory.service import MemoryService, OwnershipBusy
from here.output.ownership import FileLease, LeaseBusy
from here.output.session_writer import read_session_metadata
from here.transcription.client import TranscriptionResult
from memory_fixtures import make_session, write_marked_wav
from test_memory_delete import _files, _rows

OPAQUE_ID = "synthetic-r62-legacy-identity"
MARKER = "memory_pending.json"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    ambient = tmp_path / "ambient-data"
    monkeypatch.setenv("HERE_DATA_DIR", str(ambient))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(sessions))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-r62-key")
    monkeypatch.setenv("CLEANUP_ENABLED", "false")

    def forbidden(*args, **kwargs):
        pytest.fail("Local recovery allocated provider or network transport")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    memory = MemoryService(tmp_path / "catalog", sessions_root=sessions)
    yield memory, sessions
    assert not ambient.exists()


def _sealed_session(sessions, source_id):
    source = make_session(sessions / "legacy", meeting_id=source_id)
    metadata = read_session_metadata(source.directory)
    assert metadata is not None and metadata.meeting_id == source_id
    write_durable_json(
        source.directory / MARKER,
        {
            "schema_version": 3,
            "meeting_id": source_id,
            "revision_of": None,
            "intent": "ingest",
            "publication_id": str(uuid4()),
            "source_kind": "session",
            "hashes": dict(sorted(publication_hashes(source.directory, metadata).items())),
        },
    )
    return source.directory


def _catalog_state(memory):
    return (
        _rows(memory.data_dir, "SELECT id,status,active_revision_id FROM meetings ORDER BY id"),
        _rows(memory.data_dir, "SELECT id,meeting_id FROM transcript_revisions ORDER BY id"),
        _rows(memory.data_dir, "SELECT origin_path,meeting_id FROM ingest_sources ORDER BY id"),
    )


@contextmanager
def _observe_source(monkeypatch, directory, at_access=None):
    """Record real material access and marker writes, including identical replacements."""
    effects = []
    real_open, real_info = Path.open, sf.info
    real_replace, real_unlink = Path.replace, Path.unlink
    marker = directory / MARKER

    def record(kind, path):
        effects.append((kind, path.name))
        if at_access is not None:
            at_access(kind)

    def opened(path, mode="r", *args, **kwargs):
        if path.parent == directory:
            writing = any(flag in mode for flag in "wax+")
            if path == marker or path.name.startswith(f".{MARKER}."):
                if writing:
                    record("marker-write", path)
            elif path.name != "session.json":
                record("material-open", path)
        return real_open(path, mode, *args, **kwargs)

    def info(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).parent == directory:
            record("material-info", Path(path))
        return real_info(path, *args, **kwargs)

    def replace(path, target):
        if Path(target) == marker:
            record("marker-replace", marker)
        return real_replace(path, target)

    def unlink(path, *args, **kwargs):
        if path == marker:
            record("marker-remove", marker)
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as watch:
        watch.setattr(Path, "open", opened)
        watch.setattr(sf, "info", info)
        watch.setattr(Path, "replace", replace)
        watch.setattr(Path, "unlink", unlink)
        yield effects


def _schedule_handoff(monkeypatch, memory, directory, contend):
    """Run the contender at the first writer entry reached by the real consumer."""
    real_publish = SessionProcessor.publish_memory
    real_scope = getattr(memory, "recovery_publication", None)
    started = False

    def interleave(target):
        nonlocal started
        if target == directory and not started:
            started = True
            contend()

    def publish(processor, session_dir, *args, **kwargs):
        interleave(session_dir)
        return real_publish(processor, session_dir, *args, **kwargs)

    monkeypatch.setattr(SessionProcessor, "publish_memory", publish)
    if real_scope is not None:

        @contextmanager
        def scope(directory, declared_identity, *, expected_identity):
            interleave(directory)
            with real_scope(
                directory, declared_identity, expected_identity=expected_identity
            ) as value:
                yield value

        monkeypatch.setattr(memory, "recovery_publication", scope)


@pytest.mark.parametrize(
    "source_id,change",
    [
        pytest.param(None, "owner", id="none-mapping-owner"),
        pytest.param(OPAQUE_ID, "owner", id="opaque-mapping-owner"),
        pytest.param(None, "delete", id="none-delete"),
        pytest.param(OPAQUE_ID, "delete", id="opaque-delete"),
    ],
)
def test_handoff_denies_new_mapping_owner_or_delete_before_source_effects(
    sandbox, monkeypatch, source_id, change
):
    memory, sessions = sandbox
    directory = _sealed_session(sessions, source_id)
    contender = MemoryService(memory.data_dir, sessions_root=sessions)
    before = _files(directory)
    assert memory.retry_parent(directory, source_id) is None
    handoff = {}

    with ExitStack() as owners, ExitStack() as observation, monkeypatch.context() as schedule:

        def contend():
            identity, parent = contender.prepare_retry(directory, source_id)
            assert str(UUID(identity)) == identity and parent
            if change == "owner":
                owners.callback(contender.finish_capture, identity)
                contender.begin_processing(identity)
                with pytest.raises(OwnershipBusy):
                    MemoryService.check_owner(memory.data_dir, identity)
            else:
                ((origin_path, fingerprint),) = _rows(
                    memory.data_dir,
                    "SELECT origin_path,origin_fingerprint FROM ingest_sources WHERE meeting_id=?",
                    (identity,),
                )
                assert contender.delete_meeting(identity).status == "completed"
                tombstone = memory.data_dir / ".deletions" / f"{identity}.json"
                record = json.loads(tombstone.read_text(encoding="utf-8"))
                assert record["phase"] == "complete"
                assert record["origins"] == [
                    {"origin_path": origin_path, "origin_fingerprint": fingerprint}
                ]
                assert memory.show(identity) is None
            assert _files(directory) == before
            handoff["identity"] = identity
            handoff["catalog"] = _catalog_state(memory)
            if change == "delete":
                assert handoff["catalog"] == ([], [], [])
            handoff["effects"] = observation.enter_context(_observe_source(monkeypatch, directory))

        _schedule_handoff(schedule, memory, directory, contend)
        RecoveryService(sessions, memory_service=memory).discover()
        assert handoff, "The real consumer did not reach the scheduled writer handoff"
        if change == "owner":
            with pytest.raises(OwnershipBusy):
                MemoryService.check_owner(memory.data_dir, handoff["identity"])

    assert (handoff["effects"], _files(directory) == before) == ([], True)
    assert _catalog_state(memory) == handoff["catalog"]


@pytest.mark.parametrize("source_id", [None, OPAQUE_ID], ids=["none", "opaque"])
def test_stable_unmapped_replay_preserves_ordinary_import_policy(sandbox, source_id):
    memory, sessions = sandbox
    directory = _sealed_session(sessions, source_id)
    before = _files(directory)
    assert memory.retry_parent(directory, source_id) is None

    candidates = RecoveryService(sessions, memory_service=memory).discover()

    meetings, revisions, origins = _catalog_state(memory)
    if source_id is not None:
        assert (meetings, revisions, origins) == ([], [], [])
        assert _files(directory) == before
        return
    assert candidates == []
    ((identity, status, active),) = meetings
    assert str(UUID(identity)) == identity and status == "completed"
    assert revisions == [(active, identity)]
    assert len(origins) == 1 and origins[0][1] == identity
    del before[MARKER]
    assert _files(directory) == before


class _SavedBeforeCatalog(BaseException):
    """Interrupt a real retry after compatibility artifacts and seal are saved."""


@pytest.mark.parametrize("authority", ["parent", "audio-reservation"])
def test_missing_real_parent_or_reservation_keeps_marker_and_material_untouched(
    sandbox, tmp_path, monkeypatch, authority
):
    old_memory, sessions = sandbox
    if authority == "parent":
        directory = make_session(sessions / "legacy").directory
        identity, parent = old_memory.prepare_retry(directory, None)
    else:
        directory = sessions / "legacy-audio-only"
        directory.mkdir(parents=True)
        write_marked_wav(directory / "audio.wav")
        parent = None

    def interrupted(*args, **kwargs):
        raise _SavedBeforeCatalog

    def synthetic_result(*args, **kwargs):
        return TranscriptionResult("Synthetic saved words", "Synthetic saved words")

    with monkeypatch.context() as cut:
        cut.setattr(SessionProcessor, "publish_memory", interrupted)
        processor = SessionProcessor(transcribe=synthetic_result, memory_service=old_memory)
        with pytest.raises(_SavedBeforeCatalog):
            processor.retry(directory)

    metadata = read_session_metadata(directory)
    assert metadata is not None and metadata.status == "completed" and metadata.meeting_id is None
    pending = read_pending_intent(directory, None)
    assert pending.hashes is not None and pending.revision_of == parent
    if authority == "audio-reservation":
        assert pending.source_kind == "legacy_audio_only"
        ((identity,),) = _rows(
            old_memory.data_dir,
            "SELECT meeting_id FROM pending_operations WHERE kind='reserve_audio_only'",
        )
        assert _rows(old_memory.data_dir, "SELECT id FROM transcript_revisions") == []
    else:
        assert _rows(old_memory.data_dir, "SELECT id FROM transcript_revisions") == [(parent,)]
    assert old_memory.retry_parent(directory, None) == (identity, parent)
    MemoryService.check_owner(old_memory.data_dir, identity)

    # Loss of this catalog and its receipts leaves the original sealed source intact.
    root = old_memory.data_dir
    retired = tmp_path / "retired-catalog"
    assert root.resolve().parent == retired.resolve().parent == tmp_path.resolve()
    assert not root.is_symlink() and not root.is_junction() and not retired.exists()
    root.rename(retired)
    memory = MemoryService(root, sessions_root=sessions)
    assert memory.retry_parent(directory, None) is None
    before = _files(directory)
    catalog = _catalog_state(memory)
    assert catalog == ([], [], [])

    with _observe_source(monkeypatch, directory) as effects:
        RecoveryService(sessions, memory_service=memory).discover()

    assert (effects, _files(directory) == before) == ([], True)
    assert _catalog_state(memory) == catalog


def test_writer_scope_covers_source_effects_with_one_connection_and_no_read_transaction(
    sandbox, monkeypatch
):
    memory, sessions = sandbox
    directory = _sealed_session(sessions, None)
    assert memory.retry_parent(directory, None) is None
    opened, closed, boundaries = [], [], []
    real_open, real_close = service_module.open_library, LibraryConnection.close
    lock = memory.data_dir / ".memory.lock"

    def open_catalog(*args, **kwargs):
        library = real_open(*args, **kwargs)
        opened.append(library)
        return library

    def close_catalog(library):
        real_close(library)
        closed.append(library)

    def at_access(kind):
        try:
            with closing(FileLease(lock, timeout=0, create=False)):
                locked = False
        except LeaseBusy:
            locked = True
        in_transaction = any(lib.connection.in_transaction for lib in opened if lib not in closed)
        boundaries.append((kind, locked, in_transaction))

    with monkeypatch.context() as observe:
        observe.setattr(service_module, "open_library", open_catalog)
        observe.setattr(LibraryConnection, "close", close_catalog)
        with _observe_source(monkeypatch, directory, at_access):
            with memory.recovery_publication(directory, None, expected_identity=None):
                result = SessionProcessor(memory_service=memory).publish_memory(directory)

    assert result.status == "inserted"
    assert len(opened) == 1 and closed == opened
    assert {kind for kind, _, _ in boundaries} >= {"material-open", "marker-remove"}
    assert all(locked and not in_transaction for _, locked, in_transaction in boundaries)
    with closing(FileLease(lock, timeout=0, create=False)):
        pass
    assert not (directory / MARKER).exists()
