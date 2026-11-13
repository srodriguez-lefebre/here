"""R59 recovery visibility and caller-error boundaries with synthetic local evidence."""

from __future__ import annotations

import socket
import sqlite3
import threading
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
import soundfile as sf
from here.memory import service as service_module
from here.memory.database import Busy, LibraryConnection, LibraryError, open_library
from here.memory.service import MemoryService
from here.output.ownership import FileLease, LeaseBusy
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session
from test_memory_delete import _files, _lose_database, _record, _rows

CORRUPT_SQL = b"Synthetic R59 unusable SQLite; no private data."
OPAQUE_ID = "synthetic-r59-opaque"


def _forbidden(*args, **kwargs):
    pytest.fail("R59 read view allocated provider, network or unauthorized SQL work")


@pytest.fixture(autouse=True)
def synthetic_environment(tmp_path, monkeypatch):
    for name in (
        "HERE_PROVIDER_SMOKE_KEY",
        "HERE_WHEEL_PYTHON",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-r59-key")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", _forbidden)


@pytest.fixture
def catalog(tmp_path):
    sessions = tmp_path / "sessions"
    memory = MemoryService(tmp_path / "data", sessions_root=sessions)
    source = make_session(sessions / "known", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    return memory, source, sessions


def _origin(memory, identity=MEETING_ID):
    ((path, fingerprint),) = _rows(
        memory.data_dir,
        "SELECT origin_path,origin_fingerprint FROM ingest_sources WHERE meeting_id=?",
        (identity,),
    )
    return {"origin_path": path, "origin_fingerprint": fingerprint}


def _corrupt(memory, tmp_path):
    _lose_database(memory.data_dir, tmp_path)
    (memory.data_dir / "library.sqlite3").write_bytes(CORRUPT_SQL)


def _released(memory, library=None):
    if library is not None:
        assert not library.connection.in_transaction
        assert library.execute("SELECT 1").fetchone()[0] == 1
    with closing(FileLease(memory.data_dir / ".memory.lock", timeout=0, create=False)):
        pass


@contextmanager
def _observe_audio(monkeypatch, source):
    reads = []
    original_open, original_info = Path.open, sf.info

    def protected(path):
        try:
            actual = Path(path).absolute()
        except TypeError:
            return False
        return actual.parent == source.directory.absolute() and actual.suffix.lower() == ".wav"

    def opened(path, *args, **kwargs):
        if protected(path):
            reads.append("WAV open")
        return original_open(path, *args, **kwargs)

    def info(path, *args, **kwargs):
        if protected(path):
            reads.append("WAV inspection")
        return original_info(path, *args, **kwargs)

    with monkeypatch.context() as watch:
        watch.setattr(Path, "open", opened)
        watch.setattr(sf, "info", info)
        yield reads


@pytest.mark.parametrize("registry", ["absent", "empty"])
def test_known_uuid_without_alias_inputs_does_not_depend_on_corrupt_sql(
    catalog, tmp_path, monkeypatch, registry
):
    memory, source, _ = catalog
    if registry == "empty":
        (memory.data_dir / ".deletions").mkdir()
    before = _files(source.directory)
    _corrupt(memory, tmp_path)
    with monkeypatch.context() as watch:
        watch.setattr(service_module, "open_library", _forbidden)
        watch.setattr(sqlite3, "connect", _forbidden)
        with memory.recovery_read_view(source.directory, MEETING_ID) as identity:
            assert identity == MEETING_ID
            with pytest.raises(LeaseBusy):
                FileLease(memory.data_dir / ".memory.lock", timeout=0, create=False)
    _released(memory)
    assert (memory.data_dir / "library.sqlite3").read_bytes() == CORRUPT_SQL
    assert _files(source.directory) == before


@pytest.mark.parametrize(
    "phase,matching",
    [
        pytest.param("pending", True, id="pending-matching"),
        pytest.param("complete", True, id="complete-matching"),
        pytest.param("pending", False, id="pending-other"),
        pytest.param("complete", False, id="complete-other"),
    ],
)
def test_zero_pair_records_check_direct_ids_without_sql(
    catalog, tmp_path, monkeypatch, phase, matching
):
    memory, source, _ = catalog
    _record(memory.data_dir, identity=MEETING_ID if matching else OTHER_MEETING_ID, phase=phase)
    _corrupt(memory, tmp_path)
    before = _files(source.directory)
    with monkeypatch.context() as watch, _observe_audio(monkeypatch, source) as reads:
        watch.setattr(service_module, "open_library", _forbidden)
        watch.setattr(sqlite3, "connect", _forbidden)
        if matching:
            with pytest.raises(LibraryError):
                with memory.recovery_read_view(source.directory, MEETING_ID):
                    source.audio.read_bytes()
                    pytest.fail("Direct suppressed UUID reached protected consumption")
        else:
            with memory.recovery_read_view(source.directory, MEETING_ID) as identity:
                assert identity == MEETING_ID
    assert reads == [] and _files(source.directory) == before
    _released(memory)


@pytest.mark.parametrize("fault", ["malformed-sibling", "unsafe-role", "inaccessible"])
def test_recovery_registry_setup_failure_is_closed_before_material(catalog, monkeypatch, fault):
    memory, source, _ = catalog
    record = _record(memory.data_dir, identity=OTHER_MEETING_ID)
    if fault == "malformed-sibling":
        # A filename-only target check must not ignore another invalid final.
        (record.parent / f"{uuid4()}.json").write_bytes(b"[]")
    elif fault == "unsafe-role":
        record.unlink()
        record.mkdir()
    before = _files(source.directory)
    registry_before = _files(record.parent)
    original_open = Path.open

    def denied(path, *args, **kwargs):
        if fault == "inaccessible" and path == record:
            raise PermissionError("Synthetic R59 registry access denial")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as deny:
        deny.setattr(Path, "open", denied)
        with _observe_audio(monkeypatch, source) as reads:
            with pytest.raises(LibraryError):
                with memory.recovery_read_view(source.directory, MEETING_ID):
                    source.audio.read_bytes()
                    pytest.fail("Invalid registry became an empty successful view")
        assert reads == []
    assert _files(source.directory) == before and _files(record.parent) == registry_before
    if fault == "unsafe-role":
        assert record.is_dir()
    _released(memory)


@pytest.mark.parametrize("match", ["exact", "path-only", "fingerprint-only"])
def test_known_uuid_alias_suppression_uses_the_exact_catalog_origin_pair(catalog, tmp_path, match):
    memory, source, _ = catalog
    origin = _origin(memory)
    if match == "path-only":
        assert origin["origin_fingerprint"] != "a" * 64
        origin["origin_fingerprint"] = "a" * 64
    elif match == "fingerprint-only":
        origin["origin_path"] = service_module.normalized_origin(tmp_path / "unrelated-origin")
    _record(memory.data_dir, identity=OTHER_MEETING_ID, phase="complete", origins=[origin])
    assert MEETING_ID != OTHER_MEETING_ID
    before = _files(source.directory)
    if match == "exact":
        with pytest.raises(LibraryError):
            with memory.recovery_read_view(source.directory, MEETING_ID):
                pytest.fail("Exact catalog alias was exposed under another tombstone UUID")
    else:
        with memory.recovery_read_view(source.directory, MEETING_ID) as identity:
            assert identity == MEETING_ID
    assert _files(source.directory) == before
    _released(memory)


def test_any_origin_pair_requires_sql_even_for_a_nonmatching_known_uuid(
    catalog, tmp_path, monkeypatch
):
    memory, source, _ = catalog
    origin = _origin(memory)
    origin["origin_path"] = service_module.normalized_origin(tmp_path / "different-origin")
    _record(memory.data_dir, identity=OTHER_MEETING_ID, origins=[origin])
    _corrupt(memory, tmp_path)
    before = _files(source.directory)
    with _observe_audio(monkeypatch, source) as reads:
        with pytest.raises(LibraryError):
            with memory.recovery_read_view(source.directory, MEETING_ID):
                source.audio.read_bytes()
                pytest.fail("UUID or nonmatching current origin bypassed unresolved SQL aliases")
    assert reads == [] and _files(source.directory) == before
    _released(memory)


@pytest.mark.parametrize(
    "source_id,state",
    [
        pytest.param(None, "mapped", id="mapped-none"),
        pytest.param(OPAQUE_ID, "mapped", id="mapped-opaque"),
        pytest.param(None, "absent", id="absent-none"),
        pytest.param(OPAQUE_ID, "absent", id="absent-opaque"),
        pytest.param(None, "corrupt", id="corrupt-none"),
        pytest.param(OPAQUE_ID, "corrupt", id="corrupt-opaque"),
        pytest.param(None, "ambiguous", id="ambiguous-none"),
        pytest.param(OPAQUE_ID, "ambiguous", id="ambiguous-opaque"),
    ],
)
def test_legacy_identity_preserves_real_mapping_and_unknown_ownership(
    tmp_path, monkeypatch, source_id, state
):
    sessions = tmp_path / "sessions"
    memory = MemoryService(tmp_path / "data", sessions_root=sessions)
    source = make_session(sessions / "legacy", meeting_id=source_id)
    expected = source_id
    if state in {"mapped", "ambiguous"}:
        expected, parent = memory.prepare_retry(source.directory, source_id)
        assert expected != source_id and isinstance(expected, str)
        assert parent and memory.retry_parent(source.directory, source_id) == (expected, parent)
    if state == "ambiguous":
        other = make_session(sessions / "other", meeting_id=OTHER_MEETING_ID)
        imported = memory.ingest_session(other.directory)
        assert imported.status == "inserted"
        first_origin = _origin(memory, expected)
        second = _rows(
            memory.data_dir,
            "SELECT origin_fingerprint,revision_id FROM ingest_sources WHERE meeting_id=?",
            (OTHER_MEETING_ID,),
        )[0]
        assert first_origin["origin_fingerprint"] != second[0]
        with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                "INSERT INTO ingest_sources "
                "(id,origin_path,origin_fingerprint,meeting_id,revision_id) VALUES (?,?,?,?,?)",
                (str(uuid4()), first_origin["origin_path"], second[0], OTHER_MEETING_ID, second[1]),
            )
            connection.commit()
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    elif state == "corrupt":
        _corrupt(memory, tmp_path)
    elif state == "absent":
        _lose_database(memory.data_dir, tmp_path)
    before = _files(source.directory)
    with _observe_audio(monkeypatch, source) as reads:
        if state in {"ambiguous", "corrupt"}:
            error = ValueError if state == "ambiguous" else LibraryError
            with pytest.raises(error):
                with memory.recovery_read_view(source.directory, source_id):
                    source.audio.read_bytes()
                    pytest.fail("Unknown legacy owner was treated as unowned")
        else:
            with memory.recovery_read_view(source.directory, source_id) as identity:
                assert identity == expected
                assert identity is None or isinstance(identity, str)
    assert reads == [] and _files(source.directory) == before
    if state == "absent":
        assert not (memory.data_dir / "library.sqlite3").exists()
    _released(memory)


@pytest.mark.parametrize("error_type", [ValueError, OSError], ids=["value", "os"])
def test_catalog_caller_exception_keeps_exact_object_and_owned_resources(catalog, error_type):
    memory, _, _ = catalog
    library = open_library(memory.data_dir)
    expected = error_type("Synthetic R59 caller failure after yield")
    entered = False
    try:
        with pytest.raises(Exception) as caught:
            with service_module.catalog_read_view(library):
                entered = True
                raise expected
        _released(memory, library)
        assert entered and caught.value is expected
    finally:
        library.close()


@pytest.mark.parametrize(
    "sql,error_type",
    [
        pytest.param(False, ValueError, id="direct-value"),
        pytest.param(False, OSError, id="direct-os"),
        pytest.param(True, ValueError, id="sql-value"),
        pytest.param(True, OSError, id="sql-os"),
    ],
)
def test_recovery_caller_exception_keeps_exact_object_and_releases_scope(
    catalog, tmp_path, monkeypatch, sql, error_type
):
    memory, source, _ = catalog
    if sql:
        origin = _origin(memory)
        origin["origin_path"] = service_module.normalized_origin(tmp_path / "nonmatching")
        _record(memory.data_dir, identity=OTHER_MEETING_ID, origins=[origin])
    expected = error_type("Synthetic R59 recovery caller failure after yield")
    opened = []
    original_open = service_module.open_library

    def tracked_open(*args, **kwargs):
        library = original_open(*args, **kwargs)
        opened.append(library)
        return library

    monkeypatch.setattr(service_module, "open_library", tracked_open)
    entered = False
    context = memory.recovery_read_view(source.directory, MEETING_ID)
    with pytest.raises(Exception) as caught:
        with context as identity:
            entered = True
            assert identity == MEETING_ID
            raise expected
    _released(memory)
    assert entered and caught.value is expected
    assert len(opened) == (1 if sql else 0)
    for library in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            library.connection.execute("SELECT 1")


@pytest.mark.parametrize(
    "view,failure",
    [
        pytest.param("catalog", "lock", id="catalog-lock"),
        pytest.param("catalog", "sql", id="catalog-sql"),
        pytest.param("recovery", "lock", id="recovery-lock"),
        pytest.param("recovery", "sql", id="recovery-sql"),
    ],
)
def test_actual_guard_setup_busy_is_typed_and_releases_owned_resources(catalog, view, failure):
    memory, source, _ = catalog
    _record(memory.data_dir, identity=OTHER_MEETING_ID, origins=[_origin(memory)])
    library = open_library(memory.data_dir)
    blocker = None
    lease = None
    try:
        if failure == "lock":
            lease = FileLease(memory.data_dir / ".memory.lock")
        else:
            library.execute("PRAGMA journal_mode=DELETE")
            blocker = sqlite3.connect(memory.data_dir / "library.sqlite3", isolation_level=None)
            blocker.execute("BEGIN EXCLUSIVE")
        with pytest.raises(Busy):
            context = (
                service_module.catalog_read_view(library)
                if view == "catalog"
                else memory.recovery_read_view(source.directory, MEETING_ID)
            )
            with context:
                pytest.fail("Guard setup failure entered protected caller body")
    finally:
        if lease is not None:
            lease.close()
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        try:
            _released(memory, library)
        finally:
            library.close()


def test_recovery_sql_view_uses_one_parser_lock_connection_and_begin(
    catalog, tmp_path, monkeypatch
):
    memory, source, _ = catalog
    origin = _origin(memory)
    origin["origin_path"] = service_module.normalized_origin(tmp_path / "unrelated")
    _record(memory.data_dir, identity=OTHER_MEETING_ID, origins=[origin])
    parsed, locks, opened, closed, statements = [], [], [], [], []
    original_parser = service_module._deletion_registry
    original_lease = FileLease.__init__
    original_open = service_module.open_library
    original_close = LibraryConnection.close

    def parser(root):
        parsed.append(root)
        with pytest.raises(LeaseBusy):
            FileLease(root / ".memory.lock", timeout=0, create=False)
        return original_parser(root)

    def lease(current, path, *args, **kwargs):
        # Successful acquisitions, not the deliberately failed ownership probes.
        result = original_lease(current, path, *args, **kwargs)
        if Path(path) == memory.data_dir / ".memory.lock":
            locks.append(Path(path))
        return result

    def opened_library(*args, **kwargs):
        library = original_open(*args, **kwargs)
        opened.append(library)
        library.connection.set_trace_callback(statements.append)
        return library

    def closed_library(library):
        closed.append(library)
        return original_close(library)

    monkeypatch.setattr(service_module, "_deletion_registry", parser)
    monkeypatch.setattr(FileLease, "__init__", lease)
    monkeypatch.setattr(service_module, "open_library", opened_library)
    monkeypatch.setattr(LibraryConnection, "close", closed_library)
    monkeypatch.setattr(service_module, "catalog_read_view", _forbidden)
    monkeypatch.setattr(service_module, "migrate", _forbidden)
    monkeypatch.setattr(MemoryService, "_operation", _forbidden)
    monkeypatch.setattr(MemoryService, "__init__", _forbidden)
    with memory.recovery_read_view(source.directory, MEETING_ID) as identity:
        assert identity == MEETING_ID
        assert len(opened) == 1 and opened[0].connection.in_transaction
        with pytest.raises(LeaseBusy):
            FileLease(memory.data_dir / ".memory.lock", timeout=0, create=False)
    assert parsed == [memory.data_dir] and len(locks) == 1
    assert len(opened) == 1 and closed == opened
    begins = [
        statement.strip().upper()
        for statement in statements
        if statement.upper().startswith("BEGIN")
    ]
    assert len(begins) == 1 and begins[0] in {"BEGIN", "BEGIN DEFERRED"}
    _released(memory)


def test_final_read_guard_excludes_delete_through_audio_when_owner_files_are_absent(catalog):
    memory, source, sessions = catalog
    writer_memory = MemoryService(memory.data_dir, sessions_root=sessions)
    owners = (
        memory.data_dir / ".owners" / f"{MEETING_ID}.lock",
        sessions / ".capture_owners" / f"{MEETING_ID}.lock",
    )
    for path in owners:
        assert not path.exists()
        with pytest.raises(FileNotFoundError):
            FileLease(path, timeout=0, create=False)
    before = source.audio.read_bytes()
    finished = threading.Event()
    outcome = {}

    def delete():
        try:
            outcome["result"] = writer_memory.delete_meeting(MEETING_ID)
        except Exception as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    writer = None
    try:
        with memory.recovery_read_view(source.directory, MEETING_ID) as identity:
            assert identity == MEETING_ID
            writer = threading.Thread(target=delete, name="synthetic-r59-delete-contender")
            writer.start()
            # Wait for the real bounded lock refusal, not a guessed scheduling delay.
            assert finished.wait(5), "Delete contender did not return from its bounded guard"
            barrier_during_consumption = (
                memory.data_dir / ".deletions" / f"{MEETING_ID}.json"
            ).exists()
            consumed = (identity, source.audio.read_bytes())
    finally:
        if writer is not None:
            writer.join(5)
            assert not writer.is_alive(), "Delete contender survived read-scope closure"
    assert consumed == (MEETING_ID, before)
    assert isinstance(outcome.get("error"), Busy), outcome
    assert not barrier_during_consumption
    assert memory.show(MEETING_ID) is not None
    _released(memory)
    assert writer_memory.delete_meeting(MEETING_ID).status == "completed"
    assert memory.show(MEETING_ID) is None and source.audio.read_bytes() == before


@pytest.mark.parametrize(
    "declared_identity,registry",
    [
        pytest.param(MEETING_ID, "absent", id="uuid-absent"),
        pytest.param(MEETING_ID, "empty", id="uuid-empty"),
        pytest.param(None, "absent", id="none-absent"),
        pytest.param(None, "empty", id="none-empty"),
        pytest.param(OPAQUE_ID, "absent", id="opaque-absent"),
        pytest.param(OPAQUE_ID, "empty", id="opaque-empty"),
    ],
)
def test_module_view_without_catalog_never_constructs_service_or_creates_sqlite(
    tmp_path, monkeypatch, declared_identity, registry
):
    data = tmp_path / "composed-data"
    data.mkdir()
    source = make_session(tmp_path / "sessions" / "unmapped", meeting_id=declared_identity)
    if registry == "empty":
        (data / ".deletions").mkdir()
    before = _files(source.directory)
    with monkeypatch.context() as watch:
        watch.setattr(MemoryService, "__init__", _forbidden)
        watch.setattr(service_module, "migrate", _forbidden)
        watch.setattr(service_module, "open_library", _forbidden)
        watch.setattr(sqlite3, "connect", _forbidden)
        watch.setattr("here.config.settings.get_settings", _forbidden)
        with service_module.recovery_read_view(data, source.directory, declared_identity) as owner:
            assert owner == declared_identity
            assert owner is None or isinstance(owner, str)
            with pytest.raises(LeaseBusy):
                FileLease(data / ".memory.lock", timeout=0, create=False)
    assert _files(source.directory) == before
    for suffix in ("", "-wal", "-shm", "-journal"):
        assert not (data / ("library.sqlite3" + suffix)).exists()
    assert not (data / "meetings").exists() and not (data / ".staging").exists()
    with closing(FileLease(data / ".memory.lock", timeout=0, create=False)):
        pass


def test_service_method_delegates_to_the_same_real_module_primitive(catalog, monkeypatch):
    memory, source, _ = catalog
    original = service_module.recovery_read_view
    delegated = []

    @contextmanager
    def observed(data_dir, directory, declared_identity):
        delegated.append((data_dir, directory, declared_identity))
        with original(data_dir, directory, declared_identity) as identity:
            yield identity

    monkeypatch.setattr(service_module, "recovery_read_view", observed)
    with memory.recovery_read_view(source.directory, MEETING_ID) as identity:
        assert identity == MEETING_ID
        with pytest.raises(LeaseBusy):
            FileLease(memory.data_dir / ".memory.lock", timeout=0, create=False)
    assert delegated == [(memory.data_dir, source.directory, MEETING_ID)]
    _released(memory)
