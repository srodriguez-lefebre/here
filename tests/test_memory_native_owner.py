"""Independent R52 native ownership supplement on the exact B4 source base.

Real controller/OS-lease lifetime tests are separate from synthetic forced-ID
native-only guards. No audio hardware, provider or actual Qt integration is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from here.memory.database import Busy, LibraryError
from here.memory.service import MemoryService
from here.output.ownership import FileLease, LeaseBusy, retain_for_job
from memory_fixtures import MEETING_ID, make_session, write_json


@pytest.fixture(autouse=True)
def synthetic_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(tmp_path / "configured-sessions"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("R52 local ownership test attempted provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "data"
    source = make_session(tmp_path / "external-origin", meeting_id=MEETING_ID)
    memory = MemoryService(root)
    assert memory.ingest_session(source.directory).status == "inserted"
    return root, source


def _files(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as db:
        return db.execute(sql, parameters).fetchall()


def _native_path(root, identity=MEETING_ID):
    from here.recording.journal import CaptureJournal

    return CaptureJournal.owner_path(root, identity)


def _native_lease(root, identity=MEETING_ID):
    path = _native_path(root, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLease(path)


def _pending(root):
    origin, fingerprint = _rows(root, "SELECT origin_path,origin_fingerprint FROM ingest_sources")[
        0
    ]
    directory = root / ".deletions"
    directory.mkdir(exist_ok=True)
    write_json(
        directory / f"{MEETING_ID}.json",
        {
            "schema_version": 1,
            "meeting_id": MEETING_ID,
            "origins": [{"origin_path": origin, "origin_fingerprint": fingerprint}],
            "phase": "pending",
        },
    )


def _assert_pending(result):
    assert (result.status, result.meeting_id, result.error_category) == (
        "pending",
        MEETING_ID,
        "io",
    )
    assert result.reason in {"busy", "hook_failed", "purge_failed", "io", "cleanup_pending"}


def _assert_lock_held(path):
    with pytest.raises(LeaseBusy):
        probe = FileLease(path, timeout=0, create=False)
        probe.close()


@pytest.mark.parametrize("selection", ["explicit-default", "custom"])
def test_explicit_configured_root_allows_idle_delete_without_creating_native_paths(
    catalog, selection
):
    root, source = catalog
    sessions = root / "sessions" if selection == "explicit-default" else root.parent / "custom"
    assert not sessions.exists()
    memory = MemoryService(root, sessions_root=sessions)
    before = _files(source.directory)
    result = memory.delete_meeting(MEETING_ID)
    assert result.status == "completed"
    assert not sessions.exists()
    assert _files(source.directory) == before


def test_bare_read_and_nondestructive_repair_work_without_settings_or_native_context(
    catalog, monkeypatch
):
    root, _ = catalog

    def settings_forbidden(*args, **kwargs):
        pytest.fail("Memory core loaded settings or inferred native authority")

    monkeypatch.setattr("here.config.settings.get_settings", settings_forbidden)
    memory = MemoryService(root)
    assert memory.get_meeting(MEETING_ID).meeting.id == MEETING_ID
    assert memory.show(MEETING_ID)["meeting"]["id"] == MEETING_ID
    memory.reconcile()  # Healthy local inspection needs no destructive authority.
    assert memory.show(MEETING_ID) is not None


def test_bare_destructive_check_fails_closed_without_fabricated_busy(catalog):
    root, _ = catalog
    memory = MemoryService(root)
    before = _files(root / "meetings")
    with pytest.raises(LibraryError) as failure:
        memory.delete_meeting(MEETING_ID)
    assert not isinstance(failure.value, Busy)
    assert not (root / ".deletions" / f"{MEETING_ID}.json").exists()
    assert memory.show(MEETING_ID) is not None
    assert _files(root / "meetings") == before


def test_bare_prior_barrier_remains_pending_hidden_without_cleanup(catalog):
    root, _ = catalog
    _pending(root)
    memory = MemoryService(root)
    before = _files(root / "meetings")
    _assert_pending(memory.delete_meeting(MEETING_ID))
    assert memory.show(MEETING_ID) is None
    memory.reconcile()
    assert _files(root / "meetings") == before
    assert (
        json.loads((root / ".deletions" / f"{MEETING_ID}.json").read_text())["phase"] == "pending"
    )


def test_registered_roots_are_retained_lexical_deduplicated_and_core_never_loads_settings(
    catalog, monkeypatch
):
    root, _ = catalog
    first, second = root.parent / "first", root.parent / "second"
    for directory in (first, second):
        lease = _native_lease(directory)
        lease.close()

    def forbidden(*args, **kwargs):
        pytest.fail("Memory core attempted to load settings")

    monkeypatch.setattr("here.config.settings.get_settings", forbidden)
    memory = MemoryService(root, sessions_root=first)
    memory._register_sessions_root(first.absolute())
    memory._register_sessions_root(first / ".")
    memory._register_sessions_root(second)
    # The first root must remain authoritative after registering a second root.
    with closing(_native_lease(first)):
        with pytest.raises(Busy):
            memory.delete_meeting(MEETING_ID)
    assert not (root / ".deletions" / f"{MEETING_ID}.json").exists()
    # Duplicate lexical roots must not cause a recursive acquisition of our own lease.
    assert memory.delete_meeting(MEETING_ID).status == "completed"


@pytest.mark.parametrize("phase", ["prebarrier", "accepted"])
def test_synthetic_forced_old_id_native_only_guard_obeys_pre_and_post_barrier(catalog, phase):
    root, _ = catalog
    sessions = root.parent / "explicit-synthetic-guard-root"
    memory = MemoryService(root, sessions_root=sessions)
    if phase == "accepted":
        _pending(root)
    before = _files(root / "meetings")
    # Deliberately forced old ID. This is NOT the supported new-ID producer opening path.
    with closing(_native_lease(sessions)):
        if phase == "prebarrier":
            with pytest.raises(Busy):
                memory.delete_meeting(MEETING_ID)
            assert memory.show(MEETING_ID) is not None
        else:
            result = memory.delete_meeting(MEETING_ID)
            _assert_pending(result)
            assert result.reason == "busy"
            assert memory.show(MEETING_ID) is None
        assert _files(root / "meetings") == before
        assert _native_path(sessions).exists()


def test_real_new_capture_opening_gets_new_identity_and_cannot_block_unrelated_old_meeting(catalog):
    from here.recording.journal import CaptureJournal

    root, _ = catalog
    sessions = root.parent / "configured"
    journal = CaptureJournal.create(sessions, owned=True)
    try:
        assert journal.document.capture_id != MEETING_ID
        _assert_lock_held(CaptureJournal.owner_path(sessions, journal.document.capture_id))
        before = _files(journal.directory)
        memory = MemoryService(root, sessions_root=sessions)
        assert memory.delete_meeting(MEETING_ID).status == "completed"
        assert _files(journal.directory) == before
        _assert_lock_held(CaptureJournal.owner_path(sessions, journal.document.capture_id))
    finally:
        journal.release_ownership()


@pytest.mark.parametrize("operation", ["begin_capture", "begin_processing"])
def test_supported_canonical_lease_survives_catalog_sql_failure_using_existing_b4_api(
    catalog, monkeypatch, operation
):
    root, _ = catalog
    memory = MemoryService(root)

    @contextmanager
    def failed_sql():
        raise LibraryError("synthetic SQL unavailable after owner acquisition")
        yield  # pragma: no cover

    with monkeypatch.context() as fault:
        fault.setattr(memory, "_operation", failed_sql)
        with pytest.raises(LibraryError):
            getattr(memory, operation)(MEETING_ID)
        _assert_lock_held(root / ".owners" / f"{MEETING_ID}.lock")
        with pytest.raises(Busy):
            MemoryService.check_owner(root, MEETING_ID)
    memory.finish_capture(MEETING_ID)
    MemoryService.check_owner(root, MEETING_ID)


class _Live:
    def __init__(self, events):
        self.events = events

    def abort(self):
        self.events.append("live_abort")

    def cleanup(self):
        self.events.append("live_cleanup")

    def wait_closed(self):
        self.events.append("live_closed")


def test_real_controller_terminal_cleanup_keeps_canonical_owner_until_native_release(
    catalog, tmp_path
):
    """Reach the B4 cleanup tail, without any future constructor/registration API."""
    from here.application.controller import HereApplicationController
    from here.application.models import ApplicationState, StartRequest
    from here.recording.isolated import WindowsRecordingHandle
    from here.recording.journal import CaptureJournal
    from here.recording.models import RecordingSession

    root, _ = catalog
    memory = MemoryService(root)
    events, released_under_canonical = [], []
    captures = []
    sessions = tmp_path / "per-request-root"

    class SyntheticNativeHandle(WindowsRecordingHandle):
        def __init__(self):
            self._journal = CaptureJournal.create(sessions, owned=True)
            self._thread = None
            self._scoped_owner = retain_for_job(self.close_ownership)
            assert self._scoped_owner
            self.opened_sources = ()
            self.live_error = None

        def wait(self, timeout=None):
            events.append("native_helper_closed")
            return RecordingSession(sources=[])

        def stop(self):
            pass

        def close_ownership(self):
            owner = getattr(getattr(self, "_journal", None), "_owner", None)
            if owner is not None and not owner.stream.closed:
                assert "native_helper_closed" in events
                assert "persisted" in events and "live_closed" in events
                try:
                    MemoryService.check_owner(root, self.meeting_id)
                except Busy:
                    released_under_canonical.append(True)
                else:
                    released_under_canonical.append(False)
                events.append("native_release")
            super().close_ownership()

    def capture(request, sink):
        handle = SyntheticNativeHandle()
        captures.append(handle)
        return handle

    class Processor:
        def process(self, session, output_dir, **kwargs):
            assert "native_helper_closed" in events
            assert output_dir == sessions
            source = make_session(
                output_dir / "synthetic-persisted", meeting_id=captures[0].meeting_id
            )
            events.append("persisted")
            return SimpleNamespace(session_dir=source.directory)

    controller = HereApplicationController(
        capture_factory=capture,
        live_factory=lambda *args: _Live(events),
        processor=Processor(),
        memory_service=memory,
    )
    controller.start(StartRequest(output_dir=sessions))
    result = controller.wait_until_terminal(8)
    assert result.worker_complete and result.state is ApplicationState.COMPLETED
    assert released_under_canonical == [True], events
    assert events.index("native_release") > events.index("live_closed")
    assert controller.memory_service is memory and memory.data_dir == root.absolute()
    identity = captures[0].meeting_id
    MemoryService.check_owner(root, identity)
    with closing(FileLease(_native_path(sessions, identity), timeout=0, create=False)):
        pass
    captures[0].close_ownership()  # Existing owned_job/native cleanup remains idempotent.
    assert released_under_canonical == [True]


def test_foreign_capture_without_close_ownership_keeps_existing_controller_contract(tmp_path):
    from here.application.controller import HereApplicationController
    from here.application.models import ApplicationState, StartRequest
    from here.recording.models import RecordingSession

    events = []

    class ForeignCapture:
        def wait(self):
            return RecordingSession(sources=[])

    class Processor:
        def process(self, session, output_dir, **kwargs):
            source = make_session(output_dir / "foreign")
            return SimpleNamespace(session_dir=source.directory)

    memory = MemoryService(tmp_path / "data")
    controller = HereApplicationController(
        capture_factory=lambda *args: ForeignCapture(),
        live_factory=lambda *args: _Live(events),
        processor=Processor(),
        memory_service=memory,
    )
    controller.start(StartRequest(output_dir=tmp_path / "foreign-root"))
    result = controller.wait_until_terminal(8)
    assert result.state is ApplicationState.COMPLETED and result.worker_complete
    assert "live_closed" in events


def test_start_request_registers_same_injected_memory_before_worker_launch(catalog, monkeypatch):
    from here.application.controller import HereApplicationController
    from here.application.models import StartRequest
    from here.recording.reservation import audio_reservation

    root, _ = catalog
    memory = MemoryService(root)
    sessions = root.parent / "request-override"
    registered, launched = [], []
    register = memory._register_sessions_root  # Missing API fails here, not at collection.

    def record_root(path):
        register(path)
        registered.append(Path(path).absolute())

    monkeypatch.setattr(memory, "_register_sessions_root", record_root)
    controller = HereApplicationController(memory_service=memory)

    def launch(worker, *, release_audio=False):
        assert controller.memory_service is memory
        assert memory.data_dir == root.absolute()
        assert registered == [sessions.absolute()]
        launched.append(worker)

    monkeypatch.setattr(controller, "_launch_worker", launch)
    try:
        controller.start(StartRequest(output_dir=sessions))
        assert len(launched) == 1
    finally:
        audio_reservation.release()


def test_recovery_passes_its_known_root_before_reconciliation(catalog, monkeypatch):
    from here.application import recovery as recovery_module

    root, _ = catalog
    sessions = root.parent / "recovery-root"
    sessions.mkdir()
    received = []
    original_reconcile = MemoryService.reconcile

    def inspect(self):
        # Behaviorally require the known root by testing a held synthetic native owner.
        with closing(_native_lease(sessions)):
            with pytest.raises(Busy):
                self.delete_meeting(MEETING_ID)
        received.append(self)
        return original_reconcile(self)

    monkeypatch.setattr(MemoryService, "reconcile", inspect)
    recovery_module.RecoveryService(sessions).discover()
    assert len(received) == 1
    assert received[0].data_dir == root.absolute()


_CHILD_BOOTSTRAP = r"""
import json, os, pathlib, socket, sys, types
repo = pathlib.Path(sys.argv[1])
package = types.ModuleType('here')
package.__path__ = [str(repo / 'src')]
sys.modules['here'] = package
sounddevice = types.ModuleType('sounddevice')
sounddevice.CallbackFlags = type('CallbackFlags', (), {})
sounddevice.InputStream = type('InputStream', (), {})
sys.modules['sounddevice'] = sounddevice
def forbidden(*args, **kwargs):
    raise AssertionError('Native-owner child attempted network')
socket.socket.connect = forbidden
"""
_CLI_CHILD = (
    _CHILD_BOOTSTRAP
    + r"""
from typer.testing import CliRunner
from here.cli import app
result = CliRunner().invoke(app, json.loads(sys.argv[2]))
print(json.dumps({'exit': result.exit_code, 'stdout': result.stdout,
                  'output': result.output, 'exception': type(result.exception).__name__
                  if result.exception else None}))
"""
)
_OWNER_CHILD = (
    _CHILD_BOOTSTRAP
    + r"""
from here.memory.service import MemoryService
memory = MemoryService(pathlib.Path(sys.argv[2]))
identity, ready = sys.argv[3], pathlib.Path(sys.argv[4])
memory.begin_processing(identity)
try:
    # Real terminal publication may complete SQL while the producer still owns cleanup.
    memory.set_status(identity, 'completed')
    ready.write_text('canonical owner held', encoding='utf-8')
    sys.stdin.buffer.read(1)
finally:
    memory.finish_capture(identity)
"""
)


def _environment(root, sessions):
    env = dict(os.environ)
    env["HERE_DATA_DIR"] = str(root)
    env["TRANSCRIPTIONS_DIR"] = str(sessions)
    env["HERE_ENV_FILE"] = str(root.parent / "absent-child.env")
    env["OPENAI_API_KEY"] = "synthetic-test-key"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("HERE_PROVIDER_SMOKE_KEY", None)
    env.pop("HERE_WHEEL_PYTHON", None)
    return env


def _fresh_cli(root, sessions, command):
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            _CLI_CHILD,
            str(Path(__file__).resolve().parents[1]),
            json.dumps(["memory", *command, "--json"]),
        ],
        env=_environment(root, sessions),
        capture_output=True,
        text=True,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(process.stdout)


@pytest.mark.parametrize("configured", ["default", "custom"])
def test_fresh_cli_uses_actual_configured_sessions_root_for_native_guard(catalog, configured):
    root, _ = catalog
    sessions = root / "sessions" if configured == "default" else root.parent / "configured-other"
    # Synthetic forced old-ID guard verifies explicit CLI composition, not a producer path.
    with closing(_native_lease(sessions)):
        result = _fresh_cli(root, sessions, ["delete", MEETING_ID, "--yes"])
    assert result["exit"] == 4, result
    document = json.loads(result["stdout"])
    assert document["error_category"] == "io"
    assert _rows(root, "SELECT id FROM meetings") == [(MEETING_ID,)]
    assert not (root / ".deletions" / f"{MEETING_ID}.json").exists()


def test_fresh_cli_reconcile_does_not_purge_pending_native_owned_tree(catalog):
    root, _ = catalog
    sessions = root.parent / "configured-recovery"
    _pending(root)
    before = _files(root / "meetings")
    with closing(_native_lease(sessions)):
        result = _fresh_cli(root, sessions, ["reconcile"])
    assert result["exit"] == 0, result
    document = json.loads(result["stdout"])
    assert document["warnings"] or document["skipped"]
    assert _files(root / "meetings") == before
    assert (
        json.loads((root / ".deletions" / f"{MEETING_ID}.json").read_text())["phase"] == "pending"
    )


def test_separate_supported_processing_owner_blocks_fresh_cli_with_different_native_root(
    catalog, tmp_path
):
    root, _ = catalog
    request_root = tmp_path / "producer-request-root"
    other_root = tmp_path / "fresh-process-configured-root"
    ready = tmp_path / "owner-ready"
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _OWNER_CHILD,
            str(Path(__file__).resolve().parents[1]),
            str(root),
            MEETING_ID,
            str(ready),
        ],
        env=_environment(root, request_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file() and owner.poll() is None
        assert _rows(root, "SELECT status FROM meetings") == [("completed",)]
        _assert_lock_held(root / ".owners" / f"{MEETING_ID}.lock")
        result = _fresh_cli(root, other_root, ["delete", MEETING_ID, "--yes"])
        assert result["exit"] == 4, result
        assert _rows(root, "SELECT id FROM meetings") == [(MEETING_ID,)]
        assert not (root / ".deletions" / f"{MEETING_ID}.json").exists()
    finally:
        if owner.poll() is None:
            try:
                owner.communicate(input=b"release", timeout=5)
            except subprocess.TimeoutExpired:
                owner.kill()
                owner.communicate(timeout=5)
        else:
            owner.communicate(timeout=5)
    assert owner.returncode == 0


def test_cli_delete_gets_context_within_existing_settings_operation(catalog, monkeypatch):
    from here import cli as cli_module
    from here.config import settings as settings_module
    from typer.testing import CliRunner

    root, _ = catalog
    sessions = root.parent / "operation-snapshot-root"
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(sessions))
    seen = []
    original = cli_module.MemoryService

    def construct(data_dir, **kwargs):
        snapshot = settings_module._operation_settings.get()
        assert snapshot is not None, "CLI did not enter existing settings_operation"
        assert snapshot.TRANSCRIPTIONS_DIR == sessions
        assert kwargs.get("sessions_root") == sessions
        seen.append(Path(data_dir))
        return original(data_dir, **kwargs)

    monkeypatch.setattr(cli_module, "MemoryService", construct)
    result = CliRunner().invoke(cli_module.app, ["memory", "delete", MEETING_ID, "--yes", "--json"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert seen == [root]


def test_unsafe_registered_root_is_validated_at_use_without_resolving_symlink(catalog, tmp_path):
    root, _ = catalog
    declared = tmp_path / "declared"
    declared.mkdir()
    memory = MemoryService(root, sessions_root=declared)
    declared.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_bytes(b"synthetic external material")
    try:
        declared.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink privilege unavailable")
    try:
        with pytest.raises(LibraryError) as failure:
            memory.delete_meeting(MEETING_ID)
        assert not isinstance(failure.value, Busy)
        assert sentinel.read_bytes() == b"synthetic external material"
        assert not (outside / ".capture_owners").exists()
        assert not (root / ".deletions" / f"{MEETING_ID}.json").exists()
    finally:
        declared.unlink()


@pytest.mark.parametrize("role", ["root", "owner-directory", "owner-file"])
def test_unsafe_explicit_native_roles_fail_closed_and_preserve_outside_target(
    catalog, tmp_path, role
):
    root, _ = catalog
    sessions = tmp_path / "declared"
    sessions.mkdir()
    outside = tmp_path / "external-target"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_bytes(b"outside must remain")
    if role == "root":
        sessions.rmdir()
        link, target = sessions, outside
    elif role == "owner-directory":
        link, target = sessions / ".capture_owners", outside
    else:
        (sessions / ".capture_owners").mkdir()
        link, target = _native_path(sessions), sentinel
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        pytest.skip("Symlink privilege unavailable")
    try:
        # Lexical registration must not silently resolve the link into authority.
        try:
            memory = MemoryService(root, sessions_root=sessions)
        except LibraryError as error:
            assert not isinstance(error, Busy)
        else:
            with pytest.raises(LibraryError) as failure:
                memory.delete_meeting(MEETING_ID)
            assert not isinstance(failure.value, Busy)
        assert sentinel.read_bytes() == b"outside must remain"
        assert _rows(root, "SELECT id FROM meetings") == [(MEETING_ID,)]
    finally:
        link.unlink()


def test_origin_receipt_manifest_paths_never_grant_native_probe_authority(catalog, monkeypatch):
    root, source = catalog
    explicit = root.parent / "explicit"
    unauthorized = source.directory.parent
    memory = MemoryService(root, sessions_root=explicit)
    before = _files(source.directory)
    original = FileLease.__init__

    def guarded(self, path, *args, **kwargs):
        if ".capture_owners" in Path(path).parts:
            assert Path(path).is_relative_to(explicit), "Origin granted native-root authority"
        return original(self, path, *args, **kwargs)

    original_stat, original_open = Path.stat, Path.open

    def guarded_stat(self, *args, **kwargs):
        if ".capture_owners" in self.parts:
            assert self.is_relative_to(explicit), "Origin granted native inspection authority"
        return original_stat(self, *args, **kwargs)

    def guarded_open(self, *args, **kwargs):
        assert not self.is_relative_to(source.directory), "Delete reopened an external origin"
        return original_open(self, *args, **kwargs)

    # The external source is recorded in ingest_sources and immutable receipt/state.
    assert _rows(root, "SELECT origin_path FROM ingest_sources")
    with monkeypatch.context() as guard:
        guard.setattr(FileLease, "__init__", guarded)
        guard.setattr(Path, "stat", guarded_stat)
        guard.setattr(Path, "open", guarded_open)
        assert memory.delete_meeting(MEETING_ID).status == "completed"
    assert _files(source.directory) == before
    assert not (source.directory / ".capture_owners").exists()
    assert not (unauthorized / ".capture_owners").exists()


@pytest.mark.parametrize("accepted", [False, True])
def test_native_owner_acquired_during_hook_ack_is_revalidated(catalog, accepted):
    root, _ = catalog
    sessions = root.parent / "explicit"
    memory = MemoryService(root, sessions_root=sessions)
    if accepted:
        _pending(root)
    held = []

    def acknowledge(identity):
        assert identity == MEETING_ID
        held.append(_native_lease(sessions))

    memory.before_delete.append(acknowledge)
    before = _files(root / "meetings")
    try:
        if accepted:
            result = memory.delete_meeting(MEETING_ID)
            _assert_pending(result)
            assert result.reason == "busy"
        else:
            with pytest.raises(Busy):
                memory.delete_meeting(MEETING_ID)
            assert not (root / ".deletions" / f"{MEETING_ID}.json").exists()
        assert held and _files(root / "meetings") == before
    finally:
        for lease in held:
            lease.close()


def test_applicable_native_and_canonical_leases_are_retained_through_files_and_final_sql(
    catalog, monkeypatch
):
    root, _ = catalog
    sessions = root.parent / "explicit"
    native = _native_lease(sessions)
    native.close()  # Known inactive owner file; deletion must retain its acquired probe.
    memory = MemoryService(root, sessions_root=sessions)
    canonical = root / ".owners" / f"{MEETING_ID}.lock"
    canonical.parent.mkdir(exist_ok=True)
    with closing(FileLease(canonical)):
        pass
    observed = []
    original_unlink, original_replace = Path.unlink, os.replace

    def acknowledge(identity):
        assert identity == MEETING_ID
        assert not (root / ".deletions" / f"{MEETING_ID}.json").exists()
        # ACK is not acceptance: callback may acquire every lock the GUI needs.
        with (
            closing(FileLease(root / ".memory.lock", timeout=0)),
            closing(FileLease(canonical, timeout=0, create=False)),
            closing(FileLease(_native_path(sessions), timeout=0, create=False)),
        ):
            with closing(sqlite3.connect(root / "library.sqlite3", timeout=0)) as db:
                db.execute("BEGIN IMMEDIATE")
                db.rollback()
        observed.append("ack")

    def replace(source, destination, *args, **kwargs):
        pending = (
            Path(destination) == root / ".deletions" / f"{MEETING_ID}.json"
            and json.loads(Path(source).read_text(encoding="utf-8")).get("phase") == "pending"
        )
        if pending:
            assert "ack" in observed
            _assert_lock_held(canonical)
            _assert_lock_held(_native_path(sessions))
        result = original_replace(source, destination, *args, **kwargs)
        if pending:
            _assert_lock_held(canonical)
            _assert_lock_held(_native_path(sessions))
            observed.append("barrier")
        return result

    def unlink(self, *args, **kwargs):
        if self.is_relative_to(root / "meetings" / MEETING_ID):
            _assert_lock_held(canonical)
            _assert_lock_held(_native_path(sessions))
            observed.append("file")
        if self == _native_path(sessions):
            # A retained native lock cannot be unlinked, even during final cleanup.
            with closing(FileLease(self, timeout=0, create=False)):
                pass
        return original_unlink(self, *args, **kwargs)

    def final_sql(repo, identity):
        assert repo.connection.in_transaction
        _assert_lock_held(canonical)
        _assert_lock_held(_native_path(sessions))
        observed.append("final_sql")

    memory.before_delete.append(acknowledge)
    memory._derived_row_purgers.append(final_sql)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    assert memory.delete_meeting(MEETING_ID).status == "completed"
    assert observed.index("ack") < observed.index("barrier") < observed.index("file")
    assert observed[-1] == "final_sql"
    MemoryService.check_owner(root, MEETING_ID)
    if _native_path(sessions).exists():
        with closing(FileLease(_native_path(sessions), timeout=0, create=False)):
            pass


def _prepared_publication(root):
    operation, revision = str(uuid4()), str(uuid4())
    stage = root / ".staging" / operation
    stage.mkdir(parents=True)
    transcript = stage / "transcript.txt"
    transcript.write_bytes(b"synthetic unpublished text")
    hashes = {"transcript.txt": hashlib.sha256(transcript.read_bytes()).hexdigest()}
    write_json(
        stage / ".publication.json",
        {
            "schema_version": 1,
            "operation_id": operation,
            "hashes": hashes,
        },
    )
    with closing(sqlite3.connect(root / "library.sqlite3")) as db:
        db.execute(
            "INSERT INTO pending_operations "
            "(id,kind,phase,meeting_id,staging_path,destination_path,payload_json) "
            "VALUES (?,'publish_revision','prepared',?,?,?,?)",
            (
                operation,
                MEETING_ID,
                f".staging/{operation}",
                f"meetings/{MEETING_ID}/revisions/{revision}",
                json.dumps({"revision_id": revision, "hashes": hashes}),
            ),
        )
        db.commit()
    return operation, stage


def test_bare_existing_b4_reconcile_preserves_destructive_stage_without_native_context(catalog):
    """Existing reconcile reaches cleanup on B4: this RED need not wait for new APIs."""
    root, _ = catalog
    operation, stage = _prepared_publication(root)
    before = _files(stage)
    report = MemoryService(root).reconcile()
    assert stage.exists(), "Bare reconcile destroyed prepared files with unknown native context"
    assert _files(stage) == before
    assert _rows(root, "SELECT phase FROM pending_operations WHERE id=?", (operation,)) == [
        ("prepared",)
    ]
    assert report.warnings or any(item.identity == operation for item in report.skipped)


def test_fresh_cli_existing_reconcile_honors_native_guard_before_prepared_stage_cleanup(catalog):
    root, _ = catalog
    sessions = root.parent / "configured-cleanup-root"
    operation, stage = _prepared_publication(root)
    before = _files(stage)
    with closing(_native_lease(sessions)):
        result = _fresh_cli(root, sessions, ["reconcile"])
    assert result["exit"] == 0, result
    assert stage.exists(), "CLI did not pass its actual configured native root to reconcile"
    assert _files(stage) == before
    assert _rows(root, "SELECT phase FROM pending_operations WHERE id=?", (operation,)) == [
        ("prepared",)
    ]
    report = json.loads(result["stdout"])
    assert report["warnings"] or any(item["identity"] == operation for item in report["skipped"])


def test_explicit_root_absent_does_not_authorize_creation_or_global_native_scan(
    catalog, monkeypatch
):
    root, _ = catalog
    declared = root.parent / "nonexistent-trusted"
    unrelated = root.parent / "unrelated-neighbor"
    unrelated.mkdir()
    (unrelated / "marker.txt").write_bytes(b"synthetic neighbor")
    memory = MemoryService(root, sessions_root=declared)
    original_iterdir = Path.iterdir

    def forbid_scan(self):
        if self in {root.parent, unrelated}:
            pytest.fail("Missing declared root triggered global directory discovery")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", forbid_scan)
    assert memory.delete_meeting(MEETING_ID).status == "completed"
    assert not declared.exists()
    assert (unrelated / "marker.txt").read_bytes() == b"synthetic neighbor"


def test_native_root_junction_is_refused_without_touching_target(catalog, tmp_path):
    if os.name != "nt":
        pytest.skip("Windows junction semantics")
    root, _ = catalog
    declared, outside = tmp_path / "declared-junction", tmp_path / "outside-junction"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_bytes(b"external synthetic target")
    process = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(declared), str(outside)],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert declared.is_junction() and declared.absolute().is_relative_to(tmp_path.absolute())
    try:
        try:
            memory = MemoryService(root, sessions_root=declared)
        except LibraryError as error:
            assert not isinstance(error, Busy)
        else:
            with pytest.raises(LibraryError) as failure:
                memory.delete_meeting(MEETING_ID)
            assert not isinstance(failure.value, Busy)
        assert sentinel.read_bytes() == b"external synthetic target"
        assert not (outside / ".capture_owners").exists()
    finally:
        assert declared.is_junction() and declared.absolute().is_relative_to(tmp_path.absolute())
        declared.rmdir()


@pytest.mark.parametrize("accepted", [False, True])
def test_unverifiable_native_context_has_stage_correct_error_without_busy_claim(catalog, accepted):
    root, _ = catalog
    sessions = root.parent / "context-becomes-invalid"
    sessions.mkdir()
    memory = MemoryService(root, sessions_root=sessions)
    if accepted:
        _pending(root)
    invalid = sessions / ".capture_owners"
    invalid.write_bytes(b"ordinary file is not a native-owner directory")
    before = _files(root / "meetings")
    if accepted:
        _assert_pending(memory.delete_meeting(MEETING_ID))
        assert memory.show(MEETING_ID) is None
    else:
        with pytest.raises(LibraryError) as failure:
            memory.delete_meeting(MEETING_ID)
        assert not isinstance(failure.value, Busy)
        assert not (root / ".deletions" / f"{MEETING_ID}.json").exists()
    assert invalid.read_bytes() == b"ordinary file is not a native-owner directory"
    assert _files(root / "meetings") == before
