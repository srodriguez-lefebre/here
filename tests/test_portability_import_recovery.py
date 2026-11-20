"""Crash cutovers for real portable publication, recovery and deletion."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import sysconfig
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import pytest
from here.memory.artifacts import PORTABLE_GROUP_NAME, PORTABLE_RECEIPT_NAME
from here.memory.models import MeetingDetail
from here.memory.portability import PortabilityService
from here.memory.service import MemoryService, NotFound
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="uses native portable export")

_NATIVE_ID = "33333333-3333-4333-8333-333333333333"
_CHILD_IMPORT = r"""
import pathlib
import sys
import types

repo, data, sessions, package, ready = map(pathlib.Path, sys.argv[1:6])
cutover = sys.argv[6]
here = types.ModuleType("here")
here.__path__ = [str(repo / "src")]
sys.modules["here"] = here

from here.memory import service
from here.memory.artifacts import PORTABLE_RECEIPT_NAME
from here.memory.portability import PortabilityService
from here.memory.repository import MemoryRepository

def pause():
    ready.touch()
    if not sys.stdin.buffer.read(1):
        raise RuntimeError("Harness closed before terminating the importer")

if cutover == "after-extraction":
    original = service.MemoryService._publish_portable_import

    def publish(self, batch, *, cancelled):
        pause()
        return original(self, batch, cancelled=cancelled)

    service.MemoryService._publish_portable_import = publish

elif cutover == "before-sql-commit":
    original = MemoryRepository.insert_portable_meeting

    def insert(self, bundle, provenance):
        original(self, bundle, provenance)
        assert self.library.connection.in_transaction
        pause()

    MemoryRepository.insert_portable_meeting = insert

elif cutover == "after-first-commit":
    original = service.MemoryService._publish_portable_meeting

    def publish_one(self, repo, stage, bundle, plan):
        original(self, repo, stage, bundle, plan)
        pause()

    service.MemoryService._publish_portable_meeting = publish_one

elif cutover == "before-first-receipt":
    original = service.MemoryService._finish_portable_receipts

    def finish(self, repo, destination, plan):
        assert not repo.library.connection.in_transaction
        pause()
        return original(self, repo, destination, plan)

    service.MemoryService._finish_portable_receipts = finish

elif cutover == "before-registration-replace":
    original = pathlib.Path.replace

    def replace(self, target):
        if pathlib.Path(target).name == ".import.json":
            assert self.is_file()
            pause()
        return original(self, target)

    pathlib.Path.replace = replace

elif cutover in {"after-first-receipt", "before-group-proof", "after-group-proof"}:
    original = service.write_durable_json

    def write(path, document, *args, **kwargs):
        is_proof = path.name.startswith("portable-group-") and path.parent.name == "states"
        if cutover == "before-group-proof" and is_proof:
            pause()
        outcome = original(path, document, *args, **kwargs)
        if (cutover == "after-first-receipt" and path.name == PORTABLE_RECEIPT_NAME) or (
            cutover == "after-group-proof" and is_proof
        ):
            pause()
        return outcome

    service.write_durable_json = write

memory = service.MemoryService(data, sessions_root=sessions)
PortabilityService(memory).import_archive(package, lambda _: None, lambda: False)
"""


def _memory(directory: Path) -> MemoryService:
    sessions = directory / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    return MemoryService(directory / "data", sessions_root=sessions)


@pytest.fixture
def source(tmp_path: Path) -> MemoryService:
    memory = _memory(tmp_path / "origin")
    sessions = memory.data_dir.parent / "sessions"
    first = make_session(sessions / "first", meeting_id=MEETING_ID)
    assert memory.ingest_session(first.directory).status == "inserted"
    parent = memory.get_meeting(MEETING_ID).meeting.active_revision_id
    assert parent is not None
    retry = make_session(
        sessions / "retry", meeting_id=MEETING_ID, transcript_text="Synthetic retry evidence."
    )
    assert memory.ingest_session(retry.directory, revision_of=parent).status == "revised"
    second = make_session(sessions / "second", meeting_id=OTHER_MEETING_ID)
    assert memory.ingest_session(second.directory).status == "inserted"
    return memory


@pytest.fixture
def target(tmp_path: Path) -> MemoryService:
    return _memory(tmp_path / "destination")


def _export(memory: MemoryService, path: Path, ids: tuple[str, ...] = (MEETING_ID,)) -> Path:
    return PortabilityService(memory).export(ids, path, True, lambda _: None, lambda: False).path


@contextmanager
def _paused_import(
    target: MemoryService, package: Path, cutover: str
) -> Iterator[subprocess.Popen[bytes]]:
    ready = package.parent / "import-ready"
    sessions = target.data_dir.parent / "sessions"
    env = {
        **os.environ,
        "HERE_DATA_DIR": str(target.data_dir),
        "HERE_ENV_FILE": str(ready.with_suffix(".absent.env")),
        "OPENAI_API_KEY": "synthetic-test-key",
    }
    env.pop("HERE_PROVIDER_SMOKE_KEY", None)
    executable = sys.executable
    if sys.prefix != sys.base_prefix:
        # The Windows venv redirector has a different PID from the importer.
        executable = sys._base_executable
        dependency_paths = dict.fromkeys(sysconfig.get_path(key) for key in ("purelib", "platlib"))
        env["PYTHONPATH"] = os.pathsep.join(dependency_paths)
    with subprocess.Popen(
        [
            executable,
            "-c",
            _CHILD_IMPORT,
            str(Path(__file__).resolve().parents[1]),
            str(target.data_dir),
            str(sessions),
            str(package),
            str(ready),
            cutover,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ) as process:
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            if process.poll() is not None:
                _, stderr = process.communicate(timeout=5)
                pytest.fail(
                    f"Importer exited with {process.returncode} before {cutover}: "
                    f"{stderr.decode(errors='replace')}"
                )
            assert ready.is_file(), f"Importer did not reach {cutover}"
            yield process
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def _kill(process: subprocess.Popen[bytes]) -> None:
    assert process.poll() is None
    process.kill()
    assert process.wait(timeout=5) != 0


def _lose_database(memory: MemoryService, tmp_path: Path, process: subprocess.Popen[bytes]) -> None:
    assert process.poll() is not None
    root = memory.data_dir
    assert root == tmp_path / "destination" / "data"
    assert root.resolve() == tmp_path.resolve() / "destination" / "data"
    assert not any(path.is_symlink() or path.is_junction() for path in (root, root.parent))
    for name in (
        "library.sqlite3",
        "library.sqlite3-wal",
        "library.sqlite3-shm",
        "library.sqlite3-journal",
    ):
        path = root / name
        assert not path.is_symlink() and not path.is_junction()
        path.unlink(missing_ok=True)


def _import_stage(memory: MemoryService) -> Path:
    (stage,) = (memory.data_dir / ".staging").glob("import-*")
    return stage


def _meeting_root(memory: MemoryService, identity: str = MEETING_ID) -> Path:
    return memory.data_dir / "meetings" / identity


def _probe_path(path: Path) -> Path:
    # Test observations must work on long synthetic paths independently of product I/O.
    name = str(path)
    if name.startswith("\\\\?\\"):
        return path
    return Path("\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name)


def _archive_files(directory: Path) -> dict[Path, bytes]:
    directory = _probe_path(directory)
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _assert_restored(memory: MemoryService, expected: MeetingDetail) -> None:
    identity = expected.meeting.id
    actual = memory.get_meeting(identity)
    assert actual.meeting.active_revision_id == expected.meeting.active_revision_id
    assert [(item.id, item.revision_number, item.raw_text) for item in actual.revisions] == [
        (item.id, item.revision_number, item.raw_text) for item in expected.revisions
    ]
    assert actual.segments == expected.segments
    assert actual.revision_audio_assets == expected.revision_audio_assets
    assert [(item.id, item.sha256) for item in actual.audio_assets] == [
        (item.id, item.sha256) for item in expected.audio_assets
    ]
    assert all(
        _probe_path(memory.data_dir / item.relative_path).is_file() for item in actual.audio_assets
    )


def _journal(memory: MemoryService) -> list[tuple[str, str]]:
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
        return connection.execute(
            "SELECT meeting_id, phase FROM pending_operations "
            "WHERE kind='publish_portable' ORDER BY meeting_id"
        ).fetchall()


@pytest.mark.parametrize(
    ("cutover", "journal"),
    [("after-extraction", []), ("before-sql-commit", [(MEETING_ID, "reverted")])],
)
def test_killed_import_before_catalog_commit_leaves_no_meetings(
    source: MemoryService,
    target: MemoryService,
    tmp_path: Path,
    cutover: str,
    journal: list[tuple[str, str]],
) -> None:
    package = _export(source, tmp_path / "uncommitted.here.zip")
    destination = _meeting_root(target)
    with _paused_import(target, package, cutover) as process:
        stage = _import_stage(target)
        assert _probe_path(stage / "payload" / "meetings" / MEETING_ID / "meeting.json").is_file()
        assert destination.is_dir() is (cutover == "before-sql-commit")
        _kill(process)

    fresh = _memory(target.data_dir.parent)
    with pytest.raises(NotFound):
        fresh.delete_meeting(MEETING_ID)
    assert not (target.data_dir / ".deletions" / f"{MEETING_ID}.json").exists()
    fresh.reconcile()
    assert fresh.show(MEETING_ID) is None
    assert not destination.exists()
    assert not stage.exists()
    assert _journal(fresh) == journal


def test_killed_two_meeting_import_keeps_exactly_the_first_committed_group(
    source: MemoryService, target: MemoryService, tmp_path: Path
) -> None:
    package = _export(source, tmp_path / "batch.here.zip", (MEETING_ID, OTHER_MEETING_ID))
    first = source.get_meeting(MEETING_ID)
    with _paused_import(target, package, "after-first-commit") as process:
        stage = _import_stage(target)
        assert _meeting_root(target).is_dir()
        assert not _meeting_root(target, OTHER_MEETING_ID).exists()
        _kill(process)

    fresh = _memory(target.data_dir.parent)
    fresh.reconcile()
    _assert_restored(fresh, first)
    assert fresh.show(OTHER_MEETING_ID) is None
    assert not _meeting_root(fresh, OTHER_MEETING_ID).exists()
    assert not stage.exists()
    assert _journal(fresh) == [(MEETING_ID, "complete")]


def test_killed_import_during_registration_cleans_the_exact_owned_intent_temporary(
    source: MemoryService, target: MemoryService, tmp_path: Path
) -> None:
    package = _export(source, tmp_path / "before-registration.here.zip")
    with _paused_import(target, package, "before-registration-replace") as process:
        stage = _import_stage(target)
        assert not (stage / ".import.json").exists()
        assert not (stage / "payload").exists()
        assert len(list(stage.iterdir())) == 2
        _kill(process)

    neighbor = stage.parent / "synthetic-unowned"
    neighbor.write_bytes(b"independent synthetic staging file")
    fresh = _memory(target.data_dir.parent)
    fresh.reconcile()

    assert not stage.exists()
    assert neighbor.read_bytes() == b"independent synthetic staging file"
    assert fresh.show(MEETING_ID) is None


@pytest.mark.parametrize(
    ("cutover", "receipt_count", "complete"),
    [
        ("before-first-receipt", 0, False),
        ("after-first-receipt", 1, False),
        ("before-group-proof", 2, False),
        ("after-group-proof", 2, True),
    ],
)
def test_catalog_loss_rebuilds_only_a_complete_portable_receipt_group(
    source: MemoryService,
    target: MemoryService,
    tmp_path: Path,
    cutover: str,
    receipt_count: int,
    complete: bool,
) -> None:
    package = _export(source, tmp_path / "catalog-loss.here.zip")
    native = make_session(target.data_dir.parent / "sessions" / "native", meeting_id=_NATIVE_ID)
    assert target.ingest_session(native.directory).status == "inserted"
    native_before = target.get_meeting(_NATIVE_ID)
    imported_before = source.get_meeting(MEETING_ID)

    with _paused_import(target, package, cutover) as process:
        stage = _import_stage(target)
        imported = _meeting_root(target)
        assert (imported / PORTABLE_GROUP_NAME).is_file()
        assert len(list(_probe_path(imported).rglob(PORTABLE_RECEIPT_NAME))) == receipt_count
        assert bool(list((imported / "states").glob("portable-group-*.json"))) is complete
        _kill(process)

    assert target.show(MEETING_ID) is not None
    files = _archive_files(imported)
    _lose_database(target, tmp_path, process)
    fresh = _memory(target.data_dir.parent)
    report = fresh.reconcile()

    _assert_restored(fresh, native_before)
    if complete:
        _assert_restored(fresh, imported_before)
    else:
        assert fresh.show(MEETING_ID) is None
        assert report.warnings
    assert _archive_files(imported) == files
    assert not stage.exists()
    fresh.reconcile()
    assert (fresh.show(MEETING_ID) is not None) is complete


@pytest.mark.parametrize("cutover", ["before-first-receipt", "after-group-proof"])
def test_deleting_committed_portable_group_purges_local_copy_and_blocks_reimport(
    source: MemoryService, target: MemoryService, tmp_path: Path, cutover: str
) -> None:
    package = _export(source, tmp_path / "delete.here.zip")
    archive = package.read_bytes()
    with _paused_import(target, package, cutover) as process:
        stage = _import_stage(target)
        _kill(process)

    fresh = _memory(target.data_dir.parent)
    assert fresh.show(MEETING_ID) is not None
    assert fresh.delete_meeting(MEETING_ID).status == "completed"
    assert not _meeting_root(fresh).exists()
    assert not stage.exists()
    assert package.read_bytes() == archive
    report = PortabilityService(fresh).import_archive(package, lambda _: None, lambda: False)
    assert [(item.meeting_id, item.status) for item in report.meetings] == [
        (MEETING_ID, "rejected")
    ]
    assert fresh.show(MEETING_ID) is None


def test_reconcile_retains_unknown_files_in_abandoned_import_stage(
    source: MemoryService, target: MemoryService, tmp_path: Path
) -> None:
    package = _export(source, tmp_path / "unknown.here.zip")
    with _paused_import(target, package, "after-extraction") as process:
        stage = _import_stage(target)
        _kill(process)

    unknown = stage / "unregistered.txt"
    unknown.write_bytes(b"Synthetic foreign stage contents")
    before = _archive_files(stage)
    fresh = _memory(target.data_dir.parent)
    report = fresh.reconcile()

    assert report.warnings
    assert fresh.show(MEETING_ID) is None
    assert not _meeting_root(fresh).exists()
    assert _archive_files(stage) == before
