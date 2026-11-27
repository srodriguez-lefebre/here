"""Public CLI coverage for recoverable portable import and per-meeting reports."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from io import BufferedReader
from os import SEEK_END
from pathlib import Path
from zipfile import ZipFile

import pytest
from here.cli import app
from here.memory.database import IntegrityFailure
from here.memory.repository import MemoryRepository
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session
from typer.testing import CliRunner

MANIFEST_ORDER = (OTHER_MEETING_ID, MEETING_ID)
THIRD_MEETING_ID = "33333333-3333-4333-8333-333333333333"
PARTIAL_BATCH_ORDER = (*MANIFEST_ORDER, THIRD_MEETING_ID)
requires_native_export = pytest.mark.skipif(
    sys.platform != "win32", reason="synthetic packages use the native portable exporter"
)
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = tmp_path / "target-sessions"
    sessions.mkdir()
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "target-catalog"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(sessions))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("Portable import attempted provider or network work")

    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)


def _require_command() -> None:
    result = runner.invoke(app, ["meetings", "import", "--help"])
    assert result.exit_code == 0, result.output + repr(result.exception)


def _origin(tmp_path: Path, ids: tuple[str, ...] = MANIFEST_ORDER) -> MemoryService:
    sessions = tmp_path / "origin-sessions"
    sessions.mkdir()
    memory = MemoryService(tmp_path / "origin-catalog", sessions_root=sessions)
    for identity in ids:
        source = make_session(sessions / identity, meeting_id=identity)
        assert memory.ingest_session(source.directory).status == "inserted"
    return memory


def _package(
    origin: MemoryService,
    path: Path,
    ids: tuple[str, ...] = MANIFEST_ORDER,
    *,
    include_audio: bool = False,
) -> Path:
    from here.memory.portability import PortabilityService

    PortabilityService(origin).export(ids, path, include_audio, lambda _: None, lambda: False)
    if len(ids) > 1:
        # The exporter sorts IDs. Use a distinct manifest order to catch CLI-side sorting.
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            payloads = [
                (name, archive.read(name)) for name in archive.namelist() if name != "manifest.json"
            ]
        manifest["meeting_ids"] = list(ids)
        with ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
            for name, payload in payloads:
                archive.writestr(name, payload)
    return path


def _target(tmp_path: Path) -> MemoryService:
    return MemoryService(tmp_path / "target-catalog", sessions_root=tmp_path / "target-sessions")


def _import_json(path: Path, *, code: int | None = None) -> dict:
    result = runner.invoke(app, ["meetings", "import", str(path), "--json"])
    if code is not None:
        assert result.exit_code == code, result.output + repr(result.exception)
    else:
        # The CLI contract defines the exit categories, with mixed-report precedence pending.
        assert result.exit_code in {0, 2, 3, 4}, result.output + repr(result.exception)
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    return json.loads(result.stdout)


def _outcomes(report: dict) -> list[tuple[str, str, str]]:
    return [(item["meeting_id"], item["status"], item["reason"]) for item in report["meetings"]]


@requires_native_export
def test_import_reports_inserted_then_unchanged_in_manifest_order(tmp_path: Path) -> None:
    _require_command()
    origin = _origin(tmp_path)
    package = _package(origin, tmp_path / "ordered meeting package.here.zip")
    before = package.read_bytes()

    assert _outcomes(_import_json(package, code=0)) == [
        (OTHER_MEETING_ID, "inserted", ""),
        (MEETING_ID, "inserted", ""),
    ]
    target = _target(tmp_path)
    original_rows = tuple(target.show(identity) for identity in MANIFEST_ORDER)
    assert all(row is not None for row in original_rows)

    assert _outcomes(_import_json(package, code=0)) == [
        (OTHER_MEETING_ID, "unchanged", ""),
        (MEETING_ID, "unchanged", ""),
    ]
    assert tuple(target.show(identity) for identity in MANIFEST_ORDER) == original_rows
    assert package.read_bytes() == before


@requires_native_export
def test_import_reports_conflict_and_blocks_other_meetings_in_preflight(tmp_path: Path) -> None:
    _require_command()
    origin = _origin(tmp_path)
    initial = _package(origin, tmp_path / "without-audio.here.zip", (MEETING_ID,))
    assert _outcomes(_import_json(initial, code=0)) == [(MEETING_ID, "inserted", "")]
    target = _target(tmp_path)
    original = target.show(MEETING_ID)
    assert original is not None

    conflict = _package(
        origin, tmp_path / "with-audio.here.zip", (MEETING_ID,), include_audio=True
    )
    assert _outcomes(_import_json(conflict, code=3)) == [
        (MEETING_ID, "conflict", "different_meeting")
    ]

    batch = _package(origin, tmp_path / "batch-with-audio.here.zip", include_audio=True)
    before = batch.read_bytes()
    assert _outcomes(_import_json(batch)) == [
        (OTHER_MEETING_ID, "rejected", "batch_stopped"),
        (MEETING_ID, "conflict", "different_meeting"),
    ]
    assert target.show(MEETING_ID) == original
    assert target.show(OTHER_MEETING_ID) is None
    assert batch.read_bytes() == before


@requires_native_export
def test_import_keeps_partial_report_after_first_entity_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_command()
    package = _package(_origin(tmp_path), tmp_path / "interrupted.here.zip")
    before = package.read_bytes()
    target = _target(tmp_path)

    def fail_receipts(self, repo, destination, plan):
        assert self.data_dir == target.data_dir
        assert plan.group.meeting_id == OTHER_MEETING_ID
        raise OSError("Synthetic receipt finalization failure after entity commit")

    monkeypatch.setattr(MemoryService, "_finish_portable_receipts", fail_receipts)
    assert _outcomes(_import_json(package)) == [
        (OTHER_MEETING_ID, "inserted", "local_recovery_pending"),
        (MEETING_ID, "rejected", "batch_stopped"),
    ]
    assert target.show(OTHER_MEETING_ID) is not None
    assert target.show(MEETING_ID) is None
    assert package.read_bytes() == before


def test_import_requires_a_package_argument() -> None:
    _require_command()
    result = runner.invoke(app, ["meetings", "import", "--json"])
    assert result.exit_code == 2, result.output + repr(result.exception)


def test_import_maps_invalid_package_to_input_error(tmp_path: Path) -> None:
    _require_command()
    package = tmp_path / "invalid.here.zip"
    contents = b"synthetic invalid zip"
    package.write_bytes(contents)

    error = _import_json(package, code=2)
    assert error["error_category"] == "invalid_input"
    assert error["error"]
    assert "meetings" not in error
    assert package.read_bytes() == contents


@requires_native_export
def test_import_maps_local_integrity_error_to_io_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_command()
    package = _package(_origin(tmp_path), tmp_path / "valid.here.zip")
    before = package.read_bytes()
    target = _target(tmp_path)

    def fail_stage(self, inventory):
        assert self.data_dir == target.data_dir
        raise IntegrityFailure("Synthetic catalog integrity failure during staging")

    monkeypatch.setattr(MemoryService, "_portable_import_stage", fail_stage)
    error = _import_json(package, code=4)
    assert error["error_category"] == "io"
    assert error["error"]
    assert "meetings" not in error
    assert all(target.show(identity) is None for identity in MANIFEST_ORDER)
    assert package.read_bytes() == before


@requires_native_export
@pytest.mark.parametrize("fault", ["open", "central_directory_read", "zip_parser_seek"])
def test_import_maps_package_access_failure_to_io_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    _require_command()
    package = _package(_origin(tmp_path), tmp_path / "inaccessible.here.zip")
    original_open = Path.open
    with ZipFile(package) as archive:
        central_offset = archive.start_dir
    injected = False

    class FailingReader(BufferedReader):
        def read(self, size: int = -1) -> bytes:
            nonlocal injected
            if fault == "central_directory_read" and self.tell() == central_offset:
                injected = True
                raise OSError("Synthetic failure reading the ZIP central directory")
            return super().read(size)

        def seek(self, offset: int, whence: int = 0) -> int:
            nonlocal injected
            # Only ZipFile seeks backward from EOF. Preflight uses absolute offsets.
            if fault == "zip_parser_seek" and whence == SEEK_END and offset < 0:
                injected = True
                raise OSError("Synthetic seek failure during ZIP metadata parsing")
            return super().seek(offset, whence)

    def open_package(path, mode="r", *args, **kwargs):
        nonlocal injected
        if path == package and mode == "rb":
            if fault == "open":
                injected = True
                raise PermissionError("Synthetic failure opening the package")
            return FailingReader(original_open(path, "rb", buffering=0))
        return original_open(path, mode, *args, **kwargs)

    before = package.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_package)
        result = runner.invoke(app, ["meetings", "import", str(package), "--json"])

    assert injected
    target = _target(tmp_path)
    assert all(target.show(identity) is None for identity in MANIFEST_ORDER)
    assert package.read_bytes() == before
    assert result.exit_code == 4, result.output + repr(result.exception)
    error = json.loads(result.stdout)
    assert error["error_category"] == "io"
    assert error["error"]
    assert "meetings" not in error


@pytest.fixture
def partial_batch(tmp_path: Path) -> tuple[Path, MemoryService]:
    _require_command()
    origin = _origin(tmp_path, PARTIAL_BATCH_ORDER)
    package = _package(origin, tmp_path / "partial-report.here.zip", PARTIAL_BATCH_ORDER)
    return package, _target(tmp_path)


@requires_native_export
def test_import_reports_earlier_commit_and_later_rollback(
    partial_batch: tuple[Path, MemoryService], monkeypatch: pytest.MonkeyPatch
) -> None:
    package, target = partial_batch
    original_insert = MemoryRepository.insert_portable_meeting
    attempted = []

    def fail_second_insert(self, bundle, source):
        original_insert(self, bundle, source)
        if self.library.data_dir == target.data_dir:
            identity = bundle.detail.meeting.id
            attempted.append(identity)
            if identity == MEETING_ID:
                raise IntegrityFailure("Synthetic integrity failure before entity commit")

    with monkeypatch.context() as patch:
        patch.setattr(MemoryRepository, "insert_portable_meeting", fail_second_insert)
        report = _import_json(package, code=4)

    assert attempted == [OTHER_MEETING_ID, MEETING_ID]
    assert _outcomes(report) == [
        (OTHER_MEETING_ID, "inserted", ""),
        (MEETING_ID, "rejected", "local_io"),
        (THIRD_MEETING_ID, "rejected", "batch_stopped"),
    ]
    assert not report.get("pending_recovery_ids")
    target.reconcile()
    assert target.show(OTHER_MEETING_ID) is not None
    assert target.show(MEETING_ID) is None
    assert target.show(THIRD_MEETING_ID) is None


@requires_native_export
def test_import_reports_uncertain_commit_separately_and_retains_recovery_registration(
    partial_batch: tuple[Path, MemoryService], monkeypatch: pytest.MonkeyPatch
) -> None:
    package, target = partial_batch
    original_transaction = MemoryRepository.transaction
    injected = False

    @contextmanager
    def fail_after_second_commit(self):
        nonlocal injected
        with original_transaction(self) as repo:
            second_missing = (
                self.library.data_dir == target.data_dir and repo.get_meeting(MEETING_ID) is None
            )
            yield repo
            second_inserted = second_missing and repo.get_meeting(MEETING_ID) is not None
        if second_inserted:
            injected = True
            raise IntegrityFailure("Synthetic failure returning from durable entity commit")

    with monkeypatch.context() as patch:
        patch.setattr(MemoryRepository, "transaction", fail_after_second_commit)
        report = _import_json(package, code=4)

    assert injected
    assert _outcomes(report) == [
        (OTHER_MEETING_ID, "inserted", ""),
        (THIRD_MEETING_ID, "rejected", "batch_stopped"),
    ]
    assert report["pending_recovery_ids"] == [MEETING_ID]
    (stage,) = (target.data_dir / ".staging").glob("import-*")
    assert (stage / ".import.json").is_file()

    target.reconcile()
    assert not stage.exists()
    assert target.show(OTHER_MEETING_ID) is not None
    assert target.show(MEETING_ID) is not None
    assert target.show(THIRD_MEETING_ID) is None


@requires_native_export
def test_import_reports_commits_when_stage_cleanup_fails(
    partial_batch: tuple[Path, MemoryService], monkeypatch: pytest.MonkeyPatch
) -> None:
    package, target = partial_batch
    injected = False

    def fail_cleanup(stage_root, ownership):
        nonlocal injected
        assert stage_root.name.startswith("import-")
        assert (stage_root / ".import.json").is_file()
        injected = True
        raise OSError("Synthetic failure cleaning the registered import stage")

    with monkeypatch.context() as patch:
        patch.setattr(MemoryService, "_remove_import_stage_locked", staticmethod(fail_cleanup))
        report = _import_json(package, code=4)

    assert injected
    assert _outcomes(report) == [(identity, "inserted", "") for identity in PARTIAL_BATCH_ORDER]
    assert report["cleanup_pending"] is True
    assert not report.get("pending_recovery_ids")
    (stage,) = (target.data_dir / ".staging").glob("import-*")
    assert (stage / ".import.json").is_file()

    target.reconcile()
    assert not stage.exists()
    assert all(target.show(identity) is not None for identity in PARTIAL_BATCH_ORDER)
