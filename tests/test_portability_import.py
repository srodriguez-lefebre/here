"""Portable import round-trips synthetic meetings through the public service."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest
from here.memory import portability
from here.memory.models import MeetingPatch
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="uses native portable export")

_REVISION = f"meetings/{MEETING_ID}/revisions/1/revision.json"
_OTHER_SELECTED_ID = "00000000-0000-4000-8000-000000000001"
_CORE_TABLES = (
    "meetings",
    "transcript_revisions",
    "segments",
    "audio_assets",
    "revision_audio_assets",
    "participants",
    "meeting_participants",
    "speaker_assignments",
    "lifecycle_events",
    "ingest_sources",
    "pending_operations",
)


def _memory(directory: Path) -> MemoryService:
    sessions = directory / "sessions"
    sessions.mkdir(parents=True)
    return MemoryService(directory / "data", sessions_root=sessions)


@pytest.fixture
def origin(tmp_path: Path) -> MemoryService:
    directory = tmp_path / "origin"
    memory = _memory(directory)
    sources = directory / "sessions"
    original = make_session(sources / "first", meeting_id=MEETING_ID)
    assert memory.ingest_session(original.directory).status == "inserted"
    initial = memory.get_meeting(MEETING_ID)
    parent = initial.meeting.active_revision_id
    assert parent is not None
    retry = make_session(
        sources / "retry",
        meeting_id=MEETING_ID,
        transcript_text="Synthetic historical retry.",
        segments=[{"text": "Synthetic historical retry."}],
    )
    pcm_wav = retry.audio.read_bytes()
    retry.audio.write_bytes(pcm_wav[:-1] + bytes((pcm_wav[-1] ^ 1,)))
    assert memory.ingest_session(retry.directory, revision_of=parent).status == "revised"

    participant = memory.add_participant("Synthetic reviewer")
    memory.link_participant(MEETING_ID, participant.id, role="observer")
    speaker = initial.segments[0].speaker_key
    assert speaker is not None
    detail = memory.assign_speaker(MEETING_ID, speaker, participant.id, scope=parent)
    memory.update_meeting(
        MEETING_ID,
        MeetingPatch(title="Synthetic import", notes="Synthetic notes"),
        detail.meeting.row_version,
    )
    return memory


@pytest.fixture
def target(tmp_path: Path) -> MemoryService:
    memory = _memory(tmp_path / "target")
    (memory.data_dir / ".staging").mkdir(exist_ok=True)
    return memory


def _export(
    memory: MemoryService,
    source: Path,
    include_audio: bool = True,
    ids: tuple[str, ...] = (MEETING_ID,),
) -> Path:
    return (
        portability.PortabilityService(memory)
        .export(ids, source, include_audio, lambda _: None, lambda: False)
        .path
    )


def _import(
    memory: MemoryService,
    source: Path,
    progress: Callable[[int], None] = lambda _: None,
    cancellation: Callable[[], bool] = lambda: False,
) -> portability.ImportReport:
    return portability.PortabilityService(memory).import_archive(source, progress, cancellation)


def _outcomes(report: portability.ImportReport) -> dict[str, str]:
    assert isinstance(report.meetings, tuple)
    assert len(report.meetings) == len({item.meeting_id for item in report.meetings})
    return {item.meeting_id: item.status for item in report.meetings}


def _rows(memory: MemoryService, sql: str, parameters: tuple[object, ...] = ()) -> list[tuple]:
    with closing(sqlite3.connect(memory.data_dir / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _catalog(memory: MemoryService) -> dict[str, list[tuple]]:
    return {table: _rows(memory, f"SELECT * FROM {table} ORDER BY 1") for table in _CORE_TABLES}


def _files(directory: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _filesystem(directory: Path) -> tuple[set[Path], dict[Path, bytes]]:
    return (
        {path.relative_to(directory) for path in directory.rglob("*") if path.is_dir()},
        _files(directory),
    )


def _local_origin(memory: MemoryService) -> str:
    return os.path.normcase(str((memory.data_dir / "meetings" / MEETING_ID).resolve()))


def _repackage(
    source: Path,
    destination: Path,
    member: str,
    payload: bytes,
    *,
    refresh_manifest: bool,
) -> Path:
    with ZipFile(source) as original:
        items = {name: original.read(name) for name in original.namelist()}
    items[member] = payload
    if refresh_manifest:
        manifest = json.loads(items["manifest.json"])
        entry = next(entry for entry in manifest["files"] if entry["path"] == member)
        entry.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        items["manifest.json"] = json.dumps(manifest).encode()
    with ZipFile(destination, "x", compression=ZIP_STORED) as archive:
        for name, body in items.items():
            archive.writestr(name, body)
    return destination


@pytest.mark.parametrize("include_audio", [True, False], ids=["with-audio", "without-audio"])
def test_import_preserves_history_and_portable_relations(
    origin: MemoryService, target: MemoryService, tmp_path: Path, include_audio: bool
) -> None:
    package = _export(origin, tmp_path / "transfer.here.zip", include_audio)
    expected = origin.get_meeting(MEETING_ID)
    assert len(expected.revisions) == len(expected.audio_assets) == 2
    assert expected.speaker_assignments and expected.participant_links[0].role == "observer"

    assert _outcomes(_import(target, package)) == {MEETING_ID: "inserted"}

    actual = target.get_meeting(MEETING_ID)
    assert replace(actual.meeting, row_version=expected.meeting.row_version) == expected.meeting
    assert actual.revisions == tuple(
        replace(revision, origin=_local_origin(target)) for revision in expected.revisions
    )
    assert actual.segments == expected.segments
    assert any(segment.start_ms == 0 for segment in actual.segments)
    assert any(
        segment.start_ms is None and segment.speaker_key is None for segment in actual.segments
    )
    assert actual.revision_audio_assets == expected.revision_audio_assets
    assert actual.participants == expected.participants
    assert actual.participant_links == expected.participant_links
    assert actual.speaker_assignments == expected.speaker_assignments
    assert len(actual.audio_assets) == len(expected.audio_assets)
    for imported, old in zip(actual.audio_assets, expected.audio_assets, strict=True):
        assert imported == replace(
            old,
            relative_path=imported.relative_path,
            status="available" if include_audio else "missing",
        )
        managed = target.data_dir / imported.relative_path
        assert managed.is_relative_to(target.data_dir / "meetings" / MEETING_ID)
        assert managed.name == f"audio_{imported.id.replace('-', '')}.wav"
        if include_audio:
            assert hashlib.sha256(managed.read_bytes()).hexdigest() == old.sha256
        else:
            assert not managed.exists()

    with ZipFile(package) as archive:
        fingerprint = hashlib.sha256(archive.read("manifest.json")).hexdigest()
    assert _rows(
        target,
        "SELECT origin_path,origin_fingerprint FROM ingest_sources WHERE meeting_id=?",
        (MEETING_ID,),
    ) == [(_local_origin(target), fingerprint)]
    assert _rows(
        target, "SELECT id,kind,occurred_at,recorded_duration_ms FROM lifecycle_events ORDER BY id"
    ) == _rows(
        origin, "SELECT id,kind,occurred_at,recorded_duration_ms FROM lifecycle_events ORDER BY id"
    )
    assert not list((target.data_dir / ".staging").glob("import-*"))


@pytest.mark.parametrize(
    "with_audio", [True, False], ids=["identical-package", "no-audio-projection"]
)
def test_reimport_unchanged_preserves_local_audio_rows_and_provenance(
    origin: MemoryService, target: MemoryService, tmp_path: Path, with_audio: bool
) -> None:
    full = _export(origin, tmp_path / "original.here.zip")
    assert _outcomes(_import(target, full)) == {MEETING_ID: "inserted"}
    incoming = full if with_audio else _export(origin, tmp_path / "without-audio.here.zip", False)
    rows = _catalog(target)
    managed = _files(target.data_dir / "meetings")
    assert any(path.suffix == ".wav" for path in managed)

    assert _outcomes(_import(target, incoming)) == {MEETING_ID: "unchanged"}

    assert _catalog(target) == rows
    assert _files(target.data_dir / "meetings") == managed
    assert not list((target.data_dir / ".staging").glob("import-*"))


def test_audio_included_reimport_cannot_be_unchanged_when_local_audio_is_missing(
    origin: MemoryService, target: MemoryService, tmp_path: Path
) -> None:
    initial = _export(origin, tmp_path / "without-audio.here.zip", False)
    assert _outcomes(_import(target, initial)) == {MEETING_ID: "inserted"}
    incoming = _export(origin, tmp_path / "with-audio.here.zip")
    rows = _catalog(target)
    managed = _files(target.data_dir / "meetings")

    assert _outcomes(_import(target, incoming)) == {MEETING_ID: "conflict"}

    assert _catalog(target) == rows
    assert _files(target.data_dir / "meetings") == managed
    assert not list((target.data_dir / ".staging").glob("import-*"))


def test_import_conflict_preflights_full_batch_before_any_meeting_is_published(
    origin: MemoryService, target: MemoryService, tmp_path: Path
) -> None:
    original = _export(origin, tmp_path / "initial.here.zip")
    assert _outcomes(_import(target, original)) == {MEETING_ID: "inserted"}
    other = make_session(tmp_path / "extra-session", meeting_id=_OTHER_SELECTED_ID)
    assert origin.ingest_session(other.directory).status == "inserted"
    package = _export(origin, tmp_path / "batch.here.zip", ids=(_OTHER_SELECTED_ID, MEETING_ID))
    with ZipFile(package) as archive:
        revision = json.loads(archive.read(_REVISION))
    revision["created_at"] = "2000-01-01T00:00:00+00:00"
    incoming = _repackage(
        package,
        tmp_path / "changed.here.zip",
        _REVISION,
        json.dumps(revision).encode(),
        refresh_manifest=True,
    )
    rows = _catalog(target)
    managed = _files(target.data_dir / "meetings")

    assert _outcomes(_import(target, incoming)) == {
        _OTHER_SELECTED_ID: "rejected",
        MEETING_ID: "conflict",
    }

    assert target.show(_OTHER_SELECTED_ID) is None
    assert _catalog(target) == rows
    assert _files(target.data_dir / "meetings") == managed
    assert not list((target.data_dir / ".staging").glob("import-*"))


def test_import_rejects_same_size_payload_with_a_different_sha256(
    origin: MemoryService, target: MemoryService, tmp_path: Path
) -> None:
    original = _export(origin, tmp_path / "source.here.zip")
    with ZipFile(original) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".wav"))
        pcm_wav = archive.read(name)
    corrupted = _repackage(
        original,
        tmp_path / "corrupted.here.zip",
        name,
        pcm_wav[:-1] + bytes((pcm_wav[-1] ^ 1,)),
        refresh_manifest=False,
    )
    rows = _catalog(target)
    before = _filesystem(tmp_path)
    staged: list[Path] = []

    def progress(written: int) -> None:
        if written > 0:
            staged.extend((target.data_dir / ".staging").glob("import-*"))

    with pytest.raises(portability.InvalidPackage):
        _import(target, corrupted, progress)

    assert staged and not any(path.exists() for path in staged)
    assert _catalog(target) == rows
    assert _filesystem(tmp_path) == before


def test_cancelled_streaming_import_removes_stage_and_leaves_catalog_untouched(
    origin: MemoryService, target: MemoryService, tmp_path: Path
) -> None:
    package = _export(origin, tmp_path / "cancel.here.zip")
    rows = _catalog(target)
    before = _filesystem(tmp_path)
    staged: list[Path] = []

    def cancel_after_writing(written: int) -> None:
        if written > 0:
            staged.extend((target.data_dir / ".staging").glob("import-*"))

    with pytest.raises(portability.ImportCancelled):
        _import(target, package, cancel_after_writing, lambda: bool(staged))

    assert staged and not any(path.exists() for path in staged)
    assert _catalog(target) == rows
    assert _filesystem(tmp_path) == before


def test_import_rejects_unlisted_script_without_running_or_staging_it(
    origin: MemoryService, target: MemoryService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _export(origin, tmp_path / "plain.here.zip")
    sentinel = tmp_path / "script-ran"
    script = f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n".encode()
    incoming = _repackage(
        package, tmp_path / "script.here.zip", "hook.py", script, refresh_manifest=False
    )
    before = _filesystem(tmp_path)

    def forbid_process(*args, **kwargs):
        pytest.fail("Portable import launched a subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbid_process)
    with pytest.raises(portability.InvalidPackage):
        _import(target, incoming)

    assert not sentinel.exists()
    assert not list((target.data_dir / ".staging").glob("import-*"))
    assert _filesystem(tmp_path) == before


@pytest.mark.parametrize("scope", ["uuid", "exact-origin-fingerprint"])
def test_import_refuses_suppressed_package_even_without_catalog_alias(
    origin: MemoryService, target: MemoryService, tmp_path: Path, scope: str
) -> None:
    package = _export(origin, tmp_path / "suppressed.here.zip")
    with ZipFile(package) as archive:
        fingerprint = hashlib.sha256(archive.read("manifest.json")).hexdigest()
    registry = target.data_dir / ".deletions"
    registry.mkdir()
    identity = MEETING_ID if scope == "uuid" else OTHER_MEETING_ID
    origins = (
        []
        if scope == "uuid"
        else [{"origin_path": _local_origin(target), "origin_fingerprint": fingerprint}]
    )
    record = registry / f"{identity}.json"
    write_json(
        record,
        {"schema_version": 1, "meeting_id": identity, "phase": "complete", "origins": origins},
    )
    before = record.read_bytes()
    assert _rows(target, "SELECT meeting_id FROM ingest_sources") == []

    assert _outcomes(_import(target, package)) == {MEETING_ID: "rejected"}

    assert record.read_bytes() == before
    assert target.show(MEETING_ID) is None
    assert _rows(target, "SELECT meeting_id FROM ingest_sources") == []
    assert not (target.data_dir / "meetings" / MEETING_ID).exists()
    assert not list((target.data_dir / ".staging").glob("import-*"))
