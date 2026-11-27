"""Public CLI coverage for selected-meeting portable export."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
from here.cli import app
from here.memory.service import MemoryService
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session
from typer.testing import CliRunner

UNSELECTED_ID = "11111111-3333-4333-8333-333333333333"
MISSING_ID = "99999999-9999-4999-8999-999999999999"
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "catalog"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("Portable export attempted provider or network work")

    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)


def _require_command() -> None:
    result = runner.invoke(app, ["meetings", "export", "--help"])
    assert result.exit_code == 0, result.output + repr(result.exception)


def _seed_catalog(tmp_path: Path, *identities: str) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    memory = MemoryService(tmp_path / "catalog", sessions_root=sessions)
    for identity in identities:
        source = make_session(sessions / identity, meeting_id=identity)
        assert memory.ingest_session(source.directory).status == "inserted"


@pytest.mark.skipif(sys.platform != "win32", reason="native portable export requires Windows")
@pytest.mark.parametrize("include_audio", [True, False], ids=["default-audio", "no-audio"])
def test_export_writes_only_selected_meetings_to_requested_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_audio: bool
) -> None:
    _require_command()
    from here.memory.portability import PortabilityService

    _seed_catalog(tmp_path, MEETING_ID, OTHER_MEETING_ID, UNSELECTED_ID)
    destination = tmp_path / "selected meetings.here.zip"
    selection = (OTHER_MEETING_ID, MEETING_ID)
    real_init = MemoryService.__init__
    real_export = PortabilityService.export
    compositions = {}
    forwarded = []

    def observe_composition(memory, data_dir, *, sessions_root=None):
        real_init(memory, data_dir, sessions_root=sessions_root)
        compositions[memory] = (data_dir, sessions_root)

    def observe_export(service, ids, destination, include_audio, progress, cancellation):
        forwarded.append((service, ids, destination, include_audio))
        return real_export(service, ids, destination, include_audio, progress, cancellation)

    monkeypatch.setattr(MemoryService, "__init__", observe_composition)
    monkeypatch.setattr(PortabilityService, "export", observe_export)
    arguments = ["meetings", "export", OTHER_MEETING_ID[:8], MEETING_ID, "--out", str(destination)]
    if not include_audio:
        arguments.append("--no-audio")
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output + repr(result.exception)
    [(service, ids, actual_destination, actual_audio)] = forwarded
    assert isinstance(ids, tuple)
    assert len(ids) == len(selection)
    assert set(ids) == set(selection)
    assert actual_destination == destination
    assert actual_audio is include_audio
    assert compositions[service._memory] == (tmp_path / "catalog", tmp_path / "sessions")
    with ZipFile(destination) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert sorted(manifest["meeting_ids"]) == sorted(selection)
        assert manifest["audio_included"] is include_audio
        assert bool([name for name in archive.namelist() if name.endswith(".wav")]) is include_audio
        assert f"meetings/{UNSELECTED_ID}/meeting.json" not in archive.namelist()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--out", "selected.here.zip"),
        (MEETING_ID,),
    ],
    ids=["missing-ids", "missing-out"],
)
def test_export_requires_ids_and_out(arguments: tuple[str, ...]) -> None:
    _require_command()
    result = runner.invoke(app, ["meetings", "export", *arguments])
    assert result.exit_code == 2, result.output + repr(result.exception)


@pytest.mark.parametrize(
    ("identities", "filename", "code"),
    [
        (("not-a-uuid",), "selected.here.zip", 2),
        ((MEETING_ID[:8],), "selected.here.zip", 2),
        ((MEETING_ID, MEETING_ID), "selected.here.zip", 2),
        ((MEETING_ID,), "selected.zip", 2),
        ((MISSING_ID,), "selected.here.zip", 3),
    ],
    ids=["invalid-id", "ambiguous-id", "duplicate-id", "invalid-destination", "missing-meeting"],
)
def test_export_refuses_invalid_or_missing_selection(
    tmp_path: Path, identities: tuple[str, ...], filename: str, code: int
) -> None:
    _require_command()
    _seed_catalog(tmp_path, MEETING_ID, UNSELECTED_ID)
    destination = tmp_path / filename
    result = runner.invoke(app, ["meetings", "export", *identities, "--out", str(destination)])

    assert result.exit_code == code, result.output + repr(result.exception)
    assert not destination.exists()


def test_export_maps_insufficient_space_to_io_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_command()
    from here.memory.portability import InsufficientSpace, PortabilityService

    _seed_catalog(tmp_path, MEETING_ID)
    destination = tmp_path / "selected.here.zip"

    def refuse_export(service, ids, destination, include_audio, progress, cancellation):
        raise InsufficientSpace("Synthetic destination storage failure")

    monkeypatch.setattr(PortabilityService, "export", refuse_export)
    result = runner.invoke(app, ["meetings", "export", MEETING_ID, "--out", str(destination)])

    assert result.exit_code == 4, result.output + repr(result.exception)
    assert not destination.exists()
