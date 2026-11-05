"""Missing metadata cannot turn an opaque capture identity into a journal path."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from memory_fixtures import make_session, write_json


def _configure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Local candidate validation allocated provider/network transport")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    return importlib.import_module("here.application.recovery")


@pytest.mark.parametrize("style", ["drive-absolute", "rooted-forward-slash"])
def test_missing_manifest_rejects_path_identity_before_any_external_journal_probe(
    tmp_path, monkeypatch, style
):
    recovery = _configure(tmp_path, monkeypatch)
    root = tmp_path / "managed-sessions"
    destination = root / "stale-selection"
    destination.mkdir(parents=True)
    owned_canary = destination / "unchanged.txt"
    owned_canary.write_bytes(b"Synthetic managed bytes survive invalid identity rejection.")
    external = tmp_path / "external-synthetic-journal"
    external.mkdir()
    sentinel = external / "journal.json"
    sentinel_bytes = b'{"sentinel":"Synthetic external journal must not be probed."}\n'
    sentinel.write_bytes(sentinel_bytes)
    capture_id = (
        str(external) if style == "drive-absolute" else external.as_posix()[len(external.drive) :]
    )
    # Windows rooted paths inherit the managed root's drive but replace the rest
    # of its prefix. On POSIX both spellings denote the same native absolute path.
    assert (root / ".captures" / capture_id).absolute() == external.absolute()
    candidate = recovery.RecoveryCandidate(
        destination,
        "Synthetic stale selection",
        0.0,
        "interrupted",
        None,
        False,
        capture_id,
    )
    probes = []

    def observer(name, original):
        def delegate(path, *args, **kwargs):
            # Lexical only: resolve()/exists() here would itself invalidate the
            # no-probe measurement. Every actual observed operation delegates.
            absolute = Path(path).absolute()
            if absolute == external.absolute() or absolute.is_relative_to(external.absolute()):
                probes.append((name, str(absolute)))
            return original(path, *args, **kwargs)

        return delegate

    with monkeypatch.context() as watch:
        for name in ("stat", "lstat", "is_file", "open", "read_bytes", "read_text"):
            watch.setattr(Path, name, observer(name, getattr(Path, name)))
        with pytest.raises(ValueError):
            recovery.RecoveryService(root).materialize(candidate)
    assert sentinel.read_bytes() == sentinel_bytes
    assert list(external.iterdir()) == [sentinel]
    assert (
        owned_canary.read_bytes() == b"Synthetic managed bytes survive invalid identity rejection."
    )
    assert not (destination / "session.json").exists()
    assert probes == [], (
        f"Capture identity reached an external journal probe before validation: {probes}"
    )


def test_control_existing_manifest_with_opaque_identity_materializes_without_conversion(
    tmp_path, monkeypatch
):
    recovery = _configure(tmp_path, monkeypatch)
    identity = "synthetic-compatible-opaque-id"
    source = make_session(tmp_path / "sessions" / "legacy", meeting_id=identity)
    metadata = json.loads(source.metadata.read_text(encoding="utf-8"))
    metadata["status"] = "failed"
    metadata["failure_stage"] = "offline_transcription"
    write_json(source.metadata, metadata)
    before = {path.name: path.read_bytes() for path in source.directory.iterdir()}
    service = recovery.RecoveryService(source.directory.parent)
    (candidate,) = service.discover()
    assert candidate.capture_id == identity and candidate.can_retry
    assert service.materialize(candidate) == source.directory
    assert {path.name: path.read_bytes() for path in source.directory.iterdir()} == before
    assert json.loads(source.metadata.read_text(encoding="utf-8"))["meeting_id"] == identity
