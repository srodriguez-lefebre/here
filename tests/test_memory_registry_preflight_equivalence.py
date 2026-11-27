"""Preservation controls for batched preflight in the real R50 read view."""

import errno
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID

import pytest
from here.memory.database import LibraryError, open_library
from here.memory.models import IngestSource, SuppressionSnapshot
from here.memory.service import catalog_read_view

from search_fixtures import small_corpus
from test_memory_search import _open_catalog

FIRST = str(UUID(int=1, version=4))
SECOND = str(UUID(int=2, version=4))
STAGE_NAME = f".{SECOND}.json.Abc_1234.stage"


def _record(directory, identity, phase="pending", origins=()):
    document = {
        "schema_version": 1,
        "meeting_id": identity,
        "origins": list(origins),
        "phase": phase,
    }
    path = directory / f"{identity}.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path, document


def _fingerprint(documents, identities):
    digest = hashlib.sha256()
    for value in ("here-suppression-v1", documents, identities):
        encoded = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _observe_preflight(monkeypatch, directory):
    metadata_checks = []
    payload_opens = []
    original_lstat = Path.lstat
    original_open = Path.open

    def lstat(path):
        if path.parent == directory:
            metadata_checks.append(path.name)
        return original_lstat(path)

    def open_path(path, *args, **kwargs):
        if path.parent == directory:
            payload_opens.append((path.name, frozenset(metadata_checks)))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "open", open_path)
    return payload_opens


def test_r50_preflights_every_leaf_before_open_and_ignores_stage_contents(tmp_path, monkeypatch):
    corpus = small_corpus()
    library, repo = _open_catalog(tmp_path, corpus)
    alias_id = corpus.meeting_id(0)
    directory = tmp_path / ".deletions"
    origin = {
        "origin_path": os.path.normcase(os.path.normpath(str(tmp_path / "origen-á"))),
        "origin_fingerprint": "a" * 64,
    }
    try:
        with repo.transaction():
            repo.insert_ingest_source(
                IngestSource(
                    id=str(UUID(int=3, version=4)),
                    meeting_id=alias_id,
                    revision_id=corpus.revision_id(0),
                    **origin,
                )
            )
        with catalog_read_view(library) as absent:
            assert absent == SuppressionSnapshot((), _fingerprint([], ()))
        assert not directory.exists()

        directory.mkdir()
        second, complete = _record(directory, SECOND, "complete")
        first, pending = _record(directory, FIRST, origins=(origin,))
        (directory / STAGE_NAME).write_bytes(b"synthetic incomplete staging bytes")
        expected_ids = tuple(sorted((FIRST, SECOND, alias_id)))
        fingerprint = _fingerprint([pending, complete], expected_ids)
        expected = SuppressionSnapshot(expected_ids, fingerprint)
        required_checks = frozenset((STAGE_NAME, first.name, second.name))
        for _ in range(2):
            # Each call needs its own leaf checks even with identical registry bytes.
            with monkeypatch.context() as observer:
                opens = _observe_preflight(observer, directory)
                with catalog_read_view(library) as snapshot:
                    assert snapshot == expected
                    assert library.connection.in_transaction
                assert opens == [
                    (first.name, required_checks),
                    (second.name, required_checks),
                ]
    finally:
        library.close()


@pytest.mark.parametrize(
    "unsafe_role", ["unknown", "final_directory", "hardlinked_final", "hardlinked_stage"]
)
def test_r50_rejects_invalid_role_before_opening_any_payload(tmp_path, monkeypatch, unsafe_role):
    directory = tmp_path / ".deletions"
    directory.mkdir()
    _record(directory, FIRST)
    library = open_library(tmp_path)
    try:
        if unsafe_role == "unknown":
            (directory / "zz-unknown").write_bytes(b"synthetic unknown")
        elif unsafe_role == "final_directory":
            (directory / f"{SECOND}.json").mkdir()
        else:
            name = STAGE_NAME if unsafe_role == "hardlinked_stage" else f"{SECOND}.json"
            outside = tmp_path / "outside.json"
            outside.write_bytes(b"synthetic external payload")
            try:
                (directory / name).hardlink_to(outside)
            except OSError as exc:
                if getattr(exc, "winerror", None) in {1, 50} or exc.errno in {
                    errno.EACCES,
                    errno.ENOSYS,
                    errno.ENOTSUP,
                    errno.EPERM,
                }:
                    pytest.skip(f"Test volume does not allow hard links: {exc}")
                raise
        with monkeypatch.context() as observer:
            opens = _observe_preflight(observer, directory)
            with pytest.raises(LibraryError, match="Deletion registry unavailable"):
                with catalog_read_view(library):
                    pytest.fail("Unsafe registry yielded a read view")
            assert opens == []
        assert not library.connection.in_transaction
    finally:
        library.close()


@pytest.mark.parametrize("swap_after", ["enumeration", "leaf_resolve"])
def test_r50_revalidates_registry_identity_before_reads(tmp_path, monkeypatch, swap_after):
    """RED on the old per-entry guard if the volume reports directory identities."""
    directory = tmp_path / ".deletions"
    directory.mkdir()
    leaf, _ = _record(directory, FIRST)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    _record(replacement, FIRST)
    source_info = directory.lstat()
    replacement_info = replacement.lstat()
    if not all(
        (source_info.st_dev, source_info.st_ino, replacement_info.st_dev, replacement_info.st_ino)
    ):
        pytest.skip("Test volume does not report directory identities")
    if (source_info.st_dev, source_info.st_ino) == (
        replacement_info.st_dev,
        replacement_info.st_ino,
    ):
        pytest.skip("Test volume does not distinguish directory identities")

    original_resolve = Path.resolve
    original_iterdir = Path.iterdir
    swapped = False

    def swap_directory():
        nonlocal swapped
        directory.rename(tmp_path / "previous-registry")
        replacement.rename(directory)
        swapped = True

    def iterdir(path):
        entries = original_iterdir(path)
        if path != directory or swapped:
            return entries
        saved_entries = tuple(entries)
        swap_directory()
        return iter(saved_entries)

    def resolve(path, *args, **kwargs):
        resolved = original_resolve(path, *args, **kwargs)
        if path == leaf and not swapped:
            swap_directory()
        return resolved

    library = open_library(tmp_path)
    try:
        with monkeypatch.context() as observer:
            opens = _observe_preflight(observer, directory)
            if swap_after == "enumeration":
                observer.setattr(Path, "iterdir", iterdir)
            else:
                observer.setattr(Path, "resolve", resolve)
            with pytest.raises(LibraryError, match="Deletion registry unavailable"):
                with catalog_read_view(library):
                    pytest.fail("Registry identity changed between enumeration and reading")
            assert swapped
            assert opens == []
        assert not library.connection.in_transaction
    finally:
        library.close()


def test_r50_uses_individual_path_checks_when_directory_identity_is_unknown(tmp_path, monkeypatch):
    directory = tmp_path / ".deletions"
    directory.mkdir()
    first, pending = _record(directory, FIRST)
    second, complete = _record(directory, SECOND, "complete")
    original_lstat = Path.lstat
    checks = []

    class UnknownIdentity:
        st_dev = 0
        st_ino = 0

        def __init__(self, info):
            self.info = info

        def __getattr__(self, name):
            return getattr(self.info, name)

    def lstat(path):
        info = original_lstat(path)
        if path == directory:
            checks.append(path)
            return UnknownIdentity(info)
        if path in (first, second):
            checks.append(path)
        return info

    library = open_library(tmp_path)
    try:
        monkeypatch.setattr(Path, "lstat", lstat)
        expected_ids = (FIRST, SECOND)
        expected_digest = _fingerprint([pending, complete], expected_ids)
        with catalog_read_view(library) as snapshot:
            assert snapshot == SuppressionSnapshot(expected_ids, expected_digest)
        first_checked = checks.index(first)
        second_checked = checks.index(second)
        assert directory in checks[:first_checked]
        assert directory in checks[first_checked + 1 : second_checked]
    finally:
        library.close()
