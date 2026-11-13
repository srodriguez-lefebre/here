"""A failed R62 publication cannot register an existing destination for cleanup."""

from __future__ import annotations

from uuid import UUID

from here.application.recovery import RecoveryService
from here.memory import service as service_module
from memory_fixtures import MEETING_ID, make_session
from test_memory_delete import _files, _rows
from test_memory_recovery_publication import MARKER
from test_memory_recovery_publication import sandbox as sandbox
from test_memory_recovery_publication_review import _revisions, _sealed_intent, _sealed_retry


def test_rejected_replay_preserves_preexisting_destination(sandbox, monkeypatch):
    memory, sessions = sandbox
    root = memory.data_dir
    directory = make_session(sessions / "recorded", meeting_id=MEETING_ID).directory
    identity, parent = memory.prepare_retry(directory, MEETING_ID)
    assert identity == MEETING_ID
    _sealed_retry(memory, directory, monkeypatch)
    intent = _sealed_intent(directory, identity)
    assert intent.revision_of == parent
    assert memory.retry_parent(directory, identity) == (identity, parent)

    collision = UUID("7656c21a-fd18-44f1-b7cd-0f42993dcf30")
    revisions = root / "meetings" / identity / "revisions"
    assert str(collision) != parent
    destination = revisions / str(collision)
    destination.mkdir()
    unowned_text = b"Synthetic transcript from an unrelated writer"
    (destination / "transcript.txt").write_bytes(unowned_text)
    assert (directory / "transcript.txt").read_bytes() != unowned_text

    destination_files = _files(destination)
    parent_files = _files(revisions / parent)
    source_files = _files(directory)
    del source_files[MARKER]
    committed = _revisions(root)
    assert [(row[0], row[1]) for row in committed] == [(identity, parent)]
    prepared_for_destination = (
        "SELECT id FROM pending_operations WHERE phase='prepared' AND destination_path=?"
    )
    destination_rel = destination.relative_to(root).as_posix()
    parameters = (destination_rel,)
    assert _rows(root, prepared_for_destination, parameters) == []

    real_ingest_retry = memory.ingest_retry
    real_uuid4 = service_module.uuid4
    real_managed_path = service_module.managed_path
    revision_uuid_requests = []
    selected_destinations = []

    def ingest_retry(session_dir, *, revision_of):
        assert session_dir == directory and revision_of == parent

        def next_uuid():
            if not revision_uuid_requests:
                revision_uuid_requests.append(collision)
                return collision
            return real_uuid4()

        def managed_path(data_dir, relative, *, directory=False):
            path = real_managed_path(data_dir, relative, directory=directory)
            if (
                data_dir == root
                and directory
                and relative.startswith(f"meetings/{identity}/revisions/")
            ):
                selected_destinations.append(path)
            return path

        # A canonical source needs no provisional identity before generating the revision ID.
        # Observe which destination the real ingest actually selected.
        with monkeypatch.context() as ids:
            ids.setattr(service_module, "uuid4", next_uuid)
            ids.setattr(service_module, "managed_path", managed_path)
            return real_ingest_retry(session_dir, revision_of=revision_of)

    with monkeypatch.context() as replay:
        replay.setattr(memory, "ingest_retry", ingest_retry)
        RecoveryService(sessions, memory_service=memory).discover()

    assert revision_uuid_requests == [collision]
    assert selected_destinations and selected_destinations[0] == destination
    assert destination.is_dir() and _files(destination) == destination_files
    assert _sealed_intent(directory, identity) == intent
    before_reconcile = _rows(root, prepared_for_destination, parameters)

    memory.reconcile()

    assert _revisions(root) == committed
    assert memory.active_revision(identity) == parent
    assert _files(revisions / parent) == parent_files
    assert _sealed_intent(directory, identity) == intent
    after_source = _files(directory)
    del after_source[MARKER]
    assert after_source == source_files
    assert (before_reconcile, destination.is_dir(), _files(destination)) == (
        [],
        True,
        destination_files,
    )
