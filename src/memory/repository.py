"""Short explicit catalog transactions over one operation-owned connection."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from typing import TypeVar

from here.memory.database import IntegrityFailure, LibraryConnection, storage_error
from here.memory.models import (
    AudioAsset,
    IngestSource,
    Meeting,
    MeetingStatus,
    PendingOperation,
    Segment,
    TranscriptRevision,
)

T = TypeVar("T", Meeting, AudioAsset, TranscriptRevision, Segment, IngestSource, PendingOperation)


class MemoryRepository:
    def __init__(self, library: LibraryConnection) -> None:
        self.library = library
        self.connection = library.connection
        self._writing = False

    @contextmanager
    def transaction(self) -> Iterator[MemoryRepository]:
        if self.connection.in_transaction or self._writing:
            raise IntegrityFailure("Nested catalog transactions are not supported")
        try:
            self.library.execute("BEGIN IMMEDIATE")
            self._writing = True
            yield self
            self.connection.commit()
        except sqlite3.Error as error:
            self.connection.rollback()
            raise storage_error(error) from error
        except BaseException:
            self.connection.rollback()
            raise
        finally:
            self._writing = False

    def _require_transaction(self) -> None:
        if not self._writing or not self.connection.in_transaction:
            raise IntegrityFailure("Catalog mutation requires an explicit transaction")

    def _insert(self, table: str, value: T) -> None:
        self._require_transaction()
        values = asdict(value)
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        self.library.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values())
        )

    def _get(self, table: str, identity: str, model: type[T]) -> T | None:
        row = self.library.execute(f"SELECT * FROM {table} WHERE id=?", (identity,)).fetchone()
        return model(**dict(row)) if row is not None else None

    def insert_meeting(self, meeting: Meeting) -> None:
        self._insert("meetings", meeting)

    def get_meeting(self, meeting_id: str) -> Meeting | None:
        meeting = self._get("meetings", meeting_id, Meeting)
        if meeting is None:
            return None
        values = asdict(meeting)
        values["status"] = MeetingStatus(meeting.status)
        return Meeting(**values)

    def get_meeting_by_legacy_id(self, legacy_id: str) -> Meeting | None:
        row = self.library.execute(
            "SELECT id FROM meetings WHERE legacy_session_id=? ORDER BY created_at, id LIMIT 1",
            (legacy_id,),
        ).fetchone()
        return self.get_meeting(row[0]) if row is not None else None

    def insert_audio_asset(self, asset: AudioAsset) -> None:
        self._insert("audio_assets", asset)

    def get_audio_asset(self, asset_id: str) -> AudioAsset | None:
        return self._get("audio_assets", asset_id, AudioAsset)

    def insert_revision(self, revision: TranscriptRevision) -> None:
        self._insert("transcript_revisions", revision)

    def get_revision(self, revision_id: str) -> TranscriptRevision | None:
        return self._get("transcript_revisions", revision_id, TranscriptRevision)

    def insert_segments(self, segments: Sequence[Segment]) -> None:
        for segment in segments:
            self._insert("segments", segment)

    def get_segments(self, revision_id: str) -> tuple[Segment, ...]:
        rows = self.library.execute(
            "SELECT * FROM segments WHERE revision_id=? ORDER BY ordinal", (revision_id,)
        )
        return tuple(Segment(**dict(row)) for row in rows)

    def activate_revision(self, meeting_id: str, revision_id: str) -> None:
        self._require_transaction()
        revision = self.get_revision(revision_id)
        if revision is None or revision.meeting_id != meeting_id:
            raise IntegrityFailure("Active revision must belong to its meeting")
        result = self.library.execute(
            "UPDATE meetings SET active_revision_id=?, row_version=row_version+1 WHERE id=?",
            (revision_id, meeting_id),
        )
        if result.rowcount != 1:
            raise IntegrityFailure("Cannot activate a revision of a missing meeting")

    def insert_ingest_source(self, source: IngestSource) -> None:
        self._insert("ingest_sources", source)

    def get_ingest_source(self, origin_path: str, fingerprint: str) -> IngestSource | None:
        row = self.library.execute(
            "SELECT * FROM ingest_sources WHERE origin_path=? AND origin_fingerprint=?",
            (origin_path, fingerprint),
        ).fetchone()
        return IngestSource(**dict(row)) if row is not None else None

    def insert_pending_operation(self, operation: PendingOperation) -> None:
        self._insert("pending_operations", operation)

    def get_pending_operation(self, operation_id: str) -> PendingOperation | None:
        return self._get("pending_operations", operation_id, PendingOperation)

    def transition_pending_operation(self, operation_id: str, phase: str) -> None:
        self._require_transaction()
        if phase not in ("complete", "reverted"):
            raise IntegrityFailure("Invalid operation transition")
        result = self.library.execute(
            "UPDATE pending_operations SET phase=? WHERE id=? AND phase='prepared'",
            (phase, operation_id),
        )
        if result.rowcount != 1:
            raise IntegrityFailure("Only prepared operations can transition")
