"""Short explicit catalog transactions over one operation-owned connection."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from typing import TypeVar
from uuid import uuid4

from here.memory.database import IntegrityFailure, LibraryConnection, storage_error
from here.memory.models import (
    AudioAsset,
    IngestSource,
    Meeting,
    MeetingStatus,
    Participant,
    PendingOperation,
    RevisionAudioAsset,
    Segment,
    TranscriptRevision,
)

T = TypeVar(
    "T",
    Meeting,
    AudioAsset,
    TranscriptRevision,
    Segment,
    IngestSource,
    PendingOperation,
    RevisionAudioAsset,
    Participant,
)


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

    def insert_revision_audio_asset(self, link: RevisionAudioAsset) -> None:
        self._insert("revision_audio_assets", link)

    def get_revision_audio_assets(self, revision_id: str) -> tuple[RevisionAudioAsset, ...]:
        rows = self.library.execute(
            "SELECT * FROM revision_audio_assets WHERE revision_id=? ORDER BY ordinal",
            (revision_id,),
        )
        return tuple(RevisionAudioAsset(**dict(row)) for row in rows)

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

    def insert_participant(self, participant: Participant) -> None:
        self._insert("participants", participant)

    def purge_meeting(self, meeting_id: str) -> None:
        """Remove core rows after trusted derived purgers have used their relationships."""
        self._require_transaction()
        self.library.execute(
            "UPDATE meetings SET active_revision_id=NULL WHERE id=?", (meeting_id,)
        )
        for table in ("speaker_assignments", "revision_audio_assets", "ingest_sources"):
            self.library.execute(f"DELETE FROM {table} WHERE meeting_id=?", (meeting_id,))
        self.library.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
        # Publication payloads can contain derived content in future formats. A
        # minimal delete control row is the only per-meeting journal retained.
        self.library.execute(
            "DELETE FROM pending_operations WHERE meeting_id=? AND kind!='delete'", (meeting_id,)
        )

    def get_participant(self, participant_id: str) -> Participant | None:
        return self._get("participants", participant_id, Participant)

    def record_event(
        self, meeting_id: str, kind: str, occurred_at: str, payload: dict[str, object]
    ) -> None:
        """Keep event payloads bounded and independent of user/provider text."""
        self._require_transaction()
        meeting = self.get_meeting(meeting_id)
        if meeting is None:
            raise IntegrityFailure("Lifecycle event requires an existing meeting")
        if kind not in {
            "status_changed",
            "metadata_updated",
            "participant_linked",
            "speaker_assigned",
        }:
            raise IntegrityFailure("Unknown lifecycle event kind")
        if set(payload) - {"from", "to", "fields"}:
            raise IntegrityFailure("Unknown lifecycle event payload field")
        for name in ("from", "to"):
            value = payload.get(name)
            if value is not None:
                if not isinstance(value, str):
                    raise IntegrityFailure("Invalid lifecycle event status")
                try:
                    MeetingStatus(value)
                except (TypeError, ValueError) as exc:
                    raise IntegrityFailure("Invalid lifecycle event status") from exc
        if "fields" in payload:
            fields = payload["fields"]
            if not isinstance(fields, list) or any(
                not isinstance(value, str) or value not in {"title", "notes"} for value in fields
            ):
                raise IntegrityFailure("Invalid lifecycle event edited fields")
        self.library.execute(
            "INSERT INTO lifecycle_events VALUES (?,?,?,?,?,?)",
            (str(uuid4()), meeting_id, kind, occurred_at, meeting.duration_ms, json.dumps(payload)),
        )
