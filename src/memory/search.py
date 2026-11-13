"""Operation-owned catalog queries, safe FTS and literal keyset pagination.

The canonical catalog read view owns suppression validation, the bounded catalog
lock and the read transaction. The optional maintenance callback owns reindex resources.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, TypeVar

from here.memory.database import LibraryConnection, LibraryError
from here.memory.models import (
    Meeting,
    MeetingPage,
    MeetingQuery,
    MeetingStatus,
    MeetingSummary,
    ReindexReport,
    SearchCapabilities,
    SearchHit,
    SearchPage,
    SearchQuery,
    SuppressionSnapshot,
)
from here.memory.service import FtsUnavailable, InvalidInput, catalog_read_view, fts_available

_LIST_ORDER = "date-desc-null-last/id-asc"
_SEARCH_ORDER = "rank-asc/date-desc-null-last/id-asc/segment-asc"
_CURSOR_VERSION = 2
_ROUTES = {"fts", "hybrid", "literal", "literal-partial"}
_GENERATION_BATCH_ROWS = 256
_Query = TypeVar("_Query", MeetingQuery, SearchQuery)


def normalize_query_text(text: str) -> str:
    """Remove controls while retaining whitespace as token boundaries."""
    if not isinstance(text, str):
        raise InvalidInput("Query text must be a string")
    cleaned = "".join(
        " " if character.isspace() else character
        for character in text
        if character.isspace() or not unicodedata.category(character).startswith("C")
    )
    return " ".join(cleaned.split())


def build_fts_expression(text: str) -> str:
    return " ".join(
        '"' + token.replace('"', '""') + '"' for token in normalize_query_text(text).split()
    )


def _utc(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidInput("Date bounds must be timezone-aware ISO dates")
    try:
        measured = datetime.fromisoformat(value)
        if measured.tzinfo is None or measured.utcoffset() is None:
            raise ValueError("Timezone required")
        return measured.astimezone(timezone.utc).isoformat()
    except (ValueError, OverflowError) as error:
        raise InvalidInput("Date bounds must be timezone-aware ISO dates") from error


def _validated(query: _Query) -> _Query:
    if type(query.limit) is not int or not 1 <= query.limit <= 100:
        raise InvalidInput("Limit must be an integer from 1 to 100")
    if type(query.include_partial) is not bool:
        raise InvalidInput("include_partial must be a boolean")
    if query.participant_id is not None and (
        not isinstance(query.participant_id, str) or not query.participant_id
    ):
        raise InvalidInput("Participant identity must be a nonempty string")
    if not isinstance(query.statuses, (tuple, list)):
        raise InvalidInput("Statuses must be a sequence of meeting states")
    try:
        statuses = tuple(sorted({MeetingStatus(status) for status in query.statuses}))
    except (TypeError, ValueError) as error:
        raise InvalidInput("Unknown meeting status") from error
    start, end = _utc(query.from_utc), _utc(query.to_utc)
    if start is not None and end is not None and start >= end:
        raise InvalidInput("The start date must precede the end date")
    return replace(
        query, text=normalize_query_text(query.text), from_utc=start, to_utc=end, statuses=statuses
    )


def _scope(query: MeetingQuery | SearchQuery) -> list[object]:
    return [
        query.text,
        query.from_utc,
        query.to_utc,
        list(query.statuses),
        query.participant_id,
        query.include_partial,
    ]


def _scope_digest(scope: list[object]) -> str:
    encoded = json.dumps(scope, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _encode(value: list[object]) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _decode(cursor: object, kind: str, scope: list[object]) -> list[Any] | None:
    """Validate all saved keys before opening a connection or issuing any SQL."""
    if cursor is None:
        return None
    try:
        if not isinstance(cursor, str) or not cursor or len(cursor) > 16384:
            raise ValueError("Invalid token")
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
        expected_order = _LIST_ORDER if kind == "list" else _SEARCH_ORDER
        if (
            not isinstance(payload, list)
            or len(payload) != (6 if kind == "list" else 8)
            or payload[0] != "here-search"
            or type(payload[1]) is not int
            or payload[1] not in (1, _CURSOR_VERSION)
            or payload[2] != kind
            or payload[4] != expected_order
        ):
            raise ValueError("Cursor scope or version mismatch")
        if payload[1] == 1:
            matches_scope = json.dumps(payload[3], ensure_ascii=False) == json.dumps(
                scope, ensure_ascii=False
            )
        else:
            matches_scope = payload[3] == _scope_digest(scope)
        if not matches_scope:
            raise ValueError("Cursor scope mismatch")
        keys = payload[5]
        if not isinstance(keys, list) or len(keys) != (3 if kind == "list" else 5):
            raise ValueError("Invalid cursor keys")
        null, date, meeting = keys[:3] if kind == "list" else keys[1:4]
        if type(null) is not int or null not in (0, 1) or not _nonempty_string(meeting):
            raise ValueError("Invalid date/identity keys")
        if null == 1:
            if date is not None:
                raise ValueError("NULL flag requires NULL date")
        elif date is None or _utc(date) != date:
            raise ValueError("Non-NULL key requires canonical UTC date")
        if kind == "search":
            rank = keys[0]
            if type(rank) not in (int, float) or not math.isfinite(rank):
                raise ValueError("Rank must be finite")
            if not _nonempty_string(keys[4]) or payload[6] not in _ROUTES:
                raise ValueError("Invalid segment or backend")
            generation = payload[7]
            if (
                not isinstance(generation, str)
                or len(generation) != 64
                or any(c not in "0123456789abcdef" for c in generation)
            ):
                raise ValueError("Invalid generation")
        return payload
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        UnicodeError,
        binascii.Error,
        OverflowError,
        RecursionError,
    ) as error:
        raise InvalidInput("Malformed or mismatched cursor") from error


def _like(token: str) -> str:
    return "%" + token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _filters(
    query: MeetingQuery | SearchQuery,
    parameters: list[object],
    suppression: SuppressionSnapshot,
) -> str:
    clauses = ["m.status != 'deleting'"]
    if suppression.suppressed_meeting_ids:
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM temp.here_search_suppressed_meetings hidden "
            "WHERE hidden.id=m.id)"
        )
    if query.from_utc is not None:
        clauses.append("m.started_at >= ?")
        parameters.append(query.from_utc)
    if query.to_utc is not None:
        clauses.append("m.started_at < ?")
        parameters.append(query.to_utc)
    if query.statuses:
        clauses.append("m.status IN (" + ",".join("?" for _ in query.statuses) + ")")
        parameters.extend(query.statuses)
    if query.participant_id is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM meeting_participants p "
            "WHERE p.meeting_id=m.id AND p.participant_id=?)"
        )
        parameters.append(query.participant_id)
    return " AND ".join(clauses)


def _literal(text: str, parameters: list[object], *, title_only: bool = False) -> str:
    clauses = []
    for token in text.split():
        pattern = _like(token)
        if title_only:
            clauses.append("m.title LIKE ? ESCAPE '\\'")
            parameters.append(pattern)
        else:
            clauses.append("(m.title LIKE ? ESCAPE '\\' OR s.text LIKE ? ESCAPE '\\')")
            parameters.extend((pattern, pattern))
    return " AND ".join(clauses) or "1"


def _date_keyset(keys: list[Any], parameters: list[object], *, search: bool) -> str:
    null, date, meeting = keys[1:4] if search else keys[:3]
    identity = "meeting_id" if search else "m.id"
    date_column = "started_at" if search else "m.started_at"
    tie = f"{identity} > ?"
    tie_parameters: list[object] = [meeting]
    if search:
        tie += f" OR ({identity} = ? AND segment_id > ?)"
        tie_parameters.extend((meeting, keys[4]))
    if null:
        parameters.extend(tie_parameters)
        return f"({date_column} IS NULL AND ({tie}))"
    parameters.extend((date, date, *tie_parameters))
    return f"({date_column} IS NULL OR {date_column} < ? OR ({date_column} = ? AND ({tie})))"


def _snippet(text: str, tokens: list[str]) -> str:
    """A character cap remains necessary after FTS's token-based snippet cap."""
    if len(text) <= 160:
        return text
    # Case/diacritic folding may change lengths. Map matches back to original
    # character positions so the returned window always consists of source text.
    folded = []
    source_positions = []
    for position, character in enumerate(text):
        pieces = "".join(
            piece
            for piece in unicodedata.normalize("NFD", character.casefold())
            if not unicodedata.combining(piece)
        )
        folded.append(pieces)
        source_positions.extend([position] * len(pieces))
    searchable = "".join(folded)
    found = []
    for token in tokens:
        folded_token = "".join(
            piece
            for piece in unicodedata.normalize("NFD", token.casefold())
            if not unicodedata.combining(piece)
        )
        match = searchable.find(folded_token) if folded_token else -1
        if match >= 0:
            found.append(source_positions[match])
    start = max(0, min(found) - 64) if found else 0
    start = min(start, max(0, len(text) - 160))
    return text[start : start + 160]


class SearchService:
    def __init__(
        self,
        connection_factory: Callable[[], LibraryConnection],
        *,
        reindexer: Callable[[], ReindexReport] | None = None,
    ) -> None:
        self._connection_factory = connection_factory
        self._reindexer = reindexer

    def reindex(self) -> ReindexReport:
        reindexer = self._reindexer
        if reindexer is None:
            raise LibraryError("Library maintenance is unavailable for this search service")
        return reindexer()

    @contextmanager
    def _snapshot(self) -> Iterator[tuple[LibraryConnection, SuppressionSnapshot]]:
        library = self._connection_factory()
        try:
            with catalog_read_view(library) as suppression:
                yield library, suppression
        finally:
            library.close()

    @staticmethod
    def _prepare_suppression(library: LibraryConnection, suppression: SuppressionSnapshot) -> None:
        # One bind per INSERT keeps large sets out of page parameter
        # budgets. TEMP storage belongs to this fresh connection only; the B view
        # rolls its transaction back before our finally-close discards the table.
        if suppression.suppressed_meeting_ids:
            library.execute(
                "CREATE TEMP TABLE here_search_suppressed_meetings (id TEXT PRIMARY KEY NOT NULL)"
            )
            library.connection.executemany(
                "INSERT INTO temp.here_search_suppressed_meetings(id) VALUES (?)",
                ((identity,) for identity in suppression.suppressed_meeting_ids),
            )

    @staticmethod
    def _capabilities(library: LibraryConnection) -> SearchCapabilities:
        runtime = fts_available(library.connection)
        exists = (
            library.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", ("segments_fts",)
            ).fetchone()
            is not None
        )
        available = runtime and exists
        return SearchCapabilities(available, () if available else (FtsUnavailable(),))

    def capabilities(self) -> SearchCapabilities:
        with self._snapshot() as (library, _suppression):
            return self._capabilities(library)

    def list_meetings(self, query: MeetingQuery) -> MeetingPage:
        normalized = _validated(query)
        scope = _scope(normalized)
        cursor = _decode(normalized.cursor, "list", scope)
        with self._snapshot() as (library, suppression):
            self._prepare_suppression(library, suppression)
            parameters: list[object] = []
            where = _filters(normalized, parameters, suppression)
            # Reservations without a revision remain listable with their real NULLs.
            where += " AND (m.active_revision_id IS NULL OR r.status='complete'"
            where += " OR r.status='partial')" if normalized.include_partial else ")"
            where += " AND " + _literal(normalized.text, parameters, title_only=True)
            if cursor is not None:
                where += " AND " + _date_keyset(cursor[5], parameters, search=False)
            parameters.append(normalized.limit + 1)
            sql = (
                "SELECT m.*,r.status AS revision_status,EXISTS ("
                "SELECT 1 FROM audio_assets a WHERE a.meeting_id=m.id AND a.status='available' "
                "AND (m.active_revision_id IS NULL OR EXISTS ("
                "SELECT 1 FROM revision_audio_assets ra "
                "WHERE ra.meeting_id=m.id AND ra.revision_id=m.active_revision_id "
                "AND ra.asset_id=a.id))"
                ") AS audio_available FROM meetings m LEFT JOIN transcript_revisions r "
                "ON r.meeting_id=m.id AND r.id=m.active_revision_id WHERE "
                + where
                + " ORDER BY (m.started_at IS NULL) ASC,m.started_at DESC,m.id ASC LIMIT ?"
            )
            rows = library.execute(sql, tuple(parameters)).fetchall()
        items = []
        for row in rows[: normalized.limit]:
            fields = dict(row)
            partial = fields.pop("revision_status") == "partial"
            audio_available = bool(fields.pop("audio_available"))
            fields["status"] = MeetingStatus(fields["status"])
            items.append(MeetingSummary(Meeting(**fields), audio_available, partial))
        next_cursor = None
        if len(rows) > normalized.limit:
            last = items[-1].meeting
            next_cursor = _encode(
                [
                    "here-search",
                    _CURSOR_VERSION,
                    "list",
                    _scope_digest(scope),
                    _LIST_ORDER,
                    [int(last.started_at is None), last.started_at, last.id],
                ]
            )
        return MeetingPage(tuple(items), next_cursor)

    @staticmethod
    def _generation(
        library: LibraryConnection,
        suppression: SuppressionSnapshot,
        *,
        indexed: bool,
        partial: bool,
    ) -> str:
        """Digest actual logical documents, independent of FTS rowids/layout.

        Catalog immutable-identity assumptions alone cannot prove that an index-only
        mutation preserved rank inputs. Hash the complete actual index, including
        nonmatches, filtered and still-indexed suppressed documents. Also bind the
        actual catalog eligibility, relations and selected branch evidence. No
        cached token or connection-local counter substitutes for this snapshot.
        """
        digest = hashlib.sha256(
            f"here-search-corpus-v2/batch{_GENERATION_BATCH_ROWS}\x00".encode("ascii")
        )
        sources = [
            (
                "meetings",
                "SELECT id,title,started_at,status,active_revision_id,row_version "
                "FROM meetings ORDER BY id",
            ),
            (
                "relations",
                "SELECT meeting_id,participant_id,role FROM meeting_participants "
                "ORDER BY meeting_id,participant_id",
            ),
        ]
        if indexed:
            sources.append(
                (
                    "fts-schema",
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='segments_fts'",
                )
            )
            sources.append(
                (
                    "fts",
                    "SELECT meeting_id,revision_id,segment_id,text,title "
                    "FROM segments_fts ORDER BY meeting_id,revision_id,segment_id,text,title",
                )
            )
        allowed = "('complete','partial')" if partial else "('complete')"
        sources.append(
            (
                "active",
                "SELECT m.id,r.id,r.status,s.id,s.ordinal,s.text,s.start_ms,"
                "s.end_ms,s.timing_source FROM meetings m JOIN transcript_revisions r "
                "ON r.meeting_id=m.id AND r.id=m.active_revision_id "
                "LEFT JOIN segments s ON s.revision_id=r.id WHERE r.status IN "
                + allowed
                + " ORDER BY m.id,r.id,s.ordinal,s.id",
            )
        )
        for name, sql in sources:
            digest.update(name.encode("ascii") + b"\x00")
            cursor = library.execute(sql)
            cursor.row_factory = None
            # Fixed row batches retain every ordered cell without a JSON encoder
            # and frame per row. Working state never grows to the whole corpus.
            while rows := cursor.fetchmany(_GENERATION_BATCH_ROWS):
                encoded = json.dumps(
                    rows,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            # Positive-length JSON frames cannot collide with this source boundary.
            digest.update(b"\x00" * 8)
        encoded_suppression = json.dumps(
            ("here-search-suppression-v1", suppression.fingerprint), separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded_suppression).to_bytes(8, "big"))
        digest.update(encoded_suppression)
        return digest.hexdigest()

    @staticmethod
    def _branch(
        query: SearchQuery,
        suppression: SuppressionSnapshot,
        *,
        fts: bool,
        partial: bool,
        parameters: list[object],
    ) -> str:
        if fts:
            projection = "bm25(segments_fts) AS rank,snippet(segments_fts,3,'','','',24) AS excerpt"
            source = (
                "segments_fts JOIN meetings m ON m.id=segments_fts.meeting_id "
                "JOIN transcript_revisions r ON r.meeting_id=m.id "
                "AND r.id=m.active_revision_id AND r.id=segments_fts.revision_id "
                "JOIN segments s ON s.revision_id=r.id AND s.id=segments_fts.segment_id"
            )
            match = "segments_fts MATCH ?"
            parameters.append(build_fts_expression(query.text))
        else:
            projection = "0 AS rank,s.text AS excerpt"
            source = (
                "meetings m JOIN transcript_revisions r ON r.meeting_id=m.id "
                "AND r.id=m.active_revision_id JOIN segments s ON s.revision_id=r.id"
            )
            match = _literal(query.text, parameters)
        where = _filters(query, parameters, suppression)
        return (
            "SELECT " + projection + ",m.id AS meeting_id,r.id AS revision_id,"
            "s.id AS segment_id,m.title,m.started_at,s.start_ms,s.end_ms,s.timing_source,"
            "s.text,r.status='partial' AS partial FROM "
            + source
            + " WHERE "
            + match
            + " AND r.status='"
            + ("partial" if partial else "complete")
            + "' AND "
            + where
        )

    def search(self, query: SearchQuery) -> SearchPage | MeetingPage:
        normalized = _validated(query)
        if not normalized.text:
            return self.list_meetings(
                MeetingQuery(
                    text="",
                    from_utc=normalized.from_utc,
                    to_utc=normalized.to_utc,
                    statuses=normalized.statuses,
                    participant_id=normalized.participant_id,
                    limit=normalized.limit,
                    cursor=normalized.cursor,
                    include_partial=normalized.include_partial,
                )
            )
        scope = _scope(normalized)
        cursor = _decode(normalized.cursor, "search", scope)
        with self._snapshot() as (library, suppression):
            capability = self._capabilities(library)
            indexed = capability.fts_available
            route = (
                ("hybrid" if normalized.include_partial else "fts")
                if indexed
                else ("literal-partial" if normalized.include_partial else "literal")
            )
            if cursor is not None and cursor[6] != route:
                raise InvalidInput("Cursor backend changed; restart the query")
            generation = self._generation(
                library, suppression, indexed=indexed, partial=normalized.include_partial
            )
            if cursor is not None and cursor[7] != generation:
                raise InvalidInput(
                    "Search corpus changed; restart the query", reason="corpus_changed"
                )
            self._prepare_suppression(library, suppression)
            parameters: list[object] = []
            complete = self._branch(
                normalized, suppression, fts=indexed, partial=False, parameters=parameters
            )
            # Materialization keeps FTS auxiliary functions inside their MATCH
            # context. It also prevents branch-local LIMITs from losing global hits.
            sql = "WITH complete AS MATERIALIZED (" + complete + ")"
            if indexed:
                # A segment has one canonical row but may have multiple FTS matches.
                # Select its best match before the global keyset can split those copies.
                sql += (
                    ", ranked AS (SELECT *,row_number() OVER (PARTITION BY segment_id "
                    "ORDER BY rank,excerpt COLLATE BINARY) AS match_number FROM complete), "
                    "hits AS (SELECT rank,excerpt,meeting_id,revision_id,segment_id,title,"
                    "started_at,start_ms,end_ms,timing_source,text,partial "
                    "FROM ranked WHERE match_number=1"
                )
            else:
                sql += ", hits AS (SELECT * FROM complete"
            if normalized.include_partial:
                sql += " UNION ALL " + self._branch(
                    normalized, suppression, fts=False, partial=True, parameters=parameters
                )
            sql += ") SELECT * FROM hits"
            if cursor is not None:
                rank = cursor[5][0]
                parameters.extend((rank, rank))
                continuation = _date_keyset(cursor[5], parameters, search=True)
                sql += " WHERE (rank > ? OR (rank = ? AND " + continuation + "))"
            sql += (
                " ORDER BY rank ASC,(started_at IS NULL) ASC,started_at DESC,"
                "meeting_id ASC,segment_id ASC LIMIT ?"
            )
            parameters.append(normalized.limit + 1)
            rows = library.execute(sql, tuple(parameters)).fetchall()
        tokens = normalized.text.split()
        items = tuple(
            SearchHit(
                meeting_id=row["meeting_id"],
                revision_id=row["revision_id"],
                segment_id=row["segment_id"],
                snippet=_snippet(row["excerpt"], tokens),
                title=row["title"],
                started_at=row["started_at"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                timing_source=row["timing_source"],
                partial=bool(row["partial"]),
            )
            for row in rows[: normalized.limit]
        )
        next_cursor = None
        if len(rows) > normalized.limit:
            last = rows[normalized.limit - 1]
            next_cursor = _encode(
                [
                    "here-search",
                    _CURSOR_VERSION,
                    "search",
                    _scope_digest(scope),
                    _SEARCH_ORDER,
                    [
                        last["rank"],
                        int(last["started_at"] is None),
                        last["started_at"],
                        last["meeting_id"],
                        last["segment_id"],
                    ],
                    route,
                    generation,
                ]
            )
        return SearchPage(items, next_cursor, not indexed)
