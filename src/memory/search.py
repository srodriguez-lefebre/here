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
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import UUID

from here.memory import _native_search
from here.memory.database import LibraryConnection, LibraryError
from here.memory.models import (
    EvidenceBlock,
    EvidenceQuery,
    EvidenceSnapshot,
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
_FTS_GENERATION_SELECT = "SELECT meeting_id,revision_id,segment_id,text,title FROM segments_fts"
_FTS_SCHEMA_PREFIX = "CREATE VIRTUAL TABLE segments_fts USING fts5("
_FTS_TEXT_TYPES = (str, str, str, str, str)
_ACTIVE_COLUMNS = "m.id,r.id,r.status,s.id,s.ordinal,s.text,s.start_ms,s.end_ms,s.timing_source"
_EVIDENCE_REVISIONS = (
    "meetings m JOIN transcript_revisions r ON r.meeting_id=m.id AND r.id=m.active_revision_id"
)
_EVIDENCE_SEGMENTS = _EVIDENCE_REVISIONS + " JOIN segments s ON s.revision_id=r.id"
_EVIDENCE_METADATA = (
    "m.id AS meeting_id,r.id AS revision_id,s.id AS segment_id,r.content_hash,"
    "s.ordinal,s.start_ms,s.end_ms,s.timing_source,r.status='partial' AS partial,m.started_at,"
    "CASE WHEN instr(s.text,char(0))=0 THEN length(s.text) END AS text_chars,"
    "CASE WHEN instr(s.text,char(0))>0 THEN length(CAST(s.text AS BLOB)) END AS text_bytes"
)
_EVIDENCE_RANK_ORDER = (
    "rank ASC,(started_at IS NULL) ASC,started_at DESC,meeting_id ASC,segment_id ASC"
)
_Query = TypeVar("_Query", MeetingQuery, SearchQuery)


def _append_generation_batch(digest: Any, rows: list[tuple[object, ...]]) -> None:
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _ordered_text_batch(rows: list[tuple[object, ...]], previous: tuple[object, ...]) -> bool:
    for row in rows:
        if tuple(map(type, row)) != _FTS_TEXT_TYPES or row < previous:
            return False
        previous = row
    return True


def _ordered_fts_digest(library: LibraryConnection, digest: Any) -> Any:
    with closing(library.execute("PRAGMA encoding")) as cursor:
        cursor.row_factory = None
        if cursor.fetchone() != ("UTF-8",) or cursor.connection.text_factory is not str:
            return None

    # FTS5 provides the rowid ordering. Prove BINARY order in this view; the
    # order of UTF-8 bytes for valid text is the same as Python's str order.
    try:
        # A cheap suffix check avoids replaying nearly the whole input for
        # common late inserts or an appended replacement of an old revision.
        with closing(
            library.execute(
                _FTS_GENERATION_SELECT + " ORDER BY rowid DESC LIMIT ?", (_GENERATION_BATCH_ROWS,)
            )
        ) as cursor:
            cursor.row_factory = None
            tail = cursor.fetchall()
        tail.reverse()
        if not _ordered_text_batch(tail, ()):
            return None

        trial = digest.copy()
        trial.update(b"fts\x00")
        previous: tuple[object, ...] = ()
        with closing(library.execute(_FTS_GENERATION_SELECT + " ORDER BY rowid")) as cursor:
            cursor.row_factory = None
            while rows := cursor.fetchmany(_GENERATION_BATCH_ROWS):
                if not _ordered_text_batch(rows, previous):
                    return None
                previous = rows[-1]
                _append_generation_batch(trial, rows)
    except sqlite3.OperationalError as exc:
        # An earlier canonical frame can fail to encode before a later
        # malformed cell is decoded. Replay to restore that error ordering.
        if str(exc).startswith("Could not decode to UTF-8 column"):
            return None
        raise
    trial.update(b"\x00" * 8)
    return trial


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
    meeting_id = query.meeting_id
    if meeting_id is not None:
        try:
            canonical = isinstance(meeting_id, str) and str(UUID(meeting_id)) == meeting_id
        except ValueError:
            canonical = False
        if not canonical:
            raise InvalidInput("Meeting identity must be a canonical UUID")
    return replace(
        query, text=normalize_query_text(query.text), from_utc=start, to_utc=end, statuses=statuses
    )


def _scope(query: MeetingQuery | SearchQuery) -> list[object]:
    scope: list[object] = [
        query.text,
        query.from_utc,
        query.to_utc,
        list(query.statuses),
        query.participant_id,
        query.include_partial,
    ]
    # Retain the exact scope encoding for existing global cursors.
    if query.meeting_id is not None:
        scope.append(query.meeting_id)
    return scope


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


def _canonical_uuid(value: object) -> bool:
    try:
        return isinstance(value, str) and str(UUID(value)) == value
    except ValueError:
        return False


def _validated_evidence(query: EvidenceQuery) -> EvidenceQuery:
    if query.purpose not in ("match", "summary"):
        raise InvalidInput("Evidence purpose must be match or summary")
    if type(query.whole_library) is not bool or not isinstance(query.meeting_ids, (tuple, list)):
        raise InvalidInput("Evidence scope must be explicit")
    if bool(query.meeting_ids) == query.whole_library:
        raise InvalidInput("Select meetings or explicitly select the whole library")
    if not all(_canonical_uuid(value) for value in query.meeting_ids):
        raise InvalidInput("Meeting identity must be a canonical UUID")
    meetings = tuple(dict.fromkeys(query.meeting_ids))
    if not isinstance(query.terms, (tuple, list)):
        raise InvalidInput("Evidence terms must be a sequence")
    terms = tuple(dict.fromkeys(filter(None, map(normalize_query_text, query.terms))))
    for value in (query.per_meeting, query.total, query.evidence_chars):
        if type(value) is not int or value < 1:
            raise InvalidInput("Evidence budgets must be positive integers")
    if query.total > (2**63 - 2) // 4:
        raise InvalidInput("Evidence total is out of range")
    if query.purpose == "summary" and (len(meetings) != 1 or terms):
        raise InvalidInput("Summary requires one selected meeting and no terms")
    if query.revision_id is not None or query.after_ordinal is not None:
        if (
            query.purpose != "summary"
            or not _canonical_uuid(query.revision_id)
            or type(query.after_ordinal) is not int
            or not 0 <= query.after_ordinal < 2**63
        ):
            raise InvalidInput("Summary continuation requires a revision and ordinal")
    filters = _validated(
        MeetingQuery(
            from_utc=query.from_utc,
            to_utc=query.to_utc,
            statuses=query.statuses,
            participant_id=query.participant_id,
            include_partial=query.include_partial,
        )
    )
    return replace(
        query,
        terms=terms,
        meeting_ids=meetings,
        from_utc=filters.from_utc,
        to_utc=filters.to_utc,
        statuses=filters.statuses,
    )


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
    if query.meeting_id is not None:
        clauses.append("m.id = ?")
        parameters.append(query.meeting_id)
    return " AND ".join(clauses)


def _evidence_filter(
    query: EvidenceQuery,
    library: LibraryConnection,
    suppression: SuppressionSnapshot,
    parameters: list[object],
) -> str:
    filters = MeetingQuery(
        from_utc=query.from_utc,
        to_utc=query.to_utc,
        statuses=query.statuses,
        participant_id=query.participant_id,
    )
    where = _filters(filters, parameters, suppression)
    revision_status = (
        "r.status IN ('complete','partial')" if query.include_partial else "r.status='complete'"
    )
    where += " AND " + revision_status
    if query.meeting_ids:
        library.execute("CREATE TEMP TABLE here_evidence_scope (id TEXT PRIMARY KEY NOT NULL)")
        library.connection.executemany(
            "INSERT INTO temp.here_evidence_scope(id) VALUES (?)",
            ((identity,) for identity in query.meeting_ids),
        )
        where += " AND m.id IN (SELECT id FROM temp.here_evidence_scope)"
    return where


def _evidence_candidates(
    library: LibraryConnection,
    query: EvidenceQuery,
    where: str,
    filters: list[object],
    *,
    indexed: bool,
) -> list[sqlite3.Row]:
    # Keep BM25 inside its MATCH context. Scope restricts the rows returned by
    # MATCH while corpus statistics still come from the global FTS index.
    complete = "SELECT NULL AS id,NULL AS rank WHERE 0"
    parameters: list[object] = []
    if indexed:
        source = (
            "segments_fts JOIN meetings m ON m.id=segments_fts.meeting_id "
            "JOIN transcript_revisions r ON r.meeting_id=m.id "
            "AND r.id=m.active_revision_id AND r.id=segments_fts.revision_id "
            "JOIN segments s ON s.revision_id=r.id AND s.id=segments_fts.segment_id"
        )
        # MIN(rank) leaves a single candidate for each segment even if FTS has
        # duplicate derived rows. The canonical source provides all other fields.
        complete = (
            "SELECT s.id AS id,bm25(segments_fts) AS rank FROM "
            + source
            + " WHERE segments_fts MATCH ? AND r.status='complete' AND "
            + where
        )
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in query.terms)
        parameters.extend((expression, *filters))
    sql = f"WITH complete AS MATERIALIZED ({complete}), hits AS (SELECT id,min(rank) AS rank "
    sql += "FROM complete GROUP BY id"
    if not indexed or query.include_partial:
        # ASCII case folding matches SQLite LIKE; instr treats %, _ and backslash
        # literally and continues through NUL in the transcript.
        literal = " OR ".join(
            "(instr(lower(m.title),lower(?))>0 OR instr(lower(s.text),lower(?))>0)"
            for _ in query.terms
        )
        sql += " UNION ALL SELECT s.id,0 FROM " + _EVIDENCE_SEGMENTS + " WHERE " + where
        if indexed:
            sql += " AND r.status='partial'"
        sql += " AND (" + literal + ")"
        parameters.extend(filters)
        parameters.extend(term for term in query.terms for _ in range(2))
    sql += (
        "), candidates AS (SELECT hits.rank,"
        + _EVIDENCE_METADATA
        + " FROM hits JOIN segments s ON s.id=hits.id "
        "JOIN transcript_revisions r ON r.id=s.revision_id "
        "JOIN meetings m ON m.id=r.meeting_id), "
        "rounds AS (SELECT *,row_number() OVER (PARTITION BY meeting_id ORDER BY "
        + _EVIDENCE_RANK_ORDER
        + ") AS round FROM candidates) SELECT * FROM rounds ORDER BY round,"
        + _EVIDENCE_RANK_ORDER
        + " LIMIT ?"
    )
    # The global window also bounds each meeting. Its final row is a sentinel
    # only, whether it belongs to a new meeting or follows that meeting's window.
    parameters.append(4 * query.total + 1)
    return library.execute(sql, tuple(parameters)).fetchall()


def _read_evidence_block(
    library: LibraryConnection, row: sqlite3.Row, remaining_chars: int
) -> EvidenceBlock | None:
    length = row["text_chars"]
    if length is not None:
        if length > remaining_chars:
            return None
    elif row["text_bytes"] is None or row["text_bytes"] > 4 * remaining_chars:
        return None
    source = library.execute(
        "SELECT text FROM segments WHERE revision_id=? AND id=?",
        (row["revision_id"], row["segment_id"]),
    ).fetchone()
    if source is None:
        raise LibraryError("Evidence source disappeared inside the catalog read view")
    text = source["text"]
    if len(text) > remaining_chars:
        return None
    return EvidenceBlock(
        meeting_id=row["meeting_id"],
        revision_id=row["revision_id"],
        segment_id=row["segment_id"],
        text=text,
        content_hash=row["content_hash"],
        ordinal=row["ordinal"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        timing_source=row["timing_source"],
        partial=bool(row["partial"]),
        rank=row["rank"],
    )


def _evidence_neighbors(
    library: LibraryConnection, block: EvidenceBlock, where: str, filters: list[object]
) -> Iterator[sqlite3.Row]:
    # Gaps in legacy ordinals do not make an otherwise adjacent segment vanish.
    for comparison, order in (("<", "DESC"), (">", "ASC")):
        row = library.execute(
            f"SELECT NULL AS rank,{_EVIDENCE_METADATA} FROM {_EVIDENCE_SEGMENTS} WHERE {where}"
            + f" AND r.id=? AND s.ordinal{comparison}? ORDER BY s.ordinal {order} LIMIT 1",
            (*filters, block.revision_id, block.ordinal),
        ).fetchone()
        if row is not None:
            yield row


def _match_evidence(
    library: LibraryConnection,
    query: EvidenceQuery,
    where: str,
    filters: list[object],
    *,
    indexed: bool,
) -> EvidenceSnapshot:
    rows = _evidence_candidates(library, query, where, filters, indexed=indexed)
    window = 4 * query.total
    blocks: list[EvidenceBlock] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    remaining = query.evidence_chars
    truncated = len(rows) > window

    def include(row: sqlite3.Row) -> None:
        nonlocal remaining, truncated
        segment_id = row["segment_id"]
        if segment_id in seen:
            return
        seen.add(segment_id)
        meeting_id = row["meeting_id"]
        if len(blocks) >= query.total or counts[meeting_id] >= query.per_meeting:
            truncated = True
            return
        block = _read_evidence_block(library, row, remaining)
        if block is None:
            truncated = True
            return
        blocks.append(block)
        counts[meeting_id] += 1
        remaining -= len(block.text)

    for row in rows[:window]:
        include(row)
    for match in tuple(blocks):
        for neighbor in _evidence_neighbors(library, match, where, filters):
            include(neighbor)
    return EvidenceSnapshot(query, tuple(blocks), truncated=truncated, degraded=not indexed)


def _summarize_evidence(
    library: LibraryConnection, query: EvidenceQuery, where: str, filters: list[object]
) -> EvidenceSnapshot:
    revision = library.execute(
        "SELECT r.id FROM " + _EVIDENCE_REVISIONS + " WHERE " + where, tuple(filters)
    ).fetchone()
    revision_id = revision["id"] if revision is not None else None
    if query.revision_id is not None and query.revision_id != revision_id:
        raise InvalidInput("Evidence revision changed; restart retrieval", reason="corpus_changed")
    if revision_id is None:
        return EvidenceSnapshot(query)
    limit = min(query.total, query.per_meeting)
    after = -1 if query.after_ordinal is None else query.after_ordinal
    rows = library.execute(
        f"SELECT NULL AS rank,{_EVIDENCE_METADATA} FROM {_EVIDENCE_SEGMENTS} WHERE {where}"
        " AND r.id=? AND s.ordinal>? ORDER BY s.ordinal LIMIT ?",
        (*filters, revision_id, after, limit + 1),
    ).fetchall()
    remaining = query.evidence_chars
    truncated = len(rows) > limit
    next_ordinal = rows[limit - 1]["ordinal"] if truncated else None
    blocks = []
    for row in rows[:limit]:
        block = _read_evidence_block(library, row, remaining)
        if block is None:
            truncated = True
            continue
        blocks.append(block)
        remaining -= len(block.text)
    return EvidenceSnapshot(
        query,
        tuple(blocks),
        truncated=truncated,
        next_ordinal=next_ordinal,
        revision_id=revision_id,
    )


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

    def evidence_snapshot(self, query: EvidenceQuery) -> EvidenceSnapshot:
        """Retrieve bounded source evidence inside one guarded catalog view."""
        normalized = _validated_evidence(query)
        with self._snapshot() as (library, suppression):
            if normalized.purpose == "match" and not normalized.terms:
                return EvidenceSnapshot(normalized)
            self._prepare_suppression(library, suppression)
            filters: list[object] = []
            where = _evidence_filter(normalized, library, suppression, filters)
            if normalized.purpose == "summary":
                return _summarize_evidence(library, normalized, where, filters)
            indexed = self._capabilities(library).fts_available
            return _match_evidence(library, normalized, where, filters, indexed=indexed)

    def revalidate_evidence(self, snapshot: EvidenceSnapshot) -> bool:
        """Check exact cited source and current eligibility in a fresh guarded view."""
        query = _validated_evidence(snapshot.query)
        with self._snapshot() as (library, suppression):
            self._prepare_suppression(library, suppression)
            filters: list[object] = []
            where = _evidence_filter(query, library, suppression, filters)
            sql = (
                f"SELECT 1 FROM {_EVIDENCE_SEGMENTS} WHERE {where}"
                " AND m.id=? AND r.id=? AND s.id=? AND r.content_hash=?"
                " AND s.ordinal=? AND s.text COLLATE BINARY=?"
                " AND s.start_ms IS ? AND s.end_ms IS ? AND s.timing_source=?"
                " AND (r.status='partial')=?"
            )
            for block in snapshot.blocks:
                current = library.execute(
                    sql,
                    (
                        *filters,
                        block.meeting_id,
                        block.revision_id,
                        block.segment_id,
                        block.content_hash,
                        block.ordinal,
                        block.text,
                        block.start_ms,
                        block.end_ms,
                        block.timing_source,
                        block.partial,
                    ),
                ).fetchone()
                if current is None:
                    return False
            return True

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
                    _FTS_GENERATION_SELECT
                    + " ORDER BY meeting_id,revision_id,segment_id,text,title",
                )
            )
        allowed = "('complete','partial')" if partial else "('complete')"
        active_from = (
            " FROM meetings m JOIN transcript_revisions r "
            "ON r.meeting_id=m.id AND r.id=m.active_revision_id "
            "LEFT JOIN segments s ON s.revision_id=r.id WHERE r.status IN "
            + allowed
            + " ORDER BY m.id,r.id,s.ordinal,s.id"
        )
        sources.append(("active", f"SELECT {_ACTIVE_COLUMNS}{active_from}"))
        current_fts_schema = False
        native_loaded = False
        for name, sql in sources:
            trial = None
            if name == "fts" and current_fts_schema:
                native_loaded = _native_search.load_fts_encoder(library)
                if native_loaded:
                    trial = _native_search.fts_digest(library, digest)
                else:
                    trial = _ordered_fts_digest(library, digest)
            elif name == "active" and native_loaded:
                active_native_sql = (
                    f"SELECT here_search_v2_active_row({_ACTIVE_COLUMNS}){active_from}"
                )
                trial = _native_search.active_digest(library, digest, active_native_sql)
            if trial is not None:
                digest = trial
                continue
            digest.update(name.encode("ascii") + b"\x00")
            with closing(library.execute(sql)) as cursor:
                cursor.row_factory = None
                # Fixed row batches retain every ordered cell. Working state
                # remains bounded on both the scan and the canonical replay.
                while rows := cursor.fetchmany(_GENERATION_BATCH_ROWS):
                    if name == "fts-schema":
                        current_fts_schema = (
                            len(rows) == 1
                            and isinstance(rows[0][0], str)
                            and rows[0][0].startswith(_FTS_SCHEMA_PREFIX)
                        )
                    _append_generation_batch(digest, rows)
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
                    meeting_id=normalized.meeting_id,
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
