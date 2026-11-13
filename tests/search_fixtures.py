"""Pure synthetic M3 records; no app imports, SQL, files, audio or provider access.

small_corpus() describes 64 meetings / 256 segments; scale_corpus() describes
1,000 / 100,000. Records stream as fresh dictionaries in dependency order via
rows(table); nothing materializes the segment collection at module import.
Consumers own insertion, active-pointer ordering and derived-index construction.
These are fixture records, not new public search DTOs or a search implementation.

Each meeting has one historical segment and the remaining active segments. All
timing is absent: these authored words have no measured audio timeline. Reused
speaker_0 labels have distinct request scopes and explicit participant links.
Text cases describe authored source content, not FTS tokenizer/ranking promises.
The cohort has exactly 50 active complete segment matches; combined date/status/
participant filters select exactly five (including matches beyond limit=10).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

COHORT_TOKEN = "syntheticcohortneedle"
VISIBILITY_TOKEN = "syntheticvisibilityneedle"
HISTORICAL_TOKEN = "synthetichistoricalonlyneedle"
PARTIAL_TOKEN = "syntheticpartialonlyneedle"
DELETING_TOKEN = "syntheticdeletingonlyneedle"
MISSING_TOKEN = "syntheticneverpresentneedle"
BASE_TIME = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
TABLES = (
    "participants",
    "meetings",
    "transcript_revisions",
    "segments",
    "meeting_participants",
    "speaker_assignments",
)
LEXICAL_TEXTS = (
    'Café ÁRBOL résumé CAFÉ; literal "quoted phrase" appears here.',
    "Two letters: AI. Operator-looking words: NEAR OR AND NOT title:x col: -x * ^.",
    'An unmatched "quote and a doubled ""quote remain source text.',
    "<b>plain text</b> & <script>synthetic</script> are never markup instructions.",
    "日本語 Ελληνικά naïve jalapeño; punctuation stays in source.",
)


def _id(kind: int, ordinal: int) -> str:
    """Canonical, reproducible UUIDs in disjoint fixture-only namespaces."""
    return str(UUID(int=(kind << 96) | (ordinal + 1), version=4))


def _date(meeting_index: int) -> str:
    # Four equal dates force consumers to use a deterministic identity tie-break.
    return (BASE_TIME + timedelta(minutes=meeting_index // 4)).isoformat()


def _speaker(scope: int) -> str:
    return f"synthetic:request-{scope}:speaker_0"


@dataclass(frozen=True, slots=True)
class SearchCorpus:
    """Repeatable, lazily generated records with explicitly authored oracles.

    meeting_count >= 64 and segments_per_meeting >= 3 retain all edge cases.
    Dictionary fields follow existing M2 storage values, not proposed M3 queries.
    active_revision_id points forward; DB consumers must resolve circular FKs in
    their own real transaction. No insertion adapter or DB writes live here.
    """

    meeting_count: int = 64
    segments_per_meeting: int = 4

    def __post_init__(self) -> None:
        if self.meeting_count < 64 or self.segments_per_meeting < 3:
            raise ValueError("Corpus requires >=64 meetings and >=3 segments per meeting")

    @property
    def segment_count(self) -> int:
        return self.meeting_count * self.segments_per_meeting

    def meeting_id(self, index: int) -> str:
        return _id(1, index)

    def revision_id(self, index: int, *, historical: bool = False) -> str:
        return _id(2, index * 2 + (0 if historical else 1))

    def segment_id(self, index: int, ordinal: int, *, historical: bool = False) -> str:
        slot = 0 if historical else ordinal + 1
        return _id(3, (index << 32) | slot)

    @property
    def filter_case(self) -> dict[str, Any]:
        """Input values and IDs for 50 matches / five combined-filter survivors."""
        return {
            "text": COHORT_TOKEN,
            "from_utc": _date(40),
            "to_utc": _date(52),
            "statuses": ("completed",),
            "participant_id": _id(4, 0),
            "limit": 10,
            "unfiltered_segment_ids": tuple(self.segment_id(i, 0) for i in range(50)),
            "filtered_segment_ids": tuple(self.segment_id(i, 0) for i in range(45, 50)),
        }

    @property
    def visibility_case(self) -> dict[str, Any]:
        return {
            "text": VISIBILITY_TOKEN,
            "historical_segment_id": self.segment_id(50, 0, historical=True),
            "historical_only_text": HISTORICAL_TOKEN,
            "partial_segment_id": self.segment_id(51, 0),
            "partial_only_text": PARTIAL_TOKEN,
            "deleting_segment_id": self.segment_id(52, 0),
            "deleting_only_text": DELETING_TOKEN,
            "active_complete_segment_id": self.segment_id(53, 0),
        }

    @property
    def lexical_cases(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "meeting_id": self.meeting_id(54 + i),
                "segment_id": self.segment_id(54 + i, 0),
                "text": text,
            }
            for i, text in enumerate(LEXICAL_TEXTS)
        )

    def _status(self, index: int) -> str:
        if index == 52:
            return "deleting"
        return "failed" if 40 <= index <= 44 or index == 51 else "completed"

    def _segment_text(self, index: int, ordinal: int, historical: bool) -> str:
        if historical:
            marker = f" {VISIBILITY_TOKEN} {HISTORICAL_TOKEN}" if index == 50 else ""
            return f"Synthetic historical meeting {index:04d}.{marker}"
        if ordinal == 0:
            if index < 50:
                return f"{COHORT_TOKEN} synthetic meeting {index:04d}."
            if index in (51, 52, 53):
                extra = PARTIAL_TOKEN if index == 51 else DELETING_TOKEN if index == 52 else ""
                return f"{VISIBILITY_TOKEN} {extra} synthetic meeting {index:04d}."
            if 54 <= index < 54 + len(LEXICAL_TEXTS):
                return LEXICAL_TEXTS[index - 54]
        return f"Synthetic meeting {index:04d} segment {ordinal:03d} ordinary context."

    def _segments(self, index: int, historical: bool) -> Iterator[dict[str, Any]]:
        count = 1 if historical else self.segments_per_meeting - 1
        for ordinal in range(count):
            yield {
                "id": self.segment_id(index, ordinal, historical=historical),
                "revision_id": self.revision_id(index, historical=historical),
                "ordinal": ordinal,
                "text": self._segment_text(index, ordinal, historical),
                "start_ms": None,
                "end_ms": None,
                "timing_source": "absent",
                "speaker_key": _speaker(ordinal % 2),
            }

    def rows(self, table: str) -> Iterator[dict[str, Any]]:
        """Stream fresh rows for a whitelisted fixture table; unknown names fail."""
        if table not in TABLES:
            raise ValueError(f"Unknown fixture table: {table}")
        if table == "participants":
            yield {"id": _id(4, 0), "display_name": "Synthetic participant Ágata"}
            yield {"id": _id(4, 1), "display_name": "Synthetic participant Beto"}
            return
        for index in range(self.meeting_count):
            meeting_id = self.meeting_id(index)
            if table == "meetings":
                yield {
                    "id": meeting_id,
                    "legacy_session_id": None,
                    "title": f"Synthetic library {index:04d} — Café"
                    if index == 54
                    else f"Synthetic library {index:04d}",
                    "notes": "Synthetic user notes, never transcript evidence.",
                    "started_at": _date(index),
                    "completed_at": None,
                    "duration_ms": None,
                    "paused_ms": 0,
                    "status": self._status(index),
                    "active_revision_id": self.revision_id(index),
                    "row_version": 1,
                    "created_at": BASE_TIME.isoformat(),
                    "updated_at": BASE_TIME.isoformat(),
                }
            elif table in ("transcript_revisions", "segments"):
                for historical in (True, False):
                    segments = self._segments(index, historical)
                    if table == "segments":
                        yield from segments
                    else:
                        raw_text = "\n".join(row["text"] for row in segments)
                        yield {
                            "id": self.revision_id(index, historical=historical),
                            "meeting_id": meeting_id,
                            "revision_number": 1 if historical else 2,
                            "raw_text": raw_text,
                            "presentation_text": None,
                            "provider": "synthetic",
                            "model": "authored-corpus",
                            "origin": "import",
                            "content_hash": hashlib.sha256(raw_text.encode()).hexdigest(),
                            "status": "partial" if index == 51 and not historical else "complete",
                            "created_at": BASE_TIME.isoformat(),
                        }
            elif table == "meeting_participants":
                yield {"meeting_id": meeting_id, "participant_id": _id(4, 1), "role": None}
                if 35 <= index < 50:
                    yield {"meeting_id": meeting_id, "participant_id": _id(4, 0), "role": None}
            elif table == "speaker_assignments":
                # Same bare label, two requests, two explicit identities; never infer
                # that speaker_0 in request-0 and request-1 means the same person.
                for historical in (True, False):
                    for scope in range(1 if historical else 2):
                        yield {
                            "meeting_id": meeting_id,
                            "revision_id": self.revision_id(index, historical=historical),
                            "speaker_key": _speaker(scope),
                            "participant_id": _id(4, 0)
                            if scope == 0 and 35 <= index < 50
                            else _id(4, 1)
                            if scope == 1
                            else None,
                        }


def small_corpus() -> SearchCorpus:
    return SearchCorpus()


def scale_corpus() -> SearchCorpus:
    return SearchCorpus(meeting_count=1000, segments_per_meeting=100)
