"""Immutable requests, evidence and results for grounded intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from here.memory.models import Citation, MeetingStatus
from here.memory.service import InvalidInput

Mode = Literal["local", "provider"]
Purpose = Literal["match", "summary"]
ExtractionKind = Literal["decision", "task", "blocker"]
ClaimKind = Literal["evidence", "fact", "synthesis", "uncertainty"]

_EXTRACTION_KINDS: tuple[ExtractionKind, ...] = ("decision", "task", "blocker")


def _validate_mode(mode: Mode) -> None:
    if mode not in ("local", "provider"):
        raise InvalidInput("Unknown intelligence mode")


def _validate_meeting_id(meeting_id: str) -> None:
    try:
        valid = isinstance(meeting_id, str) and str(UUID(meeting_id)) == meeting_id
    except ValueError:
        valid = False
    if not valid:
        raise InvalidInput("Invalid meeting ID")


@dataclass(frozen=True, slots=True)
class Budget:
    per_meeting: int = 6
    total: int = 24
    evidence_chars: int = 24_000
    output_tokens: int = 2_000

    def __post_init__(self) -> None:
        for value in (self.per_meeting, self.total, self.evidence_chars, self.output_tokens):
            if type(value) is not int or value < 1:
                raise InvalidInput("Intelligence budget limits must be positive integers")


@dataclass(frozen=True, slots=True)
class QuestionRequest:
    question: str
    meeting_ids: tuple[str, ...] = ()
    whole_library: bool = False
    mode: Mode = "local"
    purpose: Purpose = "match"
    budget: Budget = Budget()
    from_utc: str | None = None
    to_utc: str | None = None
    statuses: tuple[MeetingStatus, ...] = ()
    participant_id: str | None = None
    include_partial: bool = False

    def __post_init__(self) -> None:
        _validate_mode(self.mode)
        if not isinstance(self.question, str):
            raise InvalidInput("Question must be text")
        if self.purpose not in ("match", "summary"):
            raise InvalidInput("Unknown evidence purpose")
        if (
            not isinstance(self.meeting_ids, tuple)
            or type(self.whole_library) is not bool
            or bool(self.meeting_ids) == self.whole_library
        ):
            raise InvalidInput("Choose selected meetings or explicitly choose the whole library")
        for meeting_id in self.meeting_ids:
            _validate_meeting_id(meeting_id)
        if self.purpose == "summary" and len(self.meeting_ids) != 1:
            raise InvalidInput("A meeting summary requires exactly one selected meeting")


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    meeting_id: str
    kinds: tuple[ExtractionKind, ...] = _EXTRACTION_KINDS
    mode: Mode = "local"
    budget: Budget = Budget()
    from_utc: str | None = None
    to_utc: str | None = None
    statuses: tuple[MeetingStatus, ...] = ()
    participant_id: str | None = None
    include_partial: bool = False

    def __post_init__(self) -> None:
        _validate_mode(self.mode)
        _validate_meeting_id(self.meeting_id)
        if not isinstance(self.kinds, tuple) or any(
            kind not in _EXTRACTION_KINDS for kind in self.kinds
        ):
            raise InvalidInput("Unknown extraction kind")


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    id: str
    meeting_id: str
    revision_id: str
    segment_id: str
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    content_hash: str = ""
    ordinal: int = 0
    timing_source: str = "absent"
    partial: bool = False
    rank: float | None = None


@dataclass(frozen=True, slots=True)
class Coverage:
    segments: int = 0
    meetings: int = 0
    truncated: bool = False
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class Claim:
    kind: ClaimKind
    text: str
    citations: tuple[Citation, ...]
    uncertainty: bool = False


@dataclass(frozen=True, slots=True)
class Extraction:
    kind: ExtractionKind
    text: str
    citations: tuple[Citation, ...]
    owner_text: str | None = None
    due_text: str | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    mode: Mode = "local"
    model: str | None = None
    coverage: Coverage = Coverage()
    claims: tuple[Claim, ...] = ()
    extractions: tuple[Extraction, ...] = ()
    message: str | None = None
    status: Literal["ok", "insufficient_evidence", "cancelled"] = "ok"


class ProviderUnavailable(RuntimeError):
    """The explicitly selected provider is unavailable."""


class VerificationFailed(RuntimeError):
    """Evidence could not be verified for publication."""
