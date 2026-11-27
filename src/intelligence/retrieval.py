"""Build scoped evidence queries and expose the exact guarded snapshot."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from here.intelligence.models import Coverage, EvidenceBlock, ExtractionRequest, QuestionRequest
from here.memory.models import EvidenceQuery, EvidenceSnapshot
from here.memory.search import SearchService

_WORD = re.compile(r"[^\W_]+")
_STOP_WORDS = frozenset(
    """
    about after all and any are been before being but can could did does during
    for from had has have how into its may might more most much must not now
    only our out over same should some such than that the their them then
    there these they this those through too under until very was were what when
    where which while who whom will with would you your

    algo algún alguna algunas alguno algunos ante aquí así bajo cada cómo como con
    contra cuál cuales cuándo cuando cuánto cualesquiera cuyo cuya dan del desde
    donde dónde durante ella ellas ellos entre era eran esa esas ese eso esos esta
    está estaban están estar estas este esto estos fue fueron hacia hace han hasta
    hay las les los más misma mismo muy nada nos nuestra nuestro otra otras otro
    otros para pero poca poco por porque qué que quien quién quienes quiénes
    sobre sola solo sólo son sus también tanto toda todas todo todos tras una unas
    uno unos usted ustedes
    """.split()
)


def _literal_terms(question: str) -> tuple[str, ...]:
    words = _WORD.findall(unicodedata.normalize("NFC", question).lower())
    informative = (word for word in words if len(word) >= 3 and word not in _STOP_WORDS)
    return tuple(dict.fromkeys(informative))


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    blocks: tuple[EvidenceBlock, ...]
    coverage: Coverage
    snapshot: EvidenceSnapshot


class Retriever:
    def __init__(self, search: SearchService) -> None:
        self._search = search

    def retrieve(
        self,
        request: QuestionRequest | ExtractionRequest,
        *,
        after_ordinal: int | None = None,
        revision_id: str | None = None,
    ) -> RetrievalResult:
        if isinstance(request, QuestionRequest):
            purpose = request.purpose
            terms = _literal_terms(request.question) if purpose == "match" else ()
            meeting_ids = request.meeting_ids
            whole_library = request.whole_library
        else:
            purpose = "summary"
            terms = ()
            meeting_ids = (request.meeting_id,)
            whole_library = False

        snapshot = self._search.evidence_snapshot(
            EvidenceQuery(
                purpose=purpose,
                terms=terms,
                meeting_ids=meeting_ids,
                whole_library=whole_library,
                from_utc=request.from_utc,
                to_utc=request.to_utc,
                statuses=request.statuses,
                participant_id=request.participant_id,
                include_partial=request.include_partial,
                per_meeting=request.budget.per_meeting,
                total=request.budget.total,
                evidence_chars=request.budget.evidence_chars,
                after_ordinal=after_ordinal,
                revision_id=revision_id,
            )
        )
        blocks = tuple(
            EvidenceBlock(
                id=f"E{index}",
                meeting_id=source.meeting_id,
                revision_id=source.revision_id,
                segment_id=source.segment_id,
                text=source.text,
                start_ms=source.start_ms,
                end_ms=source.end_ms,
                content_hash=source.content_hash,
                ordinal=source.ordinal,
                timing_source=source.timing_source,
                partial=source.partial,
                rank=source.rank,
            )
            for index, source in enumerate(snapshot.blocks, start=1)
        )
        coverage = Coverage(
            segments=len(blocks),
            meetings=len({block.meeting_id for block in blocks}),
            truncated=snapshot.truncated,
            degraded=snapshot.degraded,
        )
        return RetrievalResult(blocks=blocks, coverage=coverage, snapshot=snapshot)

    def revalidate(self, snapshot: EvidenceSnapshot) -> bool:
        return self._search.revalidate_evidence(snapshot)
