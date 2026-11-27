"""Contracts shared by grounded question and extraction consumers."""

from __future__ import annotations

import pytest
from here.intelligence.models import Budget, EvidenceBlock, ExtractionRequest, QuestionRequest
from here.memory.models import Citation
from here.memory.service import InvalidInput

_MEETING = "d39a6546-c405-4085-8a2a-8ca4c814907a"
_OTHER_MEETING = "3f7411b1-517f-4359-b5c5-26b622437b03"
_REVISION = "093d33d2-0df1-4c91-950b-2fd08c861649"
_SEGMENT = "53601bec-2b64-429c-9112-5802084014c2"
_SOURCE_TEXT = "A\U0001f600 café\0fin."


def test_default_budget_exposes_all_four_hard_limits() -> None:
    budget = Budget()
    assert (
        budget.per_meeting,
        budget.total,
        budget.evidence_chars,
        budget.output_tokens,
    ) == (6, 24, 24_000, 2_000)
    with pytest.raises(AttributeError):
        setattr(budget, "total", 2000)


@pytest.mark.parametrize(
    ("meeting_ids", "whole_library"),
    [
        pytest.param((), False, id="missing-scope"),
        pytest.param((_MEETING,), True, id="mixed-scopes"),
    ],
)
def test_question_rejects_missing_or_ambiguous_scope(
    meeting_ids: tuple[str, ...], whole_library: bool
) -> None:
    with pytest.raises(InvalidInput) as error:
        QuestionRequest(
            question="What was agreed?", meeting_ids=meeting_ids, whole_library=whole_library
        )
    assert error.value.reason == "invalid_input"


def test_question_defaults_to_local_matching_in_selected_meetings() -> None:
    request = QuestionRequest(question="What was agreed?", meeting_ids=(_MEETING, _OTHER_MEETING))
    assert request.question == "What was agreed?"
    assert request.meeting_ids == (_MEETING, _OTHER_MEETING)
    assert request.whole_library is False
    assert request.mode == "local"
    assert request.purpose == "match"
    with pytest.raises(AttributeError):
        setattr(request, "whole_library", True)


def test_question_accepts_express_whole_library_and_provider_mode() -> None:
    request = QuestionRequest(question="What was agreed?", whole_library=True, mode="provider")
    assert request.meeting_ids == ()
    assert request.whole_library is True
    assert request.mode == "provider"


def test_extraction_keeps_one_meeting_and_explicit_kinds() -> None:
    request = ExtractionRequest(meeting_id=_MEETING, kinds=("decision", "blocker"))
    assert request.meeting_id == _MEETING
    assert set(request.kinds) == {"decision", "blocker"}
    assert request.mode == "local"
    with pytest.raises(AttributeError):
        setattr(request, "meeting_id", _OTHER_MEETING)


def test_default_extraction_has_only_the_documented_candidate_kinds() -> None:
    request = ExtractionRequest(meeting_id=_MEETING)
    assert set(request.kinds) == {"decision", "task", "blocker"}


def test_extraction_rejects_an_unknown_candidate_kind() -> None:
    with pytest.raises(InvalidInput):
        ExtractionRequest(meeting_id=_MEETING, kinds=("task", "fact"))


@pytest.mark.parametrize("start_ms", [None, 0], ids=["untimed", "measured-zero"])
def test_evidence_preserves_original_text_and_timing_for_a_passive_citation(
    start_ms: int | None,
) -> None:
    block = EvidenceBlock(
        id="E1",
        meeting_id=_MEETING,
        revision_id=_REVISION,
        segment_id=_SEGMENT,
        text=_SOURCE_TEXT,
        start_ms=start_ms,
        end_ms=None,
    )

    assert block.id == "E1"
    assert block.text == _SOURCE_TEXT
    assert block.start_ms == start_ms
    assert block.end_ms is None
    with pytest.raises(AttributeError):
        setattr(block, "text", "rewritten evidence")

    citation = Citation(
        block.meeting_id,
        block.revision_id,
        block.segment_id,
        quote="\U0001f600 café",
        start_char=1,
        end_char=7,
    )
    assert (citation.meeting_id, citation.revision_id, citation.segment_id) == (
        _MEETING,
        _REVISION,
        _SEGMENT,
    )
    assert block.text[citation.start_char : citation.end_char] == citation.quote
