"""Deterministic synthetic input for M4 retrieval budget tests.

The tuple is in fixture order. Its first meeting is outside the default test
selection, and its first two meetings together dominate the match volume.
Passages are in transcript order. The consumer binds them to the accepted memory
APIs and obtains revision and segment IDs from that ingestion path.
"""

from dataclasses import dataclass

OR_TERMS = ("cobalto", "azafrán")
TITLE_ONLY_TERM = "jacarandá"

MEETING_COUNT = 50
MATCH_COUNT = 2_000
SELECTED_MATCH_COUNT = 400
SPARSE_MATCH_COUNT = 100
# The first sparse meeting starts with two matches bracketing the same non-hit.
LEFT_MATCH_ORDINAL = 0
SHARED_NEIGHBOR_ORDINAL = 1
RIGHT_MATCH_ORDINAL = 2


@dataclass(frozen=True)
class IntelligenceMeeting:
    """Synthetic meeting title and original source passages."""

    meeting_id: str
    title: str
    passages: tuple[str, ...]

    @property
    def transcript_text(self) -> str:
        return "\n".join(self.passages)

    def segment_rows(self) -> list[dict[str, str]]:
        """Return text-only rows in the shape accepted by memory_fixtures.make_session."""
        return [{"text": passage} for passage in self.passages]


def make_intelligence_corpus() -> tuple[IntelligenceMeeting, ...]:
    """Return all 50 meetings in a stable order, with no filesystem access.

    Use meetings[1:] for 400 selected hits and meetings[2:] for 100 sparse
    hits behind both dense meetings. For OR assertions, each dense hit uses
    both terms and each sparse hit uses exactly one. A metadata-only title
    token is available for payload and source-citation checks.

    Sparse matches and their shared neighbor have over 1,000 Unicode
    characters each. Test a tight budget against the first sparse meeting
    to count original characters and charge the shared neighbor only once.
    """
    first_term, second_term = OR_TERMS
    meetings = []
    for number in range(1, MEETING_COUNT + 1):
        if number in (1, 2):
            count = 1_600 if number == 1 else 300
            repeated_terms = f"{first_term} {second_term} " * (12 if number == 1 else 6)
            passages = tuple(f"{repeated_terms}muestra {ordinal:04d}." for ordinal in range(count))
        else:
            # Four meetings contribute three hits each, and 44 contribute two.
            prefix = "Observación sintética sobre el diseño del prototipo. " * 22
            passages = (
                f"{first_term}: {prefix}Registro inicial.",
                "Contexto vecino: " + "Revisión sintética de la lámina de demostración. " * 22,
                f"{second_term}: {prefix}Registro siguiente.",
            )
            if number <= 6:
                passages += (f"{first_term}: {prefix}Registro de cierre.",)
        meetings.append(
            IntelligenceMeeting(
                meeting_id=f"94000000-0000-4000-8000-{number:012d}",
                title=f"Reunión sintética {number:02d} · {TITLE_ONLY_TERM}",
                passages=passages,
            )
        )

    return tuple(meetings)


def make_evidence_window_corpus(
    *, oversized_before: int, oversized_after: int = 2
) -> tuple[IntelligenceMeeting, IntelligenceMeeting]:
    """Return a window fixture and a separate shared-neighbor fixture.

    ``oversized_before`` is the zero-based transcript ordinal of the fitting
    match in the first meeting. Everything before and after it is an oversized
    match, so the caller can place the fitting match on either side of its
    consideration boundary using a deterministic order established by the test.
    Transcript ordinal gives fixture order only. The caller controls search
    ranking and verifies where the fixture entries land in that order.

    At a 96-character budget, the fitting match fits and every other match in
    its meeting exceeds the budget. Each oversized match puts NUL immediately
    after a short matching prefix. The separate meeting brackets a nonmatching
    neighbor containing NUL with two short matches; all three texts together fit
    that character budget. NUL, astral Unicode and combining marks are literal
    source characters in both meetings.
    """
    oversized = "cobalto\x00" + " 🌱cafe\u0301" * 20
    fitting = "cobalto: pieza útil."
    window = IntelligenceMeeting(
        meeting_id="94000000-0000-4000-8000-000000000051",
        title="Ventana de consideración sintética",
        passages=(oversized,) * oversized_before + (fitting,) + (oversized,) * oversized_after,
    )
    shared_neighbor = IntelligenceMeeting(
        meeting_id="94000000-0000-4000-8000-000000000052",
        title="Contexto compartido sintético",
        passages=("cobalto inicial.", "📌 Nota\x00: cafe\u0301 frío.", "azafrán final."),
    )
    return window, shared_neighbor
