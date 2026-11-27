"""Synthetic provider JSON values and source anchors for the M4 verifier.

E1 and E2 denote sent blocks from distinct meetings. The test consumer binds
the meeting, revision and segment identities to its own memory snapshot.
Payloads describe mock responses. The source anchors describe original text.
The spec names the uncertainty field without fixing its JSON type. False is
the fixture value for these claims, and the eventual schema owns the type.
"""

SOURCE_TEXT = {
    "E1": "🌱 Nota: We approved  the “Café Atlas” sample. Next item.",
    "E2": "📌 Registro: El taller  confirmó el montaje. Sin más cambios.",
}

# Start inclusive, end exclusive, in original Unicode code points.
EXPECTED_SOURCE_CITATIONS = {
    "E1": {
        "quote": "We approved  the “Café Atlas” sample.",
        "start_char": 8,
        "end_char": 45,
    },
    "E2": {
        "quote": "El taller  confirmó el montaje.",
        "start_char": 12,
        "end_char": 43,
    },
}

_CROSS_MEETING_CLAIM = {
    "kind": "synthesis",
    "text": "The sample was approved, and the workshop confirmed assembly.",
    "evidence_ids": ["E1", "E2"],
    "quotes": ['WE APPROVED THE "CAFÉ ATLAS" SAMPLE.', "El taller confirmó el montaje."],
    "uncertainty": False,
}

CROSS_MEETING_RESPONSE = {"claims": [_CROSS_MEETING_CLAIM], "extractions": []}

# Every variant changes one field and keeps the other source's contribution intact.
REJECTED_RESPONSES = {
    "unknown_evidence_id": {
        "claims": [{**_CROSS_MEETING_CLAIM, "evidence_ids": ["E99", "E2"]}],
        "extractions": [],
    },
    "empty_quote": {
        "claims": [{**_CROSS_MEETING_CLAIM, "quotes": ["", "El taller confirmó el montaje."]}],
        "extractions": [],
    },
    "quote_from_wrong_block": {
        "claims": [
            {
                **_CROSS_MEETING_CLAIM,
                "quotes": ["Sin más cambios.", "El taller confirmó el montaje."],
            },
        ],
        "extractions": [],
    },
    "model_supplied_timestamp": {
        "claims": [{**_CROSS_MEETING_CLAIM, "start_ms": 42}],
        "extractions": [],
    },
    # A bounded oversized sample. The schema will specify the exact field limit.
    "oversized_claim_text": {
        "claims": [{**_CROSS_MEETING_CLAIM, "text": "synthetic text " * 4_096}],
        "extractions": [],
    },
}

EMPTY_RESPONSE = {"claims": [], "extractions": []}

# These are independent sent sets. Spans list the original source occurrences
# without prescribing how a verifier selects among repeated matches.
UNICODE_QUOTE_CASES = {
    "full_case_expansion": {
        "source_text": {"E1": "🌱 Maß. Ende."},
        "response": {
            "claims": [
                {
                    "kind": "fact",
                    "text": "The note reads Maß.",
                    "evidence_ids": ["E1"],
                    "quotes": ["MASS"],
                    "uncertainty": False,
                },
            ],
            "extractions": [],
        },
        "original_spans": ((2, 5, "Maß"),),
    },
    "partial_case_expansion": {
        "source_text": {"E1": "ß"},
        "response": {
            "claims": [
                {
                    "kind": "fact",
                    "text": "The note contains the letter s.",
                    "evidence_ids": ["E1"],
                    "quotes": ["s"],
                    "uncertainty": False,
                },
            ],
            "extractions": [],
        },
        "original_spans": (),
    },
    "repeated_literal_quote": {
        "source_text": {"E1": "🌱 Maß, Maß."},
        "response": {
            "claims": [
                {
                    "kind": "fact",
                    "text": "The note repeats Maß.",
                    "evidence_ids": ["E1"],
                    "quotes": ["Maß"],
                    "uncertainty": False,
                },
            ],
            "extractions": [],
        },
        "original_spans": ((2, 5, "Maß"), (7, 10, "Maß")),
    },
}
