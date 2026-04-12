from here.audio.text_merge import (
    TextMergeConfig,
    dedupe_overlap,
    merge_transcript_pair,
    merge_transcript_parts,
    trim_overlapping_prefix,
)


def test_trim_overlapping_prefix_removes_exact_duplicate_words() -> None:
    previous = "alpha beta gamma delta"
    next_text = "alpha beta gamma delta epsilon"

    assert trim_overlapping_prefix(previous, next_text) == "epsilon"
    assert merge_transcript_pair(previous, next_text) == "alpha beta gamma delta\nepsilon"


def test_trim_overlapping_prefix_ignores_punctuation_and_casing() -> None:
    previous = "Hello, world! This is fine."
    next_text = "hello world this is fine. And more."

    assert dedupe_overlap(previous, next_text) == "And more."


def test_trim_overlapping_prefix_handles_small_word_variations() -> None:
    config = TextMergeConfig(min_similarity=0.8)
    previous = "We need ten minutes for the demo"
    next_text = "we need 10 minutes for the demo and then notes"

    assert trim_overlapping_prefix(previous, next_text, config) == "and then notes"


def test_trim_overlapping_prefix_prefers_line_overlap_for_speaker_labels() -> None:
    previous = "Speaker 1: We should review the plan.\nSpeaker 2: Absolutely."
    next_text = "speaker 1: we should review the plan!\nSpeaker 2: Absolutely.\nSpeaker 2: I added logs."

    assert trim_overlapping_prefix(previous, next_text) == "Speaker 2: I added logs."


def test_trim_overlapping_prefix_leaves_distinct_text_untouched() -> None:
    previous = "Alpha beta gamma"
    next_text = "Completely different sentence."

    assert trim_overlapping_prefix(previous, next_text) == "Completely different sentence."
    assert merge_transcript_pair(previous, next_text) == "Alpha beta gamma\nCompletely different sentence."


def test_merge_transcript_parts_skips_empty_parts_and_dedupes_across_chunks() -> None:
    parts = [
        "",
        "Speaker 1: We should review the plan.\nSpeaker 2: Absolutely.",
        "speaker 1: we should review the plan!\nSpeaker 2: Absolutely.\nSpeaker 2: I added logs.",
        "   ",
    ]

    assert merge_transcript_parts(parts) == "Speaker 1: We should review the plan.\nSpeaker 2: Absolutely.\nSpeaker 2: I added logs."
