from __future__ import annotations

from datetime import datetime

from here.output.markdown import format_duration, render_transcript_markdown
from here.output.metadata import SessionMetadata, SourceMetadata


def test_format_duration_uses_compact_human_units() -> None:
    assert format_duration(4.4) == "4s"
    assert format_duration(125) == "2m 5s"
    assert format_duration(3725) == "1h 2m 5s"


def test_render_transcript_markdown_has_stable_sections() -> None:
    metadata = SessionMetadata(
        session_id="20260522_103000",
        started_at=datetime(2026, 5, 22, 10, 25, 0),
        completed_at=datetime(2026, 5, 22, 10, 30, 0),
        duration_seconds=300,
        sources=[
            SourceMetadata(
                label="microphone",
                device_name="Asterisk Nova",
                sample_rate=48000,
                channels=2,
                frames=14_400_000,
                duration_seconds=300,
            )
        ],
        transcription_model="gpt-4o-transcribe-diarize",
        cleanup_model="gpt-4.1-mini",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=True,
        live_pipeline_used=True,
        fallback_used=False,
        output_files=["transcript.txt", "transcript.md", "session.json"],
    )

    markdown = render_transcript_markdown(metadata, "Speaker 1: hola")

    assert markdown == (
        "# Recording 2026-05-22 10:30\n"
        "\n"
        "## Details\n"
        "\n"
        "- Session ID: `20260522_103000`\n"
        "- Started: 2026-05-22 10:25:00\n"
        "- Completed: 2026-05-22 10:30:00\n"
        "- Duration: 5m 0s\n"
        "\n"
        "## Sources\n"
        "\n"
        "- microphone (Asterisk Nova): 2 channel(s), 48000 Hz, 5m 0s\n"
        "\n"
        "## Processing\n"
        "\n"
        "- Transcription model: gpt-4o-transcribe-diarize\n"
        "- Cleanup model: gpt-4.1-mini\n"
        "- Cleanup enabled: no\n"
        "- Alternate model used: no\n"
        "- Live pipeline attempted: yes\n"
        "- Live pipeline used: yes\n"
        "- Fallback used: no\n"
        "\n"
        "## Transcript\n"
        "\n"
        "Speaker 1: hola\n"
    )
