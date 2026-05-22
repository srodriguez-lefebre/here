from __future__ import annotations

from datetime import datetime

from here.output.metadata import SessionMetadata


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S %z").strip()


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def render_transcript_markdown(metadata: SessionMetadata, transcript_text: str) -> str:
    title_time = metadata.completed_at.strftime("%Y-%m-%d %H:%M")
    source_lines = [
        (
            f"- {source.label}: {source.channels} channel(s), "
            f"{source.sample_rate} Hz, {format_duration(source.duration_seconds)}"
        )
        for source in metadata.sources
    ] or ["- none"]

    lines = [
        f"# Recording {title_time}",
        "",
        "## Details",
        "",
        f"- Session ID: `{metadata.session_id}`",
        f"- Started: {_format_timestamp(metadata.started_at)}",
        f"- Completed: {_format_timestamp(metadata.completed_at)}",
        f"- Duration: {format_duration(metadata.duration_seconds)}",
        "",
        "## Sources",
        "",
        *source_lines,
        "",
        "## Processing",
        "",
        f"- Transcription model: {metadata.transcription_model}",
        f"- Cleanup model: {metadata.cleanup_model}",
        f"- Cleanup enabled: {_yes_no(metadata.cleanup_enabled)}",
        f"- Alternate model used: {_yes_no(metadata.alt_model_used)}",
        f"- Live pipeline attempted: {_yes_no(metadata.live_pipeline_attempted)}",
        f"- Live pipeline used: {_yes_no(metadata.live_pipeline_used)}",
        f"- Fallback used: {_yes_no(metadata.fallback_used)}",
        "",
        "## Transcript",
        "",
        transcript_text.strip(),
        "",
    ]
    return "\n".join(lines)
