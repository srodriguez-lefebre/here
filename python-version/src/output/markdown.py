from __future__ import annotations

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


def render_transcript_markdown(metadata: SessionMetadata, transcript_text: str) -> str:
    source_labels = ", ".join(source.label for source in metadata.sources) or "none"
    completed_at = metadata.completed_at.strftime("%Y-%m-%d %H:%M:%S %z").strip()

    lines = [
        f"# Recording {metadata.session_id}",
        "",
        "## Details",
        "",
        f"- Completed: {completed_at}",
        f"- Duration: {format_duration(metadata.duration_seconds)}",
        f"- Sources: {source_labels}",
        f"- Transcription model: {metadata.transcription_model}",
        f"- Cleanup enabled: {'yes' if metadata.cleanup_enabled else 'no'}",
        f"- Live pipeline: {'yes' if metadata.live_pipeline_used else 'no'}",
        f"- Fallback used: {'yes' if metadata.fallback_used else 'no'}",
        "",
        "## Transcript",
        "",
        transcript_text.strip(),
        "",
    ]
    return "\n".join(lines)
