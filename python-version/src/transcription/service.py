from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger

from here.audio.chunking import (
    build_chunk_prompt,
    merge_transcript_parts,
    plan_chunk_windows,
    render_chunk_window,
)
from here.audio.mix import materialize_normalized_session
from here.audio.models import ChunkingConfig
from here.recording.models import RecordingSession
from here.transcription.client import (
    AudioTranscription,
    TranscriptionResult,
    build_client,
    coerce_audio_transcription,
    finalize_transcription,
    format_transcript_segments,
    model_supports_prompt,
    resolve_transcription_models,
    transcribe_audio_file,
)
from here.transcription.segments import SegmentTimeline, shift_segments


def _merge_chunk_transcription(
    merged_raw_text: str,
    timeline: SegmentTimeline | None,
    transcription: AudioTranscription,
    *,
    offset_seconds: float,
    config: ChunkingConfig,
) -> tuple[str, SegmentTimeline | None]:
    if timeline is not None and transcription.segments:
        timeline.extend(shift_segments(transcription.segments, offset_seconds))
        return format_transcript_segments(timeline.to_list()), timeline

    if timeline is not None and not transcription.segments:
        timeline = None

    merged_raw_text = merge_transcript_parts(
        [merged_raw_text, transcription.text],
        config,
    )
    return merged_raw_text, timeline


def transcribe(
    audio_path: Path,
    *,
    transcription_model: str | None = None,
    cleanup_model: str | None = None,
    skip_cleanup: bool = False,
    use_alt_transcription_model: bool = False,
    prompt: str | None = None,
) -> TranscriptionResult:
    """
    Transcribe a WAV file and optionally clean up the text.

    Args:
        audio_path: Path to the WAV audio file.
        transcription_model: Optional override for the transcription model.
        cleanup_model: Optional override for the cleanup model.
        skip_cleanup: When True, return the raw transcript without cleanup.
        use_alt_transcription_model: When True, use the configured alternate transcription model.
        prompt: Optional transcript context for the current audio segment.

    Returns:
        Raw and final transcription text.
    """
    client = build_client()
    resolved_transcription_model, resolved_cleanup_model, should_cleanup = resolve_transcription_models(
        transcription_model=transcription_model,
        cleanup_model=cleanup_model,
        skip_cleanup=skip_cleanup,
        use_alt_transcription_model=use_alt_transcription_model,
    )

    logger.info("Transcribing {path}...", path=audio_path.name)

    try:
        transcription = coerce_audio_transcription(
            transcribe_audio_file(
                client,
                audio_path,
                resolved_transcription_model,
                prompt=prompt,
            )
        )
    except Exception as exc:
        logger.error("Transcription failed: {exc}", exc=exc)
        raise RuntimeError("Transcription failed") from exc

    return finalize_transcription(
        client=client,
        raw_text=transcription.text,
        cleanup_model=resolved_cleanup_model,
        should_cleanup=should_cleanup,
    )


def transcribe_recording_session(
    session: RecordingSession,
    *,
    transcription_model: str | None = None,
    cleanup_model: str | None = None,
    skip_cleanup: bool = False,
    use_alt_transcription_model: bool = False,
    chunking_config: ChunkingConfig | None = None,
) -> TranscriptionResult:
    """Transcribe a recorded session, chunking audio as needed for long recordings."""
    resolved_config = chunking_config or ChunkingConfig()

    client = build_client()
    resolved_transcription_model, resolved_cleanup_model, should_cleanup = resolve_transcription_models(
        transcription_model=transcription_model,
        cleanup_model=cleanup_model,
        skip_cleanup=skip_cleanup,
        use_alt_transcription_model=use_alt_transcription_model,
    )

    merged_raw_text = ""
    segment_timeline: SegmentTimeline | None = SegmentTimeline()

    with TemporaryDirectory(prefix="here_chunks_") as temp_dir:
        working_dir = Path(temp_dir)
        try:
            normalized_session = materialize_normalized_session(session, working_dir, resolved_config)
            windows = plan_chunk_windows(normalized_session, resolved_config)
            if not windows:
                raise RuntimeError("No normalized audio was available to transcribe.")

            logger.info(
                "Normalized recording into {chunks} chunk(s) for transcription.",
                chunks=len(windows),
            )

            for index, window in enumerate(windows, start=1):
                logger.info("Rendering chunk {index}/{total}...", index=index, total=len(windows))
                chunk_path = render_chunk_window(normalized_session, window, working_dir, resolved_config)
                prompt = None
                if model_supports_prompt(resolved_transcription_model):
                    prompt = build_chunk_prompt(merged_raw_text, resolved_config.prompt_tail_words)

                logger.info("Transcribing chunk {index}/{total}...", index=index, total=len(windows))
                try:
                    chunk_transcription = coerce_audio_transcription(
                        transcribe_audio_file(
                            client,
                            chunk_path,
                            resolved_transcription_model,
                            prompt=prompt,
                        )
                    )
                finally:
                    chunk_path.unlink(missing_ok=True)

                merged_raw_text, segment_timeline = _merge_chunk_transcription(
                    merged_raw_text,
                    segment_timeline,
                    chunk_transcription,
                    offset_seconds=window.start_frame / resolved_config.target_sample_rate,
                    config=resolved_config,
                )
        except Exception as exc:
            logger.error("Chunked transcription failed: {exc}", exc=exc)
            raise RuntimeError("Transcription failed") from exc

    return finalize_transcription(
        client=client,
        raw_text=merged_raw_text,
        cleanup_model=resolved_cleanup_model,
        should_cleanup=should_cleanup,
    )
