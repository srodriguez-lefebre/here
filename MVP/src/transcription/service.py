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
    TranscriptionResult,
    build_client,
    finalize_transcription,
    resolve_transcription_models,
    transcribe_audio_file,
)


def transcribe(
    audio_path: Path,
    *,
    transcription_model: str | None = None,
    cleanup_model: str | None = None,
    skip_cleanup: bool = False,
    prompt: str | None = None,
) -> TranscriptionResult:
    """
    Transcribe a WAV file and optionally clean up the text.

    Args:
        audio_path: Path to the WAV audio file.
        transcription_model: Optional override for the transcription model.
        cleanup_model: Optional override for the cleanup model.
        skip_cleanup: When True, return the raw transcript without cleanup.
        prompt: Optional transcript context for the current audio segment.

    Returns:
        Raw and final transcription text.
    """
    client = build_client()
    resolved_transcription_model, resolved_cleanup_model, should_cleanup = resolve_transcription_models(
        transcription_model=transcription_model,
        cleanup_model=cleanup_model,
        skip_cleanup=skip_cleanup,
    )

    logger.info("Transcribing {path}...", path=audio_path.name)

    try:
        raw_text = transcribe_audio_file(
            client,
            audio_path,
            resolved_transcription_model,
            prompt=prompt,
        )
    except Exception as exc:
        logger.error("Transcription failed: {exc}", exc=exc)
        raise RuntimeError("Transcription failed") from exc

    return finalize_transcription(
        client=client,
        raw_text=raw_text,
        cleanup_model=resolved_cleanup_model,
        should_cleanup=should_cleanup,
    )


def transcribe_recording_session(
    session: RecordingSession,
    *,
    transcription_model: str | None = None,
    cleanup_model: str | None = None,
    skip_cleanup: bool = False,
    chunking_config: ChunkingConfig | None = None,
) -> TranscriptionResult:
    """Transcribe a recorded session, chunking audio as needed for long recordings."""
    resolved_config = chunking_config or ChunkingConfig()

    client = build_client()
    resolved_transcription_model, resolved_cleanup_model, should_cleanup = resolve_transcription_models(
        transcription_model=transcription_model,
        cleanup_model=cleanup_model,
        skip_cleanup=skip_cleanup,
    )

    merged_raw_text = ""

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
                prompt = build_chunk_prompt(merged_raw_text, resolved_config.prompt_tail_words)

                logger.info("Transcribing chunk {index}/{total}...", index=index, total=len(windows))
                try:
                    chunk_text = transcribe_audio_file(
                        client,
                        chunk_path,
                        resolved_transcription_model,
                        prompt=prompt,
                    )
                finally:
                    chunk_path.unlink(missing_ok=True)

                merged_raw_text = merge_transcript_parts(
                    [merged_raw_text, chunk_text],
                    resolved_config,
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
