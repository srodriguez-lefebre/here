from __future__ import annotations

import queue
import shutil
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from here.audio.chunking import build_chunk_prompt, merge_transcript_parts
from here.audio.mix import (
    downmix_to_mono,
    fit_audio_to_target_frames,
    materialize_normalized_session,
    mix_audio_blocks,
    resample_mono_audio,
)
from here.audio.models import ChunkingConfig
from here.audio.silence_boundaries import choose_silence_cut_index
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.client import (
    TranscriptionResult,
    build_client,
    coerce_audio_transcription,
    finalize_transcription,
    format_transcript_segments,
    model_supports_prompt,
    resolve_transcription_models,
    transcribe_audio_file,
)
from here.transcription.segments import SegmentTimeline, TranscriptSegment, shift_segments

PCM_SUBTYPE = "PCM_16"


@dataclass(slots=True)
class CapturedAudioBlock:
    label: str
    data: np.ndarray
    sample_rate: int
    channels: int


@dataclass(slots=True)
class SourceSegment:
    index: int
    source: RecordedAudioSource
    start_offset_seconds: float


@dataclass(slots=True)
class LiveChunkJob:
    index: int
    session: RecordingSession
    start_offset_seconds: float


def _sanitize_label(label: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in label.casefold())
    return sanitized.strip("_") or "source"


def _frame_count(audio: np.ndarray) -> int:
    if audio.ndim == 0:
        return 1
    return int(audio.shape[0])


def _tail_frames(audio: np.ndarray, count: int) -> np.ndarray:
    frame_count = _frame_count(audio)
    if count <= 0:
        shape = (0, *audio.shape[1:]) if audio.ndim > 1 else (0,)
        return np.zeros(shape, dtype=audio.dtype)
    if frame_count <= count:
        return audio.copy()
    return audio[-count:].copy()


class BufferedSourceState:
    def __init__(
        self,
        *,
        label: str,
        sample_rate: int,
        channels: int,
        working_dir: Path,
        config: ChunkingConfig,
    ) -> None:
        self.label = label
        self.sample_rate = sample_rate
        self.channels = channels
        self.working_dir = working_dir
        self.config = config
        self._label_slug = _sanitize_label(label)
        self._segment_index = 1
        self._parts: list[np.ndarray] = []
        self._cached_audio: np.ndarray | None = None
        self._frame_total = 0
        self._seed_frames = 0
        self._segment_start_seconds = 0.0
        self._overlap_frames = max(0, int(round(config.overlap_seconds * sample_rate)))

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self._frame_total / self.sample_rate

    def append_block(self, block: np.ndarray) -> None:
        copied = block.copy()
        self._parts.append(copied)
        self._cached_audio = None
        self._frame_total += _frame_count(copied)

    def _materialize_audio(self) -> np.ndarray:
        if self._cached_audio is None:
            if not self._parts:
                shape = (0, self.channels) if self.channels > 1 else (0,)
                self._cached_audio = np.zeros(shape, dtype=np.float32)
            elif len(self._parts) == 1:
                self._cached_audio = self._parts[0]
            else:
                self._cached_audio = np.concatenate(self._parts, axis=0)
            self._parts = [self._cached_audio] if _frame_count(self._cached_audio) > 0 else []
        return self._cached_audio

    def resampled_mono(self, target_sample_rate: int, target_frames: int) -> np.ndarray:
        audio = self._materialize_audio()
        mono_audio = downmix_to_mono(audio)
        resampled = resample_mono_audio(mono_audio, self.sample_rate, target_sample_rate)
        return fit_audio_to_target_frames(resampled, target_frames)

    def _segment_path(self) -> Path:
        return self.working_dir / f"{self._label_slug}_segment_{self._segment_index:04d}.wav"

    def _write_segment(self, audio: np.ndarray) -> Path:
        path = self._segment_path()
        with sf.SoundFile(
            path,
            mode="w",
            samplerate=self.sample_rate,
            channels=self.channels,
            subtype=PCM_SUBTYPE,
        ) as writer:
            writer.write(audio)
        return path

    def _segment_from_audio(self, audio: np.ndarray) -> SourceSegment | None:
        total_frames = _frame_count(audio)
        if total_frames <= 0:
            return None
        if total_frames <= self._seed_frames and self._segment_index > 1:
            return None

        path = self._write_segment(audio)
        source = RecordedAudioSource(
            path=path,
            sample_rate=self.sample_rate,
            channels=self.channels,
            frames=total_frames,
            label=self.label,
        )
        segment = SourceSegment(
            index=self._segment_index,
            source=source,
            start_offset_seconds=self._segment_start_seconds,
        )
        self._segment_index += 1
        return segment

    def finalize_chunk(self, cut_seconds: float) -> SourceSegment | None:
        audio = self._materialize_audio()
        total_frames = _frame_count(audio)
        if total_frames <= 0:
            return None

        cut_frames = int(round(cut_seconds * self.sample_rate))
        cut_frames = max(1, min(total_frames, cut_frames))
        segment_audio = audio[:cut_frames].copy()
        segment = self._segment_from_audio(segment_audio)

        overlap_seed = _tail_frames(segment_audio, self._overlap_frames)
        remainder = audio[cut_frames:].copy()
        if _frame_count(overlap_seed) > 0 and _frame_count(remainder) > 0:
            next_audio = np.concatenate([overlap_seed, remainder], axis=0)
        elif _frame_count(overlap_seed) > 0:
            next_audio = overlap_seed
        else:
            next_audio = remainder

        overlap_seconds = (_frame_count(overlap_seed) / self.sample_rate) if self.sample_rate > 0 else 0.0
        self._segment_start_seconds += (cut_frames / self.sample_rate) - overlap_seconds
        self._seed_frames = _frame_count(overlap_seed)
        self._cached_audio = next_audio
        self._parts = [next_audio] if _frame_count(next_audio) > 0 else []
        self._frame_total = _frame_count(next_audio)
        return segment

    def finish(self) -> SourceSegment | None:
        audio = self._materialize_audio()
        segment = self._segment_from_audio(audio)
        self._parts = []
        self._cached_audio = None
        self._frame_total = 0
        self._seed_frames = 0
        return segment


class LiveTranscriptionController:
    def __init__(
        self,
        *,
        expected_source_count: int,
        chunking_config: ChunkingConfig | None = None,
        transcription_model: str | None = None,
        cleanup_model: str | None = None,
        skip_cleanup: bool = False,
        use_alt_transcription_model: bool = False,
    ) -> None:
        self.expected_source_count = expected_source_count
        self.config = chunking_config or ChunkingConfig()
        self.working_dir = Path(tempfile.mkdtemp(prefix="here_live_"))
        self._capture_queue: queue.SimpleQueue[CapturedAudioBlock | None] = queue.SimpleQueue()
        self._chunk_queue: queue.SimpleQueue[LiveChunkJob | None] = queue.SimpleQueue()
        self._source_states: dict[str, BufferedSourceState] = {}
        self._capture_closed = False
        self._capture_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._error: Exception | None = None
        self._result: TranscriptionResult | None = None
        self._client = build_client()
        self._resolved_transcription_model, self._resolved_cleanup_model, self._should_cleanup = (
            resolve_transcription_models(
                transcription_model=transcription_model,
                cleanup_model=cleanup_model,
                skip_cleanup=skip_cleanup,
                use_alt_transcription_model=use_alt_transcription_model,
            )
        )

        logger.info(
            "Starting live transcription pipeline for {sources} source(s). Rolling chunks every {seconds}s with {overlap}s overlap.",
            sources=self.expected_source_count,
            seconds=self.config.live_chunk_seconds,
            overlap=self.config.overlap_seconds,
        )

        self._chunker_thread = threading.Thread(target=self._chunker_worker, daemon=True)
        self._transcriber_thread = threading.Thread(target=self._transcriber_worker, daemon=True)
        self._chunker_thread.start()
        self._transcriber_thread.start()

    def submit_block(
        self,
        label: str,
        data: np.ndarray,
        sample_rate: int,
        channels: int,
    ) -> None:
        if self._capture_closed:
            return
        self._capture_queue.put(
            CapturedAudioBlock(
                label=label,
                data=data.copy(),
                sample_rate=sample_rate,
                channels=channels,
            )
        )

    def _source_state_for(self, block: CapturedAudioBlock) -> BufferedSourceState:
        state = self._source_states.get(block.label)
        if state is None:
            logger.info(
                "Live capture connected for {label} ({sample_rate} Hz, {channels} channel(s)).",
                label=block.label,
                sample_rate=block.sample_rate,
                channels=block.channels,
            )
            state = BufferedSourceState(
                label=block.label,
                sample_rate=block.sample_rate,
                channels=block.channels,
                working_dir=self.working_dir,
                config=self.config,
            )
            self._source_states[block.label] = state
        return state

    def _ordered_states(self) -> list[BufferedSourceState]:
        return list(self._source_states.values())

    def _build_boundary_proxy(self) -> np.ndarray | None:
        states = self._ordered_states()
        if len(states) < self.expected_source_count:
            return None

        available_seconds = min(state.duration_seconds for state in states)
        if available_seconds <= 0.0:
            return None

        target_frames = int(round(available_seconds * self.config.target_sample_rate))
        if target_frames <= 0:
            return None

        proxy_blocks = [
            state.resampled_mono(self.config.target_sample_rate, target_frames)
            for state in states
        ]
        return mix_audio_blocks(proxy_blocks, target_frames)

    def _live_cut_ready(self) -> bool:
        states = self._ordered_states()
        if len(states) < self.expected_source_count:
            return False

        required_seconds = self.config.live_chunk_seconds + self.config.silence_search_seconds
        return min(state.duration_seconds for state in states) >= required_seconds

    def _enqueue_live_job(self, segments: list[SourceSegment]) -> None:
        if not segments:
            return

        index = segments[0].index
        start_offset_seconds = segments[0].start_offset_seconds
        logger.info(
            "Queued live chunk {index} with {sources} source(s) for transcription at {offset:.2f}s.",
            index=index,
            sources=len(segments),
            offset=start_offset_seconds,
        )
        self._chunk_queue.put(
            LiveChunkJob(
                index=index,
                session=RecordingSession(sources=[segment.source for segment in segments]),
                start_offset_seconds=start_offset_seconds,
            )
        )

    def _maybe_enqueue_live_chunks(self) -> None:
        while self._live_cut_ready():
            proxy_audio = self._build_boundary_proxy()
            if proxy_audio is None:
                return

            target_index = int(round(self.config.live_chunk_seconds * self.config.target_sample_rate))
            overlap_frames = int(round(self.config.overlap_seconds * self.config.target_sample_rate))
            min_index = max(1, target_index - overlap_frames)
            cut_index = choose_silence_cut_index(
                proxy_audio,
                target_index,
                sample_rate=self.config.target_sample_rate,
                min_index=min_index,
                max_index=int(proxy_audio.shape[0]),
                search_radius_seconds=self.config.silence_search_seconds,
                analysis_window_seconds=self.config.silence_window_seconds,
                candidate_step_seconds=self.config.silence_candidate_step_seconds,
                silence_threshold=self.config.silence_threshold,
            )
            cut_seconds = cut_index / self.config.target_sample_rate
            logger.info(
                "Selected live cut at {seconds:.2f}s for chunk target {target:.2f}s.",
                seconds=cut_seconds,
                target=self.config.live_chunk_seconds,
            )

            segments = [
                segment
                for state in self._ordered_states()
                for segment in [state.finalize_chunk(cut_seconds)]
                if segment is not None
            ]
            if len(segments) < self.expected_source_count:
                raise RuntimeError("Live chunking produced fewer source segments than expected.")
            self._enqueue_live_job(segments)

    def _set_error(self, exc: Exception) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _chunker_worker(self) -> None:
        try:
            while True:
                block = self._capture_queue.get()
                if block is None:
                    logger.info("Live capture finished. Flushing pending audio into final chunk(s).")
                    break
                state = self._source_state_for(block)
                state.append_block(block.data)
                self._maybe_enqueue_live_chunks()

            segments = [
                segment
                for state in self._ordered_states()
                for segment in [state.finish()]
                if segment is not None
            ]
            if len(segments) == self.expected_source_count:
                self._enqueue_live_job(segments)
            elif segments:
                logger.warning(
                    "Discarding partial final live chunk because only {count}/{expected} source(s) produced audio.",
                    count=len(segments),
                    expected=self.expected_source_count,
                )
        except Exception as exc:
            logger.error("Live chunk generation failed: {exc}", exc=exc)
            self._set_error(exc)
        finally:
            self._chunk_queue.put(None)

    def _cleanup_chunk_job(self, job: LiveChunkJob) -> None:
        try:
            job.session.cleanup()
        except Exception:
            pass

    def _merge_chunk_transcription(
        self,
        merged_raw_text: str,
        timeline: SegmentTimeline | None,
        *,
        chunk_text: str,
        chunk_segments: Sequence[TranscriptSegment],
        offset_seconds: float,
    ) -> tuple[str, SegmentTimeline | None]:
        if timeline is not None and chunk_segments:
            timeline.extend(shift_segments(chunk_segments, offset_seconds))
            return format_transcript_segments(timeline.to_list()), timeline

        if timeline is not None and not chunk_segments:
            timeline = None

        merged_raw_text = merge_transcript_parts(
            [merged_raw_text, chunk_text],
            self.config,
        )
        return merged_raw_text, timeline

    def _transcriber_worker(self) -> None:
        merged_raw_text = ""
        segment_timeline: SegmentTimeline | None = SegmentTimeline()
        failed = False
        try:
            while True:
                job = self._chunk_queue.get()
                if job is None:
                    break

                if failed:
                    self._cleanup_chunk_job(job)
                    continue

                chunk_working_dir = self.working_dir / f"chunk_{job.index:04d}"
                prompt = None
                if model_supports_prompt(self._resolved_transcription_model):
                    prompt = build_chunk_prompt(merged_raw_text, self.config.prompt_tail_words)

                normalized_session: RecordingSession | None = None
                try:
                    logger.info("Normalizing live chunk {index}...", index=job.index)
                    normalized_session = materialize_normalized_session(
                        job.session,
                        chunk_working_dir,
                        self.config,
                    )
                    normalized_path = normalized_session.sources[0].path
                    logger.info("Transcribing live chunk {index}...", index=job.index)
                    transcription = coerce_audio_transcription(
                        transcribe_audio_file(
                            self._client,
                            normalized_path,
                            self._resolved_transcription_model,
                            prompt=prompt,
                        )
                    )
                    merged_raw_text, segment_timeline = self._merge_chunk_transcription(
                        merged_raw_text,
                        segment_timeline,
                        chunk_text=transcription.text,
                        chunk_segments=transcription.segments,
                        offset_seconds=job.start_offset_seconds,
                    )
                    logger.info("Live chunk {index} transcribed.", index=job.index)
                except Exception as exc:
                    logger.error("Live chunk transcription failed: {exc}", exc=exc)
                    self._set_error(exc)
                    failed = True
                finally:
                    try:
                        if normalized_session is not None:
                            normalized_session.cleanup()
                    except Exception:
                        pass
                    self._cleanup_chunk_job(job)

            if not failed:
                logger.info("Finalizing live transcription.")
                self._result = finalize_transcription(
                    client=self._client,
                    raw_text=merged_raw_text,
                    cleanup_model=self._resolved_cleanup_model,
                    should_cleanup=self._should_cleanup,
                )
                logger.success("Live transcription complete.")
        except Exception as exc:
            logger.error("Live transcription finalization failed: {exc}", exc=exc)
            self._set_error(exc)

    def complete(self) -> TranscriptionResult:
        with self._capture_lock:
            if not self._capture_closed:
                self._capture_closed = True
                self._capture_queue.put(None)

        logger.info("Waiting for live chunking and transcription workers to finish.")
        self._chunker_thread.join()
        self._transcriber_thread.join()

        if self._error is not None:
            raise RuntimeError("Live transcription failed") from self._error
        if self._result is None:
            raise RuntimeError("Live transcription did not produce a result")
        return self._result

    def abort(self) -> None:
        logger.warning("Aborting live transcription pipeline.")
        with self._capture_lock:
            if not self._capture_closed:
                self._capture_closed = True
                self._capture_queue.put(None)
        if self._chunker_thread.is_alive():
            self._chunker_thread.join(timeout=2)
        if self._transcriber_thread.is_alive():
            self._transcriber_thread.join(timeout=2)

    def cleanup(self) -> None:
        logger.info("Cleaning live transcription workspace.")
        shutil.rmtree(self.working_dir, ignore_errors=True)
