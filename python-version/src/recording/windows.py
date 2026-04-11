import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from here.recording.models import RecordedAudioSource, RecordingSession
from here.recording.shared import build_single_source_session, open_temp_soundfile, safe_close_soundfile

WINDOWS_CAPTURE_CHUNK = 1024


def _capture_windows_stream_to_file(
    stream: object,
    *,
    chunk: int,
    sample_rate: int,
    channels: int,
    writer: sf.SoundFile,
    stop_event: threading.Event,
    errors: list[Exception],
    label: str,
    written_frames: list[int],
    start_time: float | None = None,
) -> None:
    chunk_duration = chunk / sample_rate
    silence_chunk = np.zeros((chunk, channels), dtype=np.int16)
    next_deadline = start_time if start_time is not None else time.perf_counter()

    while True:
        now = time.perf_counter()
        if stop_event.is_set() and now < next_deadline:
            break

        if now + chunk_duration < next_deadline:
            time.sleep(min(next_deadline - now, 0.01))
            continue

        while now >= next_deadline + chunk_duration:
            writer.write(silence_chunk)
            written_frames[0] += chunk
            next_deadline += chunk_duration

        frames = silence_chunk.copy()
        try:
            available = 0
            get_read_available = getattr(stream, "get_read_available", None)
            if callable(get_read_available):
                available = max(0, int(get_read_available()))

            if available > 0:
                target_read_frames = min(chunk, available)
                data = stream.read(target_read_frames, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                usable_samples = (audio.size // channels) * channels
                if usable_samples > 0:
                    reshaped = audio[:usable_samples].reshape(-1, channels)
                    frames_written = min(chunk, int(reshaped.shape[0]))
                    frames[:frames_written] = reshaped[:frames_written]

            writer.write(frames)
            written_frames[0] += chunk
        except Exception as exc:
            errors.append(exc)
            logger.error("Failed to read {label}: {exc}", label=label, exc=exc)
            stop_event.set()
            break

        next_deadline += chunk_duration


def _safe_close_stream(stream: object | None) -> None:
    if stream is None:
        return

    try:
        stream.stop_stream()
    except Exception:
        pass

    try:
        stream.close()
    except Exception:
        pass


def _get_default_windows_input_device() -> dict[str, object]:
    import pyaudiowpatch as pyaudio

    p = pyaudio.PyAudio()
    try:
        try:
            return p.get_default_input_device_info()
        except Exception as exc:
            raise RuntimeError(
                "No default microphone input device is available in Windows audio settings."
            ) from exc
    finally:
        p.terminate()


def _get_default_windows_loopback_device() -> dict[str, object]:
    import pyaudiowpatch as pyaudio

    p = pyaudio.PyAudio()
    try:
        try:
            return p.get_default_wasapi_loopback()
        except Exception as exc:
            raise RuntimeError(
                "No default WASAPI loopback device is available. Check the active Windows playback device."
            ) from exc
    finally:
        p.terminate()


def _open_windows_input_stream(
    p: object,
    pyaudio: object,
    device: dict[str, object],
    chunk: int,
) -> tuple[object, int, int]:
    channels = int(device["maxInputChannels"])
    if channels <= 0:
        raise RuntimeError(f"Device has no input channels: {device['name']}")

    sample_rate = int(float(device["defaultSampleRate"]))
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=int(device["index"]),
        frames_per_buffer=chunk,
    )
    return stream, sample_rate, channels


def _record_windows_device(label: str, device: dict[str, object]) -> RecordingSession:
    import pyaudiowpatch as pyaudio

    chunk = WINDOWS_CAPTURE_CHUNK
    p = pyaudio.PyAudio()
    stream: object | None = None
    writer: sf.SoundFile | None = None
    path: Path | None = None
    sample_rate = 0
    channels = 0
    errors: list[Exception] = []
    written_frames = [0]
    stop_event = threading.Event()
    reader: threading.Thread | None = None
    try:
        logger.info("Using {label} device: {name}", label=label, name=device["name"])
        try:
            stream, sample_rate, channels = _open_windows_input_stream(p, pyaudio, device, chunk)
        except Exception as exc:
            raise RuntimeError(f"Failed to open {label} device '{device['name']}'.") from exc

        path, writer = open_temp_soundfile(sample_rate, channels)
        capture_start_time = time.perf_counter() + 0.05
        reader = threading.Thread(
            target=_capture_windows_stream_to_file,
            kwargs={
                "stream": stream,
                "chunk": chunk,
                "sample_rate": sample_rate,
                "channels": channels,
                "writer": writer,
                "stop_event": stop_event,
                "errors": errors,
                "label": label,
                "written_frames": written_frames,
                "start_time": capture_start_time,
            },
            daemon=True,
        )

        reader.start()
        logger.info("Recording {label}... Press Enter to stop.", label=label)
        input()

        stop_event.set()
        reader.join()
    finally:
        stop_event.set()
        if reader is not None and reader.is_alive():
            reader.join(timeout=2)
        _safe_close_stream(stream)
        safe_close_soundfile(writer)
        p.terminate()

    if errors:
        if path is not None:
            path.unlink(missing_ok=True)
        raise RuntimeError(f"Recording failed for {label}.") from errors[0]

    if path is None or written_frames[0] <= 0:
        if path is not None:
            path.unlink(missing_ok=True)
        raise RuntimeError(f"No audio captured from {label}.")

    logger.info(
        "Recording stopped. {frames} frames captured from {label}.",
        frames=written_frames[0],
        label=label,
    )
    return build_single_source_session(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        frames=written_frames[0],
        label=label,
    )


def record_mic_windows() -> RecordingSession:
    device = _get_default_windows_input_device()
    return _record_windows_device("microphone", device)


def record_os_windows() -> RecordingSession:
    device = _get_default_windows_loopback_device()
    return _record_windows_device("system audio", device)


def record_both_windows() -> RecordingSession:
    import pyaudiowpatch as pyaudio

    chunk = WINDOWS_CAPTURE_CHUNK
    p = pyaudio.PyAudio()
    mic_stream: object | None = None
    os_stream: object | None = None
    mic_writer: sf.SoundFile | None = None
    os_writer: sf.SoundFile | None = None
    mic_path: Path | None = None
    os_path: Path | None = None
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    errors: list[Exception] = []
    mic_written_frames = [0]
    os_written_frames = [0]
    try:
        mic_device = _get_default_windows_input_device()
        os_device = _get_default_windows_loopback_device()

        logger.info("Using microphone device: {name}", name=mic_device["name"])
        logger.info("Using WASAPI loopback: {name}", name=os_device["name"])

        try:
            mic_stream, mic_rate, mic_channels = _open_windows_input_stream(
                p, pyaudio, mic_device, chunk
            )
            os_stream, os_rate, os_channels = _open_windows_input_stream(
                p, pyaudio, os_device, chunk
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to open microphone + system audio capture streams in Windows."
            ) from exc

        mic_path, mic_writer = open_temp_soundfile(mic_rate, mic_channels)
        os_path, os_writer = open_temp_soundfile(os_rate, os_channels)
        capture_start_time = time.perf_counter() + 0.1

        threads = [
            threading.Thread(
                target=_capture_windows_stream_to_file,
                kwargs={
                    "stream": mic_stream,
                    "chunk": chunk,
                    "sample_rate": mic_rate,
                    "channels": mic_channels,
                    "writer": mic_writer,
                    "stop_event": stop_event,
                    "errors": errors,
                    "label": "microphone",
                    "written_frames": mic_written_frames,
                    "start_time": capture_start_time,
                },
                daemon=True,
            ),
            threading.Thread(
                target=_capture_windows_stream_to_file,
                kwargs={
                    "stream": os_stream,
                    "chunk": chunk,
                    "sample_rate": os_rate,
                    "channels": os_channels,
                    "writer": os_writer,
                    "stop_event": stop_event,
                    "errors": errors,
                    "label": "system audio",
                    "written_frames": os_written_frames,
                    "start_time": capture_start_time,
                },
                daemon=True,
            ),
        ]

        for thread in threads:
            thread.start()

        logger.info("Recording microphone + system audio... Press Enter to stop.")
        input()

        stop_event.set()
        for thread in threads:
            thread.join()
    finally:
        stop_event.set()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2)
        _safe_close_stream(mic_stream)
        _safe_close_stream(os_stream)
        safe_close_soundfile(mic_writer)
        safe_close_soundfile(os_writer)
        p.terminate()

    if errors:
        if mic_path is not None:
            mic_path.unlink(missing_ok=True)
        if os_path is not None:
            os_path.unlink(missing_ok=True)
        raise RuntimeError("Recording failed while capturing microphone and system audio.") from errors[0]

    if mic_path is None or mic_written_frames[0] <= 0:
        if mic_path is not None:
            mic_path.unlink(missing_ok=True)
        if os_path is not None:
            os_path.unlink(missing_ok=True)
        raise RuntimeError("No audio captured from microphone.")

    if os_path is None or os_written_frames[0] <= 0:
        if mic_path is not None:
            mic_path.unlink(missing_ok=True)
        if os_path is not None:
            os_path.unlink(missing_ok=True)
        raise RuntimeError("No audio captured from system audio.")

    logger.info(
        "Recording stopped. mic_frames={mic_frames}, os_frames={os_frames}",
        mic_frames=mic_written_frames[0],
        os_frames=os_written_frames[0],
    )

    return RecordingSession(
        sources=[
            RecordedAudioSource(
                path=mic_path,
                sample_rate=mic_rate,
                channels=mic_channels,
                frames=mic_written_frames[0],
                label="microphone",
            ),
            RecordedAudioSource(
                path=os_path,
                sample_rate=os_rate,
                channels=os_channels,
                frames=os_written_frames[0],
                label="system audio",
            ),
        ]
    )
