import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

from here.config.settings import get_settings


def _write_temp_wav(audio: np.ndarray, sample_rate: int) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, sample_rate)
    return Path(tmp.name)


def _record_until_enter(sample_rate: int, channels: int, label: str) -> Path:
    """Record audio with sounddevice until the user presses Enter."""
    chunks: list[np.ndarray] = []

    def callback(
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time
        if status:
            logger.warning("Audio status: {status}", status=status)
        chunks.append(indata.copy())

    logger.info("Recording {label}... Press Enter to stop.", label=label)

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="float32", callback=callback):
        input()

    if not chunks:
        raise RuntimeError(f"No audio captured from {label}.")

    logger.info("Recording stopped. {n} chunks captured.", n=len(chunks))
    audio = np.concatenate(chunks, axis=0)
    return _write_temp_wav(audio, sample_rate)


def _setup_pulse() -> None:
    """Set PULSE_SERVER in the environment if configured."""
    pulse_server = get_settings().PULSE_SERVER
    if pulse_server:
        os.environ["PULSE_SERVER"] = pulse_server


def _get_monitor_source_name() -> str | None:
    """Find the PulseAudio monitor source name using pactl."""
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True,
            text=True,
            env=os.environ,
        )
        for line in result.stdout.splitlines():
            if ".monitor" in line:
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    except Exception as exc:
        logger.error("Failed to query PulseAudio sources: {exc}", exc=exc)
    return None


def _get_default_source() -> str:
    """Return the current PulseAudio default source name."""
    result = subprocess.run(
        ["pactl", "get-default-source"],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    return result.stdout.strip()


def _set_default_source(source: str) -> None:
    """Set the PulseAudio default source."""
    subprocess.run(["pactl", "set-default-source", source], env=os.environ)


def _record_os_linux(sample_rate: int) -> Path:
    _setup_pulse()

    monitor = _get_monitor_source_name()
    if not monitor:
        raise RuntimeError("No PulseAudio monitor source found. Is PulseAudio running?")

    logger.info("Using monitor source: {monitor}", monitor=monitor)

    previous_source = _get_default_source()
    _set_default_source(monitor)

    try:
        return _record_until_enter(sample_rate, channels=1, label="system audio")
    finally:
        _set_default_source(previous_source)


def _frames_to_audio(frames: list[bytes], channels: int, label: str) -> np.ndarray:
    if not frames:
        raise RuntimeError(f"No audio captured from {label}.")

    audio_int16 = np.frombuffer(b"".join(frames), dtype=np.int16)
    return (audio_int16.astype(np.float32) / 32768.0).reshape(-1, channels)


def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1, dtype=np.float32)


def _resample_mono_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32)

    target_length = max(1, int(round(audio.shape[0] * target_rate / source_rate)))
    if audio.shape[0] == 1:
        return np.full(target_length, audio[0], dtype=np.float32)

    source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, audio)
    return resampled.astype(np.float32)


def _pad_mono_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    if audio.shape[0] >= target_length:
        return audio

    padding = np.zeros(target_length - audio.shape[0], dtype=np.float32)
    return np.concatenate([audio, padding])


def _mix_mono_tracks(mic_audio: np.ndarray, os_audio: np.ndarray) -> np.ndarray:
    target_length = max(mic_audio.shape[0], os_audio.shape[0])
    mixed = 0.5 * _pad_mono_audio(mic_audio, target_length) + 0.5 * _pad_mono_audio(
        os_audio, target_length
    )

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def _capture_windows_stream(
    stream: object,
    *,
    chunk: int,
    frames: list[bytes],
    stop_event: threading.Event,
    errors: list[Exception],
    label: str,
) -> None:
    while not stop_event.is_set():
        try:
            data = stream.read(chunk, exception_on_overflow=False)
        except Exception as exc:
            errors.append(exc)
            logger.error("Failed to read {label}: {exc}", label=label, exc=exc)
            stop_event.set()
            break
        frames.append(data)


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


def _record_windows_device(label: str, device: dict[str, object]) -> Path:
    import pyaudiowpatch as pyaudio

    chunk = 1024
    p = pyaudio.PyAudio()
    stream: object | None = None
    sample_rate = 0
    channels = 0
    frames: list[bytes] = []
    errors: list[Exception] = []
    stop_event = threading.Event()
    reader: threading.Thread | None = None
    try:
        logger.info("Using {label} device: {name}", label=label, name=device["name"])
        try:
            stream, sample_rate, channels = _open_windows_input_stream(p, pyaudio, device, chunk)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open {label} device '{device['name']}'."
            ) from exc

        reader = threading.Thread(
            target=_capture_windows_stream,
            kwargs={
                "stream": stream,
                "chunk": chunk,
                "frames": frames,
                "stop_event": stop_event,
                "errors": errors,
                "label": label,
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
        p.terminate()

    if errors:
        raise RuntimeError(f"Recording failed for {label}") from errors[0]

    logger.info("Recording stopped. {n} chunks captured from {label}.", n=len(frames), label=label)
    audio = _frames_to_audio(frames, channels, label)
    return _write_temp_wav(audio, sample_rate)


def _record_mic_windows() -> Path:
    device = _get_default_windows_input_device()
    return _record_windows_device("microphone", device)


def _record_os_windows() -> Path:
    device = _get_default_windows_loopback_device()
    return _record_windows_device("system audio", device)


def _record_both_windows() -> Path:
    import pyaudiowpatch as pyaudio

    chunk = 1024
    p = pyaudio.PyAudio()
    mic_stream: object | None = None
    os_stream: object | None = None
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    mic_frames: list[bytes] = []
    os_frames: list[bytes] = []
    errors: list[Exception] = []
    try:
        try:
            mic_device = _get_default_windows_input_device()
            os_device = _get_default_windows_loopback_device()
        except RuntimeError:
            raise

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

        threads = [
            threading.Thread(
                target=_capture_windows_stream,
                kwargs={
                    "stream": mic_stream,
                    "chunk": chunk,
                    "frames": mic_frames,
                    "stop_event": stop_event,
                    "errors": errors,
                    "label": "microphone",
                },
                daemon=True,
            ),
            threading.Thread(
                target=_capture_windows_stream,
                kwargs={
                    "stream": os_stream,
                    "chunk": chunk,
                    "frames": os_frames,
                    "stop_event": stop_event,
                    "errors": errors,
                    "label": "system audio",
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
        p.terminate()

    if errors:
        raise RuntimeError("Recording failed while capturing microphone and system audio") from errors[0]

    logger.info(
        "Recording stopped. mic_chunks={mic_chunks}, os_chunks={os_chunks}",
        mic_chunks=len(mic_frames),
        os_chunks=len(os_frames),
    )

    mic_audio = _downmix_to_mono(_frames_to_audio(mic_frames, mic_channels, "microphone"))
    os_audio = _downmix_to_mono(_frames_to_audio(os_frames, os_channels, "system audio"))

    target_rate = os_rate
    mic_audio = _resample_mono_audio(mic_audio, mic_rate, target_rate)
    os_audio = _resample_mono_audio(os_audio, os_rate, target_rate)
    mixed_audio = _mix_mono_tracks(mic_audio, os_audio)

    return _write_temp_wav(mixed_audio, target_rate)


def record_mic_until_enter(sample_rate: int = 16000) -> Path:
    """
    Record audio from the microphone until the user presses Enter.

    On Windows: uses PyAudioWPatch.
    On Linux/WSL2: uses sounddevice.
    """
    if sys.platform == "win32":
        return _record_mic_windows()

    _setup_pulse()
    return _record_until_enter(sample_rate, channels=1, label="mic")


def record_os_until_enter(sample_rate: int = 16000) -> Path:
    """
    Record system audio until the user presses Enter.

    On Windows: uses WASAPI loopback via PyAudioWPatch.
    On Linux/WSL2: uses the PulseAudio monitor source.
    """
    if sys.platform == "win32":
        return _record_os_windows()
    return _record_os_linux(sample_rate)


def record_both_until_enter() -> Path:
    """Record microphone and system audio together until the user presses Enter."""
    if sys.platform != "win32":
        raise RuntimeError(
            "`here record` currently supports combined mic + system capture only on Windows."
        )

    return _record_both_windows()
