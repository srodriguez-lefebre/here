import os
import queue
import subprocess
import threading

import numpy as np
import sounddevice as sd
from loguru import logger

from here.config.settings import get_settings
from here.recording.models import RecordingSession
from here.recording.shared import build_single_source_session, open_temp_soundfile

PCM_16_MAX = 32768.0
PCM_16_SAMPLE_BYTES = 2


def _record_to_disk_until_enter(sample_rate: int, channels: int, label: str) -> RecordingSession:
    path, writer = open_temp_soundfile(sample_rate, channels)
    audio_queue: queue.SimpleQueue[np.ndarray | None] = queue.SimpleQueue()
    writer_errors: list[Exception] = []
    written_frames = [0]

    def callback(
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time
        if status:
            logger.warning("Audio status: {status}", status=status)
        audio_queue.put(indata.copy())

    def writer_worker() -> None:
        try:
            with writer:
                while True:
                    block = audio_queue.get()
                    if block is None:
                        break
                    writer.write(block)
                    written_frames[0] += int(block.shape[0])
        except Exception as exc:
            writer_errors.append(exc)

    writer_thread = threading.Thread(target=writer_worker, daemon=True)

    logger.info("Recording {label}... Press Enter to stop.", label=label)
    try:
        writer_thread.start()
        with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="float32", callback=callback):
            input()
    finally:
        audio_queue.put(None)
        writer_thread.join()

    if writer_errors:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed while writing recorded audio for {label}.") from writer_errors[0]

    if written_frames[0] <= 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"No audio captured from {label}.")

    logger.info("Recording stopped. {frames} frames captured from {label}.", frames=written_frames[0], label=label)
    return build_single_source_session(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        frames=written_frames[0],
        label=label,
    )


def _setup_pulse() -> None:
    pulse_server = get_settings().PULSE_SERVER
    if pulse_server:
        os.environ["PULSE_SERVER"] = pulse_server


def _run_pactl(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["pactl", *args],
            capture_output=True,
            text=True,
            env=os.environ,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("pactl is not installed or not available in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        message = f"pactl {' '.join(args)} failed"
        if details:
            message = f"{message}: {details}"
        raise RuntimeError(message) from exc


def _get_monitor_source_name() -> str | None:
    try:
        result = _run_pactl(["list", "sources", "short"])
        for line in result.stdout.splitlines():
            if ".monitor" in line:
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    except RuntimeError as exc:
        logger.error("Failed to query PulseAudio sources: {exc}", exc=exc)
    return None


def _get_default_source() -> str:
    return _run_pactl(["get-default-source"]).stdout.strip()


def _set_default_source(source: str) -> None:
    _run_pactl(["set-default-source", source])


def _record_pulse_source_until_enter(
    *,
    source_name: str,
    sample_rate: int,
    channels: int,
    label: str,
) -> RecordingSession:
    path, writer = open_temp_soundfile(sample_rate, channels)
    writer_errors: list[Exception] = []
    written_frames = [0]
    command = [
        "parec",
        f"--device={source_name}",
        "--format=s16le",
        f"--rate={sample_rate}",
        f"--channels={channels}",
        "--raw",
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )
    except FileNotFoundError as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError("parec is not installed or not available in PATH.") from exc

    def writer_worker() -> None:
        pending = b""
        bytes_per_frame = channels * PCM_16_SAMPLE_BYTES
        read_size = 4096 * bytes_per_frame
        try:
            if process.stdout is None:
                raise RuntimeError("parec did not expose a stdout pipe.")

            with writer:
                while True:
                    chunk = process.stdout.read(read_size)
                    if not chunk:
                        break

                    pending += chunk
                    usable_size = (len(pending) // bytes_per_frame) * bytes_per_frame
                    if usable_size <= 0:
                        continue

                    audio = np.frombuffer(pending[:usable_size], dtype=np.int16).astype(np.float32)
                    audio = (audio / PCM_16_MAX).reshape(-1, channels)
                    writer.write(audio)
                    written_frames[0] += int(audio.shape[0])
                    pending = pending[usable_size:]
        except Exception as exc:
            writer_errors.append(exc)

    writer_thread = threading.Thread(target=writer_worker, daemon=True)

    logger.info(
        "Recording {label} from PulseAudio source {source}... Press Enter to stop.",
        label=label,
        source=source_name,
    )
    try:
        writer_thread.start()
        input()
    finally:
        if process.poll() is None:
            process.terminate()
        writer_thread.join()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    stderr_output = ""
    if process.stderr is not None:
        stderr_output = process.stderr.read().decode("utf-8", errors="replace").strip()

    if writer_errors:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed while writing recorded audio for {label}.") from writer_errors[0]

    if process.returncode not in (0, -15):
        path.unlink(missing_ok=True)
        details = f" {stderr_output}" if stderr_output else ""
        raise RuntimeError(f"parec failed while recording {label}.{details}")

    if written_frames[0] <= 0:
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


def record_mic_linux(sample_rate: int = 16000) -> RecordingSession:
    _setup_pulse()
    return _record_to_disk_until_enter(sample_rate, channels=1, label="mic")


def record_os_linux(sample_rate: int = 16000) -> RecordingSession:
    _setup_pulse()

    monitor = _get_monitor_source_name()
    if not monitor:
        raise RuntimeError("No PulseAudio monitor source found. Is PulseAudio running?")

    logger.info("Using monitor source: {monitor}", monitor=monitor)
    return _record_pulse_source_until_enter(
        source_name=monitor,
        sample_rate=sample_rate,
        channels=1,
        label="system audio",
    )
