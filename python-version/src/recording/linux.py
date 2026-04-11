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


def record_mic_linux(sample_rate: int = 16000) -> RecordingSession:
    _setup_pulse()
    return _record_to_disk_until_enter(sample_rate, channels=1, label="mic")


def record_os_linux(sample_rate: int = 16000) -> RecordingSession:
    _setup_pulse()

    monitor = _get_monitor_source_name()
    if not monitor:
        raise RuntimeError("No PulseAudio monitor source found. Is PulseAudio running?")

    logger.info("Using monitor source: {monitor}", monitor=monitor)

    previous_source = _get_default_source()
    _set_default_source(monitor)

    try:
        return _record_to_disk_until_enter(sample_rate, channels=1, label="system audio")
    finally:
        if previous_source:
            _set_default_source(previous_source)
