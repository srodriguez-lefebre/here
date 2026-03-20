import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger


def record_until_enter(sample_rate: int = 16000) -> Path:
    """
    Records audio from the microphone until the user presses Enter.

    Args:
        sample_rate: Sample rate in Hz. Defaults to 16000 (optimal for Whisper).

    Returns:
        Path to a temporary WAV file with the recorded audio.
    """
    chunks: list[np.ndarray] = []

    def callback(
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio status: {status}", status=status)
        chunks.append(indata.copy())

    logger.info("Recording... Press Enter to stop.")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        input()

    logger.info("Recording stopped. {n} chunks captured.", n=len(chunks))

    audio = np.concatenate(chunks, axis=0)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)

    return Path(tmp.name)
