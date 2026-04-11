import tempfile
from pathlib import Path

import soundfile as sf

from here.recording.models import RecordedAudioSource, RecordingSession

PCM_SUBTYPE = "PCM_16"


def create_temp_wav_path() -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    return Path(tmp.name)


def open_temp_soundfile(sample_rate: int, channels: int) -> tuple[Path, sf.SoundFile]:
    path = create_temp_wav_path()
    writer = sf.SoundFile(path, mode="w", samplerate=sample_rate, channels=channels, subtype=PCM_SUBTYPE)
    return path, writer


def build_single_source_session(
    *,
    path: Path,
    sample_rate: int,
    channels: int,
    frames: int,
    label: str,
) -> RecordingSession:
    return RecordingSession(
        sources=[
            RecordedAudioSource(
                path=path,
                sample_rate=sample_rate,
                channels=channels,
                frames=frames,
                label=label,
            )
        ]
    )


def safe_close_soundfile(writer: sf.SoundFile | None) -> None:
    if writer is None:
        return

    try:
        writer.close()
    except Exception:
        pass
