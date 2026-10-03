import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from here.application.processing import SessionProcessor, session_from_audio_file
from here.transcription.client import TranscriptionResult


def test_legacy_save_real_wav_normalization_failure_has_local_retry_source(tmp_path, monkeypatch):
    import here.cli as cli

    audio = tmp_path / "original.wav"
    sf.write(audio, np.full(80, 1234, dtype=np.int16), 8000, subtype="PCM_16")
    session = session_from_audio_file(audio)

    def fail(*args, **kwargs):
        raise OSError("synthetic normalization failure")

    monkeypatch.setattr(cli, "materialize_normalized_session", fail)
    with pytest.raises(RuntimeError):
        cli._save_transcription(session, tmp_path / "sessions")
    saved = next((tmp_path / "sessions").iterdir())
    metadata = json.loads((saved / "session.json").read_text())
    local = metadata["capture_sources"][0]["audio_file"]
    assert local is not None
    audio.unlink(missing_ok=True)
    assert sf.info(saved / local).frames == 80
    error = json.loads((saved / "errors.json").read_text())["errors"][0]
    assert error["type"] == "OSError" and error["message"] == "synthetic normalization failure"
    result = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("ok", "ok")).retry(
        saved
    )
    assert result.metadata.status == "completed"


@pytest.mark.parametrize("reference", ["../outside.wav", "C:/outside.wav", "nested/file.wav"])
def test_cli_preflights_all_advertised_references_before_provider(tmp_path, monkeypatch, reference):
    import here.cli as cli

    audio = tmp_path / "original.wav"
    sf.write(audio, np.zeros(80), 8000)
    saved = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("ok", "ok")).process(
        session_from_audio_file(audio), tmp_path / "sessions"
    )
    path = saved.metadata_path
    metadata = json.loads(path.read_text())
    metadata["output_files"].append(reference)
    path.write_text(json.dumps(metadata))
    called = []

    def provider(*args, **kwargs):
        called.append(1)
        return TranscriptionResult("wrong", "wrong")

    monkeypatch.setattr(cli, "transcribe_recording_session", provider)
    with pytest.raises(ValueError):
        cli._transcribe_audio_path(saved.audio_path, tmp_path / "sessions")
    assert called == []


@pytest.mark.parametrize("name", ["session.json", "chunks.json", "errors.json", "audio.wav"])
def test_cli_redirected_neighbor_never_reads_canary(tmp_path, monkeypatch, name):
    import here.cli as cli

    audio = tmp_path / "original.wav"
    sf.write(audio, np.zeros(80), 8000)
    saved = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("ok", "ok")).process(
        session_from_audio_file(audio), tmp_path / "sessions"
    )
    canary = tmp_path / "external-canary"
    canary.write_text("private canary")
    entry = saved.session_dir / name
    entry.unlink(missing_ok=True)
    try:
        entry.symlink_to(canary)
    except OSError:
        entry.hardlink_to(canary)
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        assert path.resolve() != canary.resolve(), "read external canary"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(
        cli, "transcribe_recording_session", lambda *a, **kw: pytest.fail("provider")
    )
    with pytest.raises(ValueError):
        cli._transcribe_audio_path(saved.audio_path, tmp_path / "sessions")
    assert canary.read_bytes() == b"private canary"
