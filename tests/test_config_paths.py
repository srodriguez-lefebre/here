import sys

import pytest
from here.config import settings


def test_optional_key_and_user_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("HERE_DATA_DIR")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "missing.env"))
    value = settings.Settings()
    assert value.OPENAI_API_KEY is None
    assert value.TRANSCRIPTIONS_DIR == tmp_path / "here" / "sessions"


def test_effective_file_precedence_and_refresh(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("HERE_ENV_FILE")
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path))
    user = tmp_path / ".env"
    user.write_text("OPENAI_API_KEY=user-placeholder\n")
    assert settings.get_settings().OPENAI_API_KEY.get_secret_value() == "user-placeholder"
    user.write_text("OPENAI_API_KEY=edited-placeholder\n")
    assert settings.get_settings().OPENAI_API_KEY.get_secret_value() == "edited-placeholder"
    explicit = tmp_path / "chosen.env"
    explicit.write_text("OPENAI_API_KEY=explicit-placeholder\n")
    monkeypatch.setenv("HERE_ENV_FILE", str(explicit))
    assert settings.Settings().OPENAI_API_KEY.get_secret_value() == "explicit-placeholder"
    monkeypatch.setenv("OPENAI_API_KEY", "process-placeholder")
    assert settings.Settings().OPENAI_API_KEY.get_secret_value() == "process-placeholder"


def test_frozen_does_not_fall_back_to_project_env(tmp_path, monkeypatch):
    from here.config.paths import get_env_file

    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("HERE_ENV_FILE")
    assert get_env_file() == tmp_path / ".env"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_default_live_fails_before_temporary_allocation(tmp_path, monkeypatch, key):
    import here.live_processing as live

    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "missing.env"))
    if key is None:
        monkeypatch.delenv("OPENAI_API_KEY")
    else:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(live.tempfile, "mkdtemp", lambda **kw: pytest.fail("allocated temp"))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        live.LiveTranscriptionController(expected_source_count=1)


def test_operation_keeps_configuration_snapshot_then_refreshes(tmp_path, monkeypatch):
    from here.config.settings import settings_operation

    chosen = tmp_path / "selected.env"
    chosen.write_text("TRANSCRIPTION_MODEL=first\n")
    monkeypatch.setenv("HERE_ENV_FILE", str(chosen))

    @settings_operation
    def operation():
        assert settings.get_settings().TRANSCRIPTION_MODEL == "first"
        chosen.write_text("TRANSCRIPTION_MODEL=second\n")
        assert settings.get_settings().TRANSCRIPTION_MODEL == "first"

    operation()
    assert settings.get_settings().TRANSCRIPTION_MODEL == "second"


def test_development_fallback_and_explicit_missing_file(tmp_path, monkeypatch):
    import here.config.paths as paths

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("HERE_ENV_FILE")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=development-placeholder\n")
    assert settings.Settings().OPENAI_API_KEY.get_secret_value() == "development-placeholder"
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "missing.env"))
    assert settings.Settings().OPENAI_API_KEY is None
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(tmp_path / "chosen"))
    assert settings.Settings().TRANSCRIPTIONS_DIR == tmp_path / "chosen"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_default_start_fails_before_capture_or_live_allocation(tmp_path, monkeypatch, key):
    import here.application.controller as controller
    import here.live_processing as live
    from here.application import ApplicationState, HereApplicationController, StartRequest

    if key is None:
        monkeypatch.delenv("OPENAI_API_KEY")
    else:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(controller, "start_recording", lambda *a, **kw: pytest.fail("hardware"))
    monkeypatch.setattr(live.tempfile, "mkdtemp", lambda **kw: pytest.fail("live allocation"))
    instance = HereApplicationController()
    instance.start(StartRequest(output_dir=tmp_path / "sessions"))
    snapshot = instance.wait_until_terminal(2)
    assert snapshot.state is ApplicationState.FAILED
    assert "OPENAI_API_KEY" in snapshot.last_error.message
    assert not (tmp_path / "sessions").exists()


def test_missing_key_retry_keeps_recovery_untouched(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from here.application.processing import SessionProcessor, session_from_audio_file
    from here.transcription.client import TranscriptionResult

    original = tmp_path / "original.wav"
    sf.write(original, np.zeros(80), 8000)
    saved = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("ok", "ok")).process(
        session_from_audio_file(original), tmp_path / "sessions"
    )
    before = {p.name: p.read_bytes() for p in saved.session_dir.iterdir()}
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        SessionProcessor().retry(saved.session_dir)
    assert {p.name: p.read_bytes() for p in saved.session_dir.iterdir()} == before


def test_retry_metadata_uses_current_operation_model(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from here.application.processing import SessionProcessor, session_from_audio_file
    from here.transcription.client import TranscriptionResult

    path = tmp_path / "original.wav"
    sf.write(path, np.zeros(80), 8000)
    processor = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("ok", "ok"))
    first = processor.process(session_from_audio_file(path), tmp_path / "sessions")
    monkeypatch.setenv("TRANSCRIPTION_MODEL", "new-operation-model")
    second = processor.retry(first.session_dir)
    assert second.metadata.transcription_model == "new-operation-model"


def test_production_gui_opens_without_key_or_provider(tmp_path, monkeypatch, qtbot):
    import here.transcription.client as client
    import here.ui.gui as gui
    from here.ui.app import create_desktop
    from PySide6.QtCore import QSettings

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(client, "OpenAI", lambda **kw: pytest.fail("provider construction"))
    preferences = QSettings(str(tmp_path / "gui.ini"), QSettings.IniFormat)
    monkeypatch.setattr(
        gui,
        "create_desktop",
        lambda app, controller, **kw: create_desktop(app, controller, settings=preferences, **kw),
    )
    desktop = gui.create_production_desktop()
    qtbot.addWidget(desktop.main_window)
    qtbot.addWidget(desktop.overlay)
    desktop.show()
    assert desktop.main_window.isVisible()
    desktop.bridge.close()
