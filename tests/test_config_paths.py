import sys

import pytest
from here.config import settings


def test_optional_key_and_user_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "missing.env"))
    value = settings.Settings()
    assert value.OPENAI_API_KEY is None
    assert value.TRANSCRIPTIONS_DIR == tmp_path / "here" / "sessions"


def test_effective_file_precedence_and_refresh(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
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
