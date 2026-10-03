from pathlib import Path

from here.config.paths import get_data_dir, get_env_file
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: SecretStr | None = None
    TRANSCRIPTIONS_DIR: Path = Field(default_factory=lambda: get_data_dir() / "sessions")
    PULSE_SERVER: str | None = None
    TRANSCRIPTION_MODEL: str = "gpt-4o-transcribe-diarize"
    ALT_TRANSCRIPTION_MODEL: str = "gpt-4o-transcribe"
    CLEANUP_MODEL: str = "gpt-4.1-mini"
    CLEANUP_ENABLED: bool = False

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **values: object) -> None:
        values.setdefault("_env_file", get_env_file())
        super().__init__(**values)


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    return Settings()


def require_provider_key() -> str:
    secret = get_settings().OPENAI_API_KEY
    value = secret.get_secret_value().strip() if secret is not None else ""
    if not value:
        raise ValueError(f"Configure OPENAI_API_KEY in the environment or {get_env_file()}")
    return value
