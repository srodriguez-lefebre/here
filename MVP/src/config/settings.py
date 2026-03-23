from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# MVP/ root, from src/config/settings.py
_PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    TRANSCRIPTIONS_DIR: Path = _PROJECT_ROOT / "transcriptions"
    PULSE_SERVER: str | None = None

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )


def get_settings() -> Settings:
    return Settings()
