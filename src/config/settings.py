from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

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
_operation_settings: ContextVar[Settings | None] = ContextVar("here_settings", default=None)


def get_settings() -> Settings:
    return _operation_settings.get() or Settings()


P = ParamSpec("P")
R = TypeVar("R")


def settings_operation(function: Callable[P, R]) -> Callable[P, R]:
    """Refresh once per explicit operation; nested consumers share its snapshot."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _operation_settings.set(get_settings())
        try:
            return function(*args, **kwargs)
        finally:
            _operation_settings.reset(token)

    return wrapped


def require_provider_key() -> str:
    secret = get_settings().OPENAI_API_KEY
    value = secret.get_secret_value().strip() if secret is not None else ""
    if not value:
        raise ValueError(f"Configure OPENAI_API_KEY in the environment or {get_env_file()}")
    return value
