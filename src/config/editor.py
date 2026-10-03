"""Edit only supported local settings, preserving unrelated dotenv entries."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import dotenv_values, set_key

from .paths import get_env_file

EDITABLE_KEYS = (
    "OPENAI_API_KEY",
    "TRANSCRIPTION_MODEL",
    "ALT_TRANSCRIPTION_MODEL",
    "CLEANUP_MODEL",
    "CLEANUP_ENABLED",
    "TRANSCRIPTIONS_DIR",
)


def read_configuration() -> dict[str, str]:
    return {key: value or "" for key, value in dotenv_values(get_env_file()).items()}


def save_configuration(values: dict[str, str]) -> Path:
    for key, value in values.items():
        if key not in EDITABLE_KEYS or any(character in value for character in "\r\n\0"):
            raise ValueError("Configuración inválida: use valores de una sola línea.")
        if key.endswith("MODEL") and not value.strip():
            raise ValueError("Los nombres de los modelos no pueden estar vacíos.")
        if key == "CLEANUP_ENABLED" and value not in {"true", "false"}:
            raise ValueError("La opción de limpieza debe ser true o false.")
        if key == "TRANSCRIPTIONS_DIR" and value and not Path(value).is_absolute():
            raise ValueError("Elegí una carpeta absoluta para las sesiones.")
    target = get_env_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".here-env-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            if target.exists():
                stream.write(target.read_text(encoding="utf-8"))
        for key, value in values.items():
            set_key(str(temporary), key, value.strip(), quote_mode="always")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
