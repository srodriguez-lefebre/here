"""Bounded persistent diagnostics: structural facts, never provider payloads or secrets."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from here.config.paths import get_data_dir

_lock = threading.Lock()


def error_code(message: str, error_type: str = "") -> str:
    if "OPENAI_API_KEY" in message:
        return "missing_api_key"
    if "Authentication" in error_type or "401" in message:
        return "authentication"
    if "RateLimit" in error_type or "429" in message:
        return "quota_or_rate_limit"
    if "Timeout" in error_type:
        return "timeout"
    if "Connection" in error_type:
        return "connection"
    if "Permission" in error_type:
        return "permission"
    return "operation_failed"


def safe_message(message: str) -> str:
    from here.config.settings import get_settings

    try:
        key = get_settings().OPENAI_API_KEY
        if key is not None and (secret := key.get_secret_value()):
            message = message.replace(secret, "[clave oculta]")
    except Exception:
        pass
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[clave oculta]", message)


def explain_error(message: str, error_type: str = "", *, recoverable: bool = False) -> str:
    code = error_code(message, error_type)
    known = {
        "missing_api_key": (
            "Falta la clave de OpenAI. Abrí Configuración y guardá OPENAI_API_KEY. "
            "La grabación no comenzó."
        ),
        "authentication": "OpenAI rechazó la clave. Revisala en Configuración y volvé a intentar.",
        "quota_or_rate_limit": (
            "OpenAI informó un límite de uso o saldo. Revisá tu cuenta y reintentá más tarde."
        ),
        "timeout": (
            "La operación superó el tiempo de espera. "
            "Revisá los dispositivos o la conexión y reintentá."
        ),
        "connection": "No se pudo conectar con OpenAI. Revisá la conexión a Internet y reintentá.",
        "permission": (
            "No hay permiso para acceder al archivo o dispositivo. "
            "Revisá la carpeta y los permisos."
        ),
    }
    text = known.get(code, "No se pudo completar la operación: " + safe_message(message))
    if recoverable:
        text += " El audio está guardado y se puede reintentar."
    return text


def record_event(component: str, event: str, **facts: str) -> bool:
    # Callers supply only structural fields; raw exceptions, transcripts, requests
    # and configuration values are deliberately never serialized.
    allowed = {
        key: value
        for key, value in facts.items()
        if key in {"stage", "error_type", "code", "state"}
    }
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "event": event,
        **allowed,
    }
    try:
        with _lock:
            path = get_data_dir() / "logs" / "here.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            try:
                handler.emit(
                    logging.LogRecord("here", logging.INFO, "", 0, json.dumps(entry), (), None)
                )
                handler.flush()
            finally:
                handler.close()
        return True
    except OSError:
        return False


def record_error(stage: str, error: BaseException) -> bool:
    return record_event(
        "application",
        "error",
        stage=stage,
        error_type=type(error).__name__,
        code=error_code(str(error), type(error).__name__),
    )
