"""Per-user storage and single-file configuration selection."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    if override := os.environ.get("HERE_DATA_DIR"):
        return Path(override).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "here"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "here"


def get_env_file() -> Path:
    if "HERE_ENV_FILE" in os.environ:
        return Path(os.environ["HERE_ENV_FILE"]).expanduser()
    user_file = get_data_dir() / ".env"
    if user_file.exists() or getattr(sys, "frozen", False):
        return user_file
    development_file = PROJECT_ROOT / ".env"
    return development_file if development_file.exists() else user_file
