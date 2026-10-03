"""Production Windows GUI entry point."""

from __future__ import annotations

import os
import sys

from here.config.settings import get_settings

from ..application import create_default_controller
from .app import HereDesktop, application_instance, create_desktop


def create_production_desktop() -> HereDesktop:
    """Compose Qt with the real application controller, never the CLI or preview."""

    application = application_instance(sys.argv)
    controller = create_default_controller()
    output_dir = get_settings().TRANSCRIPTIONS_DIR
    if settings_file := os.environ.get("HERE_SETTINGS_FILE"):
        from PySide6.QtCore import QSettings

        return create_desktop(
            application,
            controller,
            output_dir=output_dir,
            settings=QSettings(settings_file, QSettings.Format.IniFormat),
        )
    return create_desktop(application, controller, output_dir=output_dir)


def main() -> int:
    application = application_instance(sys.argv)
    desktop = create_production_desktop()
    desktop.show()
    return int(application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
