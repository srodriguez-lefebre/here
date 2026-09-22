"""Interactive preview backed by the fake application controller."""

from __future__ import annotations

import sys
from pathlib import Path

from here.ui.app import application_instance, create_desktop
from here.ui.fake_controller import FakeApplicationController


def main() -> int:
    application = application_instance(sys.argv)
    desktop = create_desktop(
        application,
        FakeApplicationController(),
        output_dir=Path.cwd() / "transcriptions",
    )
    desktop.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
