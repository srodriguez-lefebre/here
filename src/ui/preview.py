"""Interactive preview backed by the fake application controller."""

from __future__ import annotations

import sys
from pathlib import Path

from here.ui.app import HereDesktop, application_instance
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController, FakeLibraryBackend


def main() -> int:
    application = application_instance(sys.argv)
    library = FakeLibraryBackend()
    desktop = HereDesktop(
        application,
        ApplicationUiAdapter(
            FakeApplicationController(),
            Path.cwd() / "transcriptions",
            memory_service=library,
            search_service=library,
        ),
    )
    desktop.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
