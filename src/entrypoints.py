"""Console/frozen entry seam: dispatch trusted helpers before product bootstrap."""

from here.audio_helper import dispatch_internal_helper


def gui_main() -> int:
    result = dispatch_internal_helper()
    if result is not None:
        return result
    from here.distribution import ApplicationMutex, dispatch_smoke_check

    result = dispatch_smoke_check()
    if result is not None:
        return result
    from here.ui.gui import main

    with ApplicationMutex():
        return main()


def cli_main() -> None:
    result = dispatch_internal_helper()
    if result is not None:
        raise SystemExit(result)
    from here.distribution import ApplicationMutex, dispatch_smoke_check

    result = dispatch_smoke_check()
    if result is not None:
        raise SystemExit(result)
    from here.cli import app

    with ApplicationMutex():
        app()


def record_main() -> None:
    result = dispatch_internal_helper()
    if result is not None:
        raise SystemExit(result)
    from here.cli import record_app

    record_app()
