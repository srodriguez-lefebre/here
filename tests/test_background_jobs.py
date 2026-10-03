import gc
import threading
import weakref
from types import SimpleNamespace

import pytest
from here.ui import bridge


@pytest.mark.parametrize("closed", [False, True])
def test_idle_waits_for_actual_final_job_thread_exit(qtbot, monkeypatch, closed):
    selected, entered, release = threading.Event(), threading.Event(), threading.Event()
    threads = []

    class TailBarrierThread(threading.Thread):
        def run(self):
            super().run()
            assert selected.wait(3)
            if self is threads[-1]:
                entered.set()
                assert release.wait(3)

    def create_thread(*args, **kwargs):
        thread = TailBarrierThread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(bridge, "threading", SimpleNamespace(Thread=create_thread))
    jobs = bridge.BackgroundJobs()
    delivered, idle = [], []
    jobs.finished.connect(lambda *args: delivered.append(args))
    jobs.idleChanged.connect(lambda: idle.append(True))
    request = jobs.submit("recovery", lambda: ["local result"])
    if closed:
        jobs.close()
    selected.set()
    try:
        qtbot.waitUntil(entered.is_set)
        # Process queued delivery while the last owned thread is provably still alive.
        qtbot.wait(30)
        assert threads[-1].is_alive()
        assert jobs.busy
        assert not delivered
        assert not idle
    finally:
        release.set()
        qtbot.waitUntil(lambda: all(not thread.is_alive() for thread in threads))
        qtbot.waitUntil(lambda: not jobs.busy)
        jobs.close()
    assert len(idle) == 1
    assert delivered == ([] if closed else [(request, "recovery", ["local result"], None)])


@pytest.mark.usefixtures("owned_desktops")
def test_test_owner_survives_pending_discovery_until_widgets_are_drained(qtbot, tmp_path):
    from here.ui.app import HereDesktop
    from here.ui.contract import ApplicationUiAdapter
    from here.ui.fake_controller import FakeApplicationController
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    entered, release = threading.Event(), threading.Event()

    class Recovery:
        def discover(self):
            entered.set()
            assert release.wait(3)
            return []

    desktop = HereDesktop(
        QApplication.instance(),
        ApplicationUiAdapter(FakeApplicationController(), tmp_path, recovery_service=Recovery()),
        settings=QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat),
    )
    jobs = desktop.jobs

    def destroyed():
        assert not jobs.busy

    desktop.main_window.destroyed.connect(destroyed)
    owner = weakref.ref(desktop)
    try:
        assert entered.wait(2)
        del desktop
        gc.collect()
        assert owner() is not None
        assert jobs.busy
    finally:
        release.set()
    # Fixture's Qtbot before-close hook must retain/drain the owner even though
    # this test intentionally returns without waiting for startup completion.
