"""Recovery started by a directly composed desktop must use its real catalog."""

from __future__ import annotations

import pytest
from here.application import HereApplicationController
from here.memory.service import MemoryService
from here.ui.app import HereDesktop
from here.ui.bridge import BackgroundJobs
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox
from test_copilot_legacy_catalog_failure import _failed_session


@pytest.mark.usefixtures("owned_desktops")
@pytest.mark.parametrize("memory_source", ["controller", "explicit"])
def test_direct_adapter_uses_same_custom_memory_for_startup_recovery(
    tmp_path, monkeypatch, qapp, qtbot, memory_source
):
    sessions = tmp_path / "sessions"
    decoy = tmp_path / "unused-catalog"
    monkeypatch.setenv("HERE_DATA_DIR", str(decoy))
    memory = MemoryService(tmp_path / "catalog")
    _failed_session(sessions / "owned", MEETING_ID)
    available = _failed_session(sessions / "available", OTHER_MEETING_ID)

    registered = []
    startup_context = []
    register = MemoryService._register_sessions_root
    submit = BackgroundJobs.submit

    def observe_registration(instance, root):
        register(instance, root)
        registered.append((instance, root.absolute()))

    def observe_submit(jobs, kind, operation):
        if kind == "recovery":
            startup_context.append(tuple(registered))
        return submit(jobs, kind, operation)

    monkeypatch.setattr(MemoryService, "_register_sessions_root", observe_registration)
    monkeypatch.setattr(BackgroundJobs, "submit", observe_submit)
    if memory_source == "explicit":
        adapter = ApplicationUiAdapter(
            FakeApplicationController(), sessions, memory_service=memory
        )
    else:
        core = HereApplicationController(memory_service=memory)
        adapter = ApplicationUiAdapter(core, sessions)

    # The catalog's real OS lease excludes this session from recovery. The decoy
    # has no owner, so an adapter using it would surface an extra candidate.
    memory.begin_processing(MEETING_ID)
    try:
        desktop = HereDesktop(
            qapp,
            adapter,
            settings=QSettings(str(tmp_path / "recovery.ini"), QSettings.Format.IniFormat),
        )
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
        selector = desktop.main_window.findChild(QComboBox, "recoverySelector")
        assert selector is not None
        candidates = [selector.itemData(index) for index in range(selector.count())]
        assert [(item.capture_id, item.session_dir, item.can_retry) for item in candidates] == [
            (OTHER_MEETING_ID, available.directory, True)
        ]
        assert startup_context
        assert any(
            instance is memory and root == sessions.absolute()
            for instance, root in startup_context[0]
        )
        assert not decoy.exists()
    finally:
        memory.finish_capture(MEETING_ID)


def test_direct_adapter_with_fake_controller_keeps_default_recovery(tmp_path):
    sessions = tmp_path / "sessions"
    available = _failed_session(sessions / "available", OTHER_MEETING_ID)

    adapter = ApplicationUiAdapter(FakeApplicationController(), sessions)

    (candidate,) = adapter.discover_recovery()
    assert candidate.capture_id == OTHER_MEETING_ID
    assert candidate.session_dir == available.directory
    assert candidate.can_retry
