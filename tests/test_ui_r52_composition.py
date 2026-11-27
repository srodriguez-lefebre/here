"""Pending R52 GUI consumers; real backend with forwarding-only observations.

Native constructor/StartRequest/lifetime controls already belong to B's existing
test_memory_native_owner.py. This file adds only the missing GUI composition
boundaries. Preparation is NOT_RUN and does not accept an R52 source handoff.
"""

from __future__ import annotations

import importlib
import json
import traceback
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("owned_desktops")


@pytest.fixture
def real_r52_composition_trace(monkeypatch, tmp_path):
    memory_type = importlib.import_module("here.memory.service").MemoryService
    assert callable(getattr(memory_type, "_register_sessions_root", None)), (
        "Pending R52 handoff: real MemoryService._register_sessions_root is required"
    )
    jobs_type = importlib.import_module("here.ui.bridge").BackgroundJobs
    adapter_type = importlib.import_module("here.ui.contract").ApplicationUiAdapter
    original_construct = memory_type.__init__
    original_register = memory_type._register_sessions_root
    original_submit = jobs_type.submit
    original_adapter = adapter_type.__init__
    instances, registered, submitted, adapters, construction_context = [], [], [], [], []

    def construct(instance, data_dir, *, sessions_root=None):
        construction_context.append(
            {
                "instance": instance,
                "data_dir_argument": str(Path(data_dir).absolute()),
                "sessions_root_argument": (
                    str(Path(sessions_root).absolute()) if sessions_root is not None else None
                ),
                "callpath": [
                    {"path": frame.filename, "function": frame.name, "line": frame.lineno}
                    for frame in traceback.extract_stack(limit=6)[:-1]
                ],
            }
        )
        original_construct(instance, data_dir, sessions_root=sessions_root)
        instances.append(instance)

    def register(instance, root):
        result = original_register(instance, root)
        registered.append((instance, Path(root).absolute()))
        return result

    def submit(jobs, kind, operation):
        # Snapshot before original submit can launch its real worker. No work is
        # replaced, deferred or suppressed, and no private backend map is read.
        submitted.append((kind, tuple(instances), tuple(registered)))
        return original_submit(jobs, kind, operation)

    def adapter(instance, controller, output_dir, **kwargs):
        original_adapter(instance, controller, output_dir, **kwargs)
        adapters.append((instance, controller))

    def forbidden(*args, **kwargs):
        pytest.fail("GUI composition attempted provider/network work")

    monkeypatch.setattr(memory_type, "__init__", construct)
    monkeypatch.setattr(memory_type, "_register_sessions_root", register)
    monkeypatch.setattr(jobs_type, "submit", submit)
    monkeypatch.setattr(adapter_type, "__init__", adapter)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    return memory_type, instances, submitted, adapters, construction_context, registered


def _assert_same_memory_context_before_jobs(controller, desktop, trace, data_dir, sessions_root):
    _, _, submitted, adapters, _, _ = trace
    memory = controller.memory_service
    assert memory.data_dir == data_dir.absolute()
    assert desktop.controller.output_dir == sessions_root
    assert any(adapter is desktop.controller and core is controller for adapter, core in adapters)
    assert submitted, "The actual desktop must reach its background recovery dispatch"
    kind, instances_before_launch, roots_before_launch = submitted[0]
    assert len(instances_before_launch) == 1 and instances_before_launch[0] is memory
    assert any(
        instance is memory and root == sessions_root.absolute()
        for instance, root in roots_before_launch
    ), f"Configured root was not registered on the injected memory before {kind} worker"


def _record_constructor_context(trace, record_property):
    _, _, submitted, _, construction_context, registered = trace
    before = submitted[0][1] if submitted else ()
    observations = []
    for context in construction_context:
        instance = context["instance"]
        observations.append(
            {
                **{key: value for key, value in context.items() if key != "instance"},
                "phase": (
                    "before_gui_dispatch"
                    if any(instance is existing for existing in before)
                    else "after_gui_dispatch"
                ),
                "actual_data_dir": str(getattr(instance, "data_dir", "UNINITIALIZED")),
                "registered_sessions_roots": [
                    str(root) for memory, root in registered if memory is instance
                ],
            }
        )
    record_property("r52_real_memory_constructor_context", json.dumps(observations))


def test_create_desktop_registers_configured_root_on_same_injected_custom_memory(
    real_r52_composition_trace, tmp_path, monkeypatch, qapp, qtbot, record_property
):
    from here.application import HereApplicationController
    from here.config.settings import get_settings
    from here.ui.app import create_desktop
    from PySide6.QtCore import QSettings

    memory_type, _, _, _, _, _ = real_r52_composition_trace
    custom_data = tmp_path / "injected-custom-catalog"
    decoy_data = tmp_path / "unused-default-catalog"
    sessions = tmp_path / "configured-recordings"
    monkeypatch.setenv("HERE_DATA_DIR", str(decoy_data))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(sessions))
    configured = get_settings().TRANSCRIPTIONS_DIR
    assert configured == sessions
    memory = memory_type(custom_data)
    controller = HereApplicationController(memory_service=memory)
    desktop = create_desktop(
        qapp,
        controller,
        output_dir=configured,
        settings=QSettings(str(tmp_path / "injected.ini"), QSettings.Format.IniFormat),
    )
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    _record_constructor_context(real_r52_composition_trace, record_property)
    assert controller.memory_service is memory
    _assert_same_memory_context_before_jobs(
        controller, desktop, real_r52_composition_trace, custom_data, sessions
    )
    assert not (decoy_data / "library.sqlite3").exists()


def test_production_gui_registers_actual_settings_root_on_existing_memory_before_workers(
    real_r52_composition_trace, tmp_path, monkeypatch, qapp, qtbot, record_property
):
    from here.config.settings import get_settings
    from here.ui import gui

    custom_data = tmp_path / "configured-custom-catalog"
    sessions = tmp_path / "actual-configured-recordings"
    monkeypatch.setenv("HERE_DATA_DIR", str(custom_data))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(sessions))
    monkeypatch.setenv("HERE_SETTINGS_FILE", str(tmp_path / "production.ini"))
    assert get_settings().TRANSCRIPTIONS_DIR == sessions
    construct_controller = gui.create_default_controller
    controllers = []

    def observe_controller():
        controller = construct_controller()
        controllers.append(controller)
        return controller

    # Observe, but still call the real production factory/controller/backend.
    # Settings, QApplication, create_desktop and recovery remain real as well.
    monkeypatch.setattr(gui, "create_default_controller", observe_controller)
    desktop = gui.create_production_desktop()
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    _record_constructor_context(real_r52_composition_trace, record_property)
    assert len(controllers) == 1
    _assert_same_memory_context_before_jobs(
        controllers[0], desktop, real_r52_composition_trace, custom_data, sessions
    )
