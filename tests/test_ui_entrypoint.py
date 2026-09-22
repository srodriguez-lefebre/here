from pathlib import Path
from types import SimpleNamespace

import here.ui.gui as gui_module
from here.ui.fake_controller import FakeApplicationController


def test_production_entrypoint_composes_real_controller_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    controller = FakeApplicationController()
    application = object()
    expected_desktop = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(gui_module, "application_instance", lambda _args: application)
    monkeypatch.setattr(gui_module, "create_default_controller", lambda: controller)
    monkeypatch.setattr(
        gui_module,
        "get_settings",
        lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path),
    )

    def create_desktop(app, core, *, output_dir):
        captured.update(app=app, core=core, output_dir=output_dir)
        return expected_desktop

    monkeypatch.setattr(gui_module, "create_desktop", create_desktop)

    assert gui_module.create_production_desktop() is expected_desktop
    assert captured == {
        "app": application,
        "core": controller,
        "output_dir": tmp_path,
    }
