import json

import pytest
from here.ui.preferences import VisualPreferences
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLineEdit, QPushButton, QSlider, QSpinBox


def test_env_editor_preserves_unknown_values_and_quotes(tmp_path, monkeypatch):
    from here.config.editor import read_configuration, save_configuration

    target = tmp_path / "config with spaces.env"
    target.write_text("# personal configuration\nOTHER_OPTION=keep\nOPENAI_API_KEY=old\n")
    monkeypatch.setenv("HERE_ENV_FILE", str(target))
    save_configuration({"OPENAI_API_KEY": "synthetic ' quoted", "CLEANUP_ENABLED": "false"})
    assert read_configuration()["OPENAI_API_KEY"] == "synthetic ' quoted"
    assert "OTHER_OPTION=keep" in target.read_text()
    assert "# personal configuration" in target.read_text()


def test_env_editor_rejects_invalid_input_before_replacement(tmp_path, monkeypatch):
    from here.config.editor import save_configuration

    target = tmp_path / "selected.env"
    target.write_text("OTHER_OPTION=keep\n")
    monkeypatch.setenv("HERE_ENV_FILE", str(target))
    previous = target.read_bytes()
    with pytest.raises(ValueError):
        save_configuration({"OPENAI_API_KEY": "key\nINJECTED=value"})
    assert target.read_bytes() == previous
    with pytest.raises(ValueError):
        save_configuration({"UNKNOWN": "value"})
    assert target.read_bytes() == previous


def test_env_editor_failed_replace_preserves_original(tmp_path, monkeypatch):
    from here.config import editor

    target = tmp_path / "selected.env"
    target.write_text("OPENAI_API_KEY=previous\n")
    monkeypatch.setenv("HERE_ENV_FILE", str(target))
    monkeypatch.setattr(
        editor.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("locked"))
    )
    with pytest.raises(OSError):
        editor.save_configuration({"OPENAI_API_KEY": "replacement"})
    assert target.read_text() == "OPENAI_API_KEY=previous\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["selected.env"]


def test_visual_size_and_opacity_persist_and_reach_overlay(tmp_path, qtbot):
    from here.ui.contract import ApplicationUiAdapter
    from here.ui.fake_controller import FakeApplicationController
    from here.ui.overlay import LiveLogoOverlay

    settings = QSettings(str(tmp_path / "visual.ini"), QSettings.Format.IniFormat)
    preferences = VisualPreferences(settings)
    overlay = LiveLogoOverlay(
        ApplicationUiAdapter(FakeApplicationController(), tmp_path), preferences
    )
    qtbot.addWidget(overlay)
    preferences.set_size(160)
    preferences.set_opacity(0.5)
    assert overlay.width() == overlay.height() == 160
    assert overlay.windowOpacity() == pytest.approx(0.5, abs=0.01)
    reloaded = VisualPreferences(settings)
    assert reloaded.size == 160
    assert reloaded.opacity == pytest.approx(0.5)


def test_corrupt_visual_preferences_remain_visible(tmp_path):
    settings = QSettings(str(tmp_path / "visual.ini"), QSettings.Format.IniFormat)
    settings.setValue("visual/size", "bad")
    settings.setValue("visual/opacity", "nan")
    preferences = VisualPreferences(settings)
    assert preferences.size == 104
    assert preferences.opacity == 1.0
    preferences.set_size(1)
    preferences.set_opacity(0)
    assert preferences.size >= 48
    assert preferences.opacity >= 0.2


def test_configuration_dialog_saves_masked_key_and_appearance(tmp_path, qtbot, monkeypatch):
    from here.config.editor import read_configuration
    from here.ui.configuration import ConfigurationDialog

    monkeypatch.delenv("OPENAI_API_KEY")
    target = tmp_path / "app.env"
    monkeypatch.setenv("HERE_ENV_FILE", str(target))
    preferences = VisualPreferences(
        QSettings(str(tmp_path / "visual.ini"), QSettings.Format.IniFormat)
    )
    dialog = ConfigurationDialog(preferences)
    qtbot.addWidget(dialog)
    key = dialog.findChild(QLineEdit, "apiKeyInput")
    assert key.echoMode() == QLineEdit.EchoMode.Password
    key.setText("synthetic-user-key")
    dialog.findChild(QSpinBox, "indicatorSizeInput").setValue(144)
    dialog.findChild(QSlider, "indicatorTransparencyInput").setValue(40)
    dialog.findChild(QPushButton, "saveConfigurationButton").click()
    assert read_configuration()["OPENAI_API_KEY"] == "synthetic-user-key"
    assert preferences.size == 144
    assert preferences.opacity == pytest.approx(0.6)
    assert dialog.result() == dialog.DialogCode.Accepted


def test_configuration_cancel_does_not_write_or_change_preferences(tmp_path, qtbot, monkeypatch):
    from here.ui.configuration import ConfigurationDialog

    target = tmp_path / "app.env"
    monkeypatch.setenv("HERE_ENV_FILE", str(target))
    preferences = VisualPreferences(
        QSettings(str(tmp_path / "visual.ini"), QSettings.Format.IniFormat)
    )
    dialog = ConfigurationDialog(preferences)
    qtbot.addWidget(dialog)
    dialog.findChild(QLineEdit, "apiKeyInput").setText("synthetic-discarded-key")
    dialog.findChild(QSpinBox, "indicatorSizeInput").setValue(144)
    dialog.reject()
    assert not target.exists()
    assert preferences.size == 104


def test_persistent_diagnostics_omit_secrets_and_session_content(tmp_path, monkeypatch):
    from here.diagnostics import record_error

    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-sensitive-key")
    record_error("startup", ValueError("synthetic-sensitive-key private transcript words"))
    lines = (tmp_path / "logs/here.log").read_text().splitlines()
    entry = json.loads(lines[-1])
    assert entry["stage"] == "startup"
    assert entry["error_type"] == "ValueError"
    assert "synthetic-sensitive-key" not in "\n".join(lines)
    assert "private transcript words" not in "\n".join(lines)


def test_missing_key_is_explained_before_recording_and_logged(tmp_path, qtbot, monkeypatch):
    from here.ui.contract import ApplicationUiAdapter
    from here.ui.fake_controller import FakeApplicationController
    from here.ui.main_window import MainWindow

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path))
    core = FakeApplicationController()
    preferences = VisualPreferences(
        QSettings(str(tmp_path / "visual.ini"), QSettings.Format.IniFormat)
    )
    window = MainWindow(ApplicationUiAdapter(core, tmp_path), preferences)
    qtbot.addWidget(window)
    window.findChild(QPushButton, "startButton").click()
    assert core.command_log == []
    detail = window.findChild(object, "detailLabel").text()
    assert "Configuración" in detail
    assert "grabación" in detail
    assert window.findChild(QPushButton, "configurationButton").isEnabled()
    assert "missing_api_key" in (tmp_path / "logs/here.log").read_text()


def test_diagnostic_logging_failure_does_not_hide_original_error(tmp_path, monkeypatch):
    from here import diagnostics

    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path))
    (tmp_path / "logs").write_text("not a directory")
    assert diagnostics.record_error("configuration", OSError("locked")) is False


@pytest.mark.usefixtures("owned_desktops")
def test_background_error_restores_window_and_explains_saved_audio(tmp_path, qtbot):
    from here.application import ApplicationEvent, ApplicationState, EventKind
    from here.ui.app import HereDesktop
    from here.ui.contract import ApplicationUiAdapter
    from here.ui.fake_controller import FakeApplicationController
    from PySide6.QtWidgets import QApplication

    core = FakeApplicationController()
    settings = QSettings(str(tmp_path / "visual.ini"), QSettings.Format.IniFormat)
    desktop = HereDesktop(
        QApplication.instance(), ApplicationUiAdapter(core, tmp_path), settings=settings
    )
    desktop.show()
    desktop.main_window.hide()
    core.fail("connection failed")
    core.publish(ApplicationEvent(kind=EventKind.ERROR_RECORDED, state=ApplicationState.FAILED))
    qtbot.waitUntil(desktop.main_window.isVisible)
    assert "audio está guardado" in desktop.main_window.findChild(object, "detailLabel").text()


def test_saved_configuration_refreshes_next_recording_destination(tmp_path, monkeypatch):
    from here.config.editor import save_configuration
    from here.ui.contract import ApplicationUiAdapter
    from here.ui.fake_controller import FakeApplicationController

    monkeypatch.delenv("TRANSCRIPTIONS_DIR", raising=False)
    adapter = ApplicationUiAdapter(FakeApplicationController(), tmp_path / "old")
    save_configuration({"TRANSCRIPTIONS_DIR": str(tmp_path / "new")})
    adapter.refresh_configuration()
    adapter.start_recording()
    assert adapter._controller.last_request.output_dir == tmp_path / "new"


def test_folder_change_routes_discovery_to_new_root(tmp_path, monkeypatch):
    from here.config.editor import save_configuration
    from here.ui import contract
    from here.ui.fake_controller import FakeApplicationController

    monkeypatch.delenv("TRANSCRIPTIONS_DIR", raising=False)
    monkeypatch.setattr(contract.RecoveryService, "discover", lambda service: [service.root])
    adapter = contract.ApplicationUiAdapter(FakeApplicationController(), tmp_path / "old")
    save_configuration({"TRANSCRIPTIONS_DIR": str(tmp_path / "new")})
    adapter.refresh_configuration()
    assert adapter.discover_recovery() == [tmp_path / "new"]


def test_configuration_refresh_refuses_active_work(tmp_path):
    from here.ui.contract import ApplicationUiAdapter
    from here.ui.fake_controller import FakeApplicationController

    adapter = ApplicationUiAdapter(FakeApplicationController(), tmp_path)
    adapter.start_recording()
    with pytest.raises(RuntimeError):
        adapter.refresh_configuration()


@pytest.mark.usefixtures("owned_desktops")
def test_saved_folder_refreshes_visible_recovery_choices(tmp_path, qtbot, monkeypatch):
    from here.application.recovery import RecoveryCandidate, RecoveryService
    from here.config.editor import save_configuration
    from here.ui.app import HereDesktop
    from here.ui.configuration import ConfigurationDialog
    from here.ui.contract import ApplicationUiAdapter
    from here.ui.fake_controller import FakeApplicationController
    from PySide6.QtWidgets import QApplication

    monkeypatch.delenv("TRANSCRIPTIONS_DIR", raising=False)
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    roots = []

    def discover(service):
        roots.append(service.root)
        return [
            RecoveryCandidate(service.root / "session", service.root.name, 1, "failed", None, True)
        ]

    monkeypatch.setattr(RecoveryService, "discover", discover)
    adapter = ApplicationUiAdapter(FakeApplicationController(), old_root)
    settings = QSettings(str(tmp_path / "visual.ini"), QSettings.Format.IniFormat)
    desktop = HereDesktop(QApplication.instance(), adapter, settings=settings)
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    selector = desktop.main_window.findChild(object, "recoverySelector")
    assert selector.currentData().display_id == "old"
    old_candidate = selector.currentData()

    def accept(dialog):
        save_configuration({"TRANSCRIPTIONS_DIR": str(new_root)})
        return dialog.DialogCode.Accepted

    monkeypatch.setattr(ConfigurationDialog, "exec", accept)
    desktop.main_window.findChild(QPushButton, "configurationButton").click()
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert roots[-1] == new_root
    assert selector.currentData().display_id == "new"
    from here.output.paths import UnsafeSessionPath

    with pytest.raises(UnsafeSessionPath):
        adapter.retry_candidate(old_candidate)
    materialized = []

    def materialize(service, candidate):
        materialized.append((service.root, candidate.session_dir))
        return candidate.session_dir

    monkeypatch.setattr(RecoveryService, "materialize", materialize)
    desktop.main_window.findChild(QPushButton, "recoverButton").click()
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert materialized == [(new_root, new_root / "session")]
    assert adapter._controller.command_log == ["retry"]
