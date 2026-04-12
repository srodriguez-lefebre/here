from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import here.cli as cli_module

runner = CliRunner()


class _FakeSession:
    def __init__(self) -> None:
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class _FrozenDateTime:
    @staticmethod
    def now() -> SimpleNamespace:
        return SimpleNamespace(strftime=lambda fmt: "20260410_220000")


class _FakeLiveController:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.aborted = False
        self.cleaned = False

    def complete(self) -> object:
        if self.error is not None:
            raise self.error
        return self.result

    def abort(self) -> None:
        self.aborted = True

    def cleanup(self) -> None:
        self.cleaned = True


def test_save_transcription_writes_file_and_cleans_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = _FakeSession()
    captured: dict[str, object] = {}

    def _transcribe_recording_session(recorded_session: object, **kwargs: object) -> SimpleNamespace:
        del recorded_session
        captured.update(kwargs)
        return SimpleNamespace(final_text="hola")

    monkeypatch.setattr(cli_module, "transcribe_recording_session", _transcribe_recording_session)
    monkeypatch.setattr(cli_module, "datetime", _FrozenDateTime)

    cli_module._save_transcription(session, tmp_path)

    output_file = tmp_path / "20260410_220000.txt"
    assert output_file.exists()
    assert output_file.read_text(encoding=cli_module.TRANSCRIPT_ENCODING) == "hola"
    assert session.cleaned
    assert captured["use_alt_transcription_model"] is False


def test_save_transcription_can_use_alt_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = _FakeSession()
    captured: dict[str, object] = {}

    def _transcribe_recording_session(recorded_session: object, **kwargs: object) -> SimpleNamespace:
        del recorded_session
        captured.update(kwargs)
        return SimpleNamespace(final_text="hola")

    monkeypatch.setattr(cli_module, "transcribe_recording_session", _transcribe_recording_session)
    monkeypatch.setattr(cli_module, "datetime", _FrozenDateTime)

    cli_module._save_transcription(session, tmp_path, use_alt_transcription_model=True)

    assert captured["use_alt_transcription_model"] is True


def test_save_transcription_preserves_audio_when_transcription_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session = _FakeSession()
    monkeypatch.setattr(
        cli_module,
        "transcribe_recording_session",
        lambda recorded_session, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        cli_module._save_transcription(session, tmp_path)

    assert not session.cleaned


def test_transcribe_session_prefers_live_result() -> None:
    live_controller = _FakeLiveController(result=SimpleNamespace(final_text="live"))

    result = cli_module._transcribe_session(
        _FakeSession(),
        live_controller=live_controller,
    )

    assert result.final_text == "live"


def test_transcribe_session_falls_back_to_offline_when_live_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    live_controller = _FakeLiveController(error=RuntimeError("live failed"))
    captured: dict[str, object] = {}

    def _transcribe_recording_session(recorded_session: object, **kwargs: object) -> SimpleNamespace:
        del recorded_session
        captured.update(kwargs)
        return SimpleNamespace(final_text="offline")

    monkeypatch.setattr(cli_module, "transcribe_recording_session", _transcribe_recording_session)

    result = cli_module._transcribe_session(
        _FakeSession(),
        use_alt_transcription_model=True,
        live_controller=live_controller,
    )

    assert result.final_text == "offline"
    assert captured["use_alt_transcription_model"] is True


def test_run_recording_wraps_runtime_errors_as_typer_exit(tmp_path: Path) -> None:
    class _Controller:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.aborted = False
            self.cleaned = False

        def submit_block(self, *args: object) -> None:
            del args

        def abort(self) -> None:
            self.aborted = True

        def cleanup(self) -> None:
            self.cleaned = True

    controller = _Controller()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli_module, "LiveTranscriptionController", lambda **kwargs: controller)
    with pytest.raises(typer.Exit) as exc_info:
        cli_module._run_recording(
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("broken")),
            tmp_path,
            expected_source_count=1,
        )
    monkeypatch.undo()

    assert exc_info.value.exit_code == 1
    assert controller.aborted
    assert controller.cleaned


def test_record_main_uses_settings_directory_when_no_subcommand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(
        capture_fn: object,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
        expected_source_count: int,
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model
        captured["expected_source_count"] = expected_source_count

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    cli_module.record_main(SimpleNamespace(invoked_subcommand=None), None)

    assert captured["capture_fn"] is cli_module.record_both_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is False
    assert captured["expected_source_count"] == 2


def test_record_main_returns_early_when_subcommand_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "_run_recording",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    cli_module.record_main(SimpleNamespace(invoked_subcommand="alt"), None)


def test_record_alt_command_uses_alt_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(
        capture_fn: object,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
        expected_source_count: int,
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model
        captured["expected_source_count"] = expected_source_count

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    result = runner.invoke(cli_module.app, ["record", "alt"])

    assert result.exit_code == 0
    assert captured["capture_fn"] is cli_module.record_both_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is True
    assert captured["expected_source_count"] == 2


def test_record_mic_command_uses_default_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(
        capture_fn: object,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
        expected_source_count: int,
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model
        captured["expected_source_count"] = expected_source_count

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    result = runner.invoke(cli_module.app, ["record", "mic"])

    assert result.exit_code == 0
    assert captured["capture_fn"] is cli_module.record_mic_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is False
    assert captured["expected_source_count"] == 1


def test_record_mic_alt_command_uses_alt_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(
        capture_fn: object,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
        expected_source_count: int,
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model
        captured["expected_source_count"] = expected_source_count

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    result = runner.invoke(cli_module.app, ["record", "mic", "alt"])

    assert result.exit_code == 0
    assert captured["capture_fn"] is cli_module.record_mic_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is True
    assert captured["expected_source_count"] == 1


def test_record_os_alt_command_uses_alt_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(
        capture_fn: object,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
        expected_source_count: int,
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model
        captured["expected_source_count"] = expected_source_count

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    result = runner.invoke(cli_module.app, ["record", "os", "alt"])

    assert result.exit_code == 0
    assert captured["capture_fn"] is cli_module.record_os_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is True
    assert captured["expected_source_count"] == 1
