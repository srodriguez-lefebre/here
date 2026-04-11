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


def test_save_transcription_cleans_up_even_when_transcription_fails(
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

    assert session.cleaned


def test_run_recording_wraps_runtime_errors_as_typer_exit(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        cli_module._run_recording(lambda: (_ for _ in ()).throw(RuntimeError("broken")), tmp_path)

    assert exc_info.value.exit_code == 1


def test_record_main_uses_settings_directory_when_no_subcommand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(
        capture_fn: object,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    cli_module.record_main(SimpleNamespace(invoked_subcommand=None), None)

    assert captured["capture_fn"] is cli_module.record_both_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is False


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
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    result = runner.invoke(cli_module.app, ["record", "alt"])

    assert result.exit_code == 0
    assert captured["capture_fn"] is cli_module.record_both_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is True


def test_record_mic_command_uses_default_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(
        capture_fn: object,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    result = runner.invoke(cli_module.app, ["record", "mic"])

    assert result.exit_code == 0
    assert captured["capture_fn"] is cli_module.record_mic_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is False


def test_record_mic_alt_command_uses_alt_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(
        capture_fn: object,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    result = runner.invoke(cli_module.app, ["record", "mic", "alt"])

    assert result.exit_code == 0
    assert captured["capture_fn"] is cli_module.record_mic_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is True


def test_record_os_alt_command_uses_alt_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(
        capture_fn: object,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
    ) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir
        captured["use_alt_transcription_model"] = use_alt_transcription_model

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    result = runner.invoke(cli_module.app, ["record", "os", "alt"])

    assert result.exit_code == 0
    assert captured["capture_fn"] is cli_module.record_os_until_enter
    assert captured["target_dir"] == tmp_path
    assert captured["use_alt_transcription_model"] is True
