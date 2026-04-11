from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

import here.cli as cli_module


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

    monkeypatch.setattr(cli_module, "transcribe_recording_session", lambda recorded_session: SimpleNamespace(final_text="hola"))
    monkeypatch.setattr(cli_module, "datetime", _FrozenDateTime)

    cli_module._save_transcription(session, tmp_path)

    output_file = tmp_path / "20260410_220000.txt"
    assert output_file.exists()
    assert output_file.read_text(encoding=cli_module.TRANSCRIPT_ENCODING) == "hola"
    assert session.cleaned


def test_save_transcription_cleans_up_even_when_transcription_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session = _FakeSession()
    monkeypatch.setattr(
        cli_module,
        "transcribe_recording_session",
        lambda recorded_session: (_ for _ in ()).throw(RuntimeError("boom")),
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

    def _run_recording(capture_fn: object, target_dir: Path) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)
    monkeypatch.setattr(cli_module, "get_settings", lambda: SimpleNamespace(TRANSCRIPTIONS_DIR=tmp_path))

    cli_module.record_main(SimpleNamespace(invoked_subcommand=None), None)

    assert captured["capture_fn"] is cli_module.record_both_until_enter
    assert captured["target_dir"] == tmp_path


def test_record_main_returns_early_when_subcommand_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "_run_recording",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    cli_module.record_main(SimpleNamespace(invoked_subcommand="mic"), None)


def test_mic_command_uses_explicit_output_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _run_recording(capture_fn: object, target_dir: Path) -> None:
        captured["capture_fn"] = capture_fn
        captured["target_dir"] = target_dir

    monkeypatch.setattr(cli_module, "_run_recording", _run_recording)

    cli_module.mic(tmp_path)

    assert captured["capture_fn"] is cli_module.record_mic_until_enter
    assert captured["target_dir"] == tmp_path
