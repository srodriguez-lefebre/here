from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import here.cli as cli_module
from here.recording.diagnostics import AudioDeviceInfo, SignalTestResult

runner = CliRunner()


class _FakeSource:
    label = "microphone"
    device_name = "Asterisk Nova"
    sample_rate = 16000
    channels = 1
    frames = 32000

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


class _FakeSession:
    def __init__(self) -> None:
        self.cleaned = False
        self.sources = [_FakeSource()]

    @property
    def duration_seconds(self) -> float:
        return max(source.duration_seconds for source in self.sources)

    def cleanup(self) -> None:
        self.cleaned = True


class _FrozenDateTime:
    @staticmethod
    def now() -> datetime:
        return datetime(2026, 4, 10, 22, 0, 0)


class _SequentialDateTime:
    calls = 0

    @classmethod
    def now(cls) -> datetime:
        cls.calls += 1
        if cls.calls == 1:
            return datetime(2026, 4, 10, 22, 0, 0)
        return datetime(2026, 4, 10, 22, 5, 0)


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

    output_dir = tmp_path / "20260410_220000"
    output_file = output_dir / "transcript.txt"
    metadata_file = output_dir / "session.json"
    markdown_file = output_dir / "transcript.md"
    chunks_file = output_dir / "chunks.json"
    assert output_file.read_text(encoding=cli_module.TRANSCRIPT_ENCODING) == "hola"
    assert metadata_file.exists()
    assert markdown_file.exists()
    assert chunks_file.exists()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["session_id"] == "20260410_220000"
    assert metadata["duration_seconds"] == 2.0
    assert metadata["sources"][0]["label"] == "microphone"
    assert metadata["sources"][0]["device_name"] == "Asterisk Nova"
    assert metadata["transcription_model"] == "gpt-4o-transcribe-diarize"
    assert metadata["alt_model_used"] is False
    assert metadata["live_pipeline_attempted"] is False
    assert metadata["fallback_used"] is False
    assert json.loads(chunks_file.read_text(encoding="utf-8")) == {"schema_version": 1, "chunks": []}
    markdown = markdown_file.read_text(encoding="utf-8")
    assert "# Recording 2026-04-10 22:00" in markdown
    assert "- Session ID: `20260410_220000`" in markdown
    assert session.cleaned
    assert captured["use_alt_transcription_model"] is False


def test_save_transcription_uses_recording_completion_time_for_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session = _FakeSession()
    _SequentialDateTime.calls = 0

    def _transcribe_recording_session(recorded_session: object, **kwargs: object) -> SimpleNamespace:
        del recorded_session, kwargs
        cli_module.datetime.now()
        return SimpleNamespace(final_text="hola")

    monkeypatch.setattr(cli_module, "transcribe_recording_session", _transcribe_recording_session)
    monkeypatch.setattr(cli_module, "datetime", _SequentialDateTime)

    cli_module._save_transcription(session, tmp_path)

    assert (tmp_path / "20260410_220000" / "session.json").exists()
    assert not (tmp_path / "20260410_220500").exists()


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


def test_devices_command_prints_windows_audio_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "get_windows_audio_devices",
        lambda: [
            AudioDeviceInfo(
                source="microphone",
                name="Mic",
                index=1,
                sample_rate=48000,
                channels=1,
            ),
            AudioDeviceInfo(
                source="system audio",
                name="Speakers",
                index=2,
                sample_rate=48000,
                channels=2,
            ),
        ],
    )

    result = runner.invoke(cli_module.app, ["devices"])

    assert result.exit_code == 0
    assert "microphone: Mic" in result.output
    assert "system audio: Speakers" in result.output


def test_audio_test_command_prints_signal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "test_windows_audio_signal",
        lambda source, *, duration_seconds: SignalTestResult(
            source=source,
            device=AudioDeviceInfo(
                source=source,
                name="Mic",
                index=1,
                sample_rate=48000,
                channels=1,
            ),
            duration_seconds=duration_seconds,
            peak=0.25,
            rms=0.1,
            has_signal=True,
        ),
    )

    result = runner.invoke(cli_module.app, ["test", "mic", "--duration", "0.5"])

    assert result.exit_code == 0
    assert "peak=0.2500" in result.output
    assert "status=signal detected" in result.output


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
