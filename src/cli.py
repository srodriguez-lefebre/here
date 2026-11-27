import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Annotated, NoReturn, cast
from uuid import UUID

import typer
from here.application import (
    ApplicationState,
    InvalidApplicationCommand,
    SourceMode,
    StartRequest,
    create_default_controller,
)
from here.application.processing import SessionProcessor, selected_retry
from here.application.recovery import RecoveryService
from here.audio.mix import materialize_normalized_session
from here.config.paths import get_data_dir
from here.config.settings import get_settings, settings_operation
from here.live_processing import LiveTranscriptionController
from here.memory.database import LibraryError, open_library
from here.memory.models import Citation, MeetingPatch, MeetingQuery, MeetingStatus, SearchQuery
from here.memory.portability import InvalidPackage, PortabilityService
from here.memory.search import SearchService
from here.memory.service import Conflict, InvalidInput, MemoryService, NotFound
from here.output.metadata import (
    ChunkMetadata,
    ErrorMetadata,
)
from here.output.paths import UnsafeSessionPath
from here.output.session_writer import (
    TRANSCRIPT_ENCODING as TRANSCRIPT_ENCODING,
)
from here.recorder import (
    RecordingSession,
    record_both_until_enter,
    record_mic_until_enter,
    record_os_until_enter,
)
from here.recording.diagnostics import (
    AudioDeviceInfo,
    SignalTestResult,
    get_windows_audio_devices,
    test_windows_audio_signal,
)
from here.transcriber import transcribe_recording_session
from here.transcription.client import TranscriptionResult
from loguru import logger

app = typer.Typer()
record_app = typer.Typer(invoke_without_command=True)
mic_app = typer.Typer(invoke_without_command=True)
os_app = typer.Typer(invoke_without_command=True)
test_app = typer.Typer()
app.add_typer(record_app, name="record")
app.add_typer(test_app, name="test")
memory_app = typer.Typer(help="Maintain the local meeting catalog without a provider.")
app.add_typer(memory_app, name="memory")
meetings_app = typer.Typer(help="Browse and manage the local meeting library.")
app.add_typer(meetings_app, name="meetings")
record_app.add_typer(mic_app, name="mic")
record_app.add_typer(os_app, name="os")
OutputDirOption = Annotated[
    Path | None,
    typer.Option(
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
]
AudioFileArgument = Annotated[
    Path,
    typer.Argument(help="Audio file to transcribe."),
]


def _memory_output(value: object, json_output: bool) -> None:
    def encode(item):
        return asdict(item) if is_dataclass(item) and not isinstance(item, type) else str(item)

    serialized = json.dumps(value, default=encode, ensure_ascii=False)
    if json_output:
        typer.echo(serialized)
    else:
        logger.info(serialized)


def _memory_error(error: Exception, json_output: bool, code: int) -> None:
    _memory_output({"error": str(error)}, json_output)
    raise typer.Exit(code)


@memory_app.command("ingest")
def memory_ingest(path: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    if not path.is_dir():
        _memory_error(ValueError("Session directory does not exist"), json_output, 2)
    try:
        result = MemoryService(get_data_dir()).ingest_session(path)
    except (LibraryError, OSError) as exc:
        _memory_error(exc, json_output, 4)
        return
    _memory_output(asdict(result), json_output)
    if result.status == "rejected":
        raise typer.Exit({"invalid_input": 2, "io": 4}.get(result.error_category or "conflict", 3))


@memory_app.command("reconcile")
@settings_operation
def memory_reconcile(json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        report = MemoryService(
            get_data_dir(), sessions_root=get_settings().TRANSCRIPTIONS_DIR
        ).reconcile()
    except (LibraryError, OSError) as exc:
        _memory_lifecycle_error(exc, json_output)
        return
    _memory_output(asdict(report), json_output)


@memory_app.command("show")
def memory_show(meeting_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        if str(UUID(meeting_id)) != meeting_id:
            raise ValueError("Use a canonical meeting UUID")
    except ValueError as exc:
        _memory_error(exc, json_output, 2)
    try:
        result = MemoryService(get_data_dir()).show(meeting_id)
    except (LibraryError, OSError) as exc:
        _memory_error(exc, json_output, 4)
        return
    if result is None:
        _memory_error(ValueError("Meeting does not exist"), json_output, 3)
    _memory_output(result, json_output)


@memory_app.command("delete")
@settings_operation
def memory_delete(
    meeting_id: str,
    yes: bool = typer.Option(
        False, "--yes", help="Confirm removal of the managed meeting and derivatives."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if not yes:
        _memory_lifecycle_error(InvalidInput("Deletion requires --yes"), json_output)
    try:
        result = MemoryService(
            get_data_dir(), sessions_root=get_settings().TRANSCRIPTIONS_DIR
        ).delete_meeting(meeting_id)
    except (LibraryError, OSError) as exc:
        _memory_lifecycle_error(exc, json_output)
        return
    _memory_output(result, json_output)
    if result.status != "completed":
        raise typer.Exit({"invalid_input": 2, "conflict": 3}.get(result.error_category or "io", 4))


@memory_app.command("reindex")
def memory_reindex(json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        result = MemoryService(get_data_dir()).reindex()
    except (LibraryError, OSError) as exc:
        _memory_lifecycle_error(exc, json_output)
        return
    _memory_output(result, json_output)


def _memory_lifecycle_error(error: Exception, json_output: bool) -> None:
    if isinstance(error, InvalidInput):
        category, code = "invalid_input", 2
    elif isinstance(error, (Conflict, NotFound)):
        category, code = "conflict", 3
    else:
        category, code = "io", 4
    _memory_output({"error": str(error), "error_category": category}, json_output)
    raise typer.Exit(code)


@memory_app.command("rename")
def memory_rename(
    meeting_id: str, title: str, json_output: bool = typer.Option(False, "--json")
) -> None:
    try:
        MemoryService._identity(meeting_id)
        memory = MemoryService(get_data_dir())
        shown = memory.show(meeting_id)
        if shown is None:
            raise NotFound("Meeting does not exist")
        meeting = cast(dict[str, object], shown["meeting"])
        result = memory.update_meeting(
            meeting_id,
            MeetingPatch(title=title),
            expected_version=cast(int, meeting["row_version"]),
        )
    except (LibraryError, OSError) as exc:
        _memory_lifecycle_error(exc, json_output)
        return
    _memory_output(result, json_output)


@memory_app.command("participant")
def memory_participant(
    meeting_id: str,
    display_name: str,
    speaker: str | None = typer.Option(None, "--speaker"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        result = MemoryService(get_data_dir()).add_meeting_participant(
            meeting_id, display_name, speaker_key=speaker
        )
    except (LibraryError, OSError) as exc:
        _memory_lifecycle_error(exc, json_output)
        return
    _memory_output(result, json_output)


class ExitCode(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 2
    CONFLICT = 3
    IO = 4
    PROVIDER = 5


_RESULT_EXIT = {
    "invalid_input": ExitCode.INVALID_INPUT,
    "conflict": ExitCode.CONFLICT,
    "io": ExitCode.IO,
}


def emit(result: object, json: bool) -> None:
    _memory_output(result, json)


def _meetings_failure(
    message: str, category: str, json_output: bool, **details: object
) -> NoReturn:
    emit({"error": message, "error_category": category, **details}, json_output)
    code = ExitCode.PROVIDER if category == "provider" else _RESULT_EXIT.get(category, ExitCode.IO)
    raise typer.Exit(code)


@contextmanager
def _meetings_errors(json_output: bool) -> Iterator[None]:
    try:
        yield
    except InvalidInput as exc:
        _meetings_failure(str(exc), "invalid_input", json_output, reason=exc.reason)
    except (Conflict, NotFound, InvalidApplicationCommand) as exc:
        _meetings_failure(str(exc), "conflict", json_output)
    except (LibraryError, OSError, UnsafeSessionPath) as exc:
        _meetings_failure(str(exc), "io", json_output)


def _resolve_meeting(memory: MemoryService, identity: str, json_output: bool) -> str:
    prefix = identity.lower()
    if (
        not prefix
        or len(prefix) > 36
        or any(
            (char != "-" if index in (8, 13, 18, 23) else char not in "0123456789abcdef")
            for index, char in enumerate(prefix)
        )
    ):
        raise InvalidInput("Use a meeting UUID or an unambiguous UUID prefix")
    matches = memory.meeting_ids_with_prefix(prefix)
    if not matches:
        raise NotFound("Meeting does not exist")
    if len(matches) > 1:
        _meetings_failure(
            "Meeting prefix is ambiguous",
            "invalid_input",
            json_output,
            reason="ambiguous_meeting_id",
            candidates=matches,
        )
    return matches[0]


def _shown_meeting(memory: MemoryService, identity: str) -> dict[str, object]:
    detail = memory.show(identity)
    if detail is None:
        raise NotFound("Meeting does not exist")
    return detail


def _meeting_search(memory: MemoryService) -> SearchService:
    return SearchService(lambda: open_library(memory.data_dir), reindexer=memory.reindex)


JsonOption = Annotated[bool, typer.Option("--json", help="Write stable JSON to stdout.")]


@meetings_app.command("list")
def meetings_list(
    from_utc: Annotated[str | None, typer.Option("--from")] = None,
    to_utc: Annotated[str | None, typer.Option("--to")] = None,
    status: Annotated[list[str] | None, typer.Option("--status")] = None,
    participant: Annotated[str | None, typer.Option("--participant")] = None,
    json_output: JsonOption = False,
) -> None:
    """List meetings with optional date, status and participant filters."""
    with _meetings_errors(json_output):
        try:
            statuses = tuple(MeetingStatus(value) for value in status or ())
        except ValueError as exc:
            raise InvalidInput("Unknown meeting status") from exc
        memory = MemoryService(get_data_dir())
        emit(
            _meeting_search(memory).list_meetings(
                MeetingQuery(
                    from_utc=from_utc,
                    to_utc=to_utc,
                    statuses=statuses,
                    participant_id=participant,
                )
            ),
            json_output,
        )


@meetings_app.command("show")
def meetings_show(meeting_id: str, json_output: JsonOption = False) -> None:
    """Show a meeting and its retained revisions."""
    with _meetings_errors(json_output):
        memory = MemoryService(get_data_dir())
        emit(_shown_meeting(memory, _resolve_meeting(memory, meeting_id, json_output)), json_output)


@meetings_app.command("search")
def meetings_search(
    text: str,
    meeting: Annotated[str | None, typer.Option("--meeting")] = None,
    json_output: JsonOption = False,
) -> None:
    """Search transcript segments, optionally within one meeting."""
    with _meetings_errors(json_output):
        memory = MemoryService(get_data_dir())
        identity = _resolve_meeting(memory, meeting, json_output) if meeting is not None else None
        emit(
            _meeting_search(memory).search(SearchQuery(text=text, meeting_id=identity)), json_output
        )


@meetings_app.command("rename")
def meetings_rename(meeting_id: str, title: str, json_output: JsonOption = False) -> None:
    """Rename a meeting."""
    with _meetings_errors(json_output):
        memory = MemoryService(get_data_dir())
        identity = _resolve_meeting(memory, meeting_id, json_output)
        meeting = cast(dict[str, object], _shown_meeting(memory, identity)["meeting"])
        emit(
            memory.update_meeting(
                identity,
                MeetingPatch(title=title),
                expected_version=cast(int, meeting["row_version"]),
            ),
            json_output,
        )


@meetings_app.command("participant")
def meetings_participant(
    meeting_id: str,
    display_name: str,
    speaker: Annotated[str | None, typer.Option("--speaker")] = None,
    json_output: JsonOption = False,
) -> None:
    """Add a participant and optionally assign an active-revision speaker."""
    with _meetings_errors(json_output):
        memory = MemoryService(get_data_dir())
        identity = _resolve_meeting(memory, meeting_id, json_output)
        emit(
            memory.add_meeting_participant(identity, display_name, speaker_key=speaker), json_output
        )


@meetings_app.command("retry")
@settings_operation
def meetings_retry(meeting_id: str, json_output: JsonOption = False) -> None:
    """Retry transcription for one recoverable meeting."""
    with _meetings_errors(json_output):
        root = get_settings().TRANSCRIPTIONS_DIR
        memory = MemoryService(get_data_dir(), sessions_root=root)
        identity = _resolve_meeting(memory, meeting_id, json_output)
        recovery = RecoveryService(root, memory_service=memory)
        candidate = recovery.find_retry_candidate(identity)
        if candidate is None or not candidate.can_retry:
            raise NotFound("No eligible recovery session exists for this meeting")
        try:
            session_dir = recovery.materialize(candidate)
        except ValueError as exc:
            raise LibraryError("Recovery material changed before retry") from exc
        controller = create_default_controller(memory_service=memory)
        with selected_retry(session_dir, identity, candidate.capture_id):
            controller.retry(session_dir)
        snapshot = controller.wait_until_terminal()
        if snapshot.state is not ApplicationState.COMPLETED or snapshot.memory_error_category:
            category = snapshot.memory_error_category or (
                snapshot.last_error.error_category if snapshot.last_error else "io"
            )
            _meetings_failure(
                "Retry could not complete",
                category,
                json_output,
                meeting_id=identity,
                status=snapshot.state.value,
            )
        emit({"meeting_id": identity, "status": snapshot.state.value}, json_output)


@meetings_app.command("play")
def meetings_play(
    meeting_id: str,
    segment: Annotated[str | None, typer.Option("--segment")] = None,
    json_output: JsonOption = False,
) -> None:
    """Open a player for the meeting or an exact transcript segment."""
    with _meetings_errors(json_output):
        memory = MemoryService(get_data_dir())
        identity = _resolve_meeting(memory, meeting_id, json_output)
        detail = _shown_meeting(memory, identity)
        meeting = cast(dict[str, object], detail["meeting"])
        segments = cast(list[dict[str, str]], detail["segments"])
        if segment is None:
            selected = next(
                (item for item in segments if item["revision_id"] == meeting["active_revision_id"]),
                None,
            )
        else:
            selected = next((item for item in segments if item["id"] == segment), None)
        if selected is None:
            raise NotFound("No matching transcript segment exists for playback")
        target = memory.resolve_citation(
            Citation(identity, selected["revision_id"], selected["id"])
        )
        emit(target, json_output)
        try:
            from here.ui.playback_widget import run_playback_window

            status = run_playback_window(target, seek_from_segment=segment is not None)
        except (ImportError, OSError, RuntimeError):
            status = ExitCode.IO
        if status:
            typer.echo("The multimedia player could not open or play the local audio.", err=True)
            raise typer.Exit(ExitCode.IO)


@meetings_app.command("import-session")
def meetings_import_session(path: Path, json_output: JsonOption = False) -> None:
    """Import an existing session into the local catalog."""
    with _meetings_errors(json_output):
        if not path.is_dir():
            raise InvalidInput("Session directory does not exist")
        result = MemoryService(get_data_dir()).ingest_session(path)
        emit(result, json_output)
        if result.status == "rejected":
            raise typer.Exit(_RESULT_EXIT.get(result.error_category or "", ExitCode.CONFLICT))


@meetings_app.command("export")
@settings_operation
def meetings_export(
    meeting_ids: Annotated[list[str], typer.Argument(metavar="ID...")],
    destination: Annotated[Path, typer.Option("--out")],
    no_audio: Annotated[bool, typer.Option("--no-audio")] = False,
) -> None:
    """Export selected meetings to a portable archive."""
    with _meetings_errors(False):
        memory = MemoryService(get_data_dir(), sessions_root=get_settings().TRANSCRIPTIONS_DIR)
        selection = tuple(_resolve_meeting(memory, identity, False) for identity in meeting_ids)
        if len(set(selection)) != len(selection):
            raise InvalidInput("Select each meeting once")
        result = PortabilityService(memory).export(
            selection,
            destination,
            include_audio=not no_audio,
            progress=lambda _: None,
            cancellation=lambda: False,
        )
        emit(result.model_dump(mode="json"), False)


@meetings_app.command("import")
@settings_operation
def meetings_import(path: Path, json_output: JsonOption = False) -> None:
    """Import a portable archive and report the result for each meeting."""
    with _meetings_errors(json_output):
        memory = MemoryService(get_data_dir(), sessions_root=get_settings().TRANSCRIPTIONS_DIR)
        try:
            report = PortabilityService(memory).import_archive(
                path, progress=lambda _: None, cancellation=lambda: False
            )
        except InvalidPackage as exc:
            _meetings_failure(str(exc), "invalid_input", json_output)

        emit(report, json_output)
        if (
            report.cleanup_pending
            or report.pending_recovery_ids
            or any(
                item.reason in {"local_io", "local_recovery_pending"} for item in report.meetings
            )
        ):
            raise typer.Exit(ExitCode.IO)
        if any(
            item.status == "conflict" or item.reason == "suppressed" for item in report.meetings
        ):
            raise typer.Exit(ExitCode.CONFLICT)
        if any(item.status == "rejected" for item in report.meetings):
            raise typer.Exit(ExitCode.IO)


@meetings_app.command("reindex")
def meetings_reindex(json_output: JsonOption = False) -> None:
    """Rebuild the local search index."""
    with _meetings_errors(json_output):
        memory = MemoryService(get_data_dir())
        emit(_meeting_search(memory).reindex(), json_output)


@meetings_app.command("delete")
@settings_operation
def meetings_delete(
    meeting_id: str,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    json_output: JsonOption = False,
) -> None:
    """Remove a managed meeting and its derivatives after confirmation."""
    with _meetings_errors(json_output):
        memory = MemoryService(get_data_dir(), sessions_root=get_settings().TRANSCRIPTIONS_DIR)
        # Only the delete operation can resume a previously accepted deletion.
        try:
            canonical = str(UUID(meeting_id)) == meeting_id
        except ValueError:
            canonical = False
        identity = (
            meeting_id if yes and canonical else _resolve_meeting(memory, meeting_id, json_output)
        )
        if not yes:
            meeting = cast(dict[str, object], _shown_meeting(memory, identity)["meeting"])
            _meetings_failure(
                "Deletion requires --yes to remove the managed meeting and its derivatives",
                "invalid_input",
                json_output,
                meeting_id=identity,
                title=meeting["title"],
            )
        result = memory.delete_meeting(identity)
        emit(result, json_output)
        if result.status != "completed":
            raise typer.Exit(_RESULT_EXIT.get(result.error_category or "", ExitCode.IO))


@dataclass(slots=True)
class _TranscriptionOutcome:
    result: TranscriptionResult
    live_pipeline_attempted: bool
    live_pipeline_used: bool
    fallback_used: bool
    errors: list[ErrorMetadata]


class _TranscriptionFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        chunks: list[ChunkMetadata],
        errors: list[ErrorMetadata],
        live_pipeline_attempted: bool,
        live_pipeline_used: bool,
        fallback_used: bool,
        failure_stage: str,
    ) -> None:
        super().__init__(message)
        self.chunks = chunks
        self.errors = errors
        self.live_pipeline_attempted = live_pipeline_attempted
        self.live_pipeline_used = live_pipeline_used
        self.fallback_used = fallback_used
        self.failure_stage = failure_stage


def _resolve_target_dir(output_dir: Path | None) -> Path:
    return output_dir or get_settings().TRANSCRIPTIONS_DIR


def _format_device_info(device: AudioDeviceInfo) -> str:
    return (
        f"{device.source}: {device.name} "
        f"(index={device.index}, {device.channels} channel(s), {device.sample_rate} Hz)"
    )


def _format_signal_result(result: SignalTestResult) -> str:
    signal = "signal detected" if result.has_signal else "no signal detected"
    return (
        f"{_format_device_info(result.device)}\n"
        f"duration={result.duration_seconds:.1f}s peak={result.peak:.4f} "
        f"rms={result.rms:.4f} status={signal}"
    )


def _run_audio_diagnostic(action: Callable[[], str]) -> None:
    try:
        typer.echo(action())
    except RuntimeError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc


@settings_operation
def _save_transcription(
    session: RecordingSession,
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
    live_controller: LiveTranscriptionController | None = None,
) -> None:
    processor = SessionProcessor(
        transcribe=transcribe_recording_session,
        normalize=materialize_normalized_session,
        clock=lambda: datetime.now().astimezone(),
        retry_delays=(),
    )
    artifacts = processor.process(
        session,
        target_dir,
        use_alt_transcription_model=use_alt_transcription_model,
        live_controller=live_controller,
    )
    logger.success("Saved to {path}", path=artifacts.session_dir)


@settings_operation
def _transcribe_audio_path(
    audio_path: Path,
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
) -> None:
    processor = SessionProcessor(
        transcribe=transcribe_recording_session,
        normalize=materialize_normalized_session,
        clock=lambda: datetime.now().astimezone(),
        retry_delays=(),
    )
    artifacts = processor.transcribe_audio_path(
        audio_path, target_dir, use_alt_transcription_model=use_alt_transcription_model
    )
    logger.success("Saved to {path}", path=artifacts.session_dir)


def _run_recording(
    capture_fn: Callable[..., RecordingSession],
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
    expected_source_count: int,
) -> None:
    del expected_source_count
    source_modes = {
        record_both_until_enter: SourceMode.BOTH,
        record_mic_until_enter: SourceMode.MICROPHONE,
        record_os_until_enter: SourceMode.SYSTEM_AUDIO,
    }
    source_mode = source_modes.get(capture_fn)
    if source_mode is None:
        raise ValueError("Unknown recording entry point")
    controller = create_default_controller()

    def finish_recording() -> None:
        try:
            if controller.snapshot.has_active_work:
                controller.cancel()
        finally:
            # A terminal state can precede live teardown and the completion observer.
            if not controller.snapshot.worker_complete:
                controller.wait_until_terminal()

    try:
        controller.start(
            StartRequest(
                output_dir=target_dir,
                source_mode=source_mode,
                use_alt_transcription_model=use_alt_transcription_model,
            )
        )
        while controller.snapshot.state is ApplicationState.PREPARING:
            time.sleep(0.01)
        if controller.snapshot.state is not ApplicationState.RECORDING:
            raise RuntimeError(
                controller.snapshot.last_error.message
                if controller.snapshot.last_error is not None
                else "Recording could not be started"
            )
        logger.info("Recording... Press Enter to stop.")
        input()
        controller.stop()
        snapshot = controller.wait_until_terminal()
        if snapshot.state is not ApplicationState.COMPLETED:
            raise RuntimeError(
                snapshot.last_error.message
                if snapshot.last_error is not None
                else f"Recording ended with state {snapshot.state.value}"
            )
        logger.success(
            "Saved to {path}",
            path=snapshot.session_dir,
        )
    except KeyboardInterrupt:
        try:
            finish_recording()
        except Exception:
            logger.exception("Failed to clean up the active recording")
        raise typer.Exit(code=130) from None
    except RuntimeError as exc:
        try:
            finish_recording()
        except Exception:
            logger.exception("Failed to clean up the active recording")
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        try:
            finish_recording()
        except Exception:
            logger.exception("Failed to clean up the active recording")
        logger.exception("Unexpected error while recording or transcribing audio")
        raise typer.Exit(code=1) from exc


def _transcribe_session(
    session: RecordingSession,
    *,
    use_alt_transcription_model: bool = False,
    live_controller: LiveTranscriptionController | None = None,
) -> TranscriptionResult:
    return _transcribe_session_outcome(
        session,
        use_alt_transcription_model=use_alt_transcription_model,
        live_controller=live_controller,
    ).result


def _transcribe_session_outcome(
    session: RecordingSession,
    *,
    use_alt_transcription_model: bool = False,
    live_controller: LiveTranscriptionController | None = None,
) -> _TranscriptionOutcome:
    failed_live_chunks: list[ChunkMetadata] = []
    errors: list[ErrorMetadata] = []
    if live_controller is not None:
        try:
            logger.info("Completing live transcription from background chunks.")
            return _TranscriptionOutcome(
                result=live_controller.complete(),
                live_pipeline_attempted=True,
                live_pipeline_used=True,
                fallback_used=False,
                errors=[],
            )
        except Exception as exc:
            logger.warning(
                "Live transcription failed: {exc}. Falling back to offline post-processing.",
                exc=exc,
            )
            errors.append(_error_metadata("live_transcription", exc))
            failed_live_chunks = _chunk_metadata_from_controller(live_controller)

    try:
        result = transcribe_recording_session(
            session,
            use_alt_transcription_model=use_alt_transcription_model,
        )
    except Exception as exc:
        offline_chunks = getattr(exc, "chunks", [])
        errors.append(_error_metadata("offline_transcription", exc))
        raise _TranscriptionFailure(
            "Transcription failed",
            chunks=failed_live_chunks + list(offline_chunks),
            errors=errors,
            live_pipeline_attempted=live_controller is not None,
            live_pipeline_used=False,
            fallback_used=live_controller is not None,
            failure_stage="offline_transcription",
        ) from exc

    if failed_live_chunks:
        result.chunks = failed_live_chunks + list(getattr(result, "chunks", []))

    return _TranscriptionOutcome(
        result=result,
        live_pipeline_attempted=live_controller is not None,
        live_pipeline_used=False,
        fallback_used=live_controller is not None,
        errors=errors,
    )


def _chunk_metadata_from_controller(
    live_controller: LiveTranscriptionController,
) -> list[ChunkMetadata]:
    chunk_metadata = getattr(live_controller, "chunk_metadata", None)
    if not callable(chunk_metadata):
        return []
    return list(chunk_metadata())


def _error_metadata(stage: str, exc: Exception) -> ErrorMetadata:
    cause = exc.__cause__ or exc.__context__
    return ErrorMetadata(
        stage=stage,
        type=type(exc).__name__,
        message=str(exc),
        cause_type=type(cause).__name__ if cause is not None else None,
        cause_message=str(cause) if cause is not None else None,
        retryable=True,
        occurred_at=datetime.now().astimezone(),
    )


@app.callback()
def main() -> None:
    """here - record audio and transcribe it."""


@app.command("devices")
def devices() -> None:
    """Show the Windows audio devices used by here."""

    def _show_devices() -> str:
        return "\n".join(_format_device_info(device) for device in get_windows_audio_devices())

    _run_audio_diagnostic(_show_devices)


@app.command("trans")
def trans(
    audio_file: AudioFileArgument,
    output_dir: OutputDirOption = None,
) -> None:
    """Transcribe an existing audio file."""
    try:
        _transcribe_audio_path(audio_file, _resolve_target_dir(output_dir))
    except RuntimeError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception("Unexpected error while transcribing audio")
        raise typer.Exit(code=1) from exc


@test_app.command("mic")
def test_mic(
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", help="Seconds to measure microphone signal."),
    ] = 3.0,
) -> None:
    """Measure signal from the default Windows microphone."""

    _run_audio_diagnostic(
        lambda: _format_signal_result(
            test_windows_audio_signal("microphone", duration_seconds=duration)
        )
    )


@test_app.command("os")
def test_os(
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", help="Seconds to measure system audio signal."),
    ] = 3.0,
) -> None:
    """Measure signal from the default Windows WASAPI loopback device."""

    _run_audio_diagnostic(
        lambda: _format_signal_result(
            test_windows_audio_signal("system audio", duration_seconds=duration)
        )
    )


@record_app.callback(invoke_without_command=True)
def record_main(
    ctx: typer.Context,
    output_dir: OutputDirOption = None,
) -> None:
    """Record audio from different sources and transcribe it."""
    if ctx.invoked_subcommand is not None:
        return

    _run_recording(
        record_both_until_enter,
        _resolve_target_dir(output_dir),
        expected_source_count=2,
    )


@record_app.command("alt")
def record_alt(
    output_dir: OutputDirOption = None,
) -> None:
    """Record microphone + system audio with the alternate transcription model."""
    _run_recording(
        record_both_until_enter,
        _resolve_target_dir(output_dir),
        use_alt_transcription_model=True,
        expected_source_count=2,
    )


@mic_app.callback(invoke_without_command=True)
def mic_main(
    ctx: typer.Context,
    output_dir: OutputDirOption = None,
) -> None:
    """Record audio from the microphone only."""
    if ctx.invoked_subcommand is not None:
        return

    _run_recording(
        record_mic_until_enter,
        _resolve_target_dir(output_dir),
        expected_source_count=1,
    )


@mic_app.command("alt")
def mic_alt(
    output_dir: OutputDirOption = None,
) -> None:
    """Record microphone audio with the alternate transcription model."""
    _run_recording(
        record_mic_until_enter,
        _resolve_target_dir(output_dir),
        use_alt_transcription_model=True,
        expected_source_count=1,
    )


@os_app.callback(invoke_without_command=True)
def os_main(
    ctx: typer.Context,
    output_dir: OutputDirOption = None,
) -> None:
    """Record system audio only."""
    if ctx.invoked_subcommand is not None:
        return

    _run_recording(
        record_os_until_enter,
        _resolve_target_dir(output_dir),
        expected_source_count=1,
    )


@os_app.command("alt")
def os_alt(
    output_dir: OutputDirOption = None,
) -> None:
    """Record system audio with the alternate transcription model."""
    _run_recording(
        record_os_until_enter,
        _resolve_target_dir(output_dir),
        use_alt_transcription_model=True,
        expected_source_count=1,
    )
