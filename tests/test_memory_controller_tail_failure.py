"""TAIL-M1: live teardown failure preserves the canonical owner on real job unwind."""

from contextlib import closing
from types import SimpleNamespace

import pytest


def test_live_finally_failure_unwinds_native_job_without_releasing_canonical_owner(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic tail test attempted provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)

    from here.application.controller import HereApplicationController
    from here.application.models import ApplicationState, StartRequest
    from here.memory.database import Busy
    from here.memory.service import MemoryService
    from here.output.ownership import FileLease, LeaseBusy, retain_for_job
    from here.recording.isolated import WindowsRecordingHandle
    from here.recording.journal import CaptureJournal
    from here.recording.models import RecordingSession
    from here.recording.reservation import audio_reservation
    from memory_fixtures import make_session

    class SyntheticLiveCloseFailure(RuntimeError):
        pass

    root = tmp_path / "data"
    sessions = tmp_path / "sessions"
    memory = MemoryService(root)
    events = []
    captures = []
    finish_calls = []
    original_finish = memory.finish_capture

    def finish_capture(identity, *, discard=False):
        finish_calls.append((identity, discard))
        return original_finish(identity, discard=discard)

    monkeypatch.setattr(memory, "finish_capture", finish_capture)

    class SyntheticNativeHandle(WindowsRecordingHandle):
        def __init__(self):
            self._journal = CaptureJournal.create(sessions, owned=True)
            self._thread = None
            self._scoped_owner = retain_for_job(self.close_ownership)
            assert self._scoped_owner
            self.opened_sources = ()
            self.live_error = None

        def wait(self, timeout=None):
            events.append("native_helper_closed")
            return RecordingSession(sources=[])

        def stop(self):
            pass

        def close_ownership(self):
            events.append("native_release")
            super().close_ownership()

    def capture(request, sink):
        handle = SyntheticNativeHandle()
        captures.append(handle)
        return handle

    class Live:
        def abort(self):
            events.append("live_abort")

        def cleanup(self):
            events.append("live_cleanup")

        def wait_closed(self):
            events.append("live_close_failed")
            raise SyntheticLiveCloseFailure("synthetic live-finally failure")

    class Processor:
        def process(self, session, output_dir, **kwargs):
            assert output_dir == sessions
            source = make_session(output_dir / "persisted", meeting_id=captures[0].meeting_id)
            events.append("persisted")
            return SimpleNamespace(session_dir=source.directory)

    controller = HereApplicationController(
        capture_factory=capture,
        live_factory=lambda *args: Live(),
        processor=Processor(),
        memory_service=memory,
    )

    def run_inline(worker, *, release_audio=False):
        # Run the actual decorated job/ExitStack, propagating its exact exception.
        # This focused control makes no thread-observer or hardware deadline claim.
        try:
            worker.run()
        finally:
            if release_audio:
                audio_reservation.release()

    monkeypatch.setattr(controller, "_launch_worker", run_inline)
    try:
        with pytest.raises(SyntheticLiveCloseFailure, match="synthetic live-finally failure"):
            controller.start(StartRequest(output_dir=sessions))
        assert controller.snapshot.state is ApplicationState.COMPLETED
        assert events == [
            "native_helper_closed",
            "persisted",
            "live_abort",
            "live_cleanup",
            "live_close_failed",
            "native_release",
        ]
        assert finish_calls == [], "Failed live teardown must not authorize catalog release"
        identity = captures[0].meeting_id
        with closing(
            FileLease(CaptureJournal.owner_path(sessions, identity), timeout=0, create=False)
        ):
            pass  # Native cleanup really released the OS lease despite the job exception.
        with pytest.raises(Busy):
            MemoryService.check_owner(root, identity)
        with pytest.raises(LeaseBusy):
            probe = FileLease(root / ".owners" / f"{identity}.lock", timeout=0, create=False)
            probe.close()
    finally:
        # Test-owned cleanup follows the retention assertions; no product retry policy.
        for handle in captures:
            handle.close_ownership()
            original_finish(handle.meeting_id)
