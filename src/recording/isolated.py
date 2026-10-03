"""Parent-owned capture with finite hardware open/read/stop deadlines."""

import base64
import queue
import threading
import time

import numpy as np
from here.audio_helper import OwnedHelper
from here.config.settings import get_settings
from here.recording.control import OpenedSource
from here.recording.journal import CaptureJournal
from here.recording.models import CaptureFailed


class WindowsRecordingHandle:
    def __init__(
        self,
        mode="both",
        *,
        block_sink=None,
        microphone_device_id=None,
        system_device_id=None,
        sessions_root=None,
        helper_factory=OwnedHelper,
        timeout=10.0,
    ):
        self.opened_sources = ()
        self.live_error = None
        self._error = None
        self._result = None
        self._stop_at = None
        self._paused = False
        self._cancelled = False
        self._ready = threading.Event()
        self._done = threading.Event()
        self._journal = CaptureJournal.create(sessions_root or get_settings().TRANSCRIPTIONS_DIR)
        self._helper = helper_factory(
            "capture",
            {
                "mode": mode,
                "microphone_device_id": microphone_device_id,
                "system_device_id": system_device_id,
                "sessions_root": str(self._journal.root),
                "capture_id": self._journal.document.capture_id,
            },
        )
        self._thread = threading.Thread(
            target=self._run, args=(block_sink, timeout), name="here-capture-owner", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            self._thread.join()
            raise self._error

    def _run(self, sink, timeout):
        helper = self._helper
        received = {}
        consumed = threading.Event()

        def consume():
            while not consumed.is_set() or not helper.messages.empty():
                try:
                    block = helper.messages.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if self.live_error is not None:
                        continue
                    label, channels = block["label"], block["channels"]
                    if block["offset"] != received.get(label, 0):
                        raise RuntimeError(
                            "Live audio IPC omitted frames; offline fallback required"
                        )
                    data = np.frombuffer(
                        base64.b64decode(block["pcm"], validate=True), dtype=np.int16
                    ).reshape(-1, channels)
                    received[label] = block["offset"] + len(data)
                    if sink is not None:
                        sink(label, data, block["rate"], channels)
                except Exception as exc:
                    self.live_error = exc

        consumer = threading.Thread(target=consume, name="here-capture-live", daemon=True)
        consumer.start()
        deadline = time.monotonic() + timeout
        last_counts = {}
        progressed = {}
        try:
            while helper.process.poll() is None:
                now = time.monotonic()
                if helper.error is not None:
                    raise helper.error
                if not self._ready.is_set():
                    if ready := helper.get("ready"):
                        self.opened_sources = tuple(
                            OpenedSource(**item) for item in ready["sources"]
                        )
                        progressed = {item.label: now for item in self.opened_sources}
                        self._ready.set()
                    elif now >= deadline:
                        raise TimeoutError("Timed out while opening Windows audio devices")
                else:
                    tick = helper.get("tick") or {}
                    counts = tick.get("counts", {})
                    for label in progressed:
                        if self._paused or counts.get(label) != last_counts.get(label):
                            progressed[label] = now
                        elif now - progressed[label] > timeout:
                            raise TimeoutError("Windows audio reader stopped responding")
                    last_counts = counts
                if self._stop_at is not None and now - self._stop_at >= timeout:
                    raise TimeoutError("Timed out closing Windows audio devices")
                time.sleep(0.01)
            helper.reader.join()
            if error := helper.get("error"):
                raise RuntimeError(error["message"])
            if helper.get("result") is None or helper.process.returncode:
                raise RuntimeError("Windows audio helper ended unexpectedly")
        except Exception as exc:
            self._error = exc
        finally:
            helper.close()
            consumed.set()
            consumer.join()  # Callback ownership cannot be abandoned on a UI timer.
            result = helper.get("result")
            final = result["value"] if result else {}
            if helper.dropped or final.get("live_dropped") or final.get("counts", {}) != received:
                self.live_error = RuntimeError(
                    "Live audio IPC incomplete; offline fallback required"
                )
            try:
                journal = CaptureJournal.load(self._journal.root, self._journal.document.capture_id)
                if self._cancelled:
                    journal.discard()
                    self._error = RuntimeError("Windows audio capture was cancelled")
                else:
                    if self._error is not None:
                        journal.finish(self._error)
                    self._result = journal.recording_session()
                    if self._error is not None and self._result.sources:
                        self._error = CaptureFailed(self._result, self._error)
            except Exception as exc:
                if self._error is None:
                    self._error = exc
            self._ready.set()
            self._done.set()

    def pause(self):
        self._paused = True
        self._helper.send({"command": "pause"})

    def resume(self):
        self._paused = False
        self._helper.send({"command": "resume"})

    def stop(self):
        if self._stop_at is None:
            self._stop_at = time.monotonic()
            self._helper.send({"command": "stop"})

    def cancel(self):
        # Parent removes only its owned journal after actual child closure.
        self._cancelled = True
        self.stop()

    def wait(self, timeout=None):
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("Timed out waiting for audio capture to stop")
        if self._error is not None:
            raise self._error
        return self._result
