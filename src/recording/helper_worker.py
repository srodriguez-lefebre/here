"""Hardware child. Only the fixed internal dispatcher imports this module."""

import base64
import queue
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from here.audio_helper import _encode, _read


def run():
    lock = threading.Lock()

    def emit(message):
        data = _encode(message)
        with lock:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    try:
        request = _read(sys.stdin.buffer)
        if request is None:
            raise ValueError("Missing audio request")
        action, parameters = request["action"], request["parameters"]
        if action == "devices":
            from here.recording.diagnostics import get_windows_audio_devices

            value = [asdict(item) for item in get_windows_audio_devices()]
        elif action == "signal":
            from here.recording.diagnostics import test_windows_audio_signal

            value = asdict(test_windows_audio_signal(**parameters))
        elif action == "capture":
            value = _capture(parameters, emit)
        else:
            raise ValueError("Unsupported internal audio action")
        emit({"kind": "result", "value": value})
        return 0
    except Exception as exc:
        emit({"kind": "error", "message": str(exc)[:2000]})
        return 1


def _capture(parameters, emit):
    from here.recording.journal import CaptureJournal
    from here.recording.windows import _ThreadedWindowsRecording

    blocks = queue.Queue(maxsize=64)
    ended = threading.Event()
    dropped = threading.Event()
    counts = {}
    count_lock = threading.Lock()

    def sink(label, data, rate, channels):
        # Primary WAV/journal never waits for IPC or live processing.
        with count_lock:
            offset = counts.get(label, 0)
            counts[label] = offset + len(data)
        if dropped.is_set():
            return
        try:
            blocks.put_nowait((label, data.copy(), rate, channels, offset))
        except queue.Full:
            dropped.set()

    def sender():
        while not ended.is_set() or not blocks.empty():
            try:
                label, data, rate, channels, offset = blocks.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                emit(
                    {
                        "kind": "block",
                        "label": label,
                        "rate": rate,
                        "channels": channels,
                        "offset": offset,
                        "pcm": base64.b64encode(data.tobytes()).decode("ascii"),
                    }
                )
            except Exception:
                dropped.set()
                return

    sender_thread = threading.Thread(target=sender, daemon=True)
    sender_thread.start()
    journal = CaptureJournal.load(
        Path(parameters.pop("sessions_root")), parameters.pop("capture_id")
    )
    handle = None
    try:
        handle = _ThreadedWindowsRecording(**parameters, block_sink=sink, journal=journal)
        emit({"kind": "ready", "sources": [asdict(item) for item in handle.opened_sources]})

        def commands():
            # Raw reads avoid a daemon owning BufferedReader's lock at interpreter exit
            # when a device fails spontaneously before the parent sends stop.
            while (message := _read(sys.stdin.buffer.raw)) is not None:
                command = message.get("command")
                if command not in {"pause", "resume", "stop", "cancel"}:
                    handle.stop()
                    return
                getattr(handle, command)()
                if command in {"stop", "cancel"}:
                    return
            handle.stop()

        threading.Thread(target=commands, daemon=True).start()
        while not handle._done_event.wait(0.2):
            with count_lock:
                current = dict(counts)
            emit({"kind": "tick", "counts": current})
        # wait joins the actual reader/writer and all hardware cleanup.
        handle.wait()
    finally:
        ended.set()
        sender_thread.join()
    return {"counts": counts, "live_dropped": dropped.is_set()}
