"""Fixed, versioned private child dispatch. Importable before Qt/provider bootstrap.

No executable expressions, pickle, public command names, or child-selected file paths.
Pipes and every queue/message are bounded. The owner always reaps before releasing audio.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

INTERNAL_FLAG = "--here-audio-helper-v1"
MAX_MESSAGE = 256 * 1024
ACTIONS = {"devices", "signal", "capture"}


def helper_command() -> list[str]:
    if getattr(sys, "frozen", False):
        # A windowed executable has no reliable Python stdio, even with pipe handles.
        return [str(Path(sys.executable).with_name("here-cli.exe")), INTERNAL_FLAG]
    return [sys.executable, "-m", "here.audio_helper", INTERNAL_FLAG]


def _encode(message):
    value = json.dumps(message, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    if len(value) > MAX_MESSAGE:
        raise ValueError("Audio helper message too large")
    return value


def _read(stream):
    line = stream.readline(MAX_MESSAGE + 1)
    if not line:
        return None
    if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ValueError("Invalid audio helper frame")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Invalid audio helper message")
    return value


class OwnedHelper:
    """One owned process; callbacks never execute in its IPC reader."""

    def __init__(self, action, parameters, *, command=None):
        if action not in ACTIONS:
            raise ValueError("Unsupported internal audio action")
        initial_request = _encode({"action": action, "parameters": parameters})
        self.messages = queue.Queue(maxsize=64)
        self.commands = queue.Queue(maxsize=16)
        self.latest = {}
        self.lock = threading.Lock()
        self.dropped = False
        self.error = None
        self.closing = threading.Event()
        self.process = subprocess.Popen(
            command or helper_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._process_owner = None
        try:
            if sys.platform == "win32":
                from here.recording.process_owner import WindowsProcessOwner

                self._process_owner = WindowsProcessOwner(self.process)
        except BaseException:
            # The child has not received its initial request, so no hardware opens.
            self.process.kill()
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            raise
        self.reader = threading.Thread(target=self._reader, name="here-audio-ipc", daemon=True)
        self.writer = threading.Thread(target=self._writer, name="here-audio-control", daemon=True)
        self.reader.start()
        self.writer.start()
        self.commands.put_nowait(initial_request)

    def send(self, message):
        # UI control calls only enqueue; they never block on a pipe or backend.
        try:
            self.commands.put_nowait(_encode(message))
        except queue.Full:
            self.error = RuntimeError("Audio helper control queue overflow")

    def _writer(self):
        try:
            while not self.closing.is_set():
                try:
                    data = self.commands.get(timeout=0.05)
                except queue.Empty:
                    continue
                self.process.stdin.write(data)
                self.process.stdin.flush()
        except (OSError, ValueError):
            pass

    def _reader(self):
        try:
            while (message := _read(self.process.stdout)) is not None:
                kind = message.get("kind")
                if kind == "block":
                    try:
                        self.messages.put_nowait(message)
                    except queue.Full:
                        self.dropped = True
                elif kind in {"ready", "tick", "result", "error"}:
                    with self.lock:
                        self.latest[kind] = message
                else:
                    raise ValueError("Unknown audio helper message")
        except Exception as exc:
            self.error = exc

    def get(self, kind):
        with self.lock:
            return self.latest.get(kind)

    def close(self):
        """Off-Qt only; no return until OS process and both pipe workers are gone."""
        self.closing.set()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        if self._process_owner is not None:
            self._process_owner.close()
        self.writer.join()
        self.reader.join()
        self.process.stdin.close()
        self.process.stdout.close()

    def result(self, timeout):
        deadline = time.monotonic() + timeout
        try:
            while self.process.poll() is None:
                if self.error is not None:
                    raise self.error
                if time.monotonic() >= deadline:
                    raise TimeoutError("Tiempo agotado al comprobar los dispositivos de audio")
                time.sleep(0.01)
            self.reader.join()
            if self.error is not None:
                raise self.error
            if error := self.get("error"):
                raise RuntimeError(error["message"])
            result = self.get("result")
            if self.process.returncode or result is None:
                raise RuntimeError("El proceso de audio terminó sin un resultado")
            return result["value"]
        finally:
            self.close()


def dispatch_internal_helper(arguments=None):
    """Frozen GUI and CLI entries call this before importing their normal bootstrap.

    Return None for ordinary launch, or the child exit code for the exact private flag.
    """
    arguments = sys.argv if arguments is None else arguments
    if arguments[1:] != [INTERNAL_FLAG]:
        return None
    from here.recording.helper_worker import run

    return run()


if __name__ == "__main__":
    raise SystemExit(dispatch_internal_helper() or 0)
