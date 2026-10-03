"""Credential-free, finite diagnostics with an owned isolated hardware backend."""

from here.audio_helper import OwnedHelper
from here.recording.diagnostics import AudioDeviceInfo, SignalTestResult
from here.recording.reservation import audio_reservation


class AudioDiagnosticsService:
    def __init__(self, *, reservation=audio_reservation, helper_factory=OwnedHelper):
        self.reservation = reservation
        self.helper_factory = helper_factory

    def _run(self, action, parameters, timeout):
        self.reservation.acquire()
        try:
            return self.helper_factory(action, parameters).result(timeout)
        finally:
            self.reservation.release()

    def devices(self) -> tuple[AudioDeviceInfo, ...]:
        return tuple(AudioDeviceInfo(**item) for item in self._run("devices", {}, 10))

    def test_signal(self, source: str) -> SignalTestResult:
        if source not in {"microphone", "system audio"}:
            raise ValueError("Unknown signal source")
        value = self._run("signal", {"source": source, "duration_seconds": 3.0}, 12)
        value["device"] = AudioDeviceInfo(**value["device"])
        return SignalTestResult(**value)
