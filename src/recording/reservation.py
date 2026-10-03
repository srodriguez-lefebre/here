"""One application-wide lease for hardware, including backend teardown."""

import threading


class AudioReservation:
    def __init__(self):
        self._lock = threading.Lock()

    def acquire(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Los dispositivos de audio están ocupados; espere a que terminen.")

    def release(self):
        self._lock.release()


audio_reservation = AudioReservation()
