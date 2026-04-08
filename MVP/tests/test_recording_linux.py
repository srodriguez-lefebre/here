import subprocess
import unittest
from unittest.mock import patch

from here.recording import linux


class RecordingLinuxTests(unittest.TestCase):
    def test_get_default_monitor_source_name_prefers_default_sink_monitor(self) -> None:
        responses = {
            ("get-default-sink",): subprocess.CompletedProcess(
                args=["pactl", "get-default-sink"],
                returncode=0,
                stdout="RDPSink\n",
                stderr="",
            ),
            ("list", "sources", "short"): subprocess.CompletedProcess(
                args=["pactl", "list", "sources", "short"],
                returncode=0,
                stdout=(
                    "7\tOtherSink.monitor\tmodule-null-sink.c\ts16le 2ch 44100Hz\tRUNNING\n"
                    "8\tRDPSink.monitor\tmodule-rdp-sink.c\ts16le 2ch 44100Hz\tRUNNING\n"
                ),
                stderr="",
            ),
        }

        def fake_run_pactl(args: list[str]) -> subprocess.CompletedProcess[str]:
            return responses[tuple(args)]

        with patch("here.recording.linux._run_pactl", side_effect=fake_run_pactl):
            monitor = linux._get_default_monitor_source_name()

        self.assertEqual(monitor, "RDPSink.monitor")

    def test_record_os_linux_uses_selected_monitor_source(self) -> None:
        with patch("here.recording.linux._setup_pulse"), patch(
            "here.recording.linux._get_default_monitor_source_name",
            return_value="RDPSink.monitor",
        ), patch(
            "here.recording.linux._record_pulse_source_until_enter",
            return_value="session",
        ) as record_pulse:
            session = linux.record_os_linux(sample_rate=22050)

        self.assertEqual(session, "session")
        record_pulse.assert_called_once_with(
            source_name="RDPSink.monitor",
            sample_rate=22050,
            channels=1,
            label="system audio",
        )


if __name__ == "__main__":
    unittest.main()
