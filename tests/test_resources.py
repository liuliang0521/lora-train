import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from trainkit.resources import ResourceMonitor


class ResourceTests(unittest.TestCase):
    def test_no_gpu_measurement_is_reported_and_failure_summary_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = ResourceMonitor(Path(directory), "pilot", interval=0.01)
            with patch("trainkit.resources.subprocess.run", side_effect=FileNotFoundError):
                item = monitor.sample()
                self.assertEqual(item["gpu_measurement"], "unavailable")
                self.assertGreater(item["system_ram_available_bytes"], 0)
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    with monitor:
                        raise RuntimeError("training failed")
            summary = json.loads((Path(directory) / "pilot-resources.json").read_text())
            self.assertFalse(summary["success"])
            self.assertIn("training failed", summary["error"])

    def test_gpu_measurement_is_parsed_in_mib(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = ResourceMonitor(Path(directory), "smoke")
            result = subprocess.CompletedProcess([], 0, "0, 12345, 16384, 97\n", "")
            with patch("trainkit.resources.subprocess.run", return_value=result):
                item = monitor.sample()
            self.assertEqual(item["gpus"][0]["used_mib"], 12345)
            self.assertEqual(monitor.peak_gpu, 12345)
