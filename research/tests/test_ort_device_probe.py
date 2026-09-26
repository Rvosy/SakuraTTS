"""A configured GPU provider cannot stand in for observed GPU execution."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/tools"))
import ort_device_probe as probe


class DeviceProbeTests(unittest.TestCase):
    def test_profile_counts_execution_events_without_session_or_fence_entries(self):
        result = probe.profile_execution([
            {"cat": "Session", "dur": 999, "args": {"provider": "DmlExecutionProvider"}},
            {"cat": "Node", "name": "conv", "dur": 10,
             "args": {"provider": "DmlExecutionProvider", "op_name": "Conv"}},
            {"cat": "Node", "name": "conv", "dur": 12,
             "args": {"provider": "DmlExecutionProvider", "op_name": "Conv"}},
            {"cat": "Node", "name": "shape", "dur": 3,
             "args": {"provider": "CPUExecutionProvider", "op_name": "Shape"}},
            {"cat": "Node", "name": "fence", "dur": 1, "args": {}}])
        self.assertEqual(result["DmlExecutionProvider"]["node_events"], 2)
        self.assertEqual(result["DmlExecutionProvider"]["profile_duration_us"], 22)
        self.assertEqual(result["DmlExecutionProvider"]["unique_node_names"], 1)
        self.assertEqual(result["CPUExecutionProvider"]["node_events"], 1)

    def test_numerically_correct_cpu_fallback_cannot_pass_a_gpu_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, output = root / "package", root / "output"
            package.mkdir()
            (package / "manifest.json").write_text(json.dumps({"graphs": {}, "weights": {}}))
            np.savez(package / "validation-0.npz", codes=np.zeros((1, 1, 1), np.int64),
                phones=np.zeros((1, 1), np.int64), ge=np.zeros((1, 1024, 1), np.float32),
                ge512=np.zeros((1, 512, 1), np.float32), noise=np.zeros((1, 192, 2), np.float32),
                noise_scale=np.float32(.5), expected_waveform=np.ones((1, 1, 4), np.float32))

            class Model:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                provider_options = {}
                sample_rate = 32000
                closed = False

                def decode(self, *args, **kwargs):
                    return np.ones((1, 1, 4), np.float32)

                def end_profiling(self):
                    path = output / "profile.json"
                    path.write_text(json.dumps([{"cat": "Node", "name": "conv", "dur": 10,
                        "args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}}]))
                    return str(path)

                def close(self):
                    self.closed = True

            model = Model()
            model.session = model
            args = SimpleNamespace(package=package, output=output, cases=None, device="directml",
                                   device_id=0, threads=2, repeats=1)
            self.assertEqual(probe.run_probe(args, loader=lambda *args, **kwargs: model), 1)
            report = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertTrue(report["numeric_passed"])
            self.assertFalse(report["execution_verified"])
            self.assertIn("DmlExecutionProvider", report["execution_error"])
            self.assertTrue(model.closed)


if __name__ == "__main__":
    unittest.main()
