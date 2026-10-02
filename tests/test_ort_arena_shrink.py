"""Arena policy is a CUDA run option, independent of numerical session options."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.backends.onnx.sovits import INPUT_NAMES, ORTSoVITS
from test_ort_sovits import manifest


class RunOptions:
    def __init__(self):
        self.entries = {}

    def add_run_config_entry(self, key, value):
        self.entries[key] = value


class ORTArenaShrinkTests(unittest.TestCase):
    def test_gpu_run_option_uses_selected_device_and_preserves_session_and_output(self):
        waveform = np.arange(3840, dtype=np.float32).reshape(1, 1, -1) / 8000
        session = Mock()
        session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session.get_provider_options.return_value = {}
        session.get_inputs.return_value = [SimpleNamespace(name=name) for name in INPUT_NAMES]
        session.get_outputs.return_value = [SimpleNamespace(name="waveform")]
        session.run.return_value = [waveform]
        ort = SimpleNamespace(SessionOptions=SimpleNamespace, RunOptions=RunOptions,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
            InferenceSession=Mock(return_value=session), get_available_providers=lambda: ["CUDAExecutionProvider"])
        inputs = (np.zeros((1, 1, 3), np.int64), np.zeros((1, 4), np.int64),
                  np.zeros((1, 1024, 1), np.float32), np.zeros((1, 512, 1), np.float32),
                  np.zeros((1, 192, 6), np.float32))
        with patch("sakuratts.backends.onnx.sovits.read_manifest", return_value=(manifest(), Path("unused"))), \
             patch.dict(sys.modules, {"onnxruntime": ort, "sakuratts.backends.cuda.runtime": SimpleNamespace(configure_cuda=Mock())}):
            baseline = ORTSoVITS.load("unused", device_id=2)
            off_options = ort.InferenceSession.call_args.kwargs
            self.assertIs(baseline.decode(*inputs), waveform)
            self.assertEqual(session.run.call_args.kwargs, {})
            candidate = ORTSoVITS.load("unused", device_id=2, acoustic_arena_shrink=True)
            on_options = ort.InferenceSession.call_args.kwargs
            for _ in range(2):
                actual = candidate.decode(*inputs)
                self.assertIs(actual, waveform)
                self.assertEqual(session.run.call_args.kwargs["run_options"].entries,
                                 {"memory.enable_memory_arena_shrinkage": "gpu:2"})
            self.assertEqual(off_options["providers"], on_options["providers"])
            self.assertEqual(vars(off_options["sess_options"]), vars(on_options["sess_options"]))
            self.assertEqual(on_options["providers"][0][1]["device_id"], "2")

if __name__ == "__main__":
    unittest.main()
