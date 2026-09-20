"""Arena policy is a CUDA run option, independent of numerical session options."""
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.ort_sovits import INPUT_NAMES, ORTSoVITS
from test_ort_sovits import manifest


class RunOptions:
    def __init__(self):
        self.entries = {}

    def add_run_config_entry(self, key, value):
        self.entries[key] = value


class ORTArenaShrinkTests(unittest.TestCase):
    def test_cpu_rejects_shrink_before_reading_package_or_loading_session(self):
        with patch("sakuratts.ort_sovits.read_manifest") as read:
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                ORTSoVITS.load("unused", device="cpu", acoustic_arena_shrink=True)
            read.assert_not_called()

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
        with patch("sakuratts.ort_sovits.read_manifest", return_value=(manifest(), Path("unused"))), \
             patch.dict(sys.modules, {"onnxruntime": ort, "sakuratts.cuda_runtime": SimpleNamespace(configure_cuda=Mock())}):
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

    def test_direct_cpu_instance_rejects_shrink(self):
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            ORTSoVITS(manifest(), session, acoustic_arena_shrink=True)

    def test_worker_forwards_shrink_and_reports_effective_policy(self):
        from sakuratts import ort_worker
        model = Mock()
        model.providers = ["CUDAExecutionProvider"]
        model.provider_options = {}
        model.encoder = SimpleNamespace(manifest={"dtype": "float16"})
        model.acoustic_arena_shrink = True
        with patch.object(sys, "argv", ["worker", "--package", "unused", "--allow-experimental-fp16", "--acoustic-arena-shrink"]), \
             patch.object(sys, "stdin", SimpleNamespace(buffer=io.BytesIO())), \
             patch.object(sys, "stdout", SimpleNamespace(buffer=io.BytesIO())), \
             patch.object(ort_worker.ORTSoVITS, "load", return_value=model) as load, \
             patch.object(ort_worker, "read_message", return_value=({"command": "close"}, {})), \
             patch.object(ort_worker, "write_message") as send, \
             patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(__version__="test")}):
            self.assertEqual(ort_worker.main(), 0)
        load.assert_called_once_with("unused", diagnostic=False, allow_experimental_fp16=True, acoustic_arena_shrink=True)
        self.assertTrue(send.call_args.args[1]["acoustic_arena_shrink"])
        model.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
