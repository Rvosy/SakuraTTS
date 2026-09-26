"""Provider selection and resource options for CPU and DirectML acoustics."""

import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from sakuratts.backends.onnx.sovits import INPUT_NAMES, ORTSoVITS
from test_ort_sovits import manifest


class SessionOptions:
    def __init__(self):
        self.entries = {}

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class ORTDirectMLTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        self.session.get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.session.get_provider_options.return_value = {"DmlExecutionProvider": {"device_id": "2"}}
        self.session.get_inputs.return_value = [SimpleNamespace(name=name) for name in INPUT_NAMES]
        self.session.get_outputs.return_value = [SimpleNamespace(name="waveform")]
        self.waveform = np.arange(3840, dtype=np.float32).reshape(1, 1, -1) / 8000
        self.session.run.return_value = [self.waveform]
        self.ort = SimpleNamespace(
            SessionOptions=SessionOptions,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
            InferenceSession=Mock(return_value=self.session),
            get_available_providers=Mock(return_value=["DmlExecutionProvider", "CPUExecutionProvider"]),
        )
        self.inputs = (np.zeros((1, 1, 3), np.int64), np.zeros((1, 4), np.int64),
                       np.zeros((1, 1024, 1), np.float32), np.zeros((1, 512, 1), np.float32),
                       np.zeros((1, 192, 6), np.float32))
        self.read = self.enterContext(patch("sakuratts.backends.onnx.sovits.read_manifest",
                                          return_value=(manifest(), Path("acoustic.onnx"))))
        self.enterContext(patch.dict(sys.modules, {"onnxruntime": self.ort}))

    def test_directml_selects_adapter_and_keeps_fp32_output(self):
        model = ORTSoVITS.load("unused", device="directml", device_id=2)
        kwargs = self.ort.InferenceSession.call_args.kwargs
        self.assertEqual(kwargs["providers"], [("DmlExecutionProvider", {"device_id": "2"}),
                                                "CPUExecutionProvider"])
        options = kwargs["sess_options"]
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(options.execution_mode, "sequential")
        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertTrue(options.enable_cpu_mem_arena)
        self.assertEqual(options.entries, {"session.intra_op.allow_spinning": "0",
                                           "session.inter_op.allow_spinning": "0"})
        self.assertFalse(self.ort.InferenceSession.call_args.kwargs["enable_fallback"])
        self.assertEqual(model.providers, ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertIs(model.decode(*self.inputs), self.waveform)
        names, feeds = self.session.run.call_args.args
        self.assertEqual(names, ["waveform"])
        self.assertEqual(feeds["ge"].dtype, np.float32)
        self.assertEqual(feeds["codes"].dtype, np.int64)
        np.testing.assert_array_equal(feeds["noise"], self.inputs[-1])

    def test_threads_cpu_arena_and_profiling_are_configurable(self):
        for device in ("cpu", "directml"):
            with self.subTest(device=device):
                self.session.get_providers.return_value = (
                    ["CPUExecutionProvider"] if device == "cpu" else ["DmlExecutionProvider", "CPUExecutionProvider"])
                ORTSoVITS.load("unused", device=device, intra_op_num_threads=3,
                               enable_cpu_mem_arena=False, profile_prefix="test-profile")
                options = self.ort.InferenceSession.call_args.kwargs["sess_options"]
                self.assertEqual(options.intra_op_num_threads, 3)
                self.assertFalse(options.enable_cpu_mem_arena)
                self.assertEqual(options.entries["session.intra_op.allow_spinning"], "0")
                self.assertTrue(options.enable_profiling)
                self.assertEqual(options.profile_file_prefix, "test-profile")

    def test_cpu_needs_no_gpu_provider_and_defaults_to_two_threads(self):
        self.session.get_providers.return_value = ["CPUExecutionProvider"]
        self.ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        model = ORTSoVITS.load("unused", device="cpu")
        kwargs = self.ort.InferenceSession.call_args.kwargs
        self.assertEqual(kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(kwargs["sess_options"].intra_op_num_threads, 2)
        self.assertIs(model.decode(*self.inputs), self.waveform)

    def test_missing_directml_provider_rejects_before_session_construction(self):
        self.ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with self.assertRaisesRegex(RuntimeError, "DirectML provider is unavailable"):
            ORTSoVITS.load("unused", device="directml")
        self.ort.InferenceSession.assert_not_called()

    def test_cpu_fallback_during_directml_initialization_is_rejected(self):
        for providers in (["CPUExecutionProvider"], []):
            with self.subTest(providers=providers):
                self.session.get_providers.return_value = providers
                with self.assertRaisesRegex(RuntimeError, "refusing silent CPU inference"):
                    ORTSoVITS.load("unused", device="directml")

    def test_directml_run_failure_is_propagated(self):
        model = ORTSoVITS.load("unused", device="directml")
        failure = RuntimeError("D3D12 device removed")
        self.session.run.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            model.decode(*self.inputs)
        self.assertIs(raised.exception, failure)
        self.session.run.assert_called_once()

    def test_rejected_initialization_does_not_retain_session_in_traceback(self):
        for failure in ("provider", "schema", "wrapper"):
            references = []
            metadata = manifest()
            if failure == "wrapper":
                metadata["config"].pop("sample_rate")
            self.read.return_value = metadata, Path("acoustic.onnx")

            class RejectedSession:
                def __init__(self, *args, **kwargs):
                    references.append(weakref.ref(self))

                def get_providers(self):
                    return ["CPUExecutionProvider"] if failure == "provider" else ["DmlExecutionProvider"]

                def disable_fallback(self):
                    pass

                def get_inputs(self):
                    return [] if failure == "schema" else [SimpleNamespace(name=name) for name in INPUT_NAMES]

                def get_outputs(self):
                    return [SimpleNamespace(name="waveform")]

                def get_provider_options(self):
                    return {}

            self.ort.InferenceSession = RejectedSession
            retained = None
            try:
                ORTSoVITS.load("unused", device="directml")
            except (RuntimeError, ValueError, KeyError) as error:
                retained = error
            self.assertIsNotNone(retained)
            self.assertIsNotNone(retained.__traceback__)
            gc.collect()
            self.assertIsNone(references[0](), failure)

    def test_invalid_directml_options_fail_before_package_loading(self):
        cases = [(dict(device_id=value), "adapter index") for value in (-1, True, "0")]
        cases += [(dict(enable_mem_pattern=True), "enable_mem_pattern=False"),
                  (dict(acoustic_arena_shrink=True), "requires CUDA")]
        for options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                ORTSoVITS.load("unused", device="directml", **options)
        self.read.assert_not_called()
        self.ort.InferenceSession.assert_not_called()

    def test_fp16_requires_device_specific_screen_and_chunked_stays_cuda_only(self):
        for device in ("cpu", "directml"):
            fp16_message = "not been screened for CPU" if device == "cpu" else "own DirectML engineering screen"
            for dtype, graph, message in (("float16", Path("acoustic.onnx"), fp16_message),
                                         ("float32", None, "Chunked.*requires CUDA")):
                with self.subTest(device=device, dtype=dtype, graph=graph):
                    metadata = manifest()
                    metadata["dtype"] = dtype
                    self.read.return_value = metadata, graph
                    with self.assertRaisesRegex(ValueError, message):
                        ORTSoVITS.load("unused", device=device, allow_experimental_fp16=True)
        self.ort.InferenceSession.assert_not_called()

    def test_cuda_session_options_keep_existing_defaults(self):
        self.session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with patch.dict(sys.modules, {"sakuratts.backends.cuda.runtime": SimpleNamespace(configure_cuda=Mock())}):
            ORTSoVITS.load("unused")
        options = self.ort.InferenceSession.call_args.kwargs["sess_options"]
        self.assertEqual(options.intra_op_num_threads, 4)
        self.assertEqual(options.entries, {})
        self.session.disable_fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
