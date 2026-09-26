"""Nonzero adapter selection and constructor errors without GPU execution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sakuratts.backends.directml.gpt import DirectMLGPT
from sakuratts.backends.directml.static_gpt import StaticDirectMLGPT
from sakuratts.backends.onnx.sovits import INPUT_NAMES, ORTSoVITS
from test_ort_sovits import manifest


@unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "ONNX Runtime is optional")
class DirectMLDeviceTests(unittest.TestCase):
    def setUp(self):
        import onnxruntime as ort
        self.ort = ort
        self.enterContext(patch.object(ort, "get_available_providers",
                                      return_value=["DmlExecutionProvider", "CPUExecutionProvider"]))
        self.enterContext(patch("sakuratts.backends.onnx.sovits.read_manifest",
                                return_value=(manifest(), Path("unused-acoustic.onnx"))))

    def construct(self, kind, adapter):
        model = SimpleNamespace(device_id=adapter, threads=4)
        if kind == "prefill":
            return DirectMLGPT._create_session(model, "unused-prefill.onnx", self.ort.SessionOptions())
        if kind == "decode":
            return StaticDirectMLGPT._create_session(model, "unused-decode.onnx")
        return ORTSoVITS.load("unused", device="directml", device_id=adapter)

    def test_all_sessions_keep_the_requested_nonzero_adapter(self):
        for adapter in (2, 5):
            for kind in ("prefill", "decode", "acoustic"):
                session = Mock()
                session.get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
                session.get_provider_options.return_value = {}
                session.get_inputs.return_value = [SimpleNamespace(name=name) for name in INPUT_NAMES]
                session.get_outputs.return_value = [SimpleNamespace(name="waveform")]
                with self.subTest(adapter=adapter, kind=kind), \
                        patch.object(self.ort, "InferenceSession", return_value=session) as create:
                    self.construct(kind, adapter)
                    self.assertEqual(create.call_args.kwargs["providers"],
                                     [("DmlExecutionProvider", {"device_id": str(adapter)}), "CPUExecutionProvider"])
                    self.assertFalse(create.call_args.kwargs["enable_fallback"])

    def assert_initialization_failure_is_not_retried(self, kind):
        attempts = []
        failure = RuntimeError("DML adapter 3 device creation failed")

        def fail(session, providers, provider_options, disabled_optimizers=None):
            attempts.append(providers)
            session._fallback_providers = ["CPUExecutionProvider"]
            raise failure

        # Exercise ORT's real Python constructor, replacing only native session
        # creation. Retrying would consume CPU weights and hide the device error.
        with patch.object(self.ort.InferenceSession, "_create_inference_session", fail):
            with self.assertRaises(RuntimeError) as raised:
                self.construct(kind, 3)
        self.assertIs(raised.exception, failure)
        self.assertEqual(attempts, [[("DmlExecutionProvider", {"device_id": "3"}), "CPUExecutionProvider"]])

    def test_prefill_initialization_failure_preserves_cause_without_cpu_retry(self):
        self.assert_initialization_failure_is_not_retried("prefill")

    def test_decode_initialization_failure_preserves_cause_without_cpu_retry(self):
        self.assert_initialization_failure_is_not_retried("decode")

    def test_acoustic_initialization_failure_preserves_cause_without_cpu_retry(self):
        self.assert_initialization_failure_is_not_retried("acoustic")


if __name__ == "__main__":
    unittest.main()
