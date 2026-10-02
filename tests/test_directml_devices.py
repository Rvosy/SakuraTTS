"""Nonzero adapter selection and constructor errors without GPU execution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sakuratts.backends.directml.gpt import DirectMLGPT
from sakuratts.backends.directml.static_gpt import StaticDirectMLGPT
from sakuratts.module.sovits import INPUT_NAMES, ORTSoVITS
from test_ort_sovits import manifest


@unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "ONNX Runtime is optional")
class DirectMLDeviceTests(unittest.TestCase):
    def setUp(self):
        import onnxruntime as ort
        self.ort = ort
        self.enterContext(patch.object(ort, "get_available_providers",
                                      return_value=["DmlExecutionProvider", "CPUExecutionProvider"]))
        self.enterContext(patch("sakuratts.module.sovits.read_manifest",
                                return_value=(manifest(), Path("unused-acoustic.onnx"))))

    def portable_probe(self):
        spec = importlib.util.spec_from_file_location("portable_probe",
            Path(__file__).resolve().parents[1] / "scripts/portable/check_runtime.py")
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        return probe

    def test_portable_probe_rejects_software_or_unknown_adapter_before_session_creation(self):
        adapters = [{"device_id": 1, "description": "Software renderer", "software": True}]
        with patch("sakuratts.backends.directml.devices.list_adapters", return_value=adapters), \
             patch.object(self.ort, "InferenceSession") as create:
            for device_id, message in ((1, "software renderer"), (4, "Unknown DXGI device_id")):
                with self.subTest(device_id=device_id), self.assertRaisesRegex(ValueError, message):
                    self.portable_probe().check_ort("directml", device_id)
            create.assert_not_called()

    def test_portable_probe_preserves_adapter_failure_without_vendor_filter_or_cpu_retry(self):
        for vendor_id in (0x8086, 0x1002):
            adapters = [{"device_id": 3, "description": "Selected hardware", "software": False,
                         "vendor_id": vendor_id}]
            attempts = []
            failure = RuntimeError("DML device 3 driver initialization failed")

            def fail(session, providers, provider_options, disabled_optimizers=None):
                attempts.append(providers)
                session._fallback_providers = ["CPUExecutionProvider"]
                raise failure

            with patch("sakuratts.backends.directml.devices.list_adapters", return_value=adapters), \
                 patch.object(self.ort.InferenceSession, "_create_inference_session", fail):
                with self.assertRaises(RuntimeError) as raised:
                    self.portable_probe().check_ort("directml", 3)
            self.assertIs(raised.exception, failure)
            self.assertEqual(attempts, [[("DmlExecutionProvider", {"device_id": "3"})]])

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
