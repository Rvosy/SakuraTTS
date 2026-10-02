"""Portable package, request validation and resource ownership checks."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.backends.onnx.sovits import ORTSoVITS


class FakeSession:
    def __init__(self):
        self.calls = []

    def get_provider_options(self):
        return {}

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feeds):
        self.calls.append((names, feeds))
        return [np.zeros((1, 1, feeds["codes"].shape[2] * 1280), np.float32)]


def manifest():
    return {"format": "sakuratts-sovits-onnx-v1", "dtype": "float32",
            "validation": {"file": "validation.json", "passed": True},
            "config": {"model": {"version": "v2ProPlus", "inter_channels": 192},
                       "sample_rate": 32000, "semantic_upsample_factor": 2,
                       "semantic_vocabulary": 1024, "phoneme_vocabulary": 732},
            "inputs": {"ge": {"shape": [1, 1024, 1]}, "ge512": {"shape": [1, 512, 1]}},
            "source": {"checkpoint_sha256": "checkpoint", "official_commit": "source"}}


class ORTSoVITSTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.model = ORTSoVITS(manifest(), self.session)
        self.inputs = [np.zeros((1, 1, 3), np.int64), np.zeros((1, 4), np.int64),
                       np.zeros((1, 1024, 1), np.float32), np.zeros((1, 512, 1), np.float32),
                       np.zeros((1, 192, 6), np.float32)]

    def test_complete_waveform_and_explicit_noise_are_preserved(self):
        self.inputs[-1][:] = np.arange(6, dtype=np.float32)
        result = self.model.decode(*self.inputs, noise_scale=0.6)
        self.assertEqual(result.shape, (1, 1, 3840))
        names, feeds = self.session.calls[-1]
        self.assertEqual(names, ["waveform"])
        np.testing.assert_array_equal(feeds["noise"], self.inputs[-1])
        self.assertEqual(feeds["noise_scale"].dtype, np.float32)
        self.assertEqual(feeds["noise_scale"].shape, ())

    def test_invalid_shape_tokens_precision_and_noise_fail_before_session(self):
        cases = [(0, np.zeros((1, 0), np.int64)), (0, np.full((1, 1, 3), 1024, np.int64)),
                 (1, np.full((1, 4), 732, np.int64)), (2, self.inputs[2].astype(np.float16)),
                 (3, np.full((1, 512, 1), np.nan, np.float32)),
                 (4, np.zeros((1, 192, 7), np.float32))]
        for index, value in cases:
            with self.subTest(index=index, shape=value.shape):
                incoming = list(self.inputs)
                incoming[index] = value
                with self.assertRaises(ValueError):
                    self.model.decode(*incoming)
        self.assertFalse(self.session.calls)

    def test_unload_drops_session_and_rejects_later_decode(self):
        self.model.release_request_state()
        self.assertIs(self.model.session, self.session)
        self.model.unload()
        self.assertIsNone(self.model.session)
        self.model.unload()
        with self.assertRaisesRegex(RuntimeError, "unloaded"):
            self.model.decode(*self.inputs)

    def test_capture_requires_explicit_diagnostic_session(self):
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            self.model.decode(*self.inputs, capture=True)
        self.assertFalse(self.session.calls)

    def test_checkpoint_and_family_mismatch_rejected(self):
        from types import SimpleNamespace
        reference = SimpleNamespace(manifest={"model_family": "v2ProPlus", "identity": {
            "sovits_checkpoint_sha256": "checkpoint", "official_commit": "source"}})
        self.model.validate_reference(reference)
        reference.manifest["model_family"] = "v2Pro"
        with self.assertRaisesRegex(ValueError, "differs"):
            self.model.validate_reference(reference)

if __name__ == "__main__":
    unittest.main()
