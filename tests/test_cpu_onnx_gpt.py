"""Real small ONNX graph tests for CPU GPT cache and package boundaries."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from test_cpu_gpt import model_data, save_package, full_prefix
from sakuratts.backends.cpu.onnx_gpt import ONNXCPUGPT


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime"))


@unittest.skipUnless(AVAILABLE, "ONNX preparation and runtime dependencies are optional")
class ONNXCPUGPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sakuratts._internal.conversion.export_gpt_onnx import export_sidecar
        cls.folder = tempfile.TemporaryDirectory()
        cls.manifest, cls.weights = model_data()
        cls.root = save_package(cls.folder.name, cls.manifest, cls.weights)
        export_sidecar(cls.root, cls.root / "onnx")
        cls.precision_metadata = {precision: export_sidecar(cls.root, cls.root / f"onnx-{precision}", precision=precision)
                                  for precision in ("fp16", "int8")}

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        self.phones = np.array([[2, 5, 1]], np.int64)
        self.prompt = np.array([[3, 0]], np.int64)
        self.bert = np.random.default_rng(2).normal(size=(1, 3, 5)).astype(np.float32)

    def test_prefill_and_delta_cache_match_independent_full_prefix(self):
        model = ONNXCPUGPT.load(self.root, capacity=12, threads=1)
        try:
            audio = self.prompt.copy()
            actual = model.prefill(self.phones, audio, self.bert)
            np.testing.assert_allclose(actual, full_prefix(self.manifest, self.weights, self.phones, audio, self.bert), atol=1e-5, rtol=1e-5)
            cache = model.keys
            for token in (2, 7, 1):
                audio = np.concatenate((audio, [[token]]), axis=1)
                actual = model.decode(token)
                np.testing.assert_allclose(actual, full_prefix(self.manifest, self.weights, self.phones, audio, self.bert), atol=1e-5, rtol=1e-5)
                self.assertIs(model.keys, cache)
            # Resetting the request must exclude rows from the earlier history.
            np.testing.assert_allclose(model.prefill(self.phones, self.prompt, self.bert),
                full_prefix(self.manifest, self.weights, self.phones, self.prompt, self.bert), atol=1e-5, rtol=1e-5)
            model.release_request_state()
            self.assertIsNone(model.keys)
            with self.assertRaisesRegex(RuntimeError, "prefill"):
                model.decode(1)
        finally:
            model.close()

    def test_capacity_validation_and_failed_run_release_request(self):
        model = ONNXCPUGPT.load(self.root, capacity=6, threads=1)
        try:
            model.prefill(self.phones, self.prompt, self.bert)
            model.decode(2)
            with self.assertRaisesRegex(ValueError, "capacity"):
                model.decode(3)
            with self.assertRaisesRegex(ValueError, "integer"):
                model.decode(True)
            model.prefill(self.phones, self.prompt, self.bert)
            with patch.object(model.session, "run", side_effect=RuntimeError("kernel failed")):
                with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                    model.decode(3)
            self.assertIsNone(model.keys)
            self.assertEqual(model.length, 0)
        finally:
            model.close()
        with self.assertRaisesRegex(RuntimeError, "unloaded"):
            model.prefill(self.phones, self.prompt, self.bert)

    def test_source_identity_and_sidecar_hash_are_required(self):
        path = self.root / "onnx" / "manifest.json"
        original = path.read_text(encoding="utf-8")
        try:
            metadata = json.loads(original)
            changed = deepcopy(metadata)
            changed["source"]["weights_sha256"] = "wrong-source"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source GPT"):
                ONNXCPUGPT.load(self.root)
            changed = deepcopy(metadata)
            changed["embedding"]["sha256"] = "corrupt"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ONNXCPUGPT.load(self.root)
        finally:
            path.write_text(original, encoding="utf-8")

    def test_chunked_prefill_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "chunked prefill is unsupported"):
            ONNXCPUGPT.load(self.root, prefill_query_chunk_size=128)

    def test_runtime_uses_checked_sidecar_without_unused_original_archive(self):
        path = self.root / "weights.npz"
        original = path.read_bytes()
        try:
            path.unlink()
            model = ONNXCPUGPT.load(self.root, threads=1)
            try:
                actual = model.prefill(self.phones, self.prompt, self.bert)
                expected = full_prefix(self.manifest, self.weights, self.phones, self.prompt, self.bert)
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            finally:
                model.close()
        finally:
            path.write_bytes(original)

    def test_constructor_failure_does_not_keep_model_in_traceback(self):
        model = ONNXCPUGPT.__new__(ONNXCPUGPT)
        embedding = {"example": np.zeros(2, np.float32)}
        invalid_session = SimpleNamespace(get_inputs=lambda: [], get_outputs=lambda: [])
        with patch("onnxruntime.InferenceSession", return_value=invalid_session):
            with self.assertRaisesRegex(ValueError, "contract"):
                model.__init__(self.manifest, {}, embedding, "unused.onnx", 8, 1)
        self.assertIsNone(model.session)
        self.assertEqual(embedding, {})

    def test_fp16_executes_on_cpu_with_half_cache_and_fp32_sampling(self):
        model = ONNXCPUGPT.load(self.root, capacity=12, threads=1, precision="fp16")
        try:
            self.assertEqual(model.precision, "fp16")
            self.assertEqual(model.session.get_providers(), ["CPUExecutionProvider"])
            actual = model.prefill(self.phones, self.prompt, self.bert)
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(model.keys.dtype, np.float16)
            np.testing.assert_allclose(actual, full_prefix(self.manifest, self.weights, self.phones, self.prompt, self.bert), atol=.005, rtol=.005)
            audio = self.prompt.copy()
            for token in (2, 7, 1):
                audio = np.concatenate((audio, [[token]]), axis=1)
                actual = model.decode(token)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, full_prefix(self.manifest, self.weights, self.phones, audio, self.bert), atol=.005, rtol=.005)
        finally:
            model.close()

    def test_int8_contains_quantized_weights_and_runs_repeatably(self):
        import onnx
        metadata = self.precision_metadata["int8"]
        graph = onnx.load(str(self.root / "onnx-int8" / metadata["graphs"]["int8"]["file"]))
        self.assertTrue(any(value.data_type == onnx.TensorProto.INT8 for value in graph.graph.initializer))
        self.assertTrue(any(node.op_type == "MatMulInteger" for node in graph.graph.node))
        model = ONNXCPUGPT.load(self.root, capacity=12, threads=1, precision="int8")
        try:
            first = model.prefill(self.phones, self.prompt, self.bert)
            decoded = model.decode(2)
            self.assertEqual(model.precision, "int8")
            self.assertEqual(model.keys.dtype, np.float32)
            self.assertTrue(np.isfinite(decoded).all())
            np.testing.assert_array_equal(first, model.prefill(self.phones, self.prompt, self.bert))
            np.testing.assert_array_equal(decoded, model.decode(2))
        finally:
            model.close()

    def test_low_precision_cannot_relabel_fp32_or_change_conversion_source(self):
        path = self.root / "onnx-int8" / "manifest.json"
        original = path.read_text(encoding="utf-8")
        try:
            metadata = json.loads(original)
            metadata["precision"] = "fp32"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "precision"):
                ONNXCPUGPT.load(self.root, precision="int8")
            metadata = json.loads(original)
            metadata["conversion"]["input_graph_sha256"] = "another-model"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "FP32 source"):
                ONNXCPUGPT.load(self.root, precision="int8")
        finally:
            path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
