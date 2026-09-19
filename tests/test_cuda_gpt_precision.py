"""CPU checks for CUDA precision selection, residency and GEMM ABI arguments."""

import ctypes
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.reference_condition import sha256_file


def load_module():
    cupy = SimpleNamespace(float16=np.float16, float32=np.float32, int32=np.int32,
        asarray=np.asarray, ascontiguousarray=np.ascontiguousarray, concatenate=np.concatenate,
        empty=np.empty, zeros=np.zeros,
        RawKernel=Mock(side_effect=lambda *args, **kwargs: Mock()),
        cuda=SimpleNamespace(Stream=Mock(return_value=Mock()),
                             get_current_stream=Mock(return_value=Mock())),
        get_default_memory_pool=Mock(return_value=Mock()))
    path = Path(__file__).resolve().parents[1] / "src/sakuratts/cuda_gpt.py"
    spec = importlib.util.spec_from_file_location("sakuratts._test_cuda_gpt_precision", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"cupy": cupy}), patch("sakuratts.cuda_runtime.configure_cuda"):
        spec.loader.exec_module(module)
    return module


cuda_gpt = load_module()


def package(root, weight=None):
    original = np.array([[.1234567, -.25], [1.2, 3.4]], dtype=np.float32) if weight is None else weight
    path = root / "weights.npz"
    np.savez(path, **{"output.weight": original})
    manifest = {"format": "sakuratts-gpt-fp32-v1", "architecture": "gpt-sovits-ar-postnorm-relu",
        "weights": {"file": path.name, "sha256": sha256_file(path)},
        "config": {"hidden_dim": 2, "heads": 1, "layers": 2, "layer_norm_epsilon": 1e-5,
                   "vocab_size": 2, "bert_dim": 2, "max_positions": 16, "phoneme_vocab_size": 4}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return original, path.read_bytes()


def array(shape, dtype):
    data = np.zeros(shape, dtype=dtype)
    return SimpleNamespace(shape=data.shape, dtype=data.dtype, ndim=data.ndim,
                           flags=data.flags, data=SimpleNamespace(ptr=data.ctypes.data), owner=data)


class CudaGptPrecisionTests(unittest.TestCase):
    def test_precision_is_explicit_and_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "fp32 or fp16"):
            cuda_gpt.CUDAGPT.load("does-not-exist", precision="automatic")

    def test_resident_weights_and_state_use_selected_precision_without_changing_package(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cuda_gpt, "_GraphBLAS"):
            root = Path(directory)
            original, package_bytes = package(root)
            for precision, dtype in (("fp32", np.float32), ("fp16", np.float16)):
                with self.subTest(precision=precision):
                    model = cuda_gpt.CUDAGPT.load(root, capacity=8, precision=precision)
                    weights = model.weights
                    np.testing.assert_array_equal(weights["output.weight"], original.astype(dtype))
                    model._allocate_state()
                    self.assertEqual(model.keys.dtype, dtype)
                    self.assertEqual(model.values.dtype, dtype)
                    self.assertEqual(model.workspace["x"].dtype, dtype)
                    self.assertEqual(model.workspace["logits"].dtype, np.float32)
                    expected = 2 * 2 * 1 * 8 * 2 * np.dtype(dtype).itemsize
                    self.assertEqual(model.keys.nbytes + model.values.nbytes, expected)
                    model.graph = object()
                    model.length = 4
                    model.release_request_state()
                    self.assertIsNone(model.graph)
                    self.assertIsNone(model.state)
                    self.assertIsNone(model.keys)
                    self.assertEqual(model.length, 0)
                    self.assertIs(model.weights, weights)
                    model._allocate_state()
                    self.assertEqual(model.keys.dtype, dtype)
                    model.close()
                    self.assertEqual(weights, {})
            self.assertEqual((root / "weights.npz").read_bytes(), package_bytes)

    def test_unrepresentable_fp16_weights_are_rejected(self):
        for value in (70000, np.inf, np.nan):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package(root, np.full((2, 2), value, dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "finite FP16"):
                    cuda_gpt.CUDAGPT.load(root, precision="fp16")

    def test_fp16_load_normalizes_column_major_archive_weights(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cuda_gpt, "_GraphBLAS"):
            root = Path(directory)
            original = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
            package(root, original)
            model = cuda_gpt.CUDAGPT.load(root, precision="fp16")
            self.assertTrue(model.weights["output.weight"].flags.c_contiguous)
            np.testing.assert_array_equal(model.weights["output.weight"], original)
            model.close()

    def test_prefill_accepts_transposed_bert_features(self):
        model = object.__new__(cuda_gpt.CUDAGPT)
        model.dtype, model.width, model.layers = np.float16, 2, 0
        model.weights = {"text_embedding": np.zeros((4, 2), np.float16),
            "audio_embedding": np.zeros((4, 2), np.float16),
            "bert.weight": np.ones((2, 3), np.float16), "bert.bias": np.zeros(2, np.float16),
            "text_alpha": np.ones(1, np.float16), "audio_alpha": np.ones(1, np.float16),
            "position_encoding": np.zeros((4, 2), np.float16),
            "output.weight": np.ones((4, 2), np.float16)}
        operands = []
        def linear(x, weight, *, output_dtype=None):
            self.assertTrue(x.flags.c_contiguous)
            self.assertEqual(x.dtype, np.float16)
            operands.append(x)
            return (x.astype(np.float32) @ weight.astype(np.float32).T).astype(output_dtype or model.dtype)
        model._linear_fp16 = linear
        bert = np.arange(6, dtype=np.float32).reshape(3, 2).T[np.newaxis]
        self.assertFalse(bert.flags.c_contiguous)
        logits = model._prefill_fp16(np.array([[1, 2]]), np.array([[1]]), bert)
        np.testing.assert_array_equal(operands[0], bert[0])
        self.assertEqual(logits.dtype, np.float32)

    def test_fp16_linear_uses_fp32_compute_and_all_prefill_rows(self):
        blas = object.__new__(cuda_gpt._GraphBLAS)
        blas.handle = ctypes.c_void_p(1)
        blas.alpha, blas.beta = ctypes.c_float(1), ctypes.c_float(0)
        blas.lib = SimpleNamespace(cublasGemmEx=Mock(return_value=0))
        x, weight, out = array((5, 7), np.float16), array((11, 7), np.float16), array((5, 11), np.float32)
        blas.linear(x, weight, out)
        args = blas.lib.cublasGemmEx.call_args.args
        self.assertEqual(args[1:6], (1, 0, 11, 5, 7))
        self.assertEqual(args[7:13], (weight.data.ptr, 2, 7, x.data.ptr, 2, 7))
        self.assertEqual(args[14:], (out.data.ptr, 0, 11, 68, -1))

    def test_fp16_batched_attention_uses_correct_matrix_strides(self):
        blas = object.__new__(cuda_gpt._GraphBLAS)
        blas.handle = ctypes.c_void_p(1)
        blas.alpha, blas.beta = ctypes.c_float(1), ctypes.c_float(0)
        blas.lib = SimpleNamespace(cublasGemmStridedBatchedEx=Mock(return_value=0))
        q, k, scores = array((3, 5, 7), np.float16), array((3, 9, 7), np.float16), array((3, 5, 9), np.float32)
        blas._gemm_fp16(q, k, scores, transpose_y=True)
        args = blas.lib.cublasGemmStridedBatchedEx.call_args.args
        self.assertEqual(args[1:6], (1, 0, 9, 5, 7))
        self.assertEqual(args[7:15], (k.data.ptr, 2, 7, 63, q.data.ptr, 2, 7, 35))
        self.assertEqual(args[16:], (scores.data.ptr, 0, 9, 45, 3, 68, -1))
        probabilities, value, attended = array((3, 5, 9), np.float16), array((3, 9, 7), np.float16), array((3, 5, 7), np.float16)
        blas._gemm_fp16(probabilities, value, attended)
        args = blas.lib.cublasGemmStridedBatchedEx.call_args.args
        self.assertEqual(args[1:6], (0, 0, 7, 5, 9))
        self.assertEqual(args[16:], (attended.data.ptr, 2, 7, 35, 3, 68, -1))


if __name__ == "__main__":
    unittest.main()
