"""CPU GPT against an independent full-prefix FP64 attention calculation."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.backends.cpu.gpt import CPUGPT
from sakuratts._internal.generation import SynthesisCancelled, generate_semantic
from sakuratts._internal.reference_condition import sha256_file


def model_data():
    config = {"hidden_dim": 12, "embedding_dim": 12, "heads": 3, "layers": 2,
        "ffn_dim": 48, "vocab_size": 9, "phoneme_vocab_size": 8, "bert_dim": 5,
        "eos": 8, "layer_norm_epsilon": 1e-5, "position_scale": 1.0, "max_positions": 32}
    rng = np.random.default_rng(917)
    shapes = {"text_embedding": (8, 12), "audio_embedding": (9, 12),
        "text_alpha": (1,), "audio_alpha": (1,), "bert.weight": (12, 5),
        "bert.bias": (12,), "output.weight": (9, 12), "position_encoding": (32, 12)}
    for layer in range(2):
        for prefix, shape in {"qkv": (36, 12), "attention_output": (12, 12),
                "ffn_in": (48, 12), "ffn_out": (12, 48)}.items():
            shapes[f"layers.{layer}.{prefix}.weight"] = shape
            shapes[f"layers.{layer}.{prefix}.bias"] = (shape[0],)
        for norm in ("norm1", "norm2"):
            shapes[f"layers.{layer}.{norm}.weight"] = (12,)
            shapes[f"layers.{layer}.{norm}.bias"] = (12,)
    weights = {name: rng.normal(0, .2, shape).astype(np.float32) for name, shape in shapes.items()}
    for layer in range(2):
        for norm in ("norm1", "norm2"):
            weights[f"layers.{layer}.{norm}.weight"] += 1
    manifest = {"format": "sakuratts-gpt-fp32-v1", "architecture": "gpt-sovits-ar-postnorm-relu",
        "dtype": "float32", "config": config,
        "source": {"checkpoint_sha256": "tiny-checkpoint", "official_commit": "tiny-source"}}
    return manifest, weights


def full_prefix(manifest, weights, phones, audio, bert):
    """Recompute every hidden row in FP64, with no cached state or runtime helpers."""
    weights = {name: value.astype(np.float64) for name, value in weights.items()}
    config = manifest["config"]
    text_length, audio_length = phones.shape[1], audio.shape[1]
    size = text_length + audio_length
    text = weights["text_embedding"][phones[0]] + bert[0] @ weights["bert.weight"].T + weights["bert.bias"]
    text += weights["text_alpha"] * weights["position_encoding"][:text_length]
    spoken = weights["audio_embedding"][audio[0]] + weights["audio_alpha"] * weights["position_encoding"][:audio_length]
    x = np.concatenate((text, spoken))
    for layer in range(config["layers"]):
        prefix = f"layers.{layer}."
        projected = x @ weights[prefix + "qkv.weight"].T + weights[prefix + "qkv.bias"]
        query, key, value = np.split(projected, 3, axis=1)
        attended = np.zeros_like(x)
        for head in range(config["heads"]):
            columns = slice(head * 4, (head + 1) * 4)
            for row in range(size):
                valid = text_length if row < text_length else row + 1
                scores = key[:valid, columns] @ query[row, columns] / np.sqrt(4)
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                attended[row, columns] = probabilities @ value[:valid, columns]
        x += attended @ weights[prefix + "attention_output.weight"].T + weights[prefix + "attention_output.bias"]
        x = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(x.var(axis=1, keepdims=True) + config["layer_norm_epsilon"])
        x = x * weights[prefix + "norm1.weight"] + weights[prefix + "norm1.bias"]
        hidden = np.maximum(x @ weights[prefix + "ffn_in.weight"].T + weights[prefix + "ffn_in.bias"], 0)
        x += hidden @ weights[prefix + "ffn_out.weight"].T + weights[prefix + "ffn_out.bias"]
        x = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(x.var(axis=1, keepdims=True) + config["layer_norm_epsilon"])
        x = x * weights[prefix + "norm2.weight"] + weights[prefix + "norm2.bias"]
    return x[-1:] @ weights["output.weight"].T


def save_package(directory, manifest, weights):
    root = Path(directory)
    weights_path = root / "weights.npz"
    np.savez(weights_path, **weights)
    manifest = deepcopy(manifest)
    manifest["weights"] = {"file": "weights.npz", "bytes": weights_path.stat().st_size,
        "sha256": sha256_file(weights_path)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class CPUGPTTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.weights = model_data()
        self.phones = np.array([[2, 5, 1]], dtype=np.int64)
        self.prompt = np.array([[3, 0]], dtype=np.int64)
        self.bert = np.random.default_rng(2).normal(size=(1, 3, 5)).astype(np.float32)

    def test_prefill_and_cached_decode_match_full_prefix_oracle(self):
        # Prepared reference features arrive as a transposed [phones, BERT] view.
        self.bert = np.ascontiguousarray(self.bert.swapaxes(1, 2)).swapaxes(1, 2)
        for chunk in (0, 1, 3, 128):
            with self.subTest(chunk=chunk):
                model = CPUGPT(self.manifest, self.weights, prefill_query_chunk_size=chunk)
                try:
                    actual = model.prefill(self.phones, self.prompt, self.bert)
                    expected = full_prefix(self.manifest, self.weights, self.phones, self.prompt, self.bert)
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-6)
                    self.assertEqual(actual.dtype, np.float32)
                    audio = self.prompt
                    for token in (6, 2, 4, 0):
                        audio = np.concatenate((audio, [[token]]), axis=1)
                        actual = model.decode(token)
                        expected = full_prefix(self.manifest, self.weights, self.phones, audio, self.bert)
                        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-6)
                finally:
                    model.close()

    def test_shorter_request_and_released_state_do_not_reuse_old_history(self):
        model = CPUGPT(self.manifest, self.weights)
        try:
            original = model.prefill(self.phones, self.prompt, self.bert)
            saved = original.copy()
            model.decode(6)
            model.decode(2)
            np.testing.assert_array_equal(original, saved)
            phones, prompt, bert = self.phones[:, :1], self.prompt[:, :1], self.bert[:, :1]
            expected = full_prefix(self.manifest, self.weights, phones, prompt, bert)
            np.testing.assert_allclose(model.prefill(phones, prompt, bert), expected, rtol=2e-5, atol=3e-6)
            model.release_request_state()
            model.release_request_state()
            with self.assertRaisesRegex(RuntimeError, "prefill"):
                model.decode(1)
            np.testing.assert_allclose(model.prefill(self.phones, self.prompt, self.bert), saved, rtol=2e-5, atol=3e-6)
        finally:
            model.close()
        model.close()

    def test_decode_failure_requires_fresh_prefill_and_does_not_leak_threads(self):
        model = CPUGPT(self.manifest, self.weights, threads=1)
        before = [(item["prefix"], item["num_threads"]) for item in threadpool_info()]
        expected = model.prefill(self.phones, self.prompt, self.bert)
        original = model._gemv
        calls = 0
        def fail_partway(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise RuntimeError("injected decode failure")
            return original(*args, **kwargs)
        with patch.object(model, "_gemv", side_effect=fail_partway):
            with self.assertRaisesRegex(RuntimeError, "injected decode failure"):
                model.decode(1)
        with self.assertRaisesRegex(RuntimeError, "prefill"):
            model.decode(1)
        self.assertEqual(before, [(item["prefix"], item["num_threads"]) for item in threadpool_info()])
        np.testing.assert_array_equal(model.prefill(self.phones, self.prompt, self.bert), expected)
        model.close()
        with self.assertRaisesRegex(RuntimeError, "unloaded"):
            model.prefill(self.phones, self.prompt, self.bert)

    def test_blas_thread_limit_is_scoped_and_restored_on_failure(self):
        model = CPUGPT(self.manifest, self.weights, threads=1)
        before = [(item["prefix"], item["num_threads"]) for item in threadpool_info()]
        with patch.object(model, "_linear", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                model.prefill(self.phones, self.prompt, self.bert)
        self.assertEqual(before, [(item["prefix"], item["num_threads"]) for item in threadpool_info()])
        with self.assertRaisesRegex(RuntimeError, "prefill"):
            model.decode(1)
        original = model._gemv
        observed = []
        def inspect_blas(*args, **kwargs):
            observed.extend(item["num_threads"] for item in threadpool_info() if item["user_api"] == "blas")
            return original(*args, **kwargs)
        model.prefill(self.phones, self.prompt, self.bert)
        with patch.object(model, "_gemv", side_effect=inspect_blas):
            model.decode(1)
        self.assertTrue(observed)
        self.assertEqual(set(observed), {1})
        self.assertEqual(before, [(item["prefix"], item["num_threads"]) for item in threadpool_info()])
        model.close()

    def test_capacity_and_audio_positions_fail_without_truncation(self):
        model = CPUGPT(self.manifest, self.weights, capacity=5)
        model.prefill(self.phones, self.prompt, self.bert)
        with self.assertRaisesRegex(ValueError, "capacity"):
            model.decode(1)
        with self.assertRaisesRegex(ValueError, "capacity"):
            model.prefill(self.phones, np.array([[1, 2, 3]], np.int64), self.bert)
        model.close()
        manifest = deepcopy(self.manifest)
        manifest["config"]["max_positions"] = 3
        weights = dict(self.weights, position_encoding=self.weights["position_encoding"][:3])
        model = CPUGPT(manifest, weights, capacity=20)
        model.prefill(self.phones, self.prompt, self.bert)
        model.decode(1)
        with self.assertRaisesRegex(ValueError, "position"):
            model.decode(2)
        model.close()

    def test_weight_package_identity_and_shapes_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = save_package(directory, self.manifest, self.weights)
            stored_checksum = sha256_file(root / "weights.npz")
            model = CPUGPT.load(root)
            self.assertEqual(model.weight_manifest["source"], self.manifest["source"])
            for name, original in self.weights.items():
                np.testing.assert_array_equal(model.weights[name], original)
            np.testing.assert_allclose(model.prefill(self.phones, self.prompt, self.bert),
                full_prefix(self.manifest, self.weights, self.phones, self.prompt, self.bert), rtol=2e-5, atol=3e-6)
            model.decode(1)
            model.close()
            self.assertEqual(sha256_file(root / "weights.npz"), stored_checksum)
            with (root / "weights.npz").open("ab") as handle:
                handle.write(b"changed")
            loaded = CPUGPT.load(root)
            np.testing.assert_allclose(loaded.prefill(self.phones, self.prompt, self.bert),
                full_prefix(self.manifest, self.weights, self.phones, self.prompt, self.bert), rtol=2e-5, atol=3e-6)
            loaded.close()
            for bad in ({"output.weight": self.weights["output.weight"][:, :-1]},
                    {"bert.bias": np.full_like(self.weights["bert.bias"], np.inf)}):
                save_package(directory, self.manifest, dict(self.weights, **bad))
                with self.assertRaisesRegex(ValueError, "finite FP32 with shape"):
                    CPUGPT.load(root)

    def test_shared_generation_keeps_limits_cancellation_and_retry(self):
        weights = dict(self.weights, **{"output.weight": np.zeros_like(self.weights["output.weight"])})
        model = CPUGPT(self.manifest, weights, capacity=20)
        options = dict(eos=8, top_k=1, repetition_penalty=1., early_stop_num=3,
            random_draw=lambda step, shape: np.ones(shape, dtype=np.float32))
        def generate(**extra):
            return generate_semantic(model, self.phones, self.prompt, self.bert, **options, **extra)
        baseline = generate()
        self.assertEqual(baseline.stop.reasons, ("early_stop_num",))
        self.assertEqual(len(baseline.sampled_tokens), 4)
        observed = []
        with self.assertRaises(SynthesisCancelled):
            generate(observer=lambda *args: observed.append(args[0]), cancel_requested=lambda: len(observed) == 2)
        model.release_request_state()
        np.testing.assert_array_equal(generate().sampled_tokens, baseline.sampled_tokens)
        model.close()


if __name__ == "__main__":
    unittest.main()
