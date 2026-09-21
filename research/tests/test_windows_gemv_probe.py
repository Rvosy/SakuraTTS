"""CPU-only GEMV probe admission, evidence integrity and numerical contracts."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/tools"))
import windows_gemv_probe as probe


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(directory):
    config = {"hidden_dim": 8, "ffn_dim": 32, "vocab_size": 17, "layers": 2,
              "bert_dim": 4, "phoneme_vocab_size": 20, "max_positions": 100}
    package = directory / "gpt"
    package.mkdir()
    contracts = probe.shape_contracts(config, 0)
    weights = {c["weight_name"]: (np.arange(np.prod(c["weight_shape"]), dtype=np.float32).reshape(c["weight_shape"]) % 7 - 3) / 8
               for c in contracts.values()}
    np.savez(package / "weights.npz", **weights)
    manifest = {"format": "sakuratts-gpt-fp32-v1", "architecture": "gpt-sovits-ar-postnorm-relu", "config": config,
        "weights": {"file": "weights.npz", "sha256": probe.sha256_file(package / "weights.npz"),
                    "bytes": (package / "weights.npz").stat().st_size}, "source": {"checkpoint_sha256": "fixture-checkpoint"}}
    write_json(package / "manifest.json", manifest)
    reference = directory / "reference"
    reference.mkdir()
    np.savez(reference / "conditions.npz", prompt_semantic=np.array([1, 2], np.int64))
    ref = {"identity": {"gpt_checkpoint_sha256": "fixture-checkpoint"}, "arrays": {"prompt_semantic": "fixture"},
           "archive": {"file": "conditions.npz", "sha256": probe.sha256_file(reference / "conditions.npz"),
                       "bytes": (reference / "conditions.npz").stat().st_size}}
    write_json(reference / "manifest.json", ref)
    capture_dir = directory / "captures"
    (capture_dir / "references" / "中性").mkdir(parents=True)
    write_json(capture_dir / "references" / "中性" / "manifest.json", ref)
    capture = capture_dir / "fixed.npz"
    np.savez(capture, gpt_all_phones=np.array([1, 2, 3], np.int64), gpt_all_bert=np.zeros((4, 3), np.float32),
             sampled_tokens=np.array([[4], [5], [6]], np.int32))
    write_json(capture.with_suffix(".json"), {"request": {"name": "fixed-history-fixture"}})
    return package, manifest, reference, capture


def input_fixture(directory, model_identity, *, dtype="float16"):
    directory.mkdir()
    arrays, info = {}, {}
    for name, contract in model_identity["selected_weights"].items():
        x = np.ones(contract["input_shape"], dtype=dtype)
        output = np.zeros(contract["output_shape"], dtype=contract["output_dtype"])
        arrays["input_" + name], arrays["output_" + name] = x, output
        info[name] = {"input_sha256": probe.array_sha256(x), "output_sha256": probe.array_sha256(output)}
    np.savez(directory / "inputs.npz", **arrays)
    record = {"format": "sakuratts-gemv-inputs-v1", "status": "captured", "model": deepcopy(model_identity),
        "sources_sha256": {"src/sakuratts/backends/cuda/gpt.py": probe.sha256_file(ROOT / "src/sakuratts/backends/cuda/gpt.py")},
        "archive_sha256": probe.sha256_file(directory / "inputs.npz"), "arrays": info}
    write_json(directory / "inputs.json", record)
    write_json(directory / "result.json", {"status": "captured", "sources_changed": [],
        "input_metadata_sha256": probe.sha256_file(directory / "inputs.json")})
    return record


class GEMVProbeTests(unittest.TestCase):
    def test_five_current_decode_contracts_preserve_fp32_logits(self):
        contracts = probe.shape_contracts({"hidden_dim": 512, "ffn_dim": 2048, "vocab_size": 1025, "layers": 24}, 12)
        self.assertEqual([contracts[n]["weight_shape"] for n in probe.SHAPES],
                         [[1536, 512], [512, 512], [2048, 512], [512, 2048], [1025, 512]])
        self.assertEqual([contracts[n]["output_dtype"] for n in probe.SHAPES], ["float16"] * 4 + ["float32"])
        self.assertEqual(contracts["ffn_out"]["weight_name"], "layers.12.ffn_out.weight")
        self.assertEqual(contracts["output"]["weight_name"], "output.weight")
        for invalid in (-1, 24, True):
            with self.assertRaises(ValueError):
                probe.shape_contracts({"hidden_dim": 512, "ffn_dim": 2048, "vocab_size": 1025, "layers": 24}, invalid)

    def test_fixed_tree_preserves_tail_and_output_rounding_on_exact_products(self):
        # Integer products/sums stay exact in FP32; the independent FP64 dot
        # therefore gives the expected tree result without replaying its loops.
        for rows, columns in ((1, 1), (5, 31), (9, 33), (1025, 65), (7, 512), (3, 2048)):
            x = ((np.arange(columns) % 5) - 2).astype(np.float16)[None]
            weight = ((np.arange(rows * columns).reshape(rows, columns) % 11) - 5).astype(np.float16)
            for output_dtype in (np.float16, np.float32):
                expected = (x.astype(np.float64) @ weight.astype(np.float64).T).astype(output_dtype)
                actual = probe.emulate_warp(x, weight, output_dtype)
                self.assertEqual(actual.dtype, np.dtype(output_dtype))
                np.testing.assert_array_equal(actual, expected)

    def test_strict_errors_and_nonfinite_values_cannot_be_hidden_by_argmax(self):
        expected = np.array([[0, 1, 2]], np.float32)
        actual = expected.copy()
        actual[0, 1] += .01
        result = probe.compare(actual, expected)
        self.assertTrue(result["argmax_equal"])
        self.assertFalse(result["strict_passed"])
        self.assertEqual(result["strict_outside_count"], 1)
        for value in (np.nan, np.inf):
            actual[0, 1] = value
            self.assertFalse(probe.compare(actual, expected)["strict_passed"])

    def test_weight_checksum_and_selected_layer_identity_are_enforced(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            package, _, _, _ = fixture(directory)
            _, weights, identity = probe.load_weights(package, list(probe.SHAPES), 0)
            self.assertEqual(set(weights), set(probe.SHAPES))
            self.assertTrue(all(w.dtype == np.float16 for w in weights.values()))
            bundle = directory / "inputs"
            input_fixture(bundle, identity)
            loaded, _ = probe.read_input_bundle(bundle, identity, list(probe.SHAPES))
            self.assertEqual(set(loaded), set(probe.SHAPES))
            wrong_layer = deepcopy(identity)
            wrong_layer["selected_weights"]["qkv"]["weight_name"] = "layers.1.qkv.weight"
            with self.assertRaisesRegex(ValueError, "weight/layer"):
                probe.read_input_bundle(bundle, wrong_layer, ["qkv"])
            with (package / "weights.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                probe.load_weights(package, ["qkv"], 0)

    def test_bundle_rejects_half_contract_violation_and_failed_capture_cleanup(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            package, _, _, _ = fixture(directory)
            _, _, identity = probe.load_weights(package, ["qkv"], 0)
            invalid = directory / "wrong-dtype"
            input_fixture(invalid, identity, dtype="float32")
            with self.assertRaisesRegex(ValueError, "dtype or shape"):
                probe.read_input_bundle(invalid, identity, ["qkv"])
            bundle = directory / "failed-capture"
            input_fixture(bundle, identity)
            result = probe.read_json(bundle / "result.json")
            result["status"] = "failed"
            write_json(bundle / "result.json", result)
            with self.assertRaisesRegex(ValueError, "finish and close"):
                probe.read_input_bundle(bundle, identity, ["qkv"])

    def test_fixed_history_bounds_reference_identity_and_cpu_only_cli(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            package, manifest, reference, capture = fixture(directory)
            _, _, history = probe.fixed_history(capture, reference, manifest, 2, 16)
            self.assertEqual(history["fixed_prefix_tokens"], [4, 5])
            for step, capacity in ((0, 16), (3, 16), (2, 6)):
                with self.assertRaises(ValueError):
                    probe.fixed_history(capture, reference, manifest, step, capacity)
            output = directory / "check"
            # A fresh process blocks any attempted CUDA backend import. It also
            # avoids relying on other unit tests' fake modules in sys.modules.
            code = """import builtins,runpy,sys
original=builtins.__import__
def blocked(name,*args,**kwargs):
 if name=='cupy' or name.startswith('cupy.') or name=='sakuratts.backends.cuda.gpt':
  raise AssertionError('CPU-only CLI tried to import CUDA: '+name)
 return original(name,*args,**kwargs)
builtins.__import__=blocked
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
            command = [sys.executable, "-c", code, str(ROOT / "research/tools/windows_gemv_probe.py"), "--mode", "record",
                "--gpt", str(package), "--reference", str(reference), "--capture", str(capture),
                "--output", str(output), "--check-only", "--capacity", "16", "--decode-step", "2"]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            record = probe.read_json(output / "result.json")
            self.assertEqual(record["status"], "configuration_verified")
            self.assertFalse(record["cuda_backend_imported"])
            self.assertFalse((output / "inputs.npz").exists())
            before = (output / "result.json").read_bytes()
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(before, (output / "result.json").read_bytes())
            ref = probe.read_json(reference / "manifest.json")
            ref["identity"]["gpt_checkpoint_sha256"] = "other-checkpoint"
            write_json(reference / "manifest.json", ref)
            with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                probe.fixed_history(capture, reference, manifest, 1, 16)


if __name__ == "__main__":
    unittest.main()
