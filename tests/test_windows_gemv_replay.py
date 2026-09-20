"""CPU admission, decode-only routing, output contracts and strict replay gates."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import windows_gemv_replay as harness


def fixture(directory):
    config = {"hidden_dim": 8, "ffn_dim": 32, "vocab_size": 17, "layers": 2,
        "bert_dim": 4, "phoneme_vocab_size": 20, "max_positions": 100}
    package, reference = directory / "gpt", directory / "reference"
    package.mkdir()
    reference.mkdir()
    weights = {name: np.ones(c["weight_shape"], np.float32) / 8 for name, c in harness.contracts(config).items()}
    np.savez(package / "weights.npz", **weights)
    manifest = {"format": "sakuratts-gpt-fp32-v1", "architecture": "gpt-sovits-ar-postnorm-relu", "config": config,
        "weights": {"file": "weights.npz", "sha256": harness.probe.sha256_file(package / "weights.npz"),
            "bytes": (package / "weights.npz").stat().st_size}, "source": {"checkpoint_sha256": "test-checkpoint"}}
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(reference / "conditions.npz", prompt_semantic=np.array([1, 2], np.int64))
    ref = {"identity": {"gpt_checkpoint_sha256": "test-checkpoint"}, "arrays": {"prompt_semantic": "fixture"},
        "archive": {"file": "conditions.npz", "sha256": harness.probe.sha256_file(reference / "conditions.npz"),
            "bytes": (reference / "conditions.npz").stat().st_size}}
    (reference / "manifest.json").write_text(json.dumps(ref), encoding="utf-8")
    captured_reference = directory / "references/中性"
    captured_reference.mkdir(parents=True)
    (captured_reference / "manifest.json").write_text(json.dumps(ref), encoding="utf-8")
    np.savez(directory / "capture.npz", gpt_all_phones=np.array([1, 2, 3], np.int64),
        gpt_all_bert=np.zeros((4, 3), np.float32), sampled_tokens=np.array([[4], [5], [6]], np.int32),
        raw_logits=np.ones((3, 17), np.float32))
    (directory / "capture.json").write_text(json.dumps({"request": {"name": "test"}}), encoding="utf-8")
    mapping = directory / "captures.json"
    mapping.write_text(json.dumps({"short": "capture.npz", "again": "capture.npz"}), encoding="utf-8")
    return ["--gpt", str(package), "--reference", str(reference), "--captures", str(mapping),
        "--output", str(directory / "result"), "--capacity", "32", "--repeats", "1"]


class Array:
    def __init__(self, shape, pointer, dtype="float16", contiguous=True):
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.data = SimpleNamespace(ptr=pointer)
        self.flags = SimpleNamespace(c_contiguous=contiguous)


class Model:
    def __init__(self):
        self.contracts = harness.contracts({"hidden_dim": 8, "ffn_dim": 32, "vocab_size": 17, "layers": 2})
        self.weights = {name: Array(c["weight_shape"], i + 1) for i, (name, c) in enumerate(self.contracts.items())}
        self.blas = SimpleNamespace(linear=Mock())
        self.precision, self.use_graph = "fp16", True
        self.graph, self.release_count = object(), 0
        self.failure = None

    def release_request_state(self):
        self.graph = self.keys = self.values = self.workspace = self.state = None
        self.release_count += 1

    def _decode_graph_body(self):
        for name, c in self.contracts.items():
            self.blas.linear(Array(c["input_shape"], 101), self.weights[name], Array(c["output_shape"], 102, c["output_dtype"]))
        if self.failure:
            raise self.failure


class GemvReplayTests(unittest.TestCase):
    def test_default_check_only_verifies_full_history_without_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = fixture(directory)
            run = subprocess.run([sys.executable, str(ROOT / "harness/windows_gemv_replay.py"), *args],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((directory / "result/result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "configuration_verified")
            self.assertFalse(report["cuda_backend_imported"])
            self.assertFalse(report["gpu_execution_requested"])
            self.assertFalse(report["quality_accepted"])
            self.assertEqual(report["captures"]["short"]["steps"], 3)
            self.assertEqual(report["captures"]["short"]["fixed_prefix_tokens"], [4, 5])
            self.assertEqual(len(report["model"]["selected_weights"]), 5)
            self.assertEqual(report["logits_arrays"], {})

    def test_admission_rejects_wrong_logits_reference_and_forbidden_shapes(self):
        for damage in ("logits", "reference", "qkv", "duplicate"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                args = fixture(directory)
                if damage == "logits":
                    with np.load(directory / "capture.npz", allow_pickle=False) as archive:
                        arrays = dict(archive)
                    arrays["raw_logits"] = arrays["raw_logits"][:2]
                    np.savez(directory / "capture.npz", **arrays)
                elif damage == "reference":
                    path = directory / "references/中性/manifest.json"
                    ref = json.loads(path.read_text(encoding="utf-8"))
                    ref["identity"]["gpt_checkpoint_sha256"] = "different"
                    path.write_text(json.dumps(ref), encoding="utf-8")
                elif damage == "qkv":
                    args += ["--shapes", "qkv"]
                else:
                    args += ["--shapes", "output", "output"]
                with self.assertRaises((ValueError, SystemExit)):
                    harness.main(args)
                self.assertFalse((directory / "result").exists())

    def test_routes_only_selected_decode_weights_and_keeps_prefill_output_on_cublas(self):
        for selected in (("attention_output",), ("ffn_out",), ("output",), harness.ALLOWED_SHAPES):
            with self.subTest(selected=selected):
                model, kernels = Model(), {"float16": Mock(), "float32": Mock()}
                original = model.blas.linear
                router = harness.DecodeLinearRouter(model, model.contracts, selected, kernels)
                router.install()
                self.assertIsNone(model.graph)
                output = model.weights["output.weight"]
                model.blas.linear(Array((1, 8), 111), output, Array((1, 17), 112, "float32"))
                self.assertEqual(original.call_count, 1)
                kernels["float32"].assert_not_called()
                model._decode_graph_body()
                expected_half = 2 * (int("attention_output" in selected) + int("ffn_out" in selected))
                self.assertEqual(kernels["float16"].call_count, expected_half)
                self.assertEqual(kernels["float32"].call_count, int("output" in selected))
                methods = {v["shape"]: v["method"] for v in router.evidence()["weights"].values()}
                self.assertEqual(methods["qkv"], "cublas")
                self.assertEqual(methods["ffn_in"], "cublas")
                self.assertEqual(router.evidence()["decode_body_invocations"], {"graph_warmup_or_capture": 1})
                model.use_graph = False
                model._decode_graph_body()
                self.assertEqual(router.evidence()["decode_body_invocations"]["eager"], 1)
                router.remove()
                self.assertIs(model.blas.linear, original)
                self.assertEqual(model.release_count, 2)

    def test_unknown_pointer_falls_back_and_alias_is_rejected(self):
        model = Model()
        original = model.blas.linear
        router = harness.DecodeLinearRouter(model, model.contracts, ("output",), {"float32": Mock()})
        router.active = True
        router.linear(Array((1, 8), 100), Array((17, 8), 999), Array((1, 17), 102, "float32"))
        original.assert_called_once()
        with self.assertRaisesRegex(ValueError, "aliases"):
            router.linear(Array((1, 8), 100), Array((17, 8), model.weights["output.weight"].data.ptr),
                Array((1, 17), 102, "float32"))
        model.weights["output.weight"].data.ptr = model.weights["layers.0.qkv.weight"].data.ptr
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            harness.DecodeLinearRouter(model, model.contracts, ("output",), {})

    def test_candidate_dtypes_shapes_and_contiguity_are_checked(self):
        for damage in ("input_dtype", "output_dtype", "shape", "contiguous"):
            with self.subTest(damage=damage):
                model, kernel = Model(), Mock()
                router = harness.DecodeLinearRouter(model, model.contracts, ("output",), {"float32": kernel})
                router.active = True
                x, out = Array((1, 8), 100), Array((1, 17), 102, "float32")
                if damage == "input_dtype": x.dtype = np.dtype("float32")
                if damage == "output_dtype": out.dtype = np.dtype("float16")
                if damage == "shape": x.shape = (2, 8)
                if damage == "contiguous": x.flags.c_contiguous = False
                with self.assertRaises(ValueError):
                    router.linear(x, model.weights["output.weight"], out)
                kernel.assert_not_called()

    def test_decode_exception_resets_scope_and_removal_releases_graph_first(self):
        model = Model()
        original_body, original_linear = model._decode_graph_body, model.blas.linear
        router = harness.DecodeLinearRouter(model, model.contracts, ("output",), {"float32": Mock()})
        router.install()
        model.failure = RuntimeError("decode failed")
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            model._decode_graph_body()
        self.assertFalse(router.active)
        count = original_linear.call_count
        model.blas.linear(Array((1, 8), 101), model.weights["output.weight"], Array((1, 17), 102, "float32"))
        self.assertEqual(original_linear.call_count, count + 1)
        release = model.release_request_state

        def checked_release():
            self.assertEqual(model.blas.linear, router.linear)
            release()

        model.release_request_state = checked_release
        router.remove()
        self.assertEqual(model._decode_graph_body, original_body)
        self.assertIs(model.blas.linear, original_linear)
        self.assertIsNone(model.graph)

    def test_replay_preserves_complete_history_and_does_not_feed_final_token(self):
        model = SimpleNamespace(prefill=Mock(return_value=np.ones((1, 17), np.float32)),
            decode=Mock(return_value=np.ones((1, 17), np.float32)))
        arrays = {"gpt_all_phones": np.array([1, 2], np.int64), "gpt_all_bert": np.zeros((4, 2), np.float32),
            "sampled_tokens": np.array([[4], [5], [6]], np.int32)}
        logits, timing = harness.replay(model, arrays, np.array([[1, 2]], np.int64))
        self.assertEqual(logits.shape, (3, 17))
        self.assertEqual(timing["steps"], 3)
        self.assertEqual([call.args[0] for call in model.decode.call_args_list], [4, 5])

    def test_strict_failure_is_not_hidden_by_official_or_lifecycle_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = fixture(directory)

            def run_variant(args, label, selected, contracts, identity, histories, prompt, outputs, report):
                report["variants"][label] = {"lifecycle_strict_passed": True, "official_fp32_strict_passed": False}
                result = {}
                for case, arrays in histories.items():
                    logits = np.ones_like(arrays["raw_logits"]) + (0.001 if selected else 0)
                    outputs[label + "__" + case] = logits
                    result[case] = logits
                return result

            with patch.object(harness, "run_variant", side_effect=run_variant), patch("builtins.print"):
                self.assertEqual(harness.main(args + ["--run-gpu", "--shapes", "output"]), 1)
            report = json.loads((directory / "result/result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "numerical_check_failed")
            self.assertFalse(report["candidate_replay_strict_passed"])
            self.assertFalse(report["official_fp32_strict_passed"])
            self.assertFalse(report["quality_accepted"])
            self.assertEqual(report["comparison"]["short"]["strict_outside_per_step"], [17, 17, 17])

    def test_every_observation_is_directly_compared_without_transitive_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = fixture(directory)

            def run_variant(args, label, selected, contracts, identity, histories, prompt, outputs, report):
                report["variants"][label] = {"lifecycle_strict_passed": True, "official_fp32_strict_passed": False}
                result = {}
                for case, arrays in histories.items():
                    result[case] = np.full_like(arrays["raw_logits"], 9e-5 if selected else 0)
                    outputs[label + "__" + case + "__run0"] = result[case]
                    outputs[label + "__" + case + "__warm"] = np.full_like(arrays["raw_logits"], 1.8e-4 if selected else 0)
                return result

            with patch.object(harness, "run_variant", side_effect=run_variant), patch("builtins.print"):
                self.assertEqual(harness.main(args + ["--run-gpu", "--shapes", "output"]), 1)
            report = json.loads((directory / "result/result.json").read_text(encoding="utf-8"))
            self.assertTrue(report["comparison"]["short"]["strict_passed"])
            self.assertFalse(report["comparison_observations"]["short__warm"]["strict_passed"])
            self.assertFalse(report["candidate_replay_strict_passed"])

    def test_invalid_observed_logits_are_preserved_and_model_is_closed(self):
        model = Model()
        model.close = Mock()
        model.weight_manifest = {"weights": {"sha256": "weights"}, "source": {"checkpoint_sha256": "checkpoint"}}
        model.config = {"layers": 2}
        identity = {"selected_weights": {}, "weights_file_sha256": "weights", "checkpoint_sha256": "checkpoint", "config": model.config}
        args = SimpleNamespace(gpt=Path("fixture"), capacity=32, attention="baseline", attention_chunk_size=256, repeats=1)
        histories = {"short": {"raw_logits": np.ones((3, 17), np.float32)}}
        invalid = np.full((3, 17), np.nan, np.float32)
        outputs, report = {}, {"variants": {}}
        with patch.dict(sys.modules, {"sakuratts.cuda_gpt": SimpleNamespace(CUDAGPT=SimpleNamespace(load=Mock(return_value=model))),
                "cupy": SimpleNamespace(RawKernel=Mock(return_value=Mock()))}), \
                patch.object(harness.probe, "gpu_environment", return_value={}), \
                patch.object(harness, "memory", return_value={}), \
                patch.object(harness, "replay", return_value=(invalid, {})):
            with self.assertRaisesRegex(ValueError, "finite"):
                harness.run_variant(args, "warp4", ("output",), model.contracts, identity,
                    histories, np.array([[1, 2]], np.int64), outputs, report)
        self.assertIs(outputs["warp4__short__warm"], invalid)
        self.assertEqual(report["variants"]["warp4"]["status"], "failed")
        model.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
