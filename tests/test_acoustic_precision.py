"""Explicit experimental-package admission and acoustic regression screening."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research/tools")]
from sakuratts.backends.onnx.sovits import FP16_EXECUTION_OPTIONS, FP16_SCREEN_VERSION, INPUT_NAMES, ORTSoVITS, STAGES, read_manifest
from sakuratts._internal.reference_condition import sha256_file
from windows_acoustic_precision import compare, engineering_screen_passed, waveform_metrics, worker


class AcousticPrecisionTests(unittest.TestCase):
    def test_screen_gates_production_execution_and_every_repeat_stage(self):
        passed = {"passed": True}
        results = {"short": {name: deepcopy(passed) for name in ("engineering_metrics", "production_vs_diagnostic",
            "baseline_production_vs_diagnostic", "production_profile_vs_production")}}
        results["short"]["baseline_against_original_capture"] = {"mean": deepcopy(passed)}
        reports = {name: {"cases": {"short": {"repeat_checks": [{"waveform": deepcopy(passed),
            "mean": deepcopy(passed)}, {"waveform": deepcopy(passed), "mean": deepcopy(passed)}]}}}
            for name in ("baseline-diagnostic", "candidate-diagnostic", "baseline-production",
                         "candidate-production", "candidate-production-profile")}
        for name in ("candidate-diagnostic", "candidate-production-profile"):
            reports[name]["execution"] = {"cuda_fp16_convolution_observed": True, "cpu_neural_compute_events": {}}
        self.assertTrue(engineering_screen_passed(results, reports))
        changed = deepcopy(reports)
        changed["candidate-diagnostic"]["cases"]["short"]["repeat_checks"][0]["mean"]["passed"] = False
        self.assertFalse(engineering_screen_passed(results, changed))
        changed = deepcopy(reports)
        changed["baseline-production"]["cases"]["short"]["repeat_checks"][0]["waveform"]["passed"] = False
        self.assertFalse(engineering_screen_passed(results, changed))
        for key, value in (("cpu_neural_compute_events", {"Conv": 1}), ("cuda_fp16_convolution_observed", False)):
            changed = deepcopy(reports)
            changed["candidate-production-profile"]["execution"][key] = value
            self.assertFalse(engineering_screen_passed(results, changed))
        results["short"]["production_profile_vs_production"]["passed"] = False
        self.assertFalse(engineering_screen_passed(results, reports))

    def test_worker_retains_a_failing_middle_repeat(self):
        for changed_stage in ("waveform", "mean"):
            with self.subTest(changed_stage=changed_stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                package = root / "package"
                package.mkdir()
                def spec(name):
                    path = package / name
                    path.write_bytes(b"CPU test stub")
                    return {"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                manifest = {"weights": spec("weights.bin"), "graphs": {"diagnostic": spec("graph.onnx")},
                            "config": {"sample_rate": 32000}}
                (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                np.savez(root / "inputs.npz", **{name: np.zeros(1, np.float32) for name in INPUT_NAMES})
                (root / "inputs.json").write_text(json.dumps({"short": str(root / "inputs.npz")}), encoding="utf-8")
                first = [np.ones((1, 1, 4), np.float32) for _ in STAGES]
                middle = [value.copy() for value in first]
                middle[STAGES.index(changed_stage)] += 1
                session = Mock()
                session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                session.get_provider_options.return_value = {}
                session.get_inputs.return_value = [SimpleNamespace(name=name, type="tensor(int64)" if index < 2 else "tensor(float)")
                                                   for index, name in enumerate(INPUT_NAMES)]
                session.run.side_effect = [first, middle, first]
                ort = SimpleNamespace(__version__="CPU test stub", SessionOptions=SimpleNamespace,
                    GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1), InferenceSession=Mock(return_value=session))
                args = SimpleNamespace(output=root / "output", package=package, inputs=root / "inputs.json",
                                       diagnostic=True, profile=False, repeats=2)
                with patch.dict(sys.modules, {"onnxruntime": ort,
                    "sakuratts.backends.cuda.runtime": SimpleNamespace(configure_cuda=Mock())}):
                    worker(args)
                row = json.loads((root / "output/report.json").read_text(encoding="utf-8"))["cases"]["short"]
                self.assertFalse(row["repeat_stage_original_tolerance"][changed_stage]["passed"])
                self.assertFalse(row["repeat_checks"][0][changed_stage]["passed"])
                self.assertTrue(row["repeat_checks"][1][changed_stage]["passed"])

    def test_fp16_rejects_untested_execution_options(self):
        for name, value in (("device_id", 1), ("arena_extend_strategy", "kNextPowerOfTwo"),
                ("cudnn_conv_algo_search", "EXHAUSTIVE"), ("cudnn_conv_use_max_workspace", True),
                ("enable_mem_pattern", True), ("intra_op_num_threads", 8)):
            with self.subTest(name=name), patch("sakuratts.backends.onnx.sovits.read_manifest", return_value=({"dtype": "float16"}, Path("unused"))):
                with self.assertRaisesRegex(ValueError, "screened CUDA/session options"):
                    ORTSoVITS.load("unused", allow_experimental_fp16=True, **{name: value})

    def test_fp32_execution_options_remain_configurable(self):
        session = Mock()
        session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session.get_inputs.return_value = [SimpleNamespace(name=name) for name in INPUT_NAMES]
        session.get_outputs.return_value = [SimpleNamespace(name="waveform")]
        ort = SimpleNamespace(SessionOptions=SimpleNamespace, GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
            InferenceSession=Mock(return_value=session), get_available_providers=lambda: ["CUDAExecutionProvider"])
        with patch("sakuratts.backends.onnx.sovits.read_manifest", return_value=({"dtype": "float32", "config": {"sample_rate": 32000}}, Path("unused"))), \
             patch.dict(sys.modules, {"onnxruntime": ort, "sakuratts.backends.cuda.runtime": SimpleNamespace(configure_cuda=Mock())}):
            ORTSoVITS.load("unused", cudnn_conv_algo_search="EXHAUSTIVE", cudnn_conv_use_max_workspace=True,
                          enable_mem_pattern=True, intra_op_num_threads=8)
        options = ort.InferenceSession.call_args.kwargs
        self.assertEqual(options["providers"][0][1]["cudnn_conv_algo_search"], "EXHAUSTIVE")
        self.assertEqual(options["providers"][0][1]["cudnn_conv_use_max_workspace"], "1")
        self.assertTrue(options["sess_options"].enable_mem_pattern)
        self.assertEqual(options["sess_options"].intra_op_num_threads, 8)

    def test_engineering_screen_does_not_relabel_original_parity(self):
        original = (0.2 * np.sin(np.arange(8192) * 0.07)).astype(np.float32)
        candidate = original * np.float32(1.002)
        self.assertFalse(compare(candidate, original)["passed"])
        self.assertTrue(waveform_metrics(candidate, original)["passed"])

    def test_screen_rejects_truncation_nonfinite_and_amplitude_growth(self):
        original = (0.2 * np.sin(np.arange(8192) * 0.07)).astype(np.float32)
        self.assertFalse(waveform_metrics(original[:-1], original)["passed"])
        self.assertFalse(waveform_metrics(np.full_like(original, np.nan), original)["passed"])
        grown = waveform_metrics(original * 1.1, original)
        self.assertFalse(grown["passed"])
        self.assertFalse(grown["checks"]["amplitude"])

    def test_spectral_screen_rejects_changed_frequency_content(self):
        original = (0.2 * np.sin(np.arange(8192) * 0.07)).astype(np.float32)
        altered = (0.2 * np.sin(np.arange(8192) * 0.075)).astype(np.float32)
        report = waveform_metrics(altered, original)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["spectral_convergence"])

    def test_fp16_requires_opt_in_and_matching_screening_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def spec(name):
                path = root / name
                return {"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            (root / "weights.bin").write_bytes(b"half-weights")
            (root / "acoustic.onnx").write_bytes(b"half-graph")
            metadata = {"format": "sakuratts-sovits-onnx-v1", "dtype": "float16",
                "config": {"model": {"version": "v2ProPlus"}, "semantic_upsample_factor": 2},
                "precision": {"profile": "fp16-mixed-v1", "keep_io_types": True,
                              "input_dtype": "float32", "output_dtype": "float32",
                              "ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": True},
                "weights": spec("weights.bin"), "graphs": {"decode": spec("acoustic.onnx"), "diagnostic": spec("acoustic.onnx")}}
            report = {"engineering_screen": {"version": FP16_SCREEN_VERSION, "passed": True},
                      "ort_execution_options": dict(FP16_EXECUTION_OPTIONS),
                      "candidate_graph_sha256": metadata["graphs"]["decode"]["sha256"],
                      "candidate_diagnostic_sha256": metadata["graphs"]["diagnostic"]["sha256"],
                      "candidate_weights_sha256": metadata["weights"]["sha256"],
                      "ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": True}
            def save():
                (root / "validation.json").write_text(json.dumps(report), encoding="utf-8")
                metadata["validation"] = dict(spec("validation.json"), kind="fp16-engineering-screen", passed=True)
                (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
            save()
            with self.assertRaisesRegex(ValueError, "allow_experimental_fp16"):
                read_manifest(root)
            read_manifest(root, allow_experimental_fp16=True)
            report["engineering_screen"]["version"] = FP16_SCREEN_VERSION - 1
            save()
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_manifest(root, allow_experimental_fp16=True)
            report["engineering_screen"]["version"] = FP16_SCREEN_VERSION
            report["ort_execution_options"]["cudnn_conv_algo_search"] = "EXHAUSTIVE"
            save()
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_manifest(root, allow_experimental_fp16=True)
            report["ort_execution_options"] = dict(FP16_EXECUTION_OPTIONS)
            report["candidate_graph_sha256"] = "different graph"
            save()
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_manifest(root, allow_experimental_fp16=True)
            report["candidate_graph_sha256"] = metadata["graphs"]["decode"]["sha256"]
            report["engineering_screen"]["passed"] = False
            save()
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_manifest(root, allow_experimental_fp16=True)


if __name__ == "__main__":
    unittest.main()
