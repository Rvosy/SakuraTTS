"""DirectML screening rejects numerical failures and neural CPU fallback."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from directml_acoustic_precision import INPUT_NAMES, STAGES, engineering_screen_passed, main, sha256_file


class DirectMLScreenTests(unittest.TestCase):
    def report(self):
        row = {"finite": True, "repeat_checks": [{"waveform": {"passed": True}}],
               "engineering_metrics": {"passed": True}, "production_compare": {"passed": True},
               "original_fp32_checks": {"waveform": {"passed": True}}}
        run = {"public_io_fp32": True, "cases": {str(i): deepcopy(row) for i in range(6)}}
        report = {"runs": {name: deepcopy(run) for name in ("fp32", "fp16", "fp16-diagnostic", "fp16-profile")}}
        report["runs"]["fp16-profile"]["profile"] = {
            "directml_fp16_convolution_observed": True, "cpu_neural_compute_events": {}}
        return report

    def test_finite_shape_and_engineering_accuracy_are_required(self):
        report = self.report()
        self.assertTrue(engineering_screen_passed(report))
        for change in ({"finite": False}, {"engineering_metrics": {"passed": False}},
                       {"repeat_checks": [{"waveform": {"passed": False}}]},
                       {"production_compare": {"passed": False}}):
            candidate = deepcopy(report)
            candidate["runs"]["fp16-profile"]["cases"]["0"].update(change)
            self.assertFalse(engineering_screen_passed(candidate))

    def test_actual_fp16_gpu_convolution_and_fp32_io_are_required(self):
        for key, value in (("directml_fp16_convolution_observed", False),
                           ("cpu_neural_compute_events", {"Conv": 1})):
            report = self.report()
            report["runs"]["fp16-profile"]["profile"][key] = value
            self.assertFalse(engineering_screen_passed(report))
        report = self.report()
        report["runs"]["fp16"]["public_io_fp32"] = False
        self.assertFalse(engineering_screen_passed(report))

    def test_baseline_failure_and_missing_short_long_cases_do_not_pass(self):
        report = self.report()
        report["runs"]["fp32"]["cases"]["0"]["original_fp32_checks"]["waveform"]["passed"] = False
        self.assertFalse(engineering_screen_passed(report))
        report = self.report()
        del report["runs"]["fp16"]["cases"]["5"]
        self.assertFalse(engineering_screen_passed(report))

    def test_screen_executes_manifest_paths_instead_of_conventional_filenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, candidate = root / "baseline", root / "candidate"
            metadata = {}
            for folder in (baseline, candidate):
                folder.mkdir()
                specs = {}
                for name in ("custom-weights.data", "custom-production.onnx", "custom-diagnostic.onnx"):
                    path = folder / name
                    path.write_bytes(name.encode())
                    specs[name] = {"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                metadata[folder] = {"dtype": "float32" if folder == baseline else "float16",
                    "weights": specs["custom-weights.data"],
                    "graphs": {"decode": specs["custom-production.onnx"], "diagnostic": specs["custom-diagnostic.onnx"]}}
            (baseline / "manifest.json").write_text(json.dumps(metadata[baseline]))
            metadata[candidate]["precision"] = {"source_manifest_sha256": sha256_file(baseline / "manifest.json"),
                "ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": False}
            (candidate / "manifest.json").write_text(json.dumps(metadata[candidate]))
            inputs = {name: np.zeros((1, 1, 2), np.float32) for name in INPUT_NAMES}
            inputs.update(codes=np.zeros((1, 1, 1), np.int64), phones=np.zeros((1, 1), np.int64))
            inputs.update({"expected_" + name: np.zeros(1, np.float32) for name in STAGES})
            for index in range(4):
                np.savez(baseline / f"validation-{index}.npz", **inputs)
            arguments = ["screen", "--baseline", str(baseline), "--candidate", str(candidate), "--output", str(root / "output")]
            with patch.object(sys, "argv", arguments), \
                    patch("directml_acoustic_precision.capture_hardware", return_value={}), \
                    patch("directml_acoustic_precision.read_manifest", return_value=(metadata[baseline], None)), \
                    patch("directml_acoustic_precision.ort.InferenceSession", side_effect=RuntimeError("session intercepted")) as session:
                with self.assertRaisesRegex(RuntimeError, "session intercepted"):
                    main()
            self.assertEqual(session.call_args.args[0], str(baseline / "custom-production.onnx"))


if __name__ == "__main__":
    unittest.main()
