"""CPU checks for precision-harness input identity and failed-startup cleanup."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("windows_gpt_precision", ROOT / "harness/windows_gpt_precision.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def fixture(root):
    gpt, reference = root / "gpt", root / "reference"
    gpt.mkdir()
    reference.mkdir()
    (gpt / "manifest.json").write_text("{}", encoding="utf-8")
    archive = reference / "conditions.npz"
    np.savez(archive, prompt_semantic=np.array([1, 2], dtype=np.int64))
    checksum = harness.sha256_file(archive)
    (reference / "manifest.json").write_text(json.dumps({"archive": {
        "file": "conditions.npz", "bytes": archive.stat().st_size, "sha256": checksum}}), encoding="utf-8")
    np.savez(root / "capture.npz", gpt_all_phones=np.array([1], dtype=np.int64),
             gpt_all_bert=np.zeros((3, 1), np.float32), sampled_tokens=np.array([2], dtype=np.int64),
             raw_logits=np.zeros((1, 4), np.float32))
    mapping = root / "captures.json"
    mapping.write_text(json.dumps({"short": "capture.npz"}), encoding="utf-8")
    args = ["windows_gpt_precision.py", "--gpt", str(gpt), "--reference", str(reference),
            "--captures", str(mapping), "--output", str(root / "output"), "--precision", "fp16"]
    return args, checksum


class WindowsGptPrecisionTests(unittest.TestCase):
    def test_modified_reference_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, checksum = fixture(root)
            prompt, actual = harness.load_reference_prompt(root / "reference")
            self.assertEqual(actual, checksum)
            np.testing.assert_array_equal(prompt, [[1, 2]])
            np.savez(root / "reference/conditions.npz", prompt_semantic=np.array([3, 4], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "Reference archive SHA-256 or size mismatch"):
                harness.load_reference_prompt(root / "reference")

    def test_model_load_failure_is_saved_without_attempting_model_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, checksum = fixture(root)
            failure = RuntimeError("model loading failed")
            model_class = Mock()
            model_class.load.side_effect = failure
            with patch.object(sys, "argv", args), patch.dict(sys.modules, {
                    "sakuratts.cuda_gpt": SimpleNamespace(CUDAGPT=model_class), "cupy": SimpleNamespace()}):
                with self.assertRaises(RuntimeError) as caught:
                    harness.main()
            self.assertIs(caught.exception, failure)
            report = json.loads((root / "output/result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["engineering_passed"])
            self.assertEqual(report["reference_archive_sha256"], checksum)
            self.assertIn("model loading failed", report["error"])
            self.assertNotIn("cleanup_error", report)
            self.assertTrue((root / "output/logits.npz").is_file())

    def test_initial_report_failure_closes_model_and_preserves_original_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = fixture(root)
            model = Mock()
            model.close.side_effect = RuntimeError("cleanup failed")
            failure = ValueError("device information unavailable")
            runtime = SimpleNamespace(getDeviceProperties=Mock(side_effect=failure))
            with patch.object(sys, "argv", args), patch.dict(sys.modules, {
                    "sakuratts.cuda_gpt": SimpleNamespace(CUDAGPT=SimpleNamespace(load=Mock(return_value=model))),
                    "cupy": SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))}):
                with self.assertRaises(ValueError) as caught:
                    harness.main()
            self.assertIs(caught.exception, failure)
            model.close.assert_called_once()
            report = json.loads((root / "output/result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertIn("device information unavailable", report["error"])
            self.assertIn("cleanup failed", report["cleanup_error"])


if __name__ == "__main__":
    unittest.main()
