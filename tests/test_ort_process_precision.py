"""The persistent acoustic worker requires explicit FP16 admission on both sides."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.ort_process import ORTProcessSoVITS


class ORTProcessPrecisionTests(unittest.TestCase):
    def test_opt_in_reaches_manifest_validation_and_worker_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            python = root / "python.exe"
            python.write_bytes(b"interpreter fixture")
            manifest = {"dtype": "float16", "config": {"sample_rate": 32000}}
            process = Mock()
            process.poll.return_value = 0
            with patch("sakuratts.ort_process.read_manifest", return_value=(manifest, package / "graph.onnx")) as read, \
                 patch("sakuratts.ort_process.subprocess.Popen", return_value=process) as launch, \
                 patch("sakuratts.ort_process.read_message", return_value=({"status": "ready", "acoustic_dtype": "float16"}, {})):
                model = ORTProcessSoVITS(package, python, diagnostic=True, allow_experimental_fp16=True,
                                         acoustic_arena_shrink=True)
                read.assert_called_once_with(package.resolve(), diagnostic=True, allow_experimental_fp16=True)
                self.assertIn("--allow-experimental-fp16", launch.call_args.args[0])
                self.assertIn("--diagnostic", launch.call_args.args[0])
                self.assertIn("--acoustic-arena-shrink", launch.call_args.args[0])
                self.assertTrue(model.acoustic_arena_shrink)
                self.assertEqual(model.encoder.manifest["dtype"], "float16")
                model.close()

    def test_manifest_rejection_prevents_worker_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "python.exe"
            python.write_bytes(b"interpreter fixture")
            with patch("sakuratts.ort_process.read_manifest", side_effect=ValueError("FP16 requires opt-in")) as read, \
                 patch("sakuratts.ort_process.subprocess.Popen") as launch:
                with self.assertRaisesRegex(ValueError, "opt-in"):
                    ORTProcessSoVITS(root, python)
                read.assert_called_once_with(root.resolve(), diagnostic=False, allow_experimental_fp16=False)
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
