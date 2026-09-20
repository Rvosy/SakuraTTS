"""The persistent acoustic worker requires explicit FP16 admission on both sides."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.backends.onnx.process import ORTProcessSoVITS


class ORTProcessPrecisionTests(unittest.TestCase):
    def test_opt_in_reaches_manifest_validation_and_worker_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            (package / "manifest.json").write_text("{}", encoding="utf-8")
            python = root / "python.exe"
            python.write_bytes(b"interpreter fixture")
            manifest = {"dtype": "float16", "config": {"sample_rate": 32000}}
            process = Mock()
            process.pid = 4321
            process.poll.return_value = None
            process.wait.return_value = 0
            from sakuratts._internal.reference_condition import sha256_file
            ready = {"status": "ready", "acoustic_dtype": "float16", "private_acoustic_process": True,
                "shared_cuda_process": False, "worker_pid": process.pid, "executable": str(python.resolve()),
                "package_manifest_sha256": sha256_file(package / "manifest.json"), "acoustic_arena_shrink": True,
                "diagnostic": True, "chunk_frames": None, "torch_imported": False, "onnx_imported": False,
                "providers": ["CUDAExecutionProvider"], "provider_options": {}}
            with patch("sakuratts.backends.onnx.process.read_manifest", return_value=(manifest, package / "graph.onnx")) as read, \
                 patch("sakuratts.backends.onnx.process.subprocess.Popen", return_value=process) as launch, \
                 patch("sakuratts.backends.onnx.process.read_message", return_value=(ready, {})):
                model = ORTProcessSoVITS(package, python, diagnostic=True, allow_experimental_fp16=True,
                                         acoustic_arena_shrink=True)
                read.assert_called_once_with(package.resolve(), diagnostic=True, allow_experimental_fp16=True,
                    acoustic_arena_shrink=True, acoustic_chunk_frames=None)
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
            with patch("sakuratts.backends.onnx.process.read_manifest", side_effect=ValueError("FP16 requires opt-in")) as read, \
                 patch("sakuratts.backends.onnx.process.subprocess.Popen") as launch:
                with self.assertRaisesRegex(ValueError, "opt-in"):
                    ORTProcessSoVITS(root, python)
                read.assert_called_once_with(root.resolve(), diagnostic=False, allow_experimental_fp16=False,
                    acoustic_arena_shrink=False, acoustic_chunk_frames=None)
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
