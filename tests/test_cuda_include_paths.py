"""NVRTC path admission is limited to installed GPT compiler headers."""

import builtins
import importlib
from importlib import metadata
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
cli = importlib.import_module("sakuratts.cli")
cuda_runtime = importlib.import_module("sakuratts.cuda_runtime")


def distribution(root, name, header):
    info = root / (name.replace("-", "_") + "-1.0.dist-info")
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: " + name + "\nVersion: 1.0\n", encoding="utf-8")
    path = root / header
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// compiler header fixture\n", encoding="utf-8")
    return metadata.Distribution.at(info)


class CudaIncludePathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sakuratts include tests ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def headers(self, *, cupy_root="cupy with spaces", runtime_root="cuda runtime with spaces"):
        return {
            "cupy-cuda12x": distribution(self.root / cupy_root, "cupy-cuda12x", "cupy/_core/include/cupy/complex.cuh"),
            "nvidia-cuda-runtime-cu12": distribution(self.root / runtime_root, "nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime/include/cuda_runtime.h"),
        }

    def test_real_metadata_locations_allow_spaces_without_restricting_resource_paths(self):
        installed = self.headers()
        with patch("importlib.metadata.distribution", side_effect=installed.__getitem__), \
                patch.object(sys, "executable", str(self.root / "日本語/python.exe")):
            result = cuda_runtime.validate_gpt_cuda_include_paths()
        self.assertEqual(result["cupy-cuda12x"], str(self.root / "cupy with spaces/cupy/_core/include"))
        self.assertEqual(result["nvidia-cuda-runtime-cu12"], str(self.root / "cuda runtime with spaces/nvidia/cuda_runtime/include"))

    def test_each_non_ascii_include_is_rejected_with_recreation_advice(self):
        for culprit in ("cupy", "runtime"):
            with self.subTest(culprit=culprit):
                installed = self.headers(cupy_root=culprit + (" 日本語" if culprit == "cupy" else " ASCII"),
                    runtime_root=culprit + (" 日本語 runtime" if culprit == "runtime" else " ASCII runtime"))
                with patch("importlib.metadata.distribution", side_effect=installed.__getitem__):
                    with self.assertRaisesRegex(RuntimeError, "ASCII-only") as caught:
                        cuda_runtime.validate_gpt_cuda_include_paths()
                message = str(caught.exception)
                self.assertIn("Recreate the Python environment", message)
                self.assertIn("spaces are supported", message)
                self.assertIn("Model and Japanese resource paths", message)
                self.assertIn("cupy-cuda12x" if culprit == "cupy" else "nvidia-cuda-runtime-cu12", message)

    def test_missing_metadata_header_fails_before_compilation(self):
        installed = self.headers()
        Path(installed["cupy-cuda12x"].locate_file("cupy/_core/include/cupy/complex.cuh")).unlink()
        with patch("importlib.metadata.distribution", side_effect=installed.__getitem__):
            with self.assertRaisesRegex(RuntimeError, "Missing GPT CUDA compiler header"):
                cuda_runtime.validate_gpt_cuda_include_paths()

    def test_gpt_import_rejects_bad_headers_before_importing_cupy(self):
        path = ROOT / "src/sakuratts/cuda_gpt.py"
        spec = importlib.util.spec_from_file_location("sakuratts._include_path_test", path)
        module = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__
        cupy_imports = []

        def importing(name, *args, **kwargs):
            if name == "cupy":
                cupy_imports.append(name)
                raise AssertionError("CuPy imported before header validation")
            return original_import(name, *args, **kwargs)

        with patch.object(cuda_runtime, "validate_gpt_cuda_include_paths", side_effect=RuntimeError("ASCII-only fixture")), \
                patch.object(cuda_runtime, "configure_cuda") as configured, patch("builtins.__import__", side_effect=importing):
            with self.assertRaisesRegex(RuntimeError, "ASCII-only fixture"):
                spec.loader.exec_module(module)
        self.assertEqual(cupy_imports, [])
        configured.assert_not_called()

    def test_doctor_nvidia_reports_header_failure_without_claiming_dependencies_ready(self):
        with patch.object(cli, "JAPANESE_MODULES", {}), patch.object(cli, "import_module"), \
                patch.object(cli.metadata, "version", return_value="test"), \
                patch.object(cli.platform, "system", return_value="Windows"), \
                patch.object(cuda_runtime, "configure_cuda"), \
                patch.object(cuda_runtime, "validate_gpt_cuda_include_paths", side_effect=RuntimeError("ASCII-only fixture")):
            result = cli.doctor(nvidia=True)
        self.assertEqual(result["gpt_cuda_headers"], {"status": "failed", "error": "ASCII-only fixture"})
        self.assertFalse(result["checks_passed"])
        self.assertFalse(result["synthesis"]["dependencies_ready"])
        self.assertFalse(result["synthesis"]["inference_tested"])

    def test_plain_doctor_and_ort_configuration_do_not_check_gpt_headers(self):
        with patch.object(cuda_runtime, "validate_gpt_cuda_include_paths", side_effect=AssertionError("GPT-only check")) as validate, \
                patch.object(cli, "import_module"), patch.object(cli.metadata, "version", return_value="test"):
            result = cli.doctor()
            with patch.object(sys, "path", []), patch.object(sys, "executable", str(self.root / "python.exe")):
                self.assertEqual(cuda_runtime.configure_cuda(), [])
        validate.assert_not_called()
        self.assertTrue(result["checks_passed"])
        self.assertNotIn("gpt_cuda_headers", result)


if __name__ == "__main__":
    unittest.main()
