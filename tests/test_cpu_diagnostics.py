"""Backend diagnostics select the active ORT distribution without GPU work."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from sakuratts import cli
from sakuratts.runtime.diagnostics import check_windows_packages, check_prepared_packages, check_worker_imports
from sakuratts.module.reference_condition import sha256_file
from test_nvidia_package_startup import fixture


class CPUDoctorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config, _, _ = fixture(self.root)
        self.available = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.distributions = {"numpy": "test", "threadpoolctl": "test", "sakuratts": "test",
                              "onnxruntime-directml": "1.24.4"}
        self.ort = SimpleNamespace(get_available_providers=lambda: self.available,
                                   InferenceSession=Mock(side_effect=AssertionError("No model execution in doctor")))
        self.imports = self.enterContext(patch("sakuratts.cli.import_module", side_effect=self.import_module))
        self.enterContext(patch("sakuratts.cli.metadata.version", side_effect=self.version))
        self.enterContext(patch("sakuratts.cli.platform.system", return_value="Windows"))
        self.resource_check = self.enterContext(patch("sakuratts.runtime.diagnostics.check_windows_packages",
            side_effect=lambda *args, backend, profile=None: {
                "status": "passed", "profile": profile or {"cpu": "int8", "directml": "fp16"}[backend]}))
        self.cuda = self.enterContext(patch("sakuratts.backends.cuda.runtime.configure_cuda",
                                            side_effect=AssertionError("CPU/DirectML must not initialize CUDA")))
        self.adapters = self.enterContext(patch("sakuratts.backends.directml.devices.list_adapters",
            return_value=[{"device_id": 0, "description": "Integrated GPU", "dedicated_video_memory_bytes": 512,
                           "luid": "0x00000000_0x00000042",
                           "dedicated_system_memory_bytes": 0, "shared_system_memory_bytes": 8192},
                          {"device_id": 1, "description": "Discrete GPU", "dedicated_video_memory_bytes": 8192,
                           "luid": "0x00000000_0x00000064",
                           "dedicated_system_memory_bytes": 0, "shared_system_memory_bytes": 8192}]))

    def import_module(self, name):
        if name == "onnxruntime":
            return self.ort
        if name not in ("numpy", "threadpoolctl"):
            raise AssertionError("Unexpected backend dependency: " + name)
        return SimpleNamespace()

    def version(self, name):
        if name not in self.distributions:
            raise cli.metadata.PackageNotFoundError(name)
        return self.distributions[name]

    def prefer(self, backend):
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["backend"] = {"preferred": backend}
        self.config.write_text(json.dumps(config), encoding="utf-8")

    def test_missing_model_resource_keeps_cpu_preference(self):
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config.pop("sovits")
        config.update(format="sakuratts-model-v1", name="test", languages=["ja"],
                      backend={"preferred": "cpu"}, acoustic="missing-acoustics")
        (self.root / "model.json").write_text(json.dumps(config), encoding="utf-8")
        self.resource_check.side_effect = check_windows_packages
        report = cli.doctor(config=self.root)
        self.assertEqual(report["synthesis"]["backend"], "cpu")
        self.assertFalse(report["synthesis"]["packages_ready"])
        self.assertTrue(report["synthesis"]["dependencies_ready"])
        self.assertIn("missing-acoustics", report["resource_check"]["error"])
        self.cuda.assert_not_called()

    def test_directml_requires_its_provider_even_when_the_distribution_is_installed(self):
        self.available = ["CPUExecutionProvider"]
        report = cli.doctor(backend="directml")
        self.assertFalse(report["checks_passed"])
        self.assertFalse(report["synthesis"]["dependencies_ready"])
        self.assertIn("DmlExecutionProvider", report["packages"]["onnxruntime-directml"]["error"])
        self.cuda.assert_not_called()

    def test_dxgi_query_failure_is_diagnostic_and_does_not_block_provider_checks(self):
        self.adapters.side_effect = RuntimeError("DXGI query unavailable")
        report = cli.doctor(backend="directml")
        self.assertTrue(report["checks_passed"])
        self.assertTrue(report["synthesis"]["dependencies_ready"])
        self.assertEqual(report["directml"]["error"], "DXGI query unavailable")
        self.assertFalse(report["directml"]["execution_tested"])
        self.ort.InferenceSession.assert_not_called()

    def test_cpu_distribution_and_conflicting_ort_installations_are_distinguished(self):
        self.distributions.pop("onnxruntime-directml")
        self.distributions["onnxruntime"] = "1.30.0"
        self.available = ["CPUExecutionProvider"]
        report = cli.doctor(backend="cpu")
        self.assertTrue(report["checks_passed"])
        self.assertIn("onnxruntime", report["packages"])
        self.distributions["onnxruntime-directml"] = "1.24.4"
        report = cli.doctor(backend="cpu")
        self.assertFalse(report["checks_passed"])
        self.assertIn("exactly one", report["packages"]["onnxruntime"]["error"])

class CPUPackageDiagnosticsTests(unittest.TestCase):
    def test_diagnostic_child_checks_selected_provider_without_cuda_initialization(self):
        actual_run = subprocess.run
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

        def run_with_stubs(command, **kwargs):
            prelude = """import sys,types
numpy=types.ModuleType('numpy'); numpy.__version__='test'
ort=types.ModuleType('onnxruntime'); ort.__version__='test'
ort.get_available_providers=lambda: PROVIDERS
cuda=types.ModuleType('sakuratts.backends.cuda.runtime')
def forbidden(): raise AssertionError('CPU/DirectML diagnostics initialized CUDA')
cuda.configure_cuda=forbidden
sys.modules.update(numpy=numpy,onnxruntime=ort)
sys.modules['sakuratts.backends.cuda.runtime']=cuda
""".replace("PROVIDERS", repr(providers))
            command = list(command)
            command[3] = prelude + command[3]
            return actual_run(command, **kwargs)

        with patch("sakuratts.runtime.diagnostics.subprocess.run", side_effect=run_with_stubs):
            for backend in ("cpu", "directml"):
                report = check_worker_imports(Path(sys.executable), {}, backend=backend)
                self.assertEqual(report["available_providers"], providers)
                self.assertFalse(report["inference_tested"])
                self.assertFalse(report["cuda_execution_tested"])
                self.assertFalse(report["torch_imported"])
            providers[:] = ["CPUExecutionProvider"]
            with self.assertRaisesRegex(RuntimeError, "no DmlExecutionProvider"):
                check_worker_imports(Path(sys.executable), {}, backend="directml")

    def test_cpu_and_directml_probe_main_acoustics_and_keep_classic_frontend_worker(self):
        for backend in ("cpu", "directml"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                config_path, _, _ = fixture(root)
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config.update(backend={"preferred": backend}, references={})
                frontend_python = root / "python.exe"
                frontend_python.touch()
                config_path.write_text(json.dumps(config), encoding="utf-8")
                weights = root / "gpt/weights.npz"
                weights.write_bytes(b"gpt")
                manifest_path = root / "gpt/manifest.json"
                gpt = json.loads(manifest_path.read_text(encoding="utf-8"))
                gpt.update(format="sakuratts-gpt-fp32-v1", architecture="gpt-sovits-ar-postnorm-relu",
                    weights={"file": weights.name, "bytes": weights.stat().st_size, "sha256": sha256_file(weights)})
                manifest_path.write_text(json.dumps(gpt), encoding="utf-8")
                acoustic = {"source": gpt["source"], "config": {"model": {"version": "v2ProPlus"}}}
                with patch("sakuratts.module.sovits.read_manifest", return_value=(acoustic, None)), \
                        patch("sakuratts.runtime.diagnostics.check_worker_imports",
                              side_effect=lambda python, profile, **kwargs: {"executable": str(python)}) as probe:
                    report = check_prepared_packages(config_path)
                self.assertEqual(report["backend"], backend)
                self.assertEqual(report["worker"]["executable"], sys.executable)
                self.assertEqual(report["frontend_worker"]["executable"], str(frontend_python))
                self.assertEqual(probe.call_args_list[0].args, (Path(sys.executable), {}))
                self.assertEqual(probe.call_args_list[0].kwargs, {"backend": backend})
                self.assertEqual(probe.call_args_list[1].kwargs, {"acoustic": False})


class GPTPackageDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config, _, _ = fixture(self.root)
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config.update(backend={"preferred": "cpu"}, references={})
        self.config.write_text(json.dumps(config), encoding="utf-8")
        (self.root / "python.exe").touch()
        weights = self.root / "gpt/weights.npz"
        weights.write_bytes(b"weights checked without loading")
        self.gpt = {"format": "sakuratts-gpt-fp32-v1", "dtype": "float32", "config": {},
            "architecture": "gpt-sovits-ar-postnorm-relu", "weights": self.spec(weights),
            "source": {"checkpoint_sha256": "checkpoint", "official_commit": "source"}}
        (self.root / "gpt/manifest.json").write_text(json.dumps(self.gpt), encoding="utf-8")
        self.acoustic = {"dtype": "float32", "source": self.gpt["source"],
            "config": {"model": {"version": "v2ProPlus"}}}
        self.acoustic_reader = self.enterContext(patch("sakuratts.module.sovits.read_manifest",
            side_effect=lambda *args, **kwargs: (self.acoustic, None)))
        self.probe = self.enterContext(patch("sakuratts.runtime.diagnostics.check_worker_imports", return_value={}))
        self.compute = self.enterContext(patch("sakuratts.backends.cpu.onnx_gpt.ONNXCPUGPT.load",
            side_effect=AssertionError("Diagnostics must not load model sessions")))

    @staticmethod
    def spec(path):
        return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


    def test_missing_selected_sidecar_fails_instead_of_checking_only_original_weights(self):
        for backend, precision in (("cpu", "int8"), ("directml", "fp16")):
            with self.subTest(backend=backend), self.assertRaisesRegex(FileNotFoundError, "onnx-" + precision):
                check_windows_packages(self.config, backend=backend)
        self.probe.assert_not_called()
        self.compute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
