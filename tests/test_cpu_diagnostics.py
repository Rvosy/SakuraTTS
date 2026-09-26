"""Backend diagnostics select the active ORT distribution without GPU work."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from sakuratts import cli
from sakuratts._internal.diagnostics import check_windows_packages, check_prepared_packages, check_worker_imports
from sakuratts._internal.reference_condition import sha256_file
from test_nvidia_package_startup import fixture


class CPUDoctorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config, _, _ = fixture(self.root)
        self.available = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.distributions = {"numpy": "test", "threadpoolctl": "test", "sakuratts": "test",
                              "onnxruntime-directml": "1.24.4"}
        self.ort = SimpleNamespace(get_available_providers=lambda: self.available,
                                   InferenceSession=Mock(side_effect=AssertionError("No model execution in doctor")))
        self.imports = self.enterContext(patch("sakuratts.cli.import_module", side_effect=self.import_module))
        self.enterContext(patch("sakuratts.cli.metadata.version", side_effect=self.version))
        self.enterContext(patch("sakuratts.cli.platform.system", return_value="Windows"))
        self.resource_check = self.enterContext(patch("sakuratts._internal.diagnostics.check_windows_packages",
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

    def test_cpu_can_use_directml_distribution_and_reports_no_inference_verification(self):
        report = cli.doctor(backend="cpu")
        self.assertTrue(report["checks_passed"])
        self.assertTrue(report["synthesis"]["dependencies_ready"])
        self.assertEqual(report["synthesis"]["backend"], "cpu")
        self.assertIn("onnxruntime-directml", report["packages"])
        self.assertNotIn("onnxruntime", report["packages"])
        self.assertEqual(report["onnxruntime"]["required_provider"], "CPUExecutionProvider")
        self.assertFalse(report["onnxruntime"]["execution_tested"])
        self.assertFalse(report["synthesis"]["inference_tested"])
        self.assertFalse(report["synthesis"]["quality_validated"])
        self.cuda.assert_not_called()
        self.ort.InferenceSession.assert_not_called()

    def test_model_preference_and_explicit_override_select_package_checks(self):
        self.prefer("directml")
        report = cli.doctor(config=self.config)
        self.assertTrue(report["checks_passed"])
        self.assertEqual(report["synthesis"]["backend"], "directml")
        self.assertEqual(report["synthesis"]["profile"], "fp16")
        self.resource_check.assert_called_with(self.config, backend="directml")
        report = cli.doctor(config=self.config, backend="cpu")
        self.assertEqual(report["synthesis"]["backend"], "cpu")
        self.assertEqual(report["synthesis"]["profile"], "int8")
        self.resource_check.assert_called_with(self.config, backend="cpu")
        self.cuda.assert_not_called()

    def test_default_and_explicit_cpu_profile_are_the_same(self):
        self.prefer("cpu")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["doctor", str(self.config), "--profile", "int8"]), 0)
        self.resource_check.assert_called_with(str(self.config), backend="cpu", profile="int8")
        implicit = cli.doctor(config=self.config)
        self.assertEqual(json.loads(output.getvalue())["synthesis"]["profile"], implicit["synthesis"]["profile"])
        self.resource_check.assert_called_with(self.config, backend="cpu")
        self.ort.InferenceSession.assert_not_called()

    def test_damaged_cpu_package_does_not_trigger_cuda_checks(self):
        self.prefer("cpu")
        self.resource_check.side_effect = ValueError("checksum mismatch")
        report = cli.doctor(config=self.config)
        self.assertFalse(report["checks_passed"])
        self.assertTrue(report["synthesis"]["dependencies_ready"])
        self.assertFalse(report["synthesis"]["packages_ready"])
        self.assertIn("checksum mismatch", report["resource_check"]["error"])
        self.cuda.assert_not_called()

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

    def test_directml_lists_dxgi_adapter_ids_and_memory_without_model_execution(self):
        report = cli.doctor(backend="directml")
        self.assertTrue(report["checks_passed"])
        self.assertEqual(report["directml"]["device_id_scheme"], "IDXGIFactory.EnumAdapters")
        self.assertEqual(report["directml"]["adapters"], self.adapters.return_value)
        self.assertFalse(report["directml"]["execution_tested"])
        self.ort.InferenceSession.assert_not_called()

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

    def test_missing_threadpoolctl_is_reported(self):
        self.distributions.pop("threadpoolctl")
        report = cli.doctor(backend="cpu")
        self.assertFalse(report["synthesis"]["dependencies_ready"])
        self.assertIn("error", report["packages"]["threadpoolctl"])

    def test_cli_backend_override_and_nvidia_conflict(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["doctor", "--backend", "directml"]), 0)
        self.assertEqual(json.loads(output.getvalue())["synthesis"]["backend"], "directml")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.main(["doctor", "--nvidia", "--backend", "cpu"])
        self.assertEqual(error.exception.code, 1)
        self.cuda.assert_not_called()


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

        with patch("sakuratts._internal.diagnostics.subprocess.run", side_effect=run_with_stubs):
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
                root = Path(directory)
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
                with patch("sakuratts.backends.onnx.sovits.read_manifest", return_value=(acoustic, None)), \
                        patch("sakuratts._internal.diagnostics.check_worker_imports",
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
        self.root = Path(temporary.name)
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
        self.acoustic_reader = self.enterContext(patch("sakuratts.backends.onnx.sovits.read_manifest",
            side_effect=lambda *args, **kwargs: (self.acoustic, None)))
        self.probe = self.enterContext(patch("sakuratts._internal.diagnostics.check_worker_imports", return_value={}))
        self.compute = self.enterContext(patch("sakuratts.backends.cpu.onnx_gpt.ONNXCPUGPT.load",
            side_effect=AssertionError("Diagnostics must not load model sessions")))

    @staticmethod
    def spec(path):
        return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}

    def sidecar(self, precision):
        if precision != "fp32" and not (self.root / "gpt/onnx/manifest.json").exists():
            self.sidecar("fp32")
        directory = self.root / "gpt" / ("onnx" if precision == "fp32" else "onnx-" + precision)
        directory.mkdir()
        graph, embedding = directory / "transformer.onnx", directory / "embedding.npz"
        graph.write_bytes(b"graph checked without creating a session")
        embedding.write_bytes(b"embedding checked without loading")
        dtype = "float16" if precision == "fp16" else "float32"
        metadata = {"format": "sakuratts-gpt-onnx-cpu-v1", "architecture": self.gpt["architecture"],
            "config": {}, "cache": "sequence-major-delta-with-masked-sentinel-v1",
            "precision": precision, "graph_io_dtype": dtype, "cache_dtype": dtype, "prefill_query_chunk_size": 0,
            "source": {"manifest_sha256": sha256_file(self.root / "gpt/manifest.json"),
                "weights_sha256": self.gpt["weights"]["sha256"], "checkpoint_sha256": "checkpoint"},
            "graphs": {precision: self.spec(graph)}, "embedding": self.spec(embedding)}
        if precision != "fp32":
            original = self.root / "gpt/onnx/manifest.json"
            metadata.update(experimental=True, conversion={"input_manifest_sha256": sha256_file(original),
                "input_graph_sha256": json.loads(original.read_text())["graphs"]["fp32"]["sha256"]})
        (directory / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        return directory, metadata

    def acoustic_fp16(self, scope="all", backend="directml"):
        evidence = self.root / f"sovits/{backend}-experiment.json"
        evidence.write_text(json.dumps({"engineering_screen": {"passed": False}}), encoding="utf-8")
        self.acoustic.update(dtype="float16", precision={"fp16_scope": scope},
            experimental_validations={backend: self.spec(evidence)})
        (self.root / "sovits/manifest.json").write_text(json.dumps(self.acoustic), encoding="utf-8")

    def test_missing_selected_sidecar_fails_instead_of_checking_only_original_weights(self):
        for backend, precision in (("cpu", "int8"), ("directml", "fp16")):
            with self.subTest(backend=backend), self.assertRaisesRegex(FileNotFoundError, "onnx-" + precision):
                check_windows_packages(self.config, backend=backend)
        self.probe.assert_not_called()
        self.compute.assert_not_called()

    def test_selected_cpu_precision_reports_verified_storage_without_execution(self):
        self.sidecar("int8")
        report = check_windows_packages(self.config, profile="int8")
        self.assertEqual(report["gpt_resources"]["precision"], "int8")
        self.assertEqual(report["gpt_resources"]["cache_dtype"], "float32")
        self.assertEqual(report["profile"], "int8")
        self.assertFalse(report["gpt_resources"]["execution_tested_by_doctor"])
        self.compute.assert_not_called()

    def test_modified_selected_graph_fails_hash_check(self):
        directory, _ = self.sidecar("int8")
        (directory / "transformer.onnx").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum or size mismatch"):
            check_windows_packages(self.config, profile="int8")
        self.probe.assert_not_called()

    def static_sidecar(self):
        directory, source = self.sidecar("fp16")
        static = self.root / "gpt/directml-fp16-cap1280"
        static.mkdir()
        graph = static / "decode.onnx"
        graph.write_bytes(b"static graph checked without execution")
        metadata = {"format": "sakuratts-gpt-directml-static-v1", "config": {}, "capacity": 1280,
            "precision": "fp16", "graph_io_dtype": "float16", "graph": self.spec(graph),
            "cache": "fixed-capacity-ping-pong-masked-sentinel-v1",
            "source": {"manifest_sha256": sha256_file(directory / "manifest.json"),
                "graph_sha256": source["graphs"]["fp16"]["sha256"]}}
        (static / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        return graph

    def test_directml_default_requires_its_fp16_static_1280_resource(self):
        graph = self.static_sidecar()
        self.acoustic_fp16()
        manifest = graph.parent / "manifest.json"
        original = manifest.read_bytes()
        manifest.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "directml-fp16-cap1280"):
            check_windows_packages(self.config, backend="directml")
        self.probe.assert_not_called()
        manifest.write_bytes(original)
        report = check_windows_packages(self.config, backend="directml")
        self.assertEqual(report["profile"], "fp16")
        self.assertEqual(report["gpt_resources"]["backend"], "directml")
        self.assertEqual(report["gpt_resources"]["graph"], str(graph))
        self.assertEqual(report["gpt_resources"]["graph_io_dtype"], "float16")
        self.assertEqual(report["gpt_resources"]["cache_dtype"], "float16")
        self.assertFalse(report["gpt_resources"]["execution_tested_by_doctor"])
        self.assertFalse(report["acoustic_validation"]["engineering_screen_passed"])

    def test_fp16_profile_does_not_accept_vocoder_only_acoustics(self):
        self.static_sidecar()
        self.acoustic_fp16("vocoder")
        with self.assertRaisesRegex(ValueError, "requires a full acoustic FP16"):
            check_windows_packages(self.config, backend="directml")
        self.probe.assert_not_called()

    def test_cuda_chunked_profiles_forward_admission_options_without_wrong_screen_kind(self):
        self.acoustic.update(dtype="float16", format="sakuratts-sovits-chunked-v1",
            validation={"kind": "sakuratts-vocoder-chunk-screen-v1", "passed": True})
        (self.root / "sovits/manifest.json").write_text(json.dumps(self.acoustic), encoding="utf-8")
        for profile, session_policy in (("fp16", "resident"), ("low-memory", "staged"), ("minimum-memory", "staged")):
            with self.subTest(profile=profile):
                report = check_windows_packages(self.config, backend="cuda", profile=profile)
                self.assertEqual(report["status"], "passed")
                self.acoustic_reader.assert_called_with((self.root / "sovits").resolve(),
                    allow_experimental_fp16=True, acoustic_chunk_frames=256,
                    acoustic_arena_shrink=True, acoustic_session_policy=session_policy)

    def test_preparation_checks_base_files_without_accepting_a_runtime_fallback(self):
        self.assertEqual(check_prepared_packages(self.config)["status"], "passed")
        with self.assertRaisesRegex(FileNotFoundError, "onnx-int8"):
            check_windows_packages(self.config)


if __name__ == "__main__":
    unittest.main()
