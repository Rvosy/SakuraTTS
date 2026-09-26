"""DirectML mixed precision requires its own checked package and explicit opt-in."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from sakuratts.backends.cpu.engine import CPUEngine, DirectMLEngine
from sakuratts.backends.onnx.sovits import (DIRECTML_FP16_EXECUTION_OPTIONS, DIRECTML_FP16_KIND,
    DIRECTML_FP16_SCREEN_VERSION, FP16_EXECUTION_OPTIONS, FP16_SCREEN_VERSION, INPUT_NAMES, ORTSoVITS, read_manifest)
from sakuratts._internal.reference_condition import sha256_file
from sakuratts._internal.diagnostics import check_prepared_packages
from test_nvidia_package_startup import fixture
from test_ort_directml import SessionOptions
from test_ort_sovits import manifest


def package(root, *, backend="directml"):
    root.mkdir(exist_ok=True)
    for name in ("weights.bin", "decode.onnx", "diagnostic.onnx"):
        (root / name).write_bytes(name.encode())

    def spec(name):
        path = root / name
        return {"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}

    metadata = manifest()
    metadata.update(dtype="float16", weights=spec("weights.bin"),
        graphs={"decode": spec("decode.onnx"), "diagnostic": spec("diagnostic.onnx")},
        precision={"profile": "fp16-mixed-v1", "fp16_scope": "vocoder", "keep_io_types": True,
                   "input_dtype": "float32", "output_dtype": "float32", "source_manifest_sha256": "fp32-source",
                   "ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": False})
    directml = backend == "directml"
    report = {"engineering_screen": {"version": DIRECTML_FP16_SCREEN_VERSION if directml else FP16_SCREEN_VERSION,
                                     "passed": True}, "backend": backend, "onnxruntime": "test-version",
              "hardware": {"device_id": 0, "description": "test-adapter", "display_drivers": [{"DriverVersion": "test"}]},
              "ort_execution_options": dict(DIRECTML_FP16_EXECUTION_OPTIONS if directml else FP16_EXECUTION_OPTIONS),
              "candidate_graph_sha256": metadata["graphs"]["decode"]["sha256"],
              "candidate_diagnostic_sha256": metadata["graphs"]["diagnostic"]["sha256"],
              "candidate_weights_sha256": metadata["weights"]["sha256"], "source_manifest_sha256": "fp32-source",
              "ort_graph_optimization_level": "ORT_ENABLE_ALL", "ort_use_deterministic_compute": False,
              "runs": {"fp16-profile": {"profile": {"directml_fp16_convolution_observed": True,
                                                   "cpu_neural_compute_events": {}}}}}

    def save():
        (root / "validation.json").write_text(json.dumps(report), encoding="utf-8")
        metadata["validation"] = dict(spec("validation.json"), passed=True,
                                     kind=DIRECTML_FP16_KIND if directml else "fp16-engineering-screen")
        (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")

    save()
    return metadata, report, save


class DirectMLPrecisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.metadata, self.report, self.save = package(self.root)
        self.session = Mock()
        self.session.get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.session.get_inputs.return_value = [SimpleNamespace(name=name, type="tensor(int64)" if i < 2 else "tensor(float)")
                                                for i, name in enumerate(INPUT_NAMES)]
        self.session.get_outputs.return_value = [SimpleNamespace(name="waveform", type="tensor(float)")]
        self.ort = SimpleNamespace(SessionOptions=SessionOptions,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
            InferenceSession=Mock(return_value=self.session),
            get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.enterContext(patch.dict(sys.modules, {"onnxruntime": self.ort}))

    def test_opt_in_loads_mixed_graph_with_fp32_io_and_no_runtime_fallback(self):
        with self.assertRaisesRegex(ValueError, "allow_experimental_fp16"):
            ORTSoVITS.load(self.root, device="directml", enable_cpu_mem_arena=False)
        model = ORTSoVITS.load(self.root, device="directml", enable_cpu_mem_arena=False,
                              allow_experimental_fp16=True)
        self.assertEqual(model.encoder.manifest["precision"]["fp16_scope"], "vocoder")
        self.assertEqual(self.ort.InferenceSession.call_args.args[0], str(self.root / "decode.onnx"))
        self.assertFalse(self.ort.InferenceSession.call_args.kwargs["enable_fallback"])

    def test_unscreened_session_options_and_half_public_io_are_rejected(self):
        for options in ({"device_id": 1}, {"intra_op_num_threads": 3}, {"enable_cpu_mem_arena": True}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "screened session options"):
                ORTSoVITS.load(self.root, device="directml", allow_experimental_fp16=True,
                              **dict({"enable_cpu_mem_arena": False}, **options))
        self.ort.InferenceSession.assert_not_called()
        self.session.get_outputs.return_value[0].type = "tensor(float16)"
        with self.assertRaisesRegex(ValueError, "FP32 public"):
            ORTSoVITS.load(self.root, device="directml", allow_experimental_fp16=True, enable_cpu_mem_arena=False)

    def test_matching_hashes_and_gpu_provenance_are_required(self):
        for change in ("graph", "gpu", "cpu_compute", "hardware"):
            with self.subTest(change=change):
                _, report, save = package(self.root)
                if change == "graph":
                    report["candidate_graph_sha256"] = "different"
                elif change == "gpu":
                    report["runs"]["fp16-profile"]["profile"]["directml_fp16_convolution_observed"] = False
                elif change == "cpu_compute":
                    report["runs"]["fp16-profile"]["profile"]["cpu_neural_compute_events"] = {"Conv": 1}
                else:
                    report.pop("hardware")
                save()
                with self.assertRaisesRegex(ValueError, "does not match|GPU execution evidence"):
                    read_manifest(self.root, allow_experimental_fp16=True)

    def test_cpu_and_cuda_cannot_reuse_directml_screen(self):
        for device, error in (("cpu", "not been screened for CPU"), ("cuda", "own CUDA engineering screen")):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, error):
                ORTSoVITS.load(self.root, device=device, allow_experimental_fp16=True)
        self.ort.InferenceSession.assert_not_called()

    def test_cuda_screen_does_not_admit_directml(self):
        package(self.root, backend="cuda")
        with self.assertRaisesRegex(ValueError, "own DirectML engineering screen"):
            ORTSoVITS.load(self.root, device="directml", allow_experimental_fp16=True, enable_cpu_mem_arena=False)
        self.ort.InferenceSession.assert_not_called()

    def test_runtime_checks_backend_screen_at_acoustic_load(self):
        for selected, screened in ((CPUEngine, "directml"), (DirectMLEngine, "cuda"), (DirectMLEngine, "directml")):
            with self.subTest(selected=selected.name, screened=screened), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _, _ = fixture(root)
                package(root / "sovits", backend=screened)
                with patch("sakuratts.frontend.runtime.load_frontend"):
                    engine = selected(config, allow_experimental_acoustic_fp16=True, load_references=False)
                    if selected.name != "directml" or screened != "directml":
                        with self.assertRaisesRegex(ValueError, "not been screened for CPU|own DirectML engineering screen"):
                            engine._load_sovits()
                        self.ort.InferenceSession.assert_not_called()
                    else:
                        engine._load_sovits()
                        self.assertIs(engine.sovits.session, self.session)
                        self.assertEqual(engine._execution_report()["acoustic_precision_scope"], "vocoder")
                    engine.close()

    def test_read_only_diagnostics_admit_screened_fp16_for_its_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _ = fixture(root)
            settings = json.loads(config.read_text())
            settings.update(references={}, backend={"preferred": "directml"})
            config.write_text(json.dumps(settings))
            (root / "python.exe").touch()
            package(root / "sovits")
            weights = root / "gpt/weights.bin"
            weights.write_bytes(b"gpt")
            path = root / "gpt/manifest.json"
            gpt = json.loads(path.read_text())
            gpt.update(format="sakuratts-gpt-fp32-v1", architecture="gpt-sovits-ar-postnorm-relu",
                       weights={"file": weights.name, "bytes": weights.stat().st_size, "sha256": sha256_file(weights)})
            path.write_text(json.dumps(gpt))
            with patch("sakuratts._internal.diagnostics.check_worker_imports", return_value={}) as probe:
                report = check_prepared_packages(config)
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["backend"], "directml")
                self.assertEqual(probe.call_args_list[0].kwargs, {"backend": "directml"})
                for backend in ("cpu", "cuda"):
                    settings["backend"]["preferred"] = backend
                    config.write_text(json.dumps(settings))
                    with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "matching backend engineering screen"):
                        check_prepared_packages(config)
            self.ort.InferenceSession.assert_not_called()


if __name__ == "__main__":
    unittest.main()
