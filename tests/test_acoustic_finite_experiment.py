"""Explicit reduced-accuracy acceptance keeps backend execution evidence intact."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
               str(Path(__file__).resolve().parent),
               str(Path(__file__).resolve().parents[1] / "research/tools")]
from sakuratts.backends.cpu.engine import CPUEngine, DirectMLEngine
from sakuratts.backends.onnx.precision_experiment import finite_execution_passed
from sakuratts.backends.onnx.sovits import INPUT_NAMES, ORTSoVITS, read_manifest
from sakuratts._internal.diagnostics import check_prepared_packages
from sakuratts._internal.reference_condition import sha256_file
from directml_acoustic_precision import publish_experiment
from test_directml_precision import package
from test_nvidia_package_startup import fixture
from test_ort_directml import SessionOptions


def experiment(root, backend="directml"):
    metadata, report, save = package(root)
    metadata["precision"].update(fp16_scope="all", source_graphs={"decode": {"sha256": "source-graph"}},
                                 source_weights={"sha256": "source-weights"})
    report.update(status="completed", backend=backend, quality_accepted=False,
                  engineering_screen={"version": 1, "passed": False},
                  finite_experiment={"version": 1, "passed": True})
    provider = "CPUExecutionProvider" if backend == "cpu" else "DmlExecutionProvider"
    row = {"finite": True, "repeat_checks": [{"waveform": {"passed": True}}],
           "shape_matches_baseline": True, "engineering_metrics": {"passed": False, "max_abs_error": .15},
           "production_compare": {"passed": True}}
    report["runs"] = {}
    for name in ("fp32", "fp16", "fp16-diagnostic", "fp16-profile"):
        report["runs"][name] = {"providers": [provider], "public_io_fp32": True,
            "graph_sha256": "source-graph" if name == "fp32" else metadata["graphs"]["diagnostic" if name == "fp16-diagnostic" else "decode"]["sha256"],
            "weights_sha256": "source-weights" if name == "fp32" else metadata["weights"]["sha256"],
            "cases": {str(index): deepcopy(row) for index in range(4)}}
    report["runs"]["fp16-profile"]["profile"] = {
        "directml_fp16_convolution_observed": backend == "directml",
        "cpu_neural_compute_events": {"Conv": 4} if backend == "cpu" else {},
        "provider_events": {provider: 4},
        "typed_operator_events": {f"{provider}/Conv/float": 4}}
    save()
    metadata["validation"]["passed"] = False
    (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return report


class AcousticFiniteExperimentTests(unittest.TestCase):
    def test_finite_acceptance_preserves_failed_accuracy_and_requires_both_opt_ins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = experiment(root)
            publish_experiment(root, report)
            metadata = json.loads((root / "manifest.json").read_text())
            self.assertFalse(metadata["validation"]["passed"])
            self.assertFalse(report["engineering_screen"]["passed"])
            with self.assertRaisesRegex(ValueError, "did not pass"):
                read_manifest(root, allow_experimental_fp16=True)
            with self.assertRaisesRegex(ValueError, "allow_experimental_fp16"):
                read_manifest(root, fp16_acceptance="finite", execution_backend="directml")
            admitted, _ = read_manifest(root, allow_experimental_fp16=True,
                                        fp16_acceptance="finite", execution_backend="directml")
            self.assertFalse(admitted["validation"]["passed"])
            for backend in ("cpu", "cuda", None):
                with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "independent cpu|explicit CPU or DirectML"):
                    read_manifest(root, allow_experimental_fp16=True,
                                  fp16_acceptance="finite", execution_backend=backend)

    def test_finite_mode_does_not_relax_hashes_io_shapes_repeats_or_placement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = experiment(root)
            changes = (
                lambda r: r.update(candidate_graph_sha256="different"),
                lambda r: r["runs"]["fp16"].update(graph_sha256="different"),
                lambda r: r["runs"]["fp16"].update(public_io_fp32=False),
                lambda r: r["runs"]["fp16"]["cases"]["0"].update(finite=False),
                lambda r: r["runs"]["fp16"]["cases"]["0"].update(shape_matches_baseline=False),
                lambda r: r["runs"]["fp16"]["cases"]["0"].update(repeat_checks=[]),
                lambda r: r["runs"]["fp16-profile"]["profile"].update(cpu_neural_compute_events={"Conv": 1}),
                lambda r: r["runs"]["fp16-profile"]["profile"].update(directml_fp16_convolution_observed=False),
            )
            for change in changes:
                report = deepcopy(original)
                change(report)
                with self.assertRaises(ValueError):
                    publish_experiment(root, report)
            self.assertNotIn("experimental_validations", json.loads((root / "manifest.json").read_text()))

    def test_cpu_uses_own_evidence_and_records_fp32_compute_without_claiming_native_fp16(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = experiment(root, "cpu")
            self.assertTrue(finite_execution_passed(report, "cpu"))
            publish_experiment(root, report)
            read_manifest(root, allow_experimental_fp16=True, fp16_acceptance="finite", execution_backend="cpu")
            session = Mock()
            session.get_providers.return_value = ["CPUExecutionProvider"]
            session.get_inputs.return_value = [SimpleNamespace(name=name, type="tensor(int64)" if i < 2 else "tensor(float)")
                                               for i, name in enumerate(INPUT_NAMES)]
            session.get_outputs.return_value = [SimpleNamespace(name="waveform", type="tensor(float)")]
            ort = SimpleNamespace(SessionOptions=SessionOptions, ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1), InferenceSession=Mock(return_value=session))
            with patch.dict(sys.modules, {"onnxruntime": ort}):
                ORTSoVITS.load(root, device="cpu", allow_experimental_fp16=True,
                              fp16_acceptance="finite", enable_cpu_mem_arena=False)
                self.assertEqual(ort.InferenceSession.call_args.kwargs["providers"], ["CPUExecutionProvider"])
                ORTSoVITS.load(root, device="cpu", allow_experimental_fp16=True,
                              fp16_acceptance="finite", enable_cpu_mem_arena=True, intra_op_num_threads=8)
                options = ort.InferenceSession.call_args.kwargs["sess_options"]
                self.assertEqual(options.intra_op_num_threads, 8)
                self.assertTrue(options.enable_cpu_mem_arena)
            report["runs"]["fp16-profile"]["profile"]["provider_events"]["DmlExecutionProvider"] = 1
            self.assertFalse(finite_execution_passed(report, "cpu"))

    def test_finite_publication_accepts_nonempty_cases_without_locking_test_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = experiment(root)
            for run in report["runs"].values():
                del run["cases"]["3"]
            report.update(quality_accepted=True, hardware={"device_id": 3},
                          ort_execution_options={"intra_op_num_threads": 8, "enable_cpu_mem_arena": True})
            publish_experiment(root, report)
            admitted, _ = read_manifest(root, allow_experimental_fp16=True,
                                        fp16_acceptance="finite", execution_backend="directml")
            self.assertTrue(admitted["experimental_validations"]["directml"]["passed"])
            report["runs"]["fp16"]["cases"] = {}
            with self.assertRaisesRegex(ValueError, "incomplete execution evidence"):
                publish_experiment(root, report)

    def test_directml_finite_load_allows_adapter_threads_and_arena_but_checks_actual_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publish_experiment(root, experiment(root))
            session = Mock()
            session.get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
            session.get_inputs.return_value = [SimpleNamespace(name=name, type="tensor(int64)" if i < 2 else "tensor(float)")
                                               for i, name in enumerate(INPUT_NAMES)]
            session.get_outputs.return_value = [SimpleNamespace(name="waveform", type="tensor(float)")]
            session.run.return_value = [np.zeros((1, 1, 10), np.float32)]
            ort = SimpleNamespace(SessionOptions=SessionOptions, ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1), InferenceSession=Mock(return_value=session),
                get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
            inputs = (np.zeros((1, 1, 3), np.int64), np.zeros((1, 4), np.int64),
                      np.zeros((1, 1024, 1), np.float32), np.zeros((1, 512, 1), np.float32),
                      np.zeros((1, 192, 6), np.float32))
            options = dict(device="directml", device_id=3, intra_op_num_threads=8, enable_cpu_mem_arena=True,
                           allow_experimental_fp16=True, fp16_acceptance="finite")
            with patch.dict(sys.modules, {"onnxruntime": ort}):
                model = ORTSoVITS.load(root, **options)
                kwargs = ort.InferenceSession.call_args.kwargs
                self.assertEqual(kwargs["providers"][0], ("DmlExecutionProvider", {"device_id": "3"}))
                self.assertEqual(kwargs["sess_options"].intra_op_num_threads, 8)
                self.assertTrue(kwargs["sess_options"].enable_cpu_mem_arena)
                self.assertFalse(kwargs["sess_options"].enable_mem_pattern)
                self.assertEqual(kwargs["sess_options"].execution_mode, "sequential")
                self.assertTrue(np.isfinite(model.decode(*inputs)).all())
                for value in (np.nan, np.inf, -np.inf):
                    session.run.return_value = [np.full((1, 1, 10), value, np.float32)]
                    with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "non-finite samples"):
                        model.decode(*inputs)
                session.get_outputs.return_value[0].type = "tensor(float16)"
                with self.assertRaisesRegex(ValueError, "FP32 public"):
                    ORTSoVITS.load(root, **options)
                session.get_providers.return_value = ["CPUExecutionProvider"]
                with self.assertRaisesRegex(RuntimeError, "refusing silent CPU inference"):
                    ORTSoVITS.load(root, **options)
            (root / "decode.onnx").write_bytes(b"different graph")
            with self.assertRaisesRegex(ValueError, "SHA-256 or size mismatch"):
                read_manifest(root, allow_experimental_fp16=True, fp16_acceptance="finite", execution_backend="directml")

    def test_runtime_checks_evidence_at_acoustic_load_and_forwards_finite_acceptance(self):
        for implementation in (CPUEngine, DirectMLEngine):
            with self.subTest(backend=implementation.name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _, _ = fixture(root)
                report = experiment(root / "sovits", implementation.name)
                with patch("sakuratts.frontend.runtime.load_frontend"):
                    engine = implementation(config, acoustic_fp16_acceptance="finite",
                                            allow_experimental_acoustic_fp16=True, load_references=False)
                    with self.assertRaisesRegex(ValueError, "independent"):
                        engine._load_sovits()
                    engine.close()
                    publish_experiment(root / "sovits", report)
                    engine = implementation(config, acoustic_fp16_acceptance="finite",
                                            allow_experimental_acoustic_fp16=True, load_references=False)
                    with patch("sakuratts.backends.onnx.sovits.ORTSoVITS.load") as load:
                        engine._load_sovits()
                        self.assertEqual(load.call_args.kwargs["fp16_acceptance"], "finite")
                    self.assertFalse(engine._execution_report()["acoustic_strict_validation_passed"])
                    engine.close()

    def test_repackaging_preserves_independent_finite_evidence_without_runtime_sidecars(self):
        from sakuratts.converter import package_model
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _ = fixture(root)
            settings = json.loads(config.read_text())
            settings.update(references={}, backend={"preferred": "directml"})
            config.write_text(json.dumps(settings))
            (root / "python.exe").touch()
            report = experiment(root / "sovits", "directml")
            publish_experiment(root / "sovits", report)
            weights = root / "gpt/weights.bin"
            weights.write_bytes(b"gpt")
            path = root / "gpt/manifest.json"
            gpt = json.loads(path.read_text())
            gpt.update(format="sakuratts-gpt-fp32-v1", architecture="gpt-sovits-ar-postnorm-relu",
                       weights={"file": weights.name, "bytes": weights.stat().st_size, "sha256": sha256_file(weights)})
            path.write_text(json.dumps(gpt))
            with patch("sakuratts._internal.diagnostics.check_worker_imports", return_value={}):
                result = check_prepared_packages(config)
                self.assertEqual(result["status"], "passed")
                self.assertFalse(result["acoustic_validation"]["engineering_screen_passed"])
                self.assertFalse(result["acoustic_validation"]["execution_tested_by_doctor"])
                packed = package_model(config, root / "packed")
                copied = json.loads((root / "packed/acoustic/manifest.json").read_text())
                self.assertFalse(copied["validation"]["passed"])
                self.assertEqual(set(copied["experimental_validations"]), {"directml"})
                self.assertIsNone(check_prepared_packages(packed.path)["gpt_resources"])
                for backend in ("cpu", "cuda"):
                    settings["backend"]["preferred"] = backend
                    config.write_text(json.dumps(settings))
                    with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "did not pass"):
                        check_prepared_packages(config)
                settings["backend"]["preferred"] = "directml"
                config.write_text(json.dumps(settings))
                for resource in (weights, root / "sovits/decode.onnx", root / "frontend/user.dict"):
                    original = resource.read_bytes()
                    resource.write_bytes(b"changed")
                    with self.subTest(resource=resource), self.assertRaisesRegex(ValueError, "mismatch"):
                        package_model(config, root / "invalid")
                    self.assertFalse((root / "invalid").exists())
                    resource.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
