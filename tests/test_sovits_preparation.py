"""Prepared DirectML acoustics publish only observed finite GPU execution."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from sakuratts._internal.conversion import validate_sovits_directml as preparation
from sakuratts._internal.conversion.directml_hardware import capture_hardware
from sakuratts._internal.reference_condition import sha256_file
from sakuratts.backends.onnx.sovits import INPUT_NAMES, STAGES, read_manifest
from test_directml_precision import package


class SoVITSPreparationTests(unittest.TestCase):
    def prepare(self, root, *, gpu_compute=True, hardware_result=None):
        baseline, candidate = root / "baseline", root / "candidate"
        source, _, _ = package(baseline)
        source["dtype"] = "float32"
        (baseline / "manifest.json").write_text(json.dumps(source), encoding="utf-8")
        manifest, _, _ = package(candidate)
        manifest["precision"].update(fp16_scope="all", source_graphs=source["graphs"],
            source_weights=source["weights"], source_manifest_sha256=sha256_file(baseline / "manifest.json"))
        manifest["validation"]["passed"] = False
        (candidate / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        values = {name: np.zeros((1, 1, 2), np.float32) for name in INPUT_NAMES}
        values.update(codes=np.zeros((1, 1, 1), np.int64), phones=np.zeros((1, 1), np.int64))
        values.update({"expected_" + name: np.zeros((1, 1, 8), np.float32) for name in STAGES})
        np.savez(baseline / "validation-0.npz", **values)
        profile = root / "profile.json"
        profile.write_text(json.dumps([{"args": {
            "provider": "DmlExecutionProvider" if gpu_compute else "CPUExecutionProvider",
            "op_name": "Conv", "output_type_shape": [{"float16": [1, 1, 8]}]}}]), encoding="utf-8")
        sessions = []

        def session(graph, *, sess_options, providers):
            self.assertEqual(providers[0], ("DmlExecutionProvider", {"device_id": "2"}))
            selected = Path(graph)
            diagnostic = selected.name == "diagnostic.onnx"
            result = Mock()
            result.get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
            result.get_inputs.return_value = [SimpleNamespace(name=name,
                type="tensor(int64)" if i < 2 else "tensor(float)") for i, name in enumerate(INPUT_NAMES)]
            result.get_outputs.return_value = [SimpleNamespace(name=name, type="tensor(float)")
                for name in (STAGES if diagnostic else ("waveform",))]
            value = 0.25 if selected.parent == candidate else 0.0
            result.run.side_effect = lambda names, _feeds: [np.full((1, 1, 8), value, np.float32) for _ in names]
            result.end_profiling.return_value = str(profile)
            sessions.append(result)
            return result

        arguments = ["--baseline", str(baseline), "--candidate", str(candidate),
                     "--output", str(root / "evidence"), "--device-id", "2"]
        with patch.object(preparation, "capture_hardware", return_value=hardware_result or {"device_id": 2}) as hardware, \
                patch.object(preparation.ort, "InferenceSession", side_effect=session), \
                contextlib.redirect_stdout(io.StringIO()):
            if gpu_compute:
                preparation.prepare_main(arguments)
            else:
                with self.assertRaisesRegex(ValueError, "incomplete execution evidence"):
                    preparation.prepare_main(arguments)
        hardware.assert_called_once_with(2)
        self.assertTrue(all(item.run.call_count == 2 for item in sessions))
        return candidate, json.loads((root / "evidence/result.json").read_text(encoding="utf-8"))

    def test_selected_adapter_is_executed_and_finite_admission_preserves_accuracy_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidate, report = self.prepare(Path(temporary))
            manifest, _ = read_manifest(candidate, allow_experimental_fp16=True,
                fp16_acceptance="finite", execution_backend="directml")
            self.assertTrue(report["finite_experiment"]["passed"])
            self.assertFalse(report["engineering_screen"]["passed"])
            self.assertFalse(manifest["validation"]["passed"])
            self.assertFalse(report["quality_accepted"])
            self.assertEqual(report["ort_execution_options"]["device_id"], 2)

    def test_cpu_neural_fallback_does_not_publish_directml_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidate, report = self.prepare(Path(temporary), gpu_compute=False)
            self.assertFalse(report["finite_experiment"]["passed"])
            manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("experimental_validations", manifest)
            self.assertFalse((candidate / "experimental-directml-finite.json").exists())

    def test_inventory_failure_is_reported_without_blocking_successful_gpu_execution(self):
        adapter = {"device_id": 2, "description": "test-adapter", "vendor_id": 0x1002, "pci_device_id": 1}
        for dxgi_error in (False, True):
            with self.subTest(dxgi_error=dxgi_error), tempfile.TemporaryDirectory() as temporary, \
                    patch("sakuratts.backends.directml.devices.list_adapters", return_value=[adapter],
                          side_effect=OSError("DXGI unavailable") if dxgi_error else None), \
                    patch("sakuratts._internal.conversion.directml_hardware.subprocess.run",
                          side_effect=FileNotFoundError("PowerShell unavailable")):
                hardware = capture_hardware(2)
                error_key = "adapter_inventory_error" if dxgi_error else "driver_inventory_error"
                self.assertIn("unavailable", hardware[error_key])
                self.assertEqual(hardware["device_id"], 2)
                if not dxgi_error:
                    self.assertEqual(hardware["description"], "test-adapter")
                candidate, report = self.prepare(Path(temporary), hardware_result=hardware)
                self.assertEqual(report["hardware"][error_key], hardware[error_key])
                self.assertTrue(report["finite_experiment"]["passed"])
                read_manifest(candidate, allow_experimental_fp16=True,
                    fp16_acceptance="finite", execution_backend="directml")


if __name__ == "__main__":
    unittest.main()
