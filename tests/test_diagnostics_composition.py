"""Package diagnostics follow the selected workers instead of assuming one ABI."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sakuratts.diagnostics.resources import check_runtime_packages
from sakuratts.module.reference_condition import sha256_file
from test_nvidia_package_startup import fixture


class DiagnosticsCompositionTests(unittest.TestCase):
    def check(self, *, separate=False, frontend_only=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config_path, _, _ = fixture(root)
            config = json.loads(config_path.read_text())
            config["references"] = {}
            acoustic_python = root / "python.exe"
            acoustic_python.touch()
            if separate or frontend_only:
                frontend_python = root / "language/python.exe"
                frontend_python.parent.mkdir()
                frontend_python.touch()
                config["frontend_python"] = str(frontend_python)
            else:
                frontend_python = acoustic_python
            if frontend_only:
                config.pop("acoustic_python")
                acoustic_python = Path(sys.executable)
            config_path.write_text(json.dumps(config))
            weights = root / "gpt/weights.npz"
            weights.write_bytes(b"gpt")
            gpt_path = root / "gpt/manifest.json"
            gpt = json.loads(gpt_path.read_text())
            gpt.update(format="sakuratts-gpt-fp32-v1", architecture="gpt-sovits-ar-postnorm-relu",
                weights={"file": weights.name, "bytes": weights.stat().st_size, "sha256": sha256_file(weights)})
            gpt_path.write_text(json.dumps(gpt))
            acoustic = {"source": gpt["source"], "config": {"model": {"version": "v2ProPlus"}}}
            with patch("sakuratts.module.sovits.read_manifest", return_value=(acoustic, None)), \
                    patch("sakuratts.diagnostics.resources.check_worker_imports",
                          side_effect=lambda python, profile, **kwargs: {"executable": str(python)}) as probe:
                result = check_runtime_packages(config_path)
            self.assertEqual(result["worker"]["executable"], str(acoustic_python))
            self.assertEqual(result["frontend_worker"]["executable"], str(frontend_python))
            if separate or frontend_only:
                self.assertEqual(probe.call_count, 2)
                self.assertEqual(probe.call_args_list[0].args, (acoustic_python, {}))
                self.assertEqual(probe.call_args_list[1].args[0], frontend_python)
                self.assertEqual(probe.call_args_list[1].kwargs, {"acoustic": False})
                self.assertIn("module_directory", probe.call_args_list[1].args[1])
            else:
                self.assertEqual(probe.call_count, 1)
                self.assertIn("module_directory", probe.call_args.args[1])

    def test_separate_frontend_is_checked_with_its_own_interpreter(self):
        self.check(separate=True)

    def test_explicit_frontend_allows_in_process_acoustic(self):
        self.check(frontend_only=True)

    def test_legacy_shared_interpreter_is_checked_once(self):
        self.check()


class MLXDiagnosticsTests(unittest.TestCase):
    def test_native_diagnostics_follow_acoustic_family_and_reference_compatibility(self):
        from sakuratts.diagnostics.mlx import check_packages
        from sakuratts.module.reference_condition import FORMAT

        for family in ("v2Pro", "v2ProPlus"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                config_path, _, _ = fixture(root)
                (root / "python.exe").touch()
                gpt = root / "gpt"
                (gpt / "weights.npz").touch()
                (gpt / "manifest.json").write_text(json.dumps({
                    "format": "sakuratts-gpt-fp32-v1", "dtype": "float32",
                    "architecture": "gpt-sovits-ar-postnorm-relu", "weights": {"file": "weights.npz"}}))
                acoustic = root / "sovits"
                np.savez(acoustic / "weights.npz", weight=np.zeros(1, np.float32))
                (acoustic / "manifest.json").write_text(json.dumps({
                    "format": "sakuratts-sovits-decode-fp32-v1", "dtype": "float32",
                    "config": {"model": {"version": family}}, "weights": {"file": "weights.npz"}}))
                reference = root / "reference"
                np.savez(reference / "arrays.npz", reference_phones=np.array([1], np.int64),
                    prompt_semantic=np.array([2], np.int64), reference_bert=np.zeros((1024, 1), np.float32),
                    ge=np.zeros((1, 1024, 1), np.float32), ge512=np.zeros((1, 512, 1), np.float32))
                manifest = {"format": FORMAT, "model_family": family, "archive": {"file": "arrays.npz"}}
                (reference / "manifest.json").write_text(json.dumps(manifest))
                with patch("sakuratts.diagnostics.mlx.check_worker_imports", return_value={}):
                    result = check_packages(config_path)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["model_family"], family)
                manifest["model_family"] = "v2ProPlus" if family == "v2Pro" else "v2Pro"
                (reference / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "family must match"):
                    check_packages(config_path)


if __name__ == "__main__":
    unittest.main()
