"""Package diagnostics follow the selected workers instead of assuming one ABI."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from sakuratts._internal.diagnostics import check_windows_packages
from sakuratts._internal.reference_condition import sha256_file
from test_nvidia_package_startup import fixture


class DiagnosticsCompositionTests(unittest.TestCase):
    def check(self, *, separate=False, frontend_only=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            with patch("sakuratts.backends.onnx.sovits.read_manifest", return_value=(acoustic, None)), \
                    patch("sakuratts._internal.diagnostics.check_worker_imports",
                          side_effect=lambda python, profile, **kwargs: {"executable": str(python)}) as probe:
                result = check_windows_packages(config_path)
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


if __name__ == "__main__":
    unittest.main()
