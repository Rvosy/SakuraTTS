"""Portable installation bindings must not retain exporter environment paths."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts._internal.portable import model_config, preparation_settings
from sakuratts._internal.diagnostics import read_windows_config


class PortableRuntimeTests(unittest.TestCase):
    def test_normal_installation_keeps_existing_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            config = {"acoustic_python": "custom/python.exe"}
            self.assertEqual(model_config(config), config)
            self.assertEqual(preparation_settings(config), config)

    def test_model_uses_relocated_worker_and_drops_external_dictionary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/acoustic").mkdir(parents=True)
            (root / "runtime/acoustic/python.exe").touch()
            (root / "runtime/portable.json").write_text('{"format":"sakuratts-portable-v1"}')
            with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)):
                result = model_config({"gpt": "gpt", "acoustic_python": "Z:/developer/python.exe", "main_dictionary": "Z:/old/dict"})
                self.assertEqual(result["acoustic_python"], str(root / "runtime/acoustic/python.exe"))
                self.assertNotIn("main_dictionary", result)
                settings = preparation_settings({"python": "Z:/old/python.exe", "official_source": "Z:/old", "gpt_checkpoint": "chosen.ckpt"})
                self.assertNotIn("python", settings)
                self.assertNotIn("official_source", settings)
                self.assertEqual(settings["gpt_checkpoint"], "chosen.ckpt")
                legacy = root / "legacy.json"
                legacy.write_text(json.dumps({"format": "sakuratts-windows-config-v1", "gpt": "gpt",
                    "sovits": "sovits", "frontend": "frontend", "acoustic_python": "Z:/old/python.exe"}))
                self.assertEqual(read_windows_config(legacy)[1]["acoustic_python"], str(root / "runtime/acoustic/python.exe"))

    def test_acoustic_worker_discovers_shared_main_cuda_libraries(self):
        from sakuratts.backends.cuda import runtime
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "runtime/main/Lib/site-packages/nvidia/cublas/bin"
            shared.mkdir(parents=True)
            (root / "runtime/portable.json").write_text('{"format":"sakuratts-portable-v1"}')
            with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)), \
                 patch.object(sys, "executable", str(root / "runtime/acoustic/python.exe")), \
                 patch.object(sys, "path", []), patch.object(runtime, "_configured_paths", set()), \
                 patch.object(runtime, "_dll_handles", []), patch.object(os, "add_dll_directory", create=True) as add:
                self.assertIn(shared, runtime.configure_cuda())
                if os.name == "nt":
                    add.assert_called_once_with(str(shared))

    def test_missing_worker_fails_without_developer_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime").mkdir()
            (root / "runtime/portable.json").write_text('{"format":"sakuratts-portable-v1"}')
            with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)):
                with self.assertRaises(FileNotFoundError):
                    model_config({"acoustic_python": sys.executable})


if __name__ == "__main__":
    unittest.main()
