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
    def preparation_bundle(self, root):
        preparation = root / "runtime/preparation"
        (preparation / "official/GPT_SoVITS/pretrained_models/fast_langdetect").mkdir(parents=True)
        (preparation / "python.exe").touch()
        (preparation / "official/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin").touch()
        config = dict(format="sakuratts-preparation-v1", python="python.exe", official_source="official",
                      language_model="official/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin")
        (preparation / "preparation.json").write_text(json.dumps(config))
        (root / "runtime/portable.json").write_text('{"format":"sakuratts-portable-v1","has_preparation":true}')
        return preparation, config

    def test_preparation_uses_current_bundle_without_changing_user_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, _ = self.preparation_bundle(root)
            with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)):
                result = preparation_settings(dict(python="Z:/old/python.exe", official_source="Z:/old",
                    language_model="Z:/old/lid.bin", cache_dir="Z:/old/cache",
                    gpt_checkpoint="models/chosen.ckpt", sovits_checkpoint="D:/Models/chosen.pth",
                    cnhubert="D:/Models/custom-hubert"))
            self.assertEqual(result["python"], str(preparation / "python.exe"))
            self.assertEqual(result["official_source"], str(preparation / "official"))
            self.assertEqual(result["cache_dir"], str(root / "cache"))
            self.assertEqual(result["gpt_checkpoint"], "models/chosen.ckpt")
            self.assertEqual(result["sovits_checkpoint"], "D:/Models/chosen.pth")
            self.assertEqual(result["cnhubert"], "D:/Models/custom-hubert")

    def test_incomplete_preparation_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, _ = self.preparation_bundle(root)
            (preparation / "preparation.json").unlink()
            with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)):
                with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                    preparation_settings({"python": sys.executable})

    def test_preparation_marker_rejects_external_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, config = self.preparation_bundle(root)
            for path in ("../main/python.exe", "C:/Python/python.exe", "..\\main\\python.exe"):
                with self.subTest(path=path), patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)):
                    (preparation / "preparation.json").write_text(json.dumps(dict(config, python=path)))
                    with self.assertRaisesRegex(ValueError, "inside runtime/preparation"):
                        preparation_settings({})

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
                self.assertEqual(result["frontend_python"], result["acoustic_python"])
                self.assertNotIn("main_dictionary", result)
                settings = preparation_settings({"python": "Z:/old/python.exe", "official_source": "Z:/old", "gpt_checkpoint": "chosen.ckpt"})
                self.assertNotIn("python", settings)
                self.assertNotIn("official_source", settings)
                self.assertEqual(settings["gpt_checkpoint"], "chosen.ckpt")
                self.assertEqual(settings["frontend_python"], settings["acoustic_python"])
                legacy = root / "legacy.json"
                legacy.write_text(json.dumps({"format": "sakuratts-windows-config-v1", "gpt": "gpt",
                    "sovits": "sovits", "frontend": "frontend", "acoustic_python": "Z:/old/python.exe"}))
                self.assertEqual(read_windows_config(legacy)[1]["acoustic_python"], str(root / "runtime/acoustic/python.exe"))

    def test_portable_marker_binds_frontend_and_acoustic_workers_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workers = {"frontend": "runtime/language/python.exe", "acoustic": "runtime/onnx/python.exe"}
            for name in workers.values():
                path = root / name
                path.parent.mkdir(parents=True)
                path.touch()
            (root / "runtime/portable.json").write_text(json.dumps({
                "format": "sakuratts-portable-v1", "workers": workers}))
            original = {"frontend_python": "Z:/old/frontend.exe", "acoustic_python": "Z:/old/acoustic.exe"}
            with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)):
                configured = model_config(original)
                preparation = preparation_settings(original)
            for role, name in workers.items():
                self.assertEqual(configured[role + "_python"], str(root / name))
                self.assertEqual(preparation[role + "_python"], str(root / name))
            self.assertEqual(original["frontend_python"], "Z:/old/frontend.exe")

    def test_portable_worker_path_cannot_escape_the_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            (root / "runtime").mkdir(parents=True)
            (root.parent / "outside.exe").touch()
            (root / "runtime/portable.json").write_text(json.dumps({
                "format": "sakuratts-portable-v1", "workers": {"frontend": "../outside.exe"}}))
            with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)):
                for configure in (model_config, preparation_settings):
                    with self.subTest(configure=configure.__name__), self.assertRaisesRegex(ValueError, "inside"):
                        configure({})

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
