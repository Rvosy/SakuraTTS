"""Public API, model relocation, and CLI behavior without optional backends."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

from sakuratts import Audio, BusyError, Engine, Model
from sakuratts.cli import main
from sakuratts.converter import package_model


def model_directory(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("gpt", "acoustic", "frontend", "references/normal"):
        (root / name).mkdir(parents=True)
        (root / name / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = {"format": "sakuratts-model-v1", "name": "テスト", "languages": ["ja"],
        "gpt": "gpt", "acoustic": "acoustic", "frontend": "frontend",
        "references": {"通常": "references/normal"}, "default_reference": "通常"}
    (root / "model.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return Model.load(root)


def audio(status="completed"):
    return Audio(np.array([-32768, 0, 32767], dtype=np.int16), 32000,
                 {"sample_rate": 32000, "status": status, "request_ms": 12.5})


class PublicApiTests(unittest.TestCase):
    def test_model_without_reference_packages_is_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            model = model_directory(Path(folder))
            manifest = dict(model.manifest)
            manifest.pop("references")
            manifest.pop("default_reference")
            model.path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = Model.load(model.path)
            self.assertEqual(loaded.references, ())
            self.assertIsNone(loaded.default_reference)

    def test_import_and_help_do_not_load_optional_or_numpy_packages(self):
        code = "import sys,sakuratts; from sakuratts import Engine,Model,start_server; assert not set(('numpy','cupy','torch','onnxruntime','fastapi','mlx')) & sys.modules.keys()"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("tts", "convert", "serve", "benchmark"):
            result = subprocess.run([sys.executable, "-m", "sakuratts", command, "--help"], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_metadata_is_readable_without_implemented_backend_or_portable_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            model = model_directory(Path(folder) / "model")
            model.path.write_text(json.dumps(dict(model.manifest, backend={"preferred": "mlx"},
                languages=["ja", "zh"], frontend_python="old/frontend/python.exe")), encoding="utf-8")
            code = """
import json, sys
from sakuratts import Model
model = Model.load(sys.argv[1])
assert model.backend == 'mlx'
assert model.languages == ('ja', 'zh')
assert model.runtime_config['frontend_python'] == 'old/frontend/python.exe'
assert not set(('numpy', 'cupy', 'torch', 'onnxruntime', 'fastapi', 'mlx',
                'sakuratts._internal.portable', 'sakuratts.backends')) & sys.modules.keys()
print(json.dumps(model.info()))
"""
            environment = dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(Path(folder) / "missing-bundle"))
            result = subprocess.run([sys.executable, "-c", code, str(model.path)],
                env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            info = json.loads(result.stdout)
            self.assertEqual(info["backend"], "mlx")
            self.assertEqual(info["languages"], ["ja", "zh"])

    def test_prepared_package_relocates_resources_and_keeps_worker_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = model_directory(root / "old")
            worker = root / "worker.exe"
            worker.write_bytes(b"fake")
            frontend_worker = root / "frontend.exe"
            frontend_worker.write_bytes(b"fake")
            model.path.write_text(json.dumps(dict(model.manifest, acoustic_python="../worker.exe",
                frontend_python="../frontend.exe", gpt="../shared-gpt")), encoding="utf-8")
            (root / "old/gpt").rename(root / "shared-gpt")
            with patch("sakuratts._internal.diagnostics.check_prepared_packages") as check:
                packed = package_model(model.path, root / "new")
            check.assert_called_once()
            self.assertEqual(packed.name, "テスト")
            self.assertEqual(packed.manifest["acoustic_python"], str(worker.resolve()))
            self.assertEqual(packed.manifest["frontend_python"], str(frontend_worker.resolve()))
            self.assertEqual(packed.references, ("通常",))
            self.assertTrue((root / "old/model.json").exists())
            with self.assertRaises(FileExistsError), patch("sakuratts._internal.diagnostics.check_prepared_packages"):
                package_model(model.path, root / "new")

    def test_failed_package_validation_does_not_publish_partial_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = model_directory(root / "old")
            with patch("sakuratts._internal.diagnostics.check_prepared_packages", side_effect=ValueError("bad hash")):
                with self.assertRaisesRegex(ValueError, "bad hash"):
                    package_model(model.path, root / "new")
            self.assertFalse((root / "new").exists())
            self.assertFalse(list(root.glob(".sakuratts-*")))

    def test_engine_lifetime_busy_recovery_and_output(self):
        runtime = Mock()
        entered, release = threading.Event(), threading.Event()
        def speak(*args, **kwargs):
            entered.set()
            release.wait(5)
            result = audio()
            return result.pcm, result.report
        runtime.synthesize.side_effect = speak
        engine = Engine(None, runtime)
        thread = threading.Thread(target=engine.synthesize, args=("一",))
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaises(BusyError):
                engine.synthesize("二")
        finally:
            release.set()
            thread.join(5)
        runtime.synthesize.side_effect = ValueError("failed")
        with self.assertRaises(ValueError): engine.synthesize("三")
        runtime.synthesize.side_effect = None
        runtime.synthesize.return_value = audio().pcm, audio().report
        self.assertEqual(engine.synthesize("四").sample_rate, 32000)
        engine.close()
        engine.close()
        runtime.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"): engine.synthesize("五")

    def test_wav_and_cli_preserve_limited_status_and_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "speech.wav"
            runtime = Mock()
            runtime.__enter__ = Mock(return_value=runtime)
            runtime.__exit__ = Mock()
            runtime.synthesize.return_value = audio("stopped_at_limit")
            with patch.object(Engine, "load", return_value=runtime), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["tts", "model", "--text", "こんにちは", "--split-method", "cut5", "--output", str(path)]), 2)
                self.assertEqual(runtime.synthesize.call_args.kwargs["split_method"], "cut5")
            with wave.open(str(path)) as wav:
                self.assertEqual(wav.getframerate(), 32000)
                self.assertEqual(wav.readframes(3), audio().pcm.astype("<i2").tobytes())
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["status"], "stopped_at_limit")
            with self.assertRaises(FileExistsError): audio().save(path)
