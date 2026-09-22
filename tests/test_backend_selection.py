"""Backend dispatch stays outside the shared inference lifecycle."""

from array import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sakuratts import Engine, Model
from sakuratts.backends import SUPPORTED_BACKENDS, create_runtime
from sakuratts.engine import Inference, read_inference_configuration


class FakeRuntime:
    name = "test-device"
    gpt_precision = acoustic_precision = "fp32"

    def __init__(self):
        self.loaded = self.closed = False
        self.packages = {name: Path(name) for name in ("gpt", "sovits", "frontend")}
        self.manifests = {name: {"source": {"checkpoint_sha256": name,
            "official_commit": "test-source"}} for name in ("gpt", "sovits")}

    def load(self):
        self.loaded = True

    def synthesize(self, text, **options):
        return array("h", [1, 2]), {"sample_rate": 32000, "text": text}

    def close(self):
        self.closed = True


class BackendSelectionTests(unittest.TestCase):
    def model(self, backend="cuda"):
        return Model(Path("model.json"), {"name": "test", "languages": ["ja"],
            "backend": {"preferred": backend}})

    def test_shared_engine_and_inference_use_backend_contract(self):
        model = self.model("test-device")
        runtime = FakeRuntime()
        with patch("sakuratts.backends.create_runtime", return_value=runtime) as create:
            inference = Inference(model)
            try:
                self.assertTrue(runtime.loaded)
                self.assertEqual(inference.info()["backend"], "test-device")
                result = inference.engine.synthesize("こんにちは")
                self.assertEqual(result.sample_rate, 32000)
                self.assertEqual(list(result.pcm), [1, 2])
                create.assert_called_once_with(model, backend=None,
                    experimental=None, load_references=False)
            finally:
                inference.close()
        self.assertTrue(runtime.closed)

    def test_public_load_forwards_backend_specific_options_unchanged(self):
        model = self.model()
        with patch("sakuratts.backends.create_runtime", return_value=FakeRuntime()) as create:
            with Engine.load(model, backend="test-device", experimental={"future-option": 7}):
                pass
        create.assert_called_once_with(model, backend="test-device",
            experimental={"future-option": 7}, load_references=True)

    def test_unimplemented_backends_do_not_fall_back_or_convert(self):
        self.assertEqual(SUPPORTED_BACKENDS, ("cuda",))
        for backend in ("cpu", "rocm", "mlx", "unknown"):
            with self.subTest(backend=backend), patch("sakuratts.backends.cuda.create_runtime") as cuda:
                with self.assertRaisesRegex(NotImplementedError, "not implemented"):
                    create_runtime(self.model(backend))
                with patch.object(Inference, "_convert_initial") as convert:
                    with self.assertRaisesRegex(NotImplementedError, "not implemented"):
                        Inference(backend=backend)
                    convert.assert_not_called()
                cuda.assert_not_called()

    def test_service_configuration_selects_backend_without_loading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tts.json"
            path.write_text(json.dumps({"custom": {"device": "cpu"}}), encoding="utf-8")
            _, settings = read_inference_configuration(tts_config=path)
            self.assertEqual(settings["backend"], "cpu")
            path.write_text(json.dumps({"custom": {"device": "cpu"},
                "sakuratts": {"backend": "cuda"}}), encoding="utf-8")
            _, settings = read_inference_configuration(tts_config=path)
            self.assertEqual(settings["backend"], "cuda")
            path.write_text(json.dumps({"custom": {"device": "cpu"}}), encoding="utf-8")
            with patch("sakuratts.backends.create_runtime", return_value=FakeRuntime()):
                inference = Inference(self.model(), tts_config=path, backend="cuda")
                self.assertEqual(inference.settings["backend"], "cuda")
                inference.close()

    def test_unsupported_selection_does_not_import_compute_modules(self):
        code = "\n".join((
            "import sys",
            "from pathlib import Path",
            "from sakuratts import Engine, Model",
            "model = Model(Path('unused.json'), {'backend': {'preferred': 'cpu'}})",
            "try:",
            "    Engine.load(model)",
            "except NotImplementedError:",
            "    pass",
            "else:",
            "    raise AssertionError('unsupported backend accepted')",
            "assert not set(('numpy', 'cupy', 'torch', 'onnxruntime', 'mlx', 'sakuratts.backends.cuda.engine')) & sys.modules.keys()",
        ))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_configuration_overrides_cached_paths_without_mutating_model(self):
        model = self.model()
        model.manifest.update(acoustic_python="old-acoustic.exe", frontend_python="old-frontend.exe")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tts.json"
            workers = {"acoustic_python": "new-acoustic.exe", "frontend_python": "new-frontend.exe"}
            path.write_text(json.dumps({"sakuratts": workers}), encoding="utf-8")
            with patch("sakuratts.backends.create_runtime", side_effect=lambda *args, **kwargs: FakeRuntime()) as create:
                inference = Inference(model, tts_config=path)
                try:
                    self.assertEqual(create.call_args.args[0].manifest["frontend_python"], str(Path(workers["frontend_python"]).resolve()))
                    self.assertEqual(create.call_args.args[0].manifest["acoustic_python"], str(Path(workers["acoustic_python"]).resolve()))
                    self.assertEqual(model.manifest["frontend_python"], "old-frontend.exe")
                    # A cached model is rebound when selected again, including after sleep.
                    inference.settings["frontend_python"] = "another-frontend.exe"
                    inference._activate(model)
                    self.assertEqual(create.call_args.args[0].manifest["frontend_python"], str(Path("another-frontend.exe").resolve()))
                finally:
                    inference.close()


if __name__ == "__main__":
    unittest.main()
