"""CLI selection reaches the runtime without eager optional dependencies."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from sakuratts.cli import main
from sakuratts.engine import Engine


class CliCompositionTests(unittest.TestCase):
    def test_original_entrypoint_forwards_launch_flags(self):
        entry = Path(__file__).resolve().parents[1] / "api_v2.py"
        with patch("sakuratts.cli.main", return_value=0) as launch, \
                patch.object(sys, "argv", [str(entry), "-a", "127.0.0.1", "-p", "9881", "-c", "voice.yaml"]):
            with self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(entry), run_name="__main__")
        self.assertEqual(stopped.exception.code, 0)
        launch.assert_called_once_with(["serve", "-a", "127.0.0.1", "-p", "9881", "-c", "voice.yaml"])

    def test_default_configuration_preserves_local_precedence_and_accepts_upstream_layout(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.chdir(temporary):
            local = Path("configs/tts_infer.yaml")
            original = Path("GPT_SoVITS/configs/tts_infer.yaml")
            original.parent.mkdir(parents=True)
            original.touch()
            start = Mock()
            with patch.dict(sys.modules, {"sakuratts.server": SimpleNamespace(start_server=start)}):
                main(["serve"])
                self.assertEqual(start.call_args.kwargs["tts_config"], original)
                local.parent.mkdir(parents=True)
                local.touch()
                main(["serve"])
                self.assertEqual(start.call_args.kwargs["tts_config"], local)
                main(["serve", "-c", "explicit.yaml"])
                self.assertEqual(start.call_args.kwargs["tts_config"], Path("explicit.yaml"))
                main(["serve", "prepared-model"])
                self.assertIsNone(start.call_args.kwargs["tts_config"])

    def test_capabilities_works_without_loading_inference_or_http(self):
        code = """
import sys
from sakuratts.cli import main
assert main(['capabilities']) == 0
assert not set(('numpy', 'torch', 'cupy', 'mlx', 'onnxruntime', 'pyopenjtalk',
                'fastapi', 'uvicorn', 'sakuratts.server', 'sakuratts.backends.cuda.engine')) & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        capabilities = json.loads(result.stdout)
        self.assertEqual(capabilities["backends"], ["cuda"])
        self.assertEqual(capabilities["languages"], ["ja"])
        self.assertEqual(capabilities["language_modes"], ["ja", "all_ja"])

    def test_tts_passes_explicit_backend_and_language_to_the_public_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = MagicMock()
            runtime.__enter__.return_value = runtime
            runtime.synthesize.return_value = SimpleNamespace(report={"status": "completed"}, save=Mock())
            output = Path(temporary) / "speech.wav"
            with patch.object(Engine, "load", return_value=runtime) as load, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = main(["tts", "model", "--text", "今日は晴れです。", "--language", "all_ja",
                               "--backend", "cuda", "--output", str(output)])
            self.assertEqual(result, 0)
            load.assert_called_once_with("model", experimental=None, backend="cuda")
            self.assertEqual(runtime.synthesize.call_args.kwargs["language"], "all_ja")
            runtime.synthesize.return_value.save.assert_called_once_with(output.resolve())

    def test_serve_passes_backend_override_in_both_lifecycle_modes(self):
        for mode in ("direct", "managed"):
            with self.subTest(mode=mode):
                start = Mock()
                with patch.dict(sys.modules, {"sakuratts.server": SimpleNamespace(start_server=start)}):
                    result = main(["serve", "model", "--backend", "cuda", "--runtime-mode", mode])
                self.assertEqual(result, 0)
                self.assertEqual(start.call_args.kwargs["backend"], "cuda")
                self.assertEqual(start.call_args.kwargs.get("runtime_mode", "direct"), mode)

    def test_benchmark_passes_backend_and_language_on_every_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = MagicMock()
            runtime.__enter__.return_value = runtime
            runtime.model.info.return_value = {"backend": "cuda", "languages": ["ja"]}
            runtime.synthesize.return_value = SimpleNamespace(report={"status": "completed"})
            output = Path(temporary) / "benchmark.json"
            with patch.object(Engine, "load", return_value=runtime) as load, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = main(["benchmark", "model", "--text", "今日は晴れです。", "--language", "all_ja",
                               "--backend", "cuda", "--repeats", "2", "--output", str(output)])
            self.assertEqual(result, 0)
            load.assert_called_once_with("model", experimental=None, backend="cuda")
            self.assertEqual(runtime.synthesize.call_count, 2)
            self.assertTrue(all(call.kwargs["language"] == "all_ja" for call in runtime.synthesize.call_args_list))
            self.assertEqual(len(json.loads(output.read_text(encoding="utf-8"))["requests"]), 2)


if __name__ == "__main__":
    unittest.main()
