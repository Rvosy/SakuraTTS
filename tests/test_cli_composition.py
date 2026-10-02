"""CLI selection reaches the runtime without eager optional dependencies."""

import builtins
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sakuratts.cli import main


class CliCompositionTests(unittest.TestCase):
    def test_serve_preserves_the_server_import_failure(self):
        original_import = builtins.__import__
        failure = "cannot import name 'MissingRuntime' from 'sakuratts.engine'"
        def import_module(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "server" and level == 1:
                raise ImportError(failure)
            return original_import(name, globals, locals, fromlist, level)
        errors = io.StringIO()
        with patch("builtins.__import__", side_effect=import_module), \
                contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
            main(["serve", "model"])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn(failure, errors.getvalue())

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
        self.assertEqual(capabilities["backends"], ["cuda", "cpu", "directml", "mlx"])
        self.assertEqual(capabilities["languages"], ["ja", "en"])
        self.assertEqual(capabilities["language_modes"], ["ja", "all_ja", "en", "auto"])

if __name__ == "__main__":
    unittest.main()
