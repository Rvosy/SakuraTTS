"""Diagnostics must distinguish installed tools from an implemented TTS backend."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sakuratts.cli import doctor, main


class EnvironmentTests(unittest.TestCase):
    def test_missing_dependency_fails_with_a_report(self):
        output = io.StringIO()
        with patch("sakuratts.cli.import_module", side_effect=ImportError("missing numpy")):
            with contextlib.redirect_stdout(output):
                code = main(["doctor"])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("missing numpy", report["packages"]["numpy"]["error"])

    def test_configured_classic_workers_do_not_require_plus_or_ort_in_main_python(self):
        resource_check = {"status": "passed", "profile": None,
                          "japanese_g2p": {"implementation": "pyopenjtalk-classic"}}
        def import_selected(module):
            if module in ("pyopenjtalk", "onnxruntime", "sudachipy", "sudachidict_core"):
                raise ImportError("This dependency belongs to another configured worker")
        with patch("sakuratts.cli.import_module", side_effect=import_selected) as imported, \
                patch("sakuratts.cli.metadata.version", return_value="test"), \
                patch("sakuratts.cli.platform.system", return_value="Windows"), \
                patch("sakuratts.backends.cuda.runtime.configure_cuda"), \
                patch("sakuratts.backends.cuda.runtime.import_cupy"), \
                patch("sakuratts.backends.cuda.runtime.validate_gpt_cuda_include_paths", return_value={}), \
                patch("sakuratts.runtime.diagnostics.check_windows_packages", return_value=resource_check):
            report = doctor(config="runtime.json")
        self.assertTrue(report["checks_passed"])
        self.assertTrue(report["synthesis"]["dependencies_ready"])
        modules = {call.args[0] for call in imported.call_args_list}
        self.assertEqual(modules, {"numpy", "split_lang", "fast_langdetect", "fasttext"})

    @unittest.skipIf(sys.platform == "darwin", "Non-Mac startup diagnostic")
    def test_synthesis_reports_missing_backend_before_writing_outputs(self):
        script = Path(__file__).resolve().parents[1] / "tools/synthesize_japanese.py"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-output/speech.wav"
            command = [sys.executable, str(script), "--text", "test", "--output", str(output)]
            for name in ("frontend", "reference", "gpt", "sovits"):
                command.extend(["--" + name + "-package", str(Path(folder) / name)])
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn("sakuratts synthesize --config", process.stderr)
            self.assertFalse(output.parent.exists())
