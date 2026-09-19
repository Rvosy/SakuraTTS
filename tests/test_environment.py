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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.cli import doctor, main
from sakuratts.diagnostics import checked_file
from sakuratts.reference_condition import sha256_file


class EnvironmentTests(unittest.TestCase):
    def test_missing_dependency_fails_with_a_report(self):
        output = io.StringIO()
        with patch("sakuratts.cli.import_module", side_effect=ImportError("missing numpy")):
            with contextlib.redirect_stdout(output):
                code = main(["doctor"])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("missing numpy", report["packages"]["numpy"]["error"])

    def test_basic_check_does_not_import_compute_backends(self):
        with patch("sakuratts.cli.import_module") as imported:
            with patch("sakuratts.cli.metadata.version", return_value="test"):
                report = doctor()
        imported.assert_called_once_with("numpy")
        self.assertTrue(report["checks_passed"])
        self.assertTrue(report["synthesis"]["windows_backend_implemented"])
        self.assertFalse(report["synthesis"]["models_checked"])
        self.assertIsNone(report["synthesis"]["dependencies_ready"])
        self.assertIsNone(report["synthesis"]["packages_ready"])
        self.assertFalse(report["synthesis"]["inference_tested"])
        self.assertFalse(report["synthesis"]["quality_validated"])

    def test_failed_package_check_does_not_claim_readiness(self):
        with patch("sakuratts.cli.JAPANESE_MODULES", {}), patch("sakuratts.cli.import_module"), \
                patch("sakuratts.cli.metadata.version", return_value="test"), \
                patch("sakuratts.cli.platform.system", return_value="Windows"), \
                patch("sakuratts.cuda_runtime.configure_cuda"), \
                patch("sakuratts.diagnostics.check_windows_packages", side_effect=ValueError("reference identity mismatch")):
            report = doctor(config="runtime.json")
        self.assertTrue(report["synthesis"]["dependencies_ready"])
        self.assertTrue(report["synthesis"]["models_checked"])
        self.assertFalse(report["synthesis"]["packages_ready"])
        self.assertFalse(report["checks_passed"])
        self.assertIn("reference identity mismatch", report["resource_check"]["error"])

    def test_resource_check_rejects_tampering_and_package_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "package"
            root.mkdir()
            file = root / "weights.bin"
            file.write_bytes(b"original")
            spec = {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
            self.assertEqual(checked_file(root, file.name, spec), file)
            file.write_bytes(b"changed!")
            with self.assertRaisesRegex(ValueError, "checksum"):
                checked_file(root, file.name, spec)
            outside = Path(folder) / "outside.bin"
            outside.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "inside its package"):
                checked_file(root, "../outside.bin", spec)

    def test_cli_help_needs_no_optional_dependencies(self):
        source = Path(__file__).resolve().parents[1] / "src"
        process = subprocess.run([sys.executable, "-m", "sakuratts", "--help"], cwd=source,
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0)
        self.assertIn("doctor", process.stdout)

    def test_malformed_windows_config_has_actionable_error_before_model_import(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "runtime.json"
            output = Path(folder) / "new-output/speech.wav"
            for value, expected in (([], "JSON object"),
                                    ({"format": "sakuratts-windows-config-v1"}, "'gpt' package path")):
                config.write_text(json.dumps(value), encoding="utf-8")
                stderr = io.StringIO()
                with patch("sakuratts.nvidia.run_cli") as run, contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        main(["synthesize", "--config", str(config), "--text", "test", "--output", str(output)])
                self.assertEqual(caught.exception.code, 1)
                self.assertIn(expected, stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertFalse(output.parent.exists())
                run.assert_not_called()

    @unittest.skipIf(sys.platform == "darwin", "Non-Mac startup diagnostic")
    def test_synthesis_reports_missing_backend_before_writing_outputs(self):
        script = Path(__file__).resolve().parents[1] / "scripts/synthesize_japanese.py"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-output/speech.wav"
            command = [sys.executable, str(script), "--text", "test", "--output", str(output)]
            for name in ("frontend", "reference", "gpt", "sovits"):
                command.extend(["--" + name + "-package", str(Path(folder) / name)])
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn("sakuratts synthesize --config", process.stderr)
            self.assertFalse(output.parent.exists())
