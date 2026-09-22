"""Conversion publishes complete packages without duplicating temporary resources."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sakuratts.converter import convert


class ConversionPublishTests(unittest.TestCase):
    def inputs(self, root):
        source = root / "official"
        source.mkdir()
        for name in ("gpt.ckpt", "sovits.pth", "reference.wav", "python.exe"):
            (root / name).write_bytes(name.encode())
        return dict(gpt=root / "gpt.ckpt", sovits=root / "sovits.pth",
                    official_source=source, python=root / "python.exe", output=root / "model")

    def run_conversion(self, command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        script = Path(command[2]).name
        if script == "prepare_windows_resources.py":
            (output / "frontend").mkdir(parents=True)
            (output / "frontend/manifest.json").write_text('{"japanese_g2p":{"implementation":"pyopenjtalk-classic"}}')
            if "--frontend-only" not in command:
                (output / "references/reference").mkdir(parents=True)
                (output / "references/reference/conditions.npz").write_bytes(b"reference")
            (output / "prepare.log").write_bytes(b"private preparation log")
        else:
            output.mkdir()
            (output / "weights.bin").write_bytes(script.encode())

    def test_publish_renames_generated_resources_and_leaves_inputs_unchanged(self):
        for with_reference in (False, True):
            with self.subTest(reference=with_reference), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                options = self.inputs(root)
                before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
                if with_reference:
                    options.update(reference=root / "reference.wav", reference_text="reference")
                with patch("sakuratts.converter.run_conversion", side_effect=self.run_conversion), \
                     patch("sakuratts.converter.shutil.copytree", side_effect=AssertionError("Conversion should not copy resources")), \
                     patch("sakuratts._internal.diagnostics.check_windows_packages") as check:
                    model = convert(**options)
                check.assert_called_once()
                self.assertEqual(model.path, root / "model/model.json")
                self.assertEqual(model.manifest["acoustic"], "acoustic")
                self.assertEqual(model.manifest["acoustic_python"], str(root / "python.exe"))
                self.assertEqual((root / "model/acoustic/weights.bin").read_bytes(), b"export_sovits_onnx.py")
                self.assertEqual(model.references, ("reference",) if with_reference else ())
                if with_reference:
                    self.assertEqual((root / "model/references/000/conditions.npz").read_bytes(), b"reference")
                self.assertFalse((root / "model/prepare.log").exists())
                self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()})
                self.assertFalse(list(root.glob(".sakuratts-*")))

    def test_failed_converter_or_final_check_does_not_publish(self):
        for fail_conversion in (True, False):
            with self.subTest(fail_conversion=fail_conversion), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                options = self.inputs(root)

                def run(command, **kwargs):
                    self.run_conversion(command, **kwargs)
                    if fail_conversion and Path(command[2]).name == "export_sovits_onnx.py":
                        raise RuntimeError("conversion failed")

                with patch("sakuratts.converter.run_conversion", side_effect=run), \
                     patch("sakuratts._internal.diagnostics.check_windows_packages", side_effect=RuntimeError("invalid package")), \
                     self.assertRaises(RuntimeError):
                    convert(**options)
                self.assertFalse((root / "model").exists())
                self.assertFalse(list(root.glob(".sakuratts-*")))

    def test_explicit_frontend_worker_does_not_select_acoustic_interpreter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = self.inputs(root)
            frontend_python = root / "frontend.exe"
            frontend_python.write_bytes(b"frontend interpreter")
            with patch("sakuratts.converter.run_conversion", side_effect=self.run_conversion), \
                 patch("sakuratts._internal.diagnostics.check_windows_packages"):
                model = convert(**options, frontend_python=frontend_python)
            self.assertEqual(model.manifest["frontend_python"], str(frontend_python))
            self.assertNotIn("acoustic_python", model.manifest)


if __name__ == "__main__":
    unittest.main()
