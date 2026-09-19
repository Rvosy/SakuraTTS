"""Check the offline runtime assembly's file and overwrite boundaries."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_ort_worker_runtime.py"
spec = importlib.util.spec_from_file_location("ort_worker_runtime_preparation", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ORTWorkerRuntimePreparationTests(unittest.TestCase):
    def test_native_libraries_are_kept_while_package_tests_and_bytecode_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("__init__.py", "core/native.pyd", ".libs/openblas.dll", "tests/test_example.py", "core/__pycache__/cache.pyc"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            files = {relative.as_posix() for _, relative in prepare.package_files(root)}
            self.assertEqual(files, {"__init__.py", "core/native.pyd", ".libs/openblas.dll"})

    def test_existing_runtime_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cuda, output = root / "source", root / "cuda", root / "output"
            for folder in (source, cuda, output):
                folder.mkdir()
            sentinel = output / "user-file"
            sentinel.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                prepare.prepare(source, cuda, output)
            self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_output_inside_source_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cuda = root / "source", root / "cuda"
            source.mkdir()
            cuda.mkdir()
            output = source / "assembled"
            with self.assertRaisesRegex(ValueError, "outside the source"):
                prepare.prepare(source, cuda, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
