"""Worker package imports must not expose another interpreter's site-packages."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sakuratts._internal.diagnostics as diagnostics


class WorkerBootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.host_site = self.root / "host-python311/site-packages"
        self.package = self.host_site / "sakuratts"
        shutil.copytree(ROOT / "src/sakuratts", self.package, ignore=shutil.ignore_patterns("__pycache__"))
        (self.package / "witness.py").write_text("ORIGIN = 'selected'\n", encoding="utf-8")
        poison = self.host_site / "numpy"
        poison.mkdir()
        (poison / "__init__.py").write_text("raise RuntimeError('host NumPy ABI was imported')\n", encoding="utf-8")
        (self.host_site / "onnxruntime.py").write_text("raise RuntimeError('host ORT ABI was imported')\n", encoding="utf-8")
        self.other_site = self.root / "worker-existing-packages"
        other = self.other_site / "sakuratts"
        other.mkdir(parents=True)
        (other / "__init__.py").write_text("ORIGIN = 'other'\n", encoding="utf-8")
        (other / "witness.py").write_text("ORIGIN = 'other'\n", encoding="utf-8")

    def isolated(self, code, *arguments):
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, *map(str, arguments)],
            cwd=self.root, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_direct_entry_imports_replace_other_package_without_host_numpy(self):
        code = """import json,runpy,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
import sakuratts.witness
assert sakuratts.witness.ORIGIN=='other'
before=list(sys.path)
runpy.run_path(sys.argv[1],run_name='worker_import_test')
import sakuratts.witness,numpy
assert sakuratts.witness.ORIGIN=='selected'
assert Path(sakuratts.__file__).resolve().parent==Path(sys.argv[1]).resolve().parents[1]
assert sys.path==before
assert str(Path(sys.argv[1]).resolve().parents[2]) not in sys.path
assert Path(sys.argv[1]).resolve().parents[2] not in Path(numpy.__file__).resolve().parents
print(json.dumps({'package':sakuratts.__file__,'numpy':numpy.__file__,'origin':sakuratts.witness.ORIGIN}))
"""
        for name in ("ort_worker.py", "classic_japanese_worker.py", "diagnostics.py"):
            with self.subTest(entry=name):
                report = self.isolated(code, self.package / "_internal" / name, self.other_site)
                self.assertEqual(report["origin"], "selected")

    def test_ordinary_module_imports_preserve_package_and_search_path(self):
        code = """import importlib,json,runpy,sys
from pathlib import Path
before=list(sys.path)
load=runpy.run_path(str(Path(sys.argv[1])/'_internal/worker.py'))['load_package']
package=load(sys.argv[1])
for name in ('ort_worker','classic_japanese_worker','diagnostics'):
    importlib.import_module('sakuratts._internal.'+name)
assert load(sys.argv[1]) is package
assert sys.modules['sakuratts'] is package
assert sys.path==before
assert package.__path__==[str(Path(sys.argv[1]).resolve())]
print(json.dumps({'same_package':True,'search_path_unchanged':True}))
"""
        self.assertTrue(self.isolated(code, self.package)["same_package"])

    def test_worker_script_help_uses_its_interpreter_numpy(self):
        for name in ("ort_worker.py", "classic_japanese_worker.py"):
            with self.subTest(entry=name):
                result = subprocess.run([sys.executable, "-I", "-B", str(self.package / "_internal" / name), "--help"],
                    cwd=self.root, capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_failed_package_initializer_restores_previous_modules(self):
        broken = self.root / "broken-package"
        broken.mkdir()
        (broken / "__init__.py").write_text("import sakuratts.partial\nraise RuntimeError('initializer failed')\n", encoding="utf-8")
        (broken / "partial.py").write_text("VALUE = 1\n", encoding="utf-8")
        code = """import json,runpy,sys
sys.path.insert(0,sys.argv[2])
import sakuratts,sakuratts.witness
original_package,original_child=sakuratts,sakuratts.witness
before=list(sys.path)
load=runpy.run_path(sys.argv[1])['load_package']
try:
    load(sys.argv[3])
except RuntimeError as error:
    assert str(error)=='initializer failed'
else:
    raise AssertionError('expected initializer failure')
assert sys.modules['sakuratts'] is original_package
assert sys.modules['sakuratts.witness'] is original_child
assert 'sakuratts.partial' not in sys.modules
assert sys.path==before
print(json.dumps({'restored':True}))
"""
        self.assertTrue(self.isolated(code, self.package / "_internal/worker.py", self.other_site, broken)["restored"])

    def test_diagnostic_child_loads_own_numpy_and_ort_without_host_site(self):
        worker_site = self.root / "worker-abi-packages"
        worker_site.mkdir()
        (worker_site / "onnxruntime.py").write_text(
            "__version__ = 'worker-ort-test'\ndef get_available_providers(): return ['CUDAExecutionProvider']\n", encoding="utf-8")
        actual_run = subprocess.run

        def isolated_child(command, **kwargs):
            self.assertEqual(command[4], str(self.package.resolve()))
            code = "import sys\nsys.path.insert(0," + repr(str(worker_site)) + ")\n" + command[3]
            return actual_run([command[0], "-I", *command[1:3], code, *command[4:]], **kwargs)

        with patch.object(diagnostics, "__file__", str(self.package / "_internal/diagnostics.py")), \
                patch.object(diagnostics.subprocess, "run", side_effect=isolated_child):
            report = diagnostics.check_worker_imports(Path(sys.executable), {})
        self.assertEqual(report["onnxruntime"], "worker-ort-test")
        self.assertFalse(report["cuda_execution_tested"])
        self.assertFalse(report["torch_imported"])


if __name__ == "__main__":
    unittest.main()
