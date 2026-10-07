"""Selected virtual environments survive preparation, packaging and worker launch."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv

from sakuratts.prepare import prepare_resources
from sakuratts.prepare.cache import prepare_initial_model
from sakuratts.prepare.converter import convert, package_model
from sakuratts.TTS_infer_pack.TTS import Inference
import test_backend_selection as backend_fixture
import test_conversion_publish as conversion_fixture
import test_ort_process_lifecycle as process_fixture


@unittest.skipUnless(os.name == "posix", "POSIX virtual environments use symlinked interpreters")
class InterpreterPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.environment = self.root / "selected environment"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.environment)
        self.python = self.environment / "bin/python"
        self.assertTrue(self.python.is_symlink())
        result = subprocess.run([str(self.python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                                capture_output=True, text=True, check=True)
        self.site = Path(result.stdout.strip())
        (self.site / "selected_environment.py").write_text("VALUE = 'selected environment'\n")
        self.environment_patch = patch.dict(os.environ)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        os.environ.pop("SAKURATTS_BUNDLE_ROOT", None)
        os.environ.pop("PYTHONPATH", None)

    def assert_environment(self, python):
        result = subprocess.run([str(python), "-B", "-c",
            "import json,sys,selected_environment; print(json.dumps([sys.prefix, selected_environment.VALUE]))"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [str(self.environment), "selected environment"])

    def test_frontend_preflight_imports_from_selected_environment(self):
        module = self.site / "pyopenjtalk"
        (module / "dictionary").mkdir(parents=True)
        (module / "dictionary/sys.dic").touch()
        (module / "__init__.py").write_text("""import sys
from pathlib import Path
from selected_environment import VALUE
__version__ = sys.prefix
OPEN_JTALK_DICT_DIR = str(Path(__file__).parent / 'dictionary')
""")
        metadata = self.site / "pyopenjtalk_plus-1.2.3.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: pyopenjtalk-plus\nVersion: 1.2.3\n")
        result = prepare_resources.inspect_frontend(os.path.relpath(self.python))
        self.assertEqual(result["module_version"], str(self.environment))
        self.assertEqual(Path(result["module_directory"]), module)
        self.assertEqual(result["executable"], str(self.python))

    def test_preparation_worker_runs_in_selected_environment(self):
        source, frontend = self.root / "official", self.root / "frontend"
        source.mkdir()
        frontend.mkdir()
        (frontend / "manifest.json").write_text('{"official_commit":"fixture"}')
        inputs = {name: str(self.root / name) for name in ("gpt", "sovits")}
        audio = self.root / "audio.wav"
        for path in (*inputs.values(), audio):
            Path(path).write_bytes(b"fixture")
        inputs["references"] = [{"audio": str(audio), "text": "reference", "language": "ja", "tone": "reference"}]
        request = self.root / "inputs.json"
        request.write_text(json.dumps(inputs))
        script = self.root / "worker.py"
        script.write_text("""import json,sys
from pathlib import Path
from selected_environment import VALUE
Path(sys.argv[-1]).with_name('environment.json').write_text(json.dumps([sys.prefix, VALUE]))
""")
        output = self.root / "prepared"
        launch = subprocess.Popen

        def start_worker(*args, **kwargs):
            process = launch(*args, **kwargs)
            self.addCleanup(process.stdout.close)
            return process

        with patch.object(prepare_resources, "inspect_frontend", return_value={}), \
                patch.object(prepare_resources, "__file__", str(script)), \
                patch.object(prepare_resources.subprocess, "Popen", side_effect=start_worker), \
                contextlib.redirect_stdout(io.StringIO()):
            result = prepare_resources.main(["--official-source", str(source), "--inputs", str(request),
                "--output", str(output), "--frontend", str(frontend), "--python", str(self.python)])
        self.assertEqual(result, 0)
        reports = list(output.glob("prepare-*/environment.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text()), [str(self.environment), "selected environment"])

    def test_conversion_packaging_and_activation_keep_worker_environment(self):
        fixture = conversion_fixture.ConversionPublishTests()
        options = fixture.inputs(self.root)
        options.update(python=self.python, acoustic_python=self.python, frontend_python=self.python)

        def run(command, **kwargs):
            self.assert_environment(command[0])
            fixture.run_conversion(command, **kwargs)

        with patch("sakuratts.prepare.converter.run_conversion", side_effect=run), \
                patch("sakuratts.diagnostics.resources.check_prepared_packages"):
            model = convert(**options)
            model = package_model(model.path, self.root / "repacked")
        for key in ("acoustic_python", "frontend_python"):
            self.assert_environment(model.manifest[key])
        config = self.root / "tts.json"
        config.write_text(json.dumps({"sakuratts": {name: str(self.python)
            for name in ("acoustic_python", "frontend_python")}}))
        with patch("sakuratts.backends.create_runtime", side_effect=lambda *args, **kwargs: backend_fixture.FakeRuntime()) as create:
            inference = Inference(model, tts_config=config)
            try:
                for key in ("acoustic_python", "frontend_python"):
                    self.assert_environment(create.call_args.args[0].manifest[key])
            finally:
                inference.close()

    def test_initial_cache_distinguishes_environments_sharing_base_python(self):
        fixture = conversion_fixture.ConversionPublishTests()
        inputs = fixture.inputs(self.root)
        source = inputs["official_source"] / "GPT_SoVITS/TTS_infer_pack"
        source.mkdir(parents=True)
        (source / "TTS.py").write_bytes(b"fixture")
        other_environment = self.root / "another environment"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(other_environment)
        other_python = other_environment / "bin/python"
        self.assertEqual(self.python.resolve(), other_python.resolve())
        settings = {"gpt_checkpoint": inputs["gpt"], "sovits_checkpoint": inputs["sovits"],
            "official_source": inputs["official_source"], "cache_dir": self.root / "cache"}
        with patch("sakuratts.prepare.converter.convert", side_effect=lambda **kwargs: kwargs["output"].mkdir(parents=True)) as conversion, \
                patch("sakuratts.prepare.cache.Model.load", side_effect=lambda path: path):
            first = prepare_initial_model(dict(settings, python=self.python))
            self.assertEqual(prepare_initial_model(dict(settings, python=self.python)), first)
            second = prepare_initial_model(dict(settings, python=other_python))
        self.assertNotEqual(first, second)
        self.assertEqual(conversion.call_count, 2)

    def test_acoustic_and_classic_workers_launch_selected_environment(self):
        from sakuratts.runtime.ort_process import ORTProcessSoVITS
        from sakuratts.text.classic_japanese import ClassicJapaneseG2P

        interpreters = []

        def child(command, **kwargs):
            interpreters.append(command[0])
            return process_fixture.Child(process_fixture.frames(({
                "status": "ready", "worker_pid": process_fixture.Child.pid,
                "providers": ["CUDAExecutionProvider"], "provider_options": {}}, {})))

        with patch("sakuratts.runtime.ort_process.read_manifest", return_value=({"config": {"sample_rate": 32000}}, None)), \
                patch("sakuratts.runtime.ort_process.subprocess.Popen", side_effect=child):
            acoustic = ORTProcessSoVITS(self.root, self.python)
            acoustic.close()
        dictionary = self.root / "user.dict"
        dictionary.touch()
        with patch("sakuratts.text.classic_japanese.subprocess.Popen", side_effect=child):
            frontend = ClassicJapaneseG2P(self.python, self.root, self.root, dictionary)
            frontend.close()
        for interpreter in interpreters:
            self.assert_environment(interpreter)


if __name__ == "__main__":
    unittest.main()
