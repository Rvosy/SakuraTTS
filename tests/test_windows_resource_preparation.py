"""Protect source resources and existing output when preparing Windows packages."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "src/sakuratts/_internal/conversion/prepare_windows_resources.py"
spec = importlib.util.spec_from_file_location("windows_resource_preparation", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class WindowsResourcePreparationTests(unittest.TestCase):
    def test_worker_network_guard_blocks_connections_before_they_are_made(self):
        code = ("import runpy,socket; m=runpy.run_path(" + repr(str(SCRIPT)) + "); "
                "m['disable_network'](); socket.socket().connect(('127.0.0.1', 1))")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Reference preparation is offline", result.stderr)

    def fixture(self, directory):
        root = Path(directory) / "official"
        user = root / "GPT_SoVITS/text/ja_userdic"
        user.mkdir(parents=True)
        (user / "userdict.csv").write_bytes(b"existing user dictionary source")
        (user / "userdict.md5").write_text(hashlib.md5((user / "userdict.csv").read_bytes()).hexdigest())
        (user / "user.dict").write_bytes(b"compiled dictionary")
        (root / "GPT_SoVITS/text/symbols2.py").write_text("symbols = ['_', 'a', 'N']\n")
        (root / "GPT_SoVITS/TTS_infer_pack").mkdir()
        (root / prepare.SOURCE_FILE).write_text("# source identity\n")
        language = root / "GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin"
        language.parent.mkdir(parents=True)
        language.write_bytes(b"full language model fixture")
        return root, user, language

    def test_stale_user_dictionary_is_rejected_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as directory:
            root, user, _ = self.fixture(directory)
            (user / "userdict.md5").write_text("stale")
            before = {p.name: p.read_bytes() for p in user.iterdir()}
            output = Path(directory) / "frontend"
            with self.assertRaisesRegex(ValueError, "needs rebuilding"):
                prepare.prepare_frontend(root, output)
            self.assertFalse(output.exists())
            self.assertEqual(before, {p.name: p.read_bytes() for p in user.iterdir()})

    def test_reusing_frontend_rejects_changed_original_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            root, user, language = self.fixture(directory)
            output = Path(directory) / "frontend"
            with patch.object(prepare, "LID_BYTES", language.stat().st_size), patch.object(prepare, "LID_SHA256", prepare.digest(language)):
                prepare.prepare_frontend(root, output)
                original = (output / "user.dict").read_bytes()
                (user / "user.dict").write_bytes(b"replacement dictionary")
                with self.assertRaisesRegex(ValueError, "differs from the official source"):
                    prepare.prepare_frontend(root, output)
                self.assertEqual(original, (output / "user.dict").read_bytes())

    def test_runtime_paths_are_portable_and_existing_configuration_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            refs = [{"tone": "中性"}, {"tone": "惊讶"}]
            path = prepare.write_runtime_config(directory, refs)
            config = prepare.read_json(path)
            self.assertEqual(config["references"]["惊讶"], "references/惊讶")
            self.assertEqual(config["default_reference"], "中性")
            config["gpt"] = "user-selected-model"
            prepare.write_json(path, config)
            with self.assertRaises(FileExistsError):
                prepare.write_runtime_config(directory, refs)
            self.assertEqual(prepare.read_json(path)["gpt"], "user-selected-model")

    def classic_fixture(self, root):
        site = root / "runtime/Lib/site-packages"
        module = site / "pyopenjtalk"
        dictionary = module / "open_jtalk_dic_utf_8-1.11"
        metadata = site / "pyopenjtalk-0.3.4.dist-info"
        dictionary.mkdir(parents=True)
        metadata.mkdir()
        for path, data in ((module / "__init__.py", b"# original module"),
                           (module / "openjtalk.cp39-win_amd64.pyd", b"native module fixture"),
                           (dictionary / "sys.dic", b"original main dictionary"),
                           (dictionary / "COPYING", b"dictionary notice"),
                           (metadata / "LICENSE.md", b"module license")):
            path.write_bytes(data)
        for cache in ("__pycache__", "~_pycache__"):
            (module / cache).mkdir()
            (module / cache / "cached.pyc").write_bytes(b"not portable")
        return {"implementation": "pyopenjtalk-classic", "version": "0.3.4", "module_version": "0.3.4",
                "module_directory": str(module), "main_dictionary": str(dictionary),
                "distribution_directory": str(metadata)}

    def test_classic_native_module_dictionary_and_license_are_hashed_without_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, language = self.fixture(directory)
            preflight = self.classic_fixture(root)
            output = Path(directory) / "frontend"
            before = {str(p): prepare.digest(p) for p in root.rglob("*") if p.is_file()}
            with patch.object(prepare, "LID_BYTES", language.stat().st_size), patch.object(prepare, "LID_SHA256", prepare.digest(language)):
                manifest = prepare.prepare_frontend(root, output, preflight=preflight)
                self.assertEqual(manifest["japanese_g2p"]["main_dictionary"],
                                 "classic-python/pyopenjtalk/open_jtalk_dic_utf_8-1.11")
                names = set(manifest["files"])
                self.assertIn("classic-python/pyopenjtalk-0.3.4.dist-info/LICENSE.md", names)
                self.assertIn("classic-python/pyopenjtalk/openjtalk.cp39-win_amd64.pyd", names)
                self.assertIn("classic-python/pyopenjtalk/open_jtalk_dic_utf_8-1.11/sys.dic", names)
                self.assertFalse(any("pycache" in name for name in names))
                self.assertEqual(prepare.prepare_frontend(root, output, preflight=preflight), manifest)
                (output / "classic-python/pyopenjtalk/openjtalk.cp39-win_amd64.pyd").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "damaged"):
                    prepare.prepare_frontend(root, output, preflight=preflight)
            self.assertEqual(before, {str(p): prepare.digest(p) for p in root.rglob("*") if p.is_file()})

    def test_classic_profile_does_not_overwrite_existing_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, language = self.fixture(directory)
            preflight = self.classic_fixture(root)
            output = Path(directory) / "frontend"
            with patch.object(prepare, "LID_BYTES", language.stat().st_size), patch.object(prepare, "LID_SHA256", prepare.digest(language)):
                prepare.prepare_frontend(root, output)
                before = {str(p): prepare.digest(p) for p in output.rglob("*") if p.is_file()}
                with self.assertRaisesRegex(ValueError, "different Japanese G2P profile"):
                    prepare.prepare_frontend(root, output, preflight=preflight)
                self.assertEqual(before, {str(p): prepare.digest(p) for p in output.rglob("*") if p.is_file()})

    def test_reference_preparation_uses_current_files_after_frontend_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            root, _, language = self.fixture(original)
            preflight = self.classic_fixture(root)
            frontend = original / "frontend"
            expected_language_hash = prepare.digest(language)
            expected_language_bytes = language.stat().st_size
            with patch.object(prepare, "LID_BYTES", expected_language_bytes), \
                 patch.object(prepare, "LID_SHA256", expected_language_hash):
                manifest = prepare.prepare_frontend(root, frontend, preflight=preflight)
            sv = root / "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
            sv.parent.mkdir()
            sv.write_bytes(b"speaker embedding fixture")
            config = root / "GPT_SoVITS/configs/tts_infer.yaml"
            config.parent.mkdir()
            config.write_bytes(b"config fixture")
            for name in ("gpt.ckpt", "sovits.pth", "reference.wav"):
                (original / name).write_bytes(name.encode())
            relocated = Path(directory) / "relocated"
            original.rename(relocated)
            root = relocated / "official"
            frontend = relocated / "frontend"
            preflight = dict(preflight)
            for field in ("module_directory", "main_dictionary", "distribution_directory"):
                preflight[field] = str(relocated / Path(preflight[field]).relative_to(original))
            inputs = relocated / "inputs.json"
            prepare.write_json(inputs, {"gpt": str(relocated / "gpt.ckpt"), "sovits": str(relocated / "sovits.pth"),
                "references": [{"audio": str(relocated / "reference.wav"), "text": "こんにちは。",
                                "language": "ja", "tone": "reference"}]})
            jobs = []

            def start_worker(command, **_kwargs):
                job = prepare.read_json(command[-1])
                jobs.append(job)
                prepare.verify_protected(job["protected"])
                return Mock(stdout=[], wait=Mock(return_value=0))

            args = [str(SCRIPT), "--official-source", str(root), "--inputs", str(inputs),
                    "--output", str(relocated / "prepared"), "--frontend", str(frontend),
                    "--language-model", str(frontend / "lid.176.bin")]
            with patch.object(prepare, "LID_BYTES", expected_language_bytes), \
                 patch.object(prepare, "LID_SHA256", expected_language_hash), \
                 patch.object(prepare, "inspect_frontend", return_value=preflight), \
                 patch.object(prepare.subprocess, "Popen", side_effect=start_worker), \
                 patch.object(sys, "argv", args), patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(prepare.main(), 0)
            self.assertFalse(original.exists())
            self.assertFalse(Path(manifest["sources"]["language_model"]["path"]).exists())
            self.assertEqual(prepare.read_json(frontend / "manifest.json"), manifest)
            protected = jobs[0]["protected"]
            self.assertIn(str(frontend / "manifest.json"), protected)
            for name in manifest["files"]:
                self.assertIn(str(frontend / name), protected)
            for path in prepare.classic_frontend_files(preflight).values():
                self.assertIn(str(path), protected)
            self.assertTrue(all(Path(path).is_relative_to(relocated) for path in protected))
            (frontend / "user.dict").write_bytes(b"changed during preparation")
            with self.assertRaisesRegex(RuntimeError, "Preparation source changed"):
                prepare.verify_protected(protected)

    def test_duplicate_tones_do_not_silently_replace_a_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("gpt.ckpt", "sovits.pth", "a.ogg", "b.ogg"):
                (root / name).write_bytes(b"fixture")
            (root / "character.json").write_text(json.dumps({"voice": {
                "gpt_model": "gpt.ckpt", "sovits_model": "sovits.pth", "tone_refs": "ref.txt"}}), encoding="utf-8")
            (root / "ref.txt").write_text("a.ogg|JA|こんにちは。|中性\nb.ogg|JA|おはよう。|中性\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate reference tone"):
                prepare.character_inputs(root)


if __name__ == "__main__":
    unittest.main()
