"""Protect source resources and existing output when preparing reference and frontend packages."""
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

SCRIPT = Path(__file__).resolve().parents[1] / "sakuratts/prepare/prepare_resources.py"
spec = importlib.util.spec_from_file_location("resource_preparation", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ResourcePreparationTests(unittest.TestCase):
    def test_new_reference_preparation_reuses_english_frontend_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, language = self.fixture(directory)
            output = Path(directory).resolve() / "frontend"
            manifest = prepare.prepare_frontend(root, output)
            self.assertEqual(manifest["english_g2p"]["directory"], "english")
            for name in ("g2p.json", "checkpoint.npz"):
                self.assertEqual((output / "english" / name).read_bytes(),
                                 (root / "english" / name).read_bytes())
            before = (output / "manifest.json").read_bytes()
            self.assertEqual(prepare.prepare_frontend(root, output), manifest)
            self.assertEqual((output / "manifest.json").read_bytes(), before)
            (output / "english/checkpoint.npz").write_bytes(b"broken")
            self.assertEqual(prepare.prepare_frontend(root, output), manifest)

    def test_worker_network_guard_blocks_connections_before_they_are_made(self):
        code = ("import runpy,socket; m=runpy.run_path(" + repr(str(SCRIPT)) + "); "
                "m['disable_network'](); socket.socket().connect(('127.0.0.1', 1))")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Reference preparation is offline", result.stderr)

    def fixture(self, directory):
        root = Path(directory).resolve() / "official"
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
        (root / "english").mkdir()
        for name in ("g2p.json", "checkpoint.npz"):
            (root / "english" / name).write_bytes(b"english fixture")
        return root, user, language

    def test_custom_language_model_and_dictionary_are_copied_without_hash_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root, user, language = self.fixture(directory)
            (user / "userdict.md5").write_text("stale")
            before = {p.name: p.read_bytes() for p in user.iterdir()}
            output = Path(directory).resolve() / "frontend"
            prepare.prepare_frontend(root, output)
            self.assertEqual((output / "user.dict").read_bytes(), before["user.dict"])
            self.assertEqual((output / "lid.176.bin").read_bytes(), language.read_bytes())
            self.assertEqual(before, {p.name: p.read_bytes() for p in user.iterdir()})

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
            output = Path(directory).resolve() / "frontend"
            before = {str(p): prepare.digest(p) for p in root.rglob("*") if p.is_file()}
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
            self.assertEqual(prepare.prepare_frontend(root, output, preflight=preflight), manifest)
            self.assertEqual(before, {str(p): prepare.digest(p) for p in root.rglob("*") if p.is_file()})

    def test_reference_preparation_uses_current_files_after_frontend_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory).resolve() / "original"
            root, _, language = self.fixture(original)
            preflight = self.classic_fixture(root)
            frontend = original / "frontend"
            manifest = prepare.prepare_frontend(root, frontend, preflight=preflight)
            sv = root / "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
            sv.parent.mkdir()
            sv.write_bytes(b"speaker embedding fixture")
            config = root / "GPT_SoVITS/configs/tts_infer.yaml"
            config.parent.mkdir()
            config.write_bytes(b"config fixture")
            for name in ("gpt.ckpt", "sovits.pth", "reference.wav"):
                (original / name).write_bytes(name.encode())
            relocated = Path(directory).resolve() / "relocated"
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
                return Mock(stdout=[], wait=Mock(return_value=0))

            args = [str(SCRIPT), "--official-source", str(root), "--inputs", str(inputs),
                    "--output", str(relocated / "prepared"), "--frontend", str(frontend),
                    "--language-model", str(frontend / "lid.176.bin")]
            with patch.object(prepare, "inspect_frontend", return_value=preflight), \
                 patch.object(prepare.subprocess, "Popen", side_effect=start_worker), \
                 patch.object(sys, "argv", args), patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(prepare.main(), 0)
            self.assertFalse(original.exists())
            self.assertFalse(Path(manifest["sources"]["language_model"]["path"]).exists())
            self.assertEqual(prepare.read_json(frontend / "manifest.json"), manifest)
            self.assertTrue(all(Path(path).is_relative_to(relocated) for path in jobs[0]["source_hashes"]))
            self.assertEqual(Path(jobs[0]["frontend"]), frontend)

    def test_duplicate_tones_do_not_silently_replace_a_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ("gpt.ckpt", "sovits.pth", "a.ogg", "b.ogg"):
                (root / name).write_bytes(b"fixture")
            (root / "character.json").write_text(json.dumps({"voice": {
                "gpt_model": "gpt.ckpt", "sovits_model": "sovits.pth", "tone_refs": "ref.txt"}}), encoding="utf-8")
            (root / "ref.txt").write_text("a.ogg|JA|こんにちは。|中性\nb.ogg|JA|おはよう。|中性\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate reference tone"):
                prepare.character_inputs(root)


if __name__ == "__main__":
    unittest.main()
