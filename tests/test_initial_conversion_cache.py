"""Preparation caches survive moves and track portable component updates."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.engine import Inference
from sakuratts.reference import ReferenceCache
from sakuratts._internal.reference_condition import sha256_file


class InitialConversionCacheTests(unittest.TestCase):
    def fixture(self, root):
        preparation = root / "runtime/preparation"
        preparation.mkdir(parents=True)
        (preparation / "python.exe").write_bytes(b"preparation interpreter")
        (root / "runtime/portable.json").write_text('{"format":"sakuratts-portable-v1"}')
        (preparation / "preparation.json").write_text(json.dumps({
            "format": "sakuratts-preparation-v1", "python": "python.exe", "official_source": "official",
            "language_model": "official/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin"}))
        (preparation / "preparation-manifest.json").write_text(json.dumps({
            "format": "sakuratts-preparation-bundle-v1", "files": {
                "official/GPT_SoVITS/text/ja_userdic/user.dict": {"sha256": "initial-dictionary"},
                "python.exe": {"sha256": "same-interpreter"}}}))
        source = preparation / "official/GPT_SoVITS/TTS_infer_pack"
        source.mkdir(parents=True)
        (source / "TTS.py").write_bytes(b"official source fixture")
        for name in ("gpt.ckpt", "sovits.pth"):
            (root / name).write_bytes(name.encode())

    def inference(self, root):
        inference = Inference.__new__(Inference)
        inference.logger = Mock()
        inference._activate = Mock()
        inference.settings = {
            "gpt_checkpoint": str(root / "gpt.ckpt"), "sovits_checkpoint": str(root / "sovits.pth"),
            "official_source": str(root / "runtime/preparation/official"),
            "python": str(root / "runtime/preparation/python.exe"), "cache_dir": str(root / "cache")}
        return inference

    def test_portable_cache_survives_move_and_changes_with_preparation_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "original"
            self.fixture(original)
            with patch("sakuratts.converter.convert", side_effect=lambda **kwargs: kwargs["output"].mkdir(parents=True)) as convert, \
                 patch("sakuratts.engine.Model.load", side_effect=lambda path: path):
                with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(original)):
                    first = self.inference(original)
                    first._convert_initial()
                self.assertEqual(convert.call_count, 1)
                key = first._activate.call_args.args[0].name
                relocated = Path(temporary) / "relocated"
                original.rename(relocated)
                self.assertFalse(original.exists())
                with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(relocated)):
                    moved = self.inference(relocated)
                    moved._convert_initial()
                    self.assertEqual(convert.call_count, 1)
                    self.assertEqual(moved._activate.call_args.args[0], relocated / "cache/models" / key)
                    self.assertEqual(len(list((relocated / "cache/models").iterdir())), 1)
                    manifest = relocated / "runtime/preparation/preparation-manifest.json"
                    data = json.loads(manifest.read_text())
                    data["files"]["official/GPT_SoVITS/text/ja_userdic/user.dict"]["sha256"] = "updated-dictionary"
                    manifest.write_text(json.dumps(data))
                    changed = self.inference(relocated)
                    changed._convert_initial()
                    self.assertEqual(convert.call_count, 2)
                    self.assertNotEqual(changed._activate.call_args.args[0].name, key)

    def test_source_installation_preserves_interpreter_path_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with patch.dict(os.environ, {}, clear=True), \
                 patch("sakuratts.converter.convert", side_effect=lambda **kwargs: kwargs["output"].mkdir(parents=True)) as convert, \
                 patch("sakuratts.engine.Model.load", side_effect=lambda path: path):
                inference = self.inference(root)
                inference._convert_initial()
                inference._convert_initial()
                self.assertEqual(convert.call_count, 1)
                inference.settings["python"] = str(root / "another/python.exe")
                inference._convert_initial()
                self.assertEqual(convert.call_count, 2)

    def update_component(self, root):
        manifest = root / "runtime/preparation/preparation-manifest.json"
        data = json.loads(manifest.read_text())
        data["files"]["python.exe"]["sha256"] = "updated-interpreter"
        manifest.write_text(json.dumps(data))

    def test_single_weight_cache_tracks_component_only_in_portable_mode(self):
        for portable in (False, True):
            for kind, name in (("gpt", "gpt.ckpt"), ("sovits", "sovits.pth")):
                with self.subTest(portable=portable, kind=kind), tempfile.TemporaryDirectory() as temporary:
                    original = Path(temporary) / "original"
                    self.fixture(original)

                    def convert_checkpoint(kind, _checkpoint, output, **_kwargs):
                        output.mkdir(parents=True)
                        format = "sakuratts-gpt-fp32-v1" if kind == "gpt" else "sakuratts-sovits-onnx-v1"
                        (output / "manifest.json").write_text(json.dumps({"format": format}))

                    def set_weights(root):
                        inference = self.inference(root)
                        inference.engine = None
                        inference._log_weights = Mock()
                        inference.model = SimpleNamespace(path=root / "model.json",
                            runtime_config={"gpt": "gpt", "sovits": "sovits", "frontend": "frontend"})
                        with patch.dict(os.environ, {"SAKURATTS_BUNDLE_ROOT": str(root)} if portable else {}, clear=True):
                            inference.set_weights(kind, root / name)
                        return Path(inference._activate.call_args.args[0].manifest[kind]).name

                    with patch("sakuratts.converter.convert_checkpoint", side_effect=convert_checkpoint) as convert:
                        key = set_weights(original)
                        self.assertEqual(convert.call_count, 1)
                        relocated = Path(temporary) / "relocated"
                        original.rename(relocated)
                        self.assertEqual(set_weights(relocated), key)
                        self.assertEqual(convert.call_count, 1)
                        self.update_component(relocated)
                        changed_key = set_weights(relocated)
                        self.assertEqual(convert.call_count, 2 if portable else 1)
                        self.assertEqual(changed_key == key, not portable)

    def test_reference_cache_tracks_component_only_in_portable_mode(self):
        for portable in (False, True):
            with self.subTest(portable=portable), tempfile.TemporaryDirectory() as temporary:
                original = Path(temporary) / "original"
                self.fixture(original)
                (original / "frontend").mkdir()
                (original / "frontend/manifest.json").write_text('{"format":"frontend-fixture"}')
                (original / "audio.wav").write_bytes(b"audio fixture")

                def prepare_audio(root):
                    settings = self.inference(root).settings
                    engine = SimpleNamespace(model=SimpleNamespace(path=root / "model.json", runtime_config={}),
                        _runtime=SimpleNamespace(manifests={kind: {"source": {
                            "checkpoint_sha256": sha256_file(settings[kind + "_checkpoint"]),
                            "official_commit": "same-source"}} for kind in ("gpt", "sovits")},
                            packages={"frontend": root / "frontend"}))
                    with patch.dict(os.environ, {"SAKURATTS_BUNDLE_ROOT": str(root)} if portable else {}, clear=True):
                        ReferenceCache(engine, settings).prepare_audio(root / "audio.wav")

                with patch("sakuratts.converter.prepare_reference", side_effect=lambda **kwargs: kwargs["output"].mkdir(parents=True)) as prepare, \
                     patch("sakuratts.reference.PreparedReference.load") as load:
                    prepare_audio(original)
                    self.assertEqual(prepare.call_count, 1)
                    key = load.call_args.args[0].name
                    relocated = Path(temporary) / "relocated"
                    original.rename(relocated)
                    prepare_audio(relocated)
                    self.assertEqual(prepare.call_count, 1)
                    self.assertEqual(load.call_args.args[0], relocated / "cache/references" / key)
                    self.update_component(relocated)
                    prepare_audio(relocated)
                    self.assertEqual(prepare.call_count, 2 if portable else 1)
                    self.assertEqual(load.call_args.args[0].name == key, not portable)

    def test_portable_reference_uses_component_identity_and_tracks_custom_hubert(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / "frontend").mkdir()
            (root / "frontend/manifest.json").write_text('{}')
            (root / "audio.wav").write_bytes(b"audio fixture")
            bundled = root / "runtime/preparation/official/GPT_SoVITS/pretrained_models/chinese-hubert-base"
            bundled.mkdir(parents=True)
            (bundled / "weights.bin").write_bytes(b"bundled hubert")
            custom = root / "custom-hubert"
            custom.mkdir()
            (custom / "weights.bin").write_bytes(b"custom hubert")
            settings = self.inference(root).settings
            engine = SimpleNamespace(model=SimpleNamespace(path=root / "model.json", runtime_config={}),
                _runtime=SimpleNamespace(manifests={kind: {"source": {
                    "checkpoint_sha256": sha256_file(settings[kind + "_checkpoint"]),
                    "official_commit": "same-source"}} for kind in ("gpt", "sovits")},
                    packages={"frontend": root / "frontend"}))
            with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)), \
                 patch("sakuratts.reference.sha256_file", wraps=sha256_file) as digest, \
                 patch("sakuratts.converter.prepare_reference", side_effect=lambda **kwargs: kwargs["output"].mkdir(parents=True)) as prepare, \
                 patch("sakuratts.reference.PreparedReference.load"):
                ReferenceCache(engine, settings).prepare_audio(root / "audio.wav")
                self.assertNotIn(bundled / "weights.bin", [Path(call.args[0]) for call in digest.call_args_list])
                settings["cnhubert"] = str(custom)
                ReferenceCache(engine, settings).prepare_audio(root / "audio.wav")
                self.assertEqual(prepare.call_count, 2)
                ReferenceCache(engine, settings).prepare_audio(root / "audio.wav")
                self.assertEqual(prepare.call_count, 2)
                (custom / "weights.bin").write_bytes(b"new custom hubert")
                ReferenceCache(engine, settings).prepare_audio(root / "audio.wav")
                self.assertEqual(prepare.call_count, 3)


if __name__ == "__main__":
    unittest.main()
