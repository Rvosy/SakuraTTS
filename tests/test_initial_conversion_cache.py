"""Preparation caches survive moves and track portable component updates."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sakuratts.TTS_infer_pack.TTS import Inference
from sakuratts.TTS_infer_pack.reference import ReferenceCache
from sakuratts.module.reference_condition import sha256_file


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
        inference.experimental = None
        inference.settings = {
            "gpt_checkpoint": str(root / "gpt.ckpt"), "sovits_checkpoint": str(root / "sovits.pth"),
            "official_source": str(root / "runtime/preparation/official"),
            "python": str(root / "runtime/preparation/python.exe"), "cache_dir": str(root / "cache")}
        return inference

    def test_portable_cache_survives_move_and_changes_with_preparation_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "original"
            self.fixture(original)
            with patch("sakuratts.prepare.converter.convert", side_effect=lambda **kwargs: kwargs["output"].mkdir(parents=True)) as convert, \
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

    def update_component(self, root):
        manifest = root / "runtime/preparation/preparation-manifest.json"
        data = json.loads(manifest.read_text())
        data["files"]["python.exe"]["sha256"] = "updated-interpreter"
        manifest.write_text(json.dumps(data))

    def test_initial_cache_separates_backends_and_directml_capacity_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            inference = self.inference(root)
            with patch("sakuratts.prepare.converter.convert", side_effect=lambda **kwargs: kwargs["output"].mkdir(parents=True)) as convert, \
                    patch("sakuratts.engine.Model.load", side_effect=lambda path: path):
                for backend, options, count in (
                    ("cuda", None, 1), ("cpu", None, 2), ("cpu", {"threads": 3}, 2),
                    ("directml", None, 3), ("directml", {"threads": 3, "device_id": 1}, 3),
                    ("directml", {"capacity": 1024}, 4), ("directml", {"capacity": 1280}, 4)):
                    with self.subTest(backend=backend, options=options):
                        inference.settings["backend"] = backend
                        inference.experimental = options
                        inference._convert_initial()
                        self.assertEqual(convert.call_count, count)
                self.assertEqual([call.kwargs["backend"] for call in convert.call_args_list],
                                 ["cuda", "cpu", "directml", "directml"])
                self.assertEqual(convert.call_args.kwargs["experimental"], {"capacity": 1024})

    def test_shared_checkpoint_loader_updates_invalidate_model_and_weight_caches(self):
        from sakuratts.prepare.cache import prepare_checkpoint, prepare_initial_model

        for backend in ("cuda", "mlx"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.fixture(root)
                settings = dict(self.inference(root).settings, backend=backend)
                loader_revision = "initial-loader"

                def digest(path):
                    return loader_revision if Path(path).name == "sovits_checkpoint.py" else sha256_file(path)

                def prepare_weight(_kind, _path, output, **_kwargs):
                    output.mkdir(parents=True)

                def cached_paths():
                    return (prepare_initial_model(settings),
                            prepare_checkpoint("sovits", root / "sovits.pth", "same-weights", settings,
                                               backend=backend))

                with patch.dict(os.environ, SAKURATTS_BUNDLE_ROOT=str(root)), \
                        patch("sakuratts.module.reference_condition.sha256_file", side_effect=digest), \
                        patch("sakuratts.prepare.converter.convert", side_effect=lambda **kwargs: kwargs["output"].mkdir(parents=True)) as convert, \
                        patch("sakuratts.prepare.converter.convert_checkpoint", side_effect=prepare_weight) as convert_weight, \
                        patch("sakuratts.engine.Model.load", side_effect=lambda path: path):
                    original = cached_paths()
                    self.assertEqual(cached_paths(), original)
                    self.assertEqual((convert.call_count, convert_weight.call_count), (1, 1))
                    loader_revision = "updated-loader"
                    updated = cached_paths()
                    self.assertTrue(all(new != old for new, old in zip(updated, original)))
                    self.assertEqual((convert.call_count, convert_weight.call_count), (2, 2))

    def test_weight_cache_prepares_for_model_backend_and_explicit_override(self):
        from sakuratts.model import Model
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            inference = self.inference(root)
            inference.engine = None
            inference._log_weights = Mock()
            inference.model = Model(root / "model.json", {"gpt": "gpt", "sovits": "sovits",
                "frontend": "frontend", "backend": {"preferred": "directml"}})

            def prepare(kind, _checkpoint, output, **_kwargs):
                output.mkdir(parents=True)
                (output / "manifest.json").write_text('{"format":"sakuratts-gpt-fp32-v1"}')

            with patch("sakuratts.prepare.converter.convert_checkpoint", side_effect=prepare) as convert:
                inference.set_weights("gpt", root / "gpt.ckpt")
                self.assertEqual(convert.call_args.kwargs["backend"], "directml")
                inference.experimental = {"capacity": 1280, "device_id": 1, "threads": 3}
                inference.set_weights("gpt", root / "gpt.ckpt")
                self.assertEqual(convert.call_count, 1)
                inference.experimental = {"capacity": 1024}
                inference.set_weights("gpt", root / "gpt.ckpt")
                self.assertEqual(convert.call_count, 2)
                self.assertEqual(convert.call_args.kwargs["experimental"], {"capacity": 1024})
                inference.settings["backend"] = "cpu"
                inference.set_weights("gpt", root / "gpt.ckpt")
                self.assertEqual(convert.call_count, 3)
                self.assertEqual(convert.call_args.kwargs["backend"], "cpu")

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
                 patch("sakuratts.TTS_infer_pack.reference.sha256_file", wraps=sha256_file) as digest, \
                 patch("sakuratts.prepare.converter.prepare_reference", side_effect=lambda **kwargs: kwargs["output"].mkdir(parents=True)) as prepare, \
                 patch("sakuratts.TTS_infer_pack.reference.PreparedReference.load"):
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
