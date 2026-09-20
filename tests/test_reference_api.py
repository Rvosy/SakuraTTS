"""Reference-path requests cannot accidentally reuse another transcript/model."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np

from sakuratts.engine import Inference
from sakuratts.model import Model
from sakuratts.reference import ReferenceCache
from sakuratts._internal.reference_condition import PreparedReference


class ReferenceApiTests(unittest.TestCase):
    def test_http_startup_does_not_silently_defer_staged_weights(self):
        with self.assertRaisesRegex(ValueError, "staged policy"):
            Inference(experimental={"policy": "staged"})

    def test_same_audio_reuses_condition_but_rebuilds_text_and_language(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "audio.wav").write_bytes(b"audio-content")
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            audio_hash = hashlib.sha256(b"audio-content").hexdigest()
            identity = dict(gpt_checkpoint_sha256="gpt", sovits_checkpoint_sha256="sovits",
                official_commit="source", audio_sha256=audio_hash, reference_text="old", reference_language="ja")
            base = PreparedReference(dict(model_family="v2ProPlus", identity=identity),
                np.array([1]), np.array([2]), np.zeros((1024, 1), dtype=np.float32),
                np.zeros((1, 1024, 1), dtype=np.float32), np.zeros((1, 512, 1), dtype=np.float32))
            segment = Mock(side_effect=lambda text, language: dict(phones=list(range(len(text))),
                bert_features=np.zeros((1024, len(text)), dtype=np.float32), norm_text=text))
            engine = SimpleNamespace(model=Model(root / "model.json", {"references": {}}),
                _runtime=SimpleNamespace(manifests={kind: {"source": {"checkpoint_sha256": kind,
                "official_commit": "source"}} for kind in ("gpt", "sovits")},
                packages={"frontend": root}, frontend=SimpleNamespace(segment=segment)))
            cache = ReferenceCache(engine, {})
            cache.audio_cache[audio_hash] = base
            first = cache.resolve(root / "audio.wav", "こんにちは", "ja")
            second = cache.resolve(root / "audio.wav", "おはよう", "all_ja")
            self.assertIs(first.ge, second.ge)
            self.assertIs(first.prompt_semantic, second.prompt_semantic)
            self.assertNotEqual(first.reference_phones.size, second.reference_phones.size)
            self.assertEqual(second.manifest["identity"]["reference_language"], "all_ja")
            self.assertEqual(base.manifest["identity"]["reference_text"], "old")
            self.assertEqual(segment.call_args.args, ("おはよう。", "all_ja"))
            self.assertFalse(second.reference_phones.flags.writeable)
            (root / "audio.wav").write_bytes(b"changed-audio")
            with self.assertRaisesRegex(ValueError, "preparation is not configured"):
                cache.resolve(root / "audio.wav", "こんにちは", "ja")
            fresh = ReferenceCache(engine, {})
            self.assertFalse(fresh.audio_cache)

    def test_startup_failure_after_loading_releases_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "tts.json"
            config.write_text(json.dumps({"custom": {"t2s_weights_path": "bad.ckpt"},
                "sakuratts": {"model": "prepared"}}), encoding="utf-8")
            runtime = Mock()
            def activate(instance, model):
                instance.engine = runtime
            with patch.object(Model, "load", return_value=Mock()), \
                    patch.object(Inference, "_activate", activate), \
                    patch.object(Inference, "set_weights", side_effect=ValueError("bad weights")):
                with self.assertRaisesRegex(ValueError, "bad weights"):
                    Inference(tts_config=config)
            runtime.close.assert_called_once()

    def test_switch_initialization_failure_preserves_current_engine(self):
        current = Inference()
        old = Mock()
        current.engine = old
        with patch("sakuratts.engine.Engine.load", side_effect=ValueError("different source")):
            with self.assertRaisesRegex(ValueError, "different source"):
                current._activate(SimpleNamespace(path="candidate"))
        self.assertIs(current.engine, old)
        old.close.assert_not_called()

    def test_gpu_load_failure_closes_candidate_and_clears_loaded_state(self):
        current = Inference()
        current.engine = old = Mock()
        candidate = Mock()
        candidate._runtime.load.side_effect = RuntimeError("out of memory")
        with patch("sakuratts.engine.Engine.load", return_value=candidate), \
                patch("sakuratts.reference.ReferenceCache"):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                current._activate(SimpleNamespace(path="candidate"))
        old.close.assert_called_once()
        candidate.close.assert_called_once()
        self.assertIsNone(current.info())
