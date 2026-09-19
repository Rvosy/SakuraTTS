"""Request-boundary checks without inference models or historical fixtures."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.reference_condition import PreparedReference
from sakuratts.synthesis import prepare_text, synthesize, synthesize_prepared


class Frontend:
    def prepare_target(self, text, language, split_method):
        self.request = (text, language, split_method)
        return [dict(phones=[4, 5], bert_features=np.full((1024, 2), 7, dtype=np.float32), norm_text=text)]


class GPT:
    config = {"eos": 2}
    weight_manifest = {"source": {"checkpoint_sha256": "gpt", "official_commit": "commit"}}

    def prefill(self, phones, prompt, bert):
        self.inputs = tuple(value.copy() for value in (phones, prompt, bert))
        self.step = 0
        return np.asarray([[0, 20, -20]], dtype=np.float32)

    def decode(self, token):
        self.step += 1
        return np.asarray([[0, -20, 20] if self.step == 11 else [0, 20, -20]], dtype=np.float32)


class SoVITS:
    sample_rate = 32000
    encoder = SimpleNamespace(manifest={
        "source": {"checkpoint_sha256": "sovits", "official_commit": "commit"},
        "config": {"model": {"version": "v2Pro", "inter_channels": 192}, "semantic_upsample_factor": 2},
    })

    def decode(self, semantic, phones, ge, ge512, noise, **parameters):
        self.inputs = dict(semantic=semantic.copy(), phones=phones.copy(), noise=noise.copy(), parameters=parameters)
        return np.asarray([[[0, .5, -.5, 0]]], dtype=np.float32)


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.reference = PreparedReference(
            dict(model_family="v2Pro", identity=dict(gpt_checkpoint_sha256="gpt", sovits_checkpoint_sha256="sovits", official_commit="commit", reference_language="ja")),
            np.asarray([1, 2, 3], dtype=np.int64), np.asarray([0, 1, 0], dtype=np.int64),
            np.full((1024, 3), 2, dtype=np.float32), np.zeros((1, 1024, 1), dtype=np.float32),
            np.zeros((1, 512, 1), dtype=np.float32),
        )
        self.frontend, self.gpt, self.sovits = Frontend(), GPT(), SoVITS()

    def request(self, **overrides):
        options = dict(frontend=self.frontend, gpt=self.gpt, sovits=self.sovits, early_stop_num=2700,
                       semantic_random_draw=lambda index, shape: np.ones(shape, dtype=np.float32))
        options.update(overrides)
        return synthesize("こんにちは。", "ja", self.reference, **options)

    def test_original_text_and_current_generated_tokens_reach_the_models(self):
        result = self.request()
        self.assertEqual(self.frontend.request, ("こんにちは。", "ja", "cut0"))
        phones, prompt, bert = self.gpt.inputs
        np.testing.assert_array_equal(phones, [[1, 2, 3, 4, 5]])
        np.testing.assert_array_equal(prompt, [[0, 1, 0]])
        self.assertTrue(np.all(bert[:, :3] == 2) and np.all(bert[:, 3:] == 7))
        np.testing.assert_array_equal(self.sovits.inputs["phones"], [[4, 5]])
        np.testing.assert_array_equal(self.sovits.inputs["semantic"], result.generation.semantic)
        self.assertEqual(self.sovits.inputs["noise"].shape, (1, 192, result.generation.semantic.shape[-1] * 2))
        np.testing.assert_array_equal(result.pcm[:4], [0, 16384, -16384, 0])
        self.assertEqual(result.pcm.size, 4 + 9600)
        self.assertTrue(np.all(result.pcm[4:] == 0))

    def test_loaded_model_identity_mismatch_is_rejected_before_frontend(self):
        self.reference.manifest["identity"]["sovits_checkpoint_sha256"] = "another-model"
        with self.assertRaisesRegex(ValueError, "Loaded sovits"):
            self.request()
        self.assertFalse(hasattr(self.frontend, "request"))

    def test_replay_noise_cannot_force_a_different_semantic_length(self):
        with self.assertRaisesRegex(ValueError, "generated-history shape"):
            self.request(acoustic_noise=np.zeros((1, 192, 1), dtype=np.float32))
        self.assertFalse(hasattr(self.sovits, "inputs"))

    def test_multiple_fragments_are_not_silently_truncated(self):
        original = self.frontend.prepare_target
        self.frontend.prepare_target = lambda *args, **kwargs: original(*args, **kwargs) * 2
        with self.assertRaisesRegex(NotImplementedError, "exactly one"):
            self.request()
        self.assertFalse(hasattr(self.gpt, "inputs"))

    def test_text_preparation_does_not_need_or_retain_synthesis_models(self):
        prepared = prepare_text("こんにちは。", "ja", self.frontend)
        self.frontend = None
        result = synthesize_prepared(prepared, self.reference, gpt=self.gpt, sovits=self.sovits,
                                     early_stop_num=2700,
                                     semantic_random_draw=lambda index, shape: np.ones(shape, dtype=np.float32))
        self.assertEqual(prepared.text, "こんにちは。")
        np.testing.assert_array_equal(self.sovits.inputs["semantic"], result.generation.semantic)

    def test_japanese_modes_are_explicit_and_chinese_is_rejected(self):
        self.assertEqual(prepare_text("今日は晴れです。", "all_ja", self.frontend).language, "all_ja")
        with self.assertRaisesRegex(ValueError, "Japanese"):
            prepare_text("你好。", "zh", self.frontend)

    def test_unvalidated_reference_language_is_rejected_before_frontend(self):
        self.reference.manifest["identity"]["reference_language"] = "zh"
        with self.assertRaisesRegex(ValueError, "Japanese reference"):
            self.request()
        self.assertFalse(hasattr(self.frontend, "request"))


if __name__ == "__main__":
    unittest.main()
