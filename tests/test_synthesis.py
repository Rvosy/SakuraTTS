"""Request-boundary checks without inference models or historical fixtures."""

from pathlib import Path
import gc
import io
import logging
import sys
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts._internal.reference_condition import PreparedReference
from sakuratts._internal.generation import SynthesisCancelled
from sakuratts._internal.synthesis import (generate_prepared_semantic, prepare_text, prepare_text_request, synthesize,
                                 synthesize_acoustic, synthesize_prepared)
from sakuratts.frontend.text_frontend import TextFrontend
from sakuratts.frontend.processors import JapaneseProcessor


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

    def release_request_state(self):
        self.released = True


class SoVITS:
    sample_rate = 32000
    encoder = SimpleNamespace(manifest={
        "source": {"checkpoint_sha256": "sovits", "official_commit": "commit"},
        "config": {"model": {"version": "v2Pro", "inter_channels": 192}, "semantic_upsample_factor": 2},
    })

    def validate_reference(self, reference):
        pass

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

    def test_progress_logging_preserves_tokens_audio_and_rng_state(self):
        from sakuratts._internal.generation import logger
        quiet_rng = np.random.default_rng(1234)
        verbose_rng = np.random.default_rng(1234)
        with patch.object(logger, "level", logging.WARNING):
            quiet = self.request(rng=quiet_rng)
        with self.assertLogs("sakuratts.inference", level="INFO"), \
                patch("sakuratts._internal.generation.sys.stderr", io.StringIO()):
            verbose = self.request(rng=verbose_rng)
        np.testing.assert_array_equal(quiet.generation.sampled_tokens, verbose.generation.sampled_tokens)
        np.testing.assert_array_equal(quiet.pcm, verbose.pcm)
        np.testing.assert_array_equal(quiet.waveform, verbose.waveform)
        self.assertEqual(quiet_rng.bit_generator.state, verbose_rng.bit_generator.state)

    def test_loaded_model_identity_mismatch_is_rejected_before_frontend(self):
        self.reference.manifest["identity"]["sovits_checkpoint_sha256"] = "another-model"
        with self.assertRaisesRegex(ValueError, "Loaded sovits"):
            self.request()
        self.assertFalse(hasattr(self.frontend, "request"))

    def test_proplus_reference_cannot_be_used_with_pro_acoustic_graph(self):
        self.reference.manifest["model_family"] = "v2ProPlus"
        with self.assertRaisesRegex(ValueError, "must match"):
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

    def test_complete_text_preparation_keeps_order_and_does_not_restart_frontend(self):
        original = self.frontend.prepare_target
        calls = []

        def fragments(text, language, split_method):
            calls.append((text, language, split_method))
            first = original(text, language, split_method)[0]
            return [dict(first, norm_text="first"), dict(first, norm_text="second")]

        self.frontend.prepare_target = fragments
        text = "こんにちは。\n今日はいい天気ですね。"
        request = prepare_text_request(text, "ja", self.frontend)
        self.assertEqual(calls, [(text, "ja", "cut0")])
        self.assertEqual([p.target["norm_text"] for p in request.fragments], ["first", "second"])
        self.assertTrue(all(p.text == text and p.seconds == 0 for p in request.fragments))
        self.assertGreaterEqual(request.seconds, 0)
        self.assertFalse(hasattr(self.gpt, "inputs"))

    def test_later_invalid_fragment_fails_before_any_synthesis(self):
        original = self.frontend.prepare_target

        def fragments(*args, **kwargs):
            first = original(*args, **kwargs)[0]
            return [first, dict(first, bert_features=np.zeros((1024, 1), dtype=np.float32))]

        self.frontend.prepare_target = fragments
        with self.assertRaisesRegex(ValueError, "aligned"):
            prepare_text_request("こんにちは。\n今日はいい天気ですね。", "ja", self.frontend)
        self.assertFalse(hasattr(self.gpt, "inputs"))

    def test_explicit_cut2_uses_full_original_text_and_preserves_fragment_order(self):
        frontend = TextFrontend(
            processors={"ja": JapaneseProcessor(SimpleNamespace(normalize=lambda text: text, g2p=lambda text: ["a"] * 6))},
            symbols=["UNK", "a"], segmenter=lambda text, *args: [{"lang": "ja", "text": text}])
        sentence = "今日はいい天気ですね。"
        text = sentence * 10
        default = prepare_text_request(text, "ja", frontend)
        grouped = prepare_text_request(text, "ja", frontend, split_method="cut2")
        self.assertEqual([p.target["norm_text"] for p in default.fragments], [text])
        self.assertEqual([p.target["norm_text"] for p in grouped.fragments], [sentence * 5, sentence * 5])
        self.assertTrue(all(p.text == text for p in grouped.fragments))
        self.assertEqual(prepare_text(text, "ja", frontend).target["norm_text"], text)
        self.assertFalse(hasattr(self.gpt, "inputs"))

    def test_split_method_is_explicit_and_invalid_methods_fail_before_frontend(self):
        text = "こんにちは。"
        prepare_text_request(text, "all_ja", self.frontend, split_method="cut2")
        self.assertEqual(self.frontend.request, (text, "all_ja", "cut2"))
        frontend = Frontend()
        with self.assertRaisesRegex(ValueError, "cut0 through cut5"):
            prepare_text_request(text, "ja", frontend, split_method="cut6")
        with self.assertRaises(TypeError):
            prepare_text_request(text, "ja", frontend, "cut2")
        self.assertFalse(hasattr(frontend, "request"))

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
        with self.assertRaisesRegex(ValueError, "Unsupported prepared reference language: zh"):
            self.request()
        self.assertFalse(hasattr(self.frontend, "request"))

    def test_optional_gpt_state_release_precedes_acoustic_and_preserves_semantic(self):
        decode = self.sovits.decode

        def observe_release(*args, **kwargs):
            self.assertTrue(self.gpt.released)
            return decode(*args, **kwargs)

        self.sovits.decode = observe_release
        result = self.request(release_gpt_state=True)
        np.testing.assert_array_equal(self.sovits.inputs["semantic"], result.generation.semantic)

    def test_semantic_failure_releases_state_without_creating_audio(self):
        def fail(token):
            raise ValueError("semantic capacity exhausted")

        self.gpt.decode = fail
        with self.assertRaisesRegex(ValueError, "semantic capacity exhausted"):
            self.request(release_gpt_state=True)
        self.assertTrue(self.gpt.released)
        self.assertFalse(hasattr(self.sovits, "inputs"))

    def test_default_does_not_change_caller_owned_gpt_state_lifetime(self):
        self.request()
        self.assertFalse(hasattr(self.gpt, "released"))

    def test_staged_request_does_not_retain_gpt_and_preserves_rng_consumption(self):
        prepared = prepare_text("こんにちは。", "ja", self.frontend)
        composed_rng = np.random.default_rng(12)
        composed = synthesize_prepared(prepared, self.reference, gpt=self.gpt, sovits=self.sovits,
                                       early_stop_num=2700, rng=composed_rng)
        expected_noise = self.sovits.inputs["noise"].copy()
        staged_rng = np.random.default_rng(12)
        pending = generate_prepared_semantic(prepared, self.reference, gpt=self.gpt,
                                             early_stop_num=2700, rng=staged_rng)
        model_reference = weakref.ref(self.gpt)
        self.gpt = None
        gc.collect()
        self.assertIsNone(model_reference())
        actual = synthesize_acoustic(pending, sovits=self.sovits)
        np.testing.assert_array_equal(composed.generation.sampled_tokens, actual.generation.sampled_tokens)
        np.testing.assert_array_equal(expected_noise, self.sovits.inputs["noise"])
        self.assertEqual(composed_rng.bit_generator.state, staged_rng.bit_generator.state)
        np.testing.assert_array_equal(composed.pcm, actual.pcm)

    def test_acoustic_identity_mismatch_is_rejected_before_rng_or_decode(self):
        prepared = prepare_text("こんにちは。", "ja", self.frontend)
        rng = np.random.default_rng(8)
        pending = generate_prepared_semantic(prepared, self.reference, gpt=self.gpt,
                                             early_stop_num=2700, rng=rng)
        state = rng.bit_generator.state
        other = SoVITS()
        other.encoder = SimpleNamespace(manifest={
            "source": {"checkpoint_sha256": "other", "official_commit": "commit"},
        })
        with self.assertRaisesRegex(ValueError, "Loaded sovits"):
            synthesize_acoustic(pending, sovits=other)
        self.assertEqual(state, rng.bit_generator.state)
        self.assertFalse(hasattr(other, "inputs"))

    def test_bound_acoustic_reference_mismatch_precedes_rng_and_decode(self):
        prepared = prepare_text("こんにちは。", "ja", self.frontend)
        rng = np.random.default_rng(8)
        pending = generate_prepared_semantic(prepared, self.reference, gpt=self.gpt,
                                             early_stop_num=2700, rng=rng)
        state = rng.bit_generator.state

        def reject(reference):
            self.assertIs(reference, pending.reference)
            raise ValueError("Acoustic reference binding mismatch")

        self.sovits.validate_reference = reject
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            synthesize_acoustic(pending, sovits=self.sovits)
        self.assertEqual(state, rng.bit_generator.state)
        self.assertFalse(hasattr(self.sovits, "inputs"))

    def test_explicit_replay_does_not_consume_shared_rng(self):
        rng = np.random.default_rng(42)
        state = rng.bit_generator.state
        result = self.request(rng=rng, acoustic_noise=np.zeros((1, 192, 22), dtype=np.float32))
        self.assertEqual(state, rng.bit_generator.state)
        self.assertEqual(result.generation.semantic.shape[-1], 11)

    def test_semantic_stage_binds_reference_identity_and_target_phones(self):
        prepared = prepare_text("こんにちは。", "ja", self.frontend)
        pending = generate_prepared_semantic(prepared, self.reference, gpt=self.gpt,
                                             early_stop_num=2700)
        prepared.target["phones"][0] = 99
        result = synthesize_acoustic(pending, sovits=self.sovits)
        np.testing.assert_array_equal(self.sovits.inputs["phones"], [[4, 5]])
        np.testing.assert_array_equal(result.target["phones"], self.sovits.inputs["phones"][0])
        self.reference.manifest["identity"]["audio_sha256"] = "changed-reference"
        with self.assertRaisesRegex(ValueError, "Reference identity changed"):
            synthesize_acoustic(pending, sovits=self.sovits)

    def test_cancel_before_prefill_releases_state_without_computation_or_rng_use(self):
        rng = np.random.default_rng(12)
        state = rng.bit_generator.state
        with self.assertRaises(SynthesisCancelled) as caught:
            self.request(cancel_requested=lambda: True, release_gpt_state=True, rng=rng)
        self.assertEqual(caught.exception.stage, "before_prefill")
        self.assertTrue(self.gpt.released)
        self.assertFalse(hasattr(self.gpt, "inputs"))
        self.assertFalse(hasattr(self.sovits, "inputs"))
        self.assertEqual(state, rng.bit_generator.state)

    def test_cancel_during_semantics_releases_state_and_same_models_can_retry(self):
        expected = self.request(rng=np.random.default_rng(12))
        expected_noise = self.sovits.inputs["noise"].copy()
        for boundary, expected_step in (("after_prefill", 0), ("after_decode", 3), ("semantic_step", 11)):
            with self.subTest(boundary=boundary):
                cancelled = Event()
                original_prefill, original_decode = self.gpt.prefill, self.gpt.decode

                def prefill(*args):
                    result = original_prefill(*args)
                    if boundary == "after_prefill":
                        cancelled.set()
                    return result

                def decode(token):
                    result = original_decode(token)
                    if boundary == "after_decode" and self.gpt.step == 3:
                        cancelled.set()
                    return result

                def draw(index, shape):
                    if boundary == "semantic_step" and index == 11:
                        cancelled.set()
                    return np.ones(shape, dtype=np.float32)

                self.gpt.prefill, self.gpt.decode = prefill, decode
                self.gpt.released = False
                del self.sovits.inputs
                try:
                    with self.assertRaises(SynthesisCancelled) as caught:
                        self.request(cancel_requested=cancelled.is_set, semantic_random_draw=draw,
                                     release_gpt_state=True, rng=np.random.default_rng(12))
                    self.assertEqual(caught.exception.stage, boundary)
                    self.assertEqual(self.gpt.step, expected_step)
                    self.assertTrue(self.gpt.released)
                    self.assertFalse(hasattr(self.sovits, "inputs"))
                finally:
                    self.gpt.prefill, self.gpt.decode = original_prefill, original_decode
                cancelled.clear()
                actual = self.request(cancel_requested=cancelled.is_set, release_gpt_state=True,
                                      rng=np.random.default_rng(12))
                np.testing.assert_array_equal(expected.generation.sampled_tokens, actual.generation.sampled_tokens)
                np.testing.assert_array_equal(expected_noise, self.sovits.inputs["noise"])
                np.testing.assert_array_equal(expected.pcm, actual.pcm)

    def test_cancel_between_phases_preserves_rng_and_does_not_decode_audio(self):
        prepared = prepare_text("こんにちは。", "ja", self.frontend)
        rng = np.random.default_rng(12)
        pending = generate_prepared_semantic(prepared, self.reference, gpt=self.gpt,
                                             early_stop_num=2700, rng=rng)
        state = rng.bit_generator.state
        with self.assertRaises(SynthesisCancelled) as caught:
            synthesize_acoustic(pending, sovits=self.sovits, cancel_requested=lambda: True)
        self.assertEqual(caught.exception.stage, "before_acoustic")
        self.assertEqual(state, rng.bit_generator.state)
        self.assertFalse(hasattr(self.sovits, "inputs"))

    def test_cancel_during_acoustic_discards_waveform_before_pcm_and_allows_retry(self):
        expected = self.request(rng=np.random.default_rng(12))
        cancelled = Event()
        original_decode = self.sovits.decode

        def decode(*args, **kwargs):
            result = original_decode(*args, **kwargs)
            cancelled.set()
            return result

        self.sovits.decode = decode
        with patch("sakuratts._internal.synthesis.single_fragment_pcm") as make_pcm:
            with self.assertRaises(SynthesisCancelled) as caught:
                self.request(cancel_requested=cancelled.is_set, release_gpt_state=True,
                             rng=np.random.default_rng(12))
            self.assertEqual(caught.exception.stage, "after_acoustic")
            self.assertTrue(self.gpt.released)
            make_pcm.assert_not_called()
        self.sovits.decode = original_decode
        cancelled.clear()
        actual = self.request(cancel_requested=cancelled.is_set, release_gpt_state=True,
                              rng=np.random.default_rng(12))
        np.testing.assert_array_equal(expected.generation.sampled_tokens, actual.generation.sampled_tokens)
        np.testing.assert_array_equal(expected.pcm, actual.pcm)

    def test_false_cancellation_predicate_preserves_default_output_and_rng(self):
        default_rng, checked_rng = np.random.default_rng(12), np.random.default_rng(12)
        expected = self.request(semantic_random_draw=None, rng=default_rng)
        expected_noise = self.sovits.inputs["noise"].copy()
        actual = self.request(semantic_random_draw=None, rng=checked_rng, cancel_requested=lambda: False)
        np.testing.assert_array_equal(expected.generation.sampled_tokens, actual.generation.sampled_tokens)
        np.testing.assert_array_equal(expected_noise, self.sovits.inputs["noise"])
        np.testing.assert_array_equal(expected.pcm, actual.pcm)
        self.assertEqual(default_rng.bit_generator.state, checked_rng.bit_generator.state)

    def test_cancellation_does_not_replace_predicate_or_model_errors(self):
        original_error = ValueError("original failure")

        def fail():
            raise original_error

        with self.assertRaises(ValueError) as caught:
            self.request(cancel_requested=fail, release_gpt_state=True)
        self.assertIs(caught.exception, original_error)
        self.assertTrue(self.gpt.released)
        cancelled = Event()

        def decode(token):
            cancelled.set()
            raise original_error

        self.gpt.decode = decode
        self.gpt.released = False
        with self.assertRaises(ValueError) as caught:
            self.request(cancel_requested=cancelled.is_set, release_gpt_state=True)
        self.assertIs(caught.exception, original_error)
        self.assertTrue(self.gpt.released)
        self.assertFalse(hasattr(self.sovits, "inputs"))

    def test_phase_result_does_not_retain_cancellation_closure_or_its_model(self):
        prepared = prepare_text("こんにちは。", "ja", self.frontend)
        predicate = lambda model=self.gpt: model is None
        predicate_reference, model_reference = weakref.ref(predicate), weakref.ref(self.gpt)
        pending = generate_prepared_semantic(prepared, self.reference, gpt=self.gpt,
                                             early_stop_num=2700, cancel_requested=predicate)
        predicate = self.gpt = None
        gc.collect()
        self.assertIsNone(predicate_reference())
        self.assertIsNone(model_reference())
        self.assertIsNotNone(synthesize_acoustic(pending, sovits=self.sovits).pcm)


if __name__ == "__main__":
    unittest.main()
