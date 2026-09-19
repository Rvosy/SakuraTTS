"""Exercise the official stopping protocol through the real generation loop."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.generation import generate_semantic


class ScriptedGPT:
    """Predict a distinct first token, repeated later tokens, and optional EOS."""

    def __init__(self, eos_step=None):
        self.eos_step = eos_step

    def logits(self):
        token = 1 if self.step == 0 else 2
        if self.step == self.eos_step:
            token = 3
        logits = np.full((1, 4), -20, dtype=np.float32)
        logits[0, token] = 20
        return logits

    def prefill(self, phones, prompt, bert):
        self.step = 0
        return self.logits()

    def decode(self, token):
        self.step += 1
        return self.logits()


class GenerationLimitTests(unittest.TestCase):
    def setUp(self):
        self.prompt = np.asarray([[0, 2]], dtype=np.int64)

    def generate(self, *, early_stop_num=-1, eos_step=None):
        return generate_semantic(
            ScriptedGPT(eos_step), np.asarray([[0]], dtype=np.int64), self.prompt,
            np.zeros((1, 1, 1024), dtype=np.float32), eos=3,
            top_k=1, repetition_penalty=1.0, early_stop_num=early_stop_num,
            random_draw=lambda step, shape: np.ones(shape, dtype=np.float32),
        )

    def test_iteration_limit_keeps_history_but_slices_off_first_generated_token(self):
        for threshold in (-1, 2700):
            with self.subTest(early_stop_num=threshold):
                result = self.generate(early_stop_num=threshold)
                expected_tokens = np.full(1500, 2, dtype=np.int64)
                expected_tokens[0] = 1
                np.testing.assert_array_equal(result.sampled_tokens, expected_tokens)
                self.assertTrue(result.stop.stopped)
                self.assertEqual(set(result.stop.reasons), {"iteration_limit"})
                self.assertEqual(result.stop.returned_index, 1499)
                np.testing.assert_array_equal(
                    result.stop.history, np.concatenate((self.prompt[0], expected_tokens)))
                np.testing.assert_array_equal(result.semantic, np.full((1, 1, 1499), 2))

    def test_early_stop_occurs_only_after_the_threshold_is_exceeded(self):
        result = self.generate(early_stop_num=3)
        np.testing.assert_array_equal(result.sampled_tokens, [1, 2, 2, 2])
        self.assertTrue(result.stop.stopped)
        self.assertEqual(set(result.stop.reasons), {"early_stop_num"})
        self.assertEqual(result.stop.returned_index, 3)
        np.testing.assert_array_equal(result.stop.history, [0, 2, 1, 2, 2, 2])
        np.testing.assert_array_equal(result.semantic, [[[2, 2, 2]]])

    def test_zero_threshold_keeps_the_official_zero_index_slice(self):
        result = self.generate(early_stop_num=0)
        np.testing.assert_array_equal(result.sampled_tokens, [1])
        self.assertEqual(set(result.stop.reasons), {"early_stop_num"})
        self.assertEqual(result.stop.returned_index, 0)
        np.testing.assert_array_equal(result.stop.history, [0, 2, 1])
        np.testing.assert_array_equal(result.semantic, [[[0, 2, 1]]])

    def test_eos_overlaps_limits_without_discarding_an_extra_token(self):
        for eos_step in (11, 1499):
            with self.subTest(eos_step=eos_step):
                result = self.generate(early_stop_num=eos_step, eos_step=eos_step)
                kept_tokens = np.full(eos_step, 2, dtype=np.int64)
                kept_tokens[0] = 1
                np.testing.assert_array_equal(
                    result.sampled_tokens, np.concatenate((kept_tokens, [3])))
                expected_reasons = {"early_stop_num", "argmax_eos", "sample_eos"}
                if eos_step == 1499:
                    expected_reasons.add("iteration_limit")
                self.assertTrue(result.stop.stopped)
                self.assertEqual(set(result.stop.reasons), expected_reasons)
                self.assertEqual(result.stop.returned_index, eos_step)
                np.testing.assert_array_equal(
                    result.stop.history, np.concatenate((self.prompt[0], kept_tokens)))
                np.testing.assert_array_equal(result.semantic, kept_tokens[None, None, :])


if __name__ == "__main__":
    unittest.main()
