"""A failed low-precision kernel must not silently turn NaN logits into token 0."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from sakuratts._internal.generation import generate_semantic
from test_generation_limits import ScriptedGPT


class GenerationPrecisionTests(unittest.TestCase):
    def generate(self, model, observer=None):
        return generate_semantic(model, np.array([[0]], np.int64), np.array([[0, 2]], np.int64),
            np.zeros((1, 1, 1024), np.float32), eos=3, top_k=1, repetition_penalty=1., early_stop_num=3,
            random_draw=lambda step, shape: np.ones(shape, np.float32), observer=observer)

    def test_prefill_and_decode_nonfinite_logits_fail_before_sampling_or_eos_exclusion(self):
        for bad_step in (0, 2):
            for value in (np.nan, np.inf, -np.inf):
                for column in (0, 3):
                    with self.subTest(step=bad_step, value=value, column=column):
                        class BrokenGPT(ScriptedGPT):
                            def logits(self):
                                result = super().logits()
                                if self.step == bad_step:
                                    result[0, column] = value
                                return result

                        observed = []
                        with self.assertRaisesRegex(RuntimeError, f"non-finite logits at semantic step {bad_step}"):
                            self.generate(BrokenGPT(), observer=lambda *row: observed.append(row))
                        self.assertEqual(len(observed), bad_step)

    def test_sampling_can_still_mask_finite_logits_with_negative_infinity(self):
        observed = []
        result = self.generate(ScriptedGPT(), observer=lambda *row: observed.append(row))
        np.testing.assert_array_equal(result.sampled_tokens, [1, 2, 2, 2])
        for _, raw, _, probabilities, _ in observed:
            self.assertTrue(np.isfinite(raw).all())
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertEqual(np.count_nonzero(probabilities), 1)


if __name__ == "__main__":
    unittest.main()
