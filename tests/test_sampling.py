"""Regression tests for non-obvious official sampling and suffix semantics."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts._internal.sampling import exclude_initial_eos, finish_nonstream_step, logits_to_probs, sample


class SamplingTests(unittest.TestCase):
    def test_repetition_mutates_once_and_changes_stop_argmax(self):
        logits = np.array([[3, -3, 1, 2]], dtype=np.float32)
        logits_to_probs(logits, np.array([[0, 0, 1]]), repetition_penalty=2, top_k=1, top_p=0.9)
        np.testing.assert_array_equal(logits, [[1.5, -6, 1, 2]])
        stop = finish_nonstream_step([0, 1], 2, logits[0], eos=3, step_index=11, prefix_length=2)
        self.assertEqual(stop.reasons, ("argmax_eos",))
        np.testing.assert_array_equal(stop.history, [0, 1])

    def test_top_p_excludes_crossing_token_before_temperature(self):
        logits = np.array([[4, 3, 2, 1]], dtype=np.float32)
        probabilities = logits_to_probs(logits, top_p=0.8, temperature=10)
        np.testing.assert_array_equal(probabilities, [[1, 0, 0, 0]])
        np.testing.assert_array_equal(logits, [[4, 3, 2, 1]])

    def test_top_k_keeps_every_tie_at_pivot(self):
        probabilities = logits_to_probs(np.array([[3, 2, 2, 1]], dtype=np.float32), top_k=2)
        np.testing.assert_array_equal(probabilities > 0, [[True, True, True, False]])

    def test_explicit_noise_and_first_eleven_eos_exclusions(self):
        logits = np.array([[1, 2, 9]], dtype=np.float32)
        self.assertEqual(exclude_initial_eos(logits, 10, 2).shape[-1], 2)
        self.assertEqual(exclude_initial_eos(logits, 11, 2).shape[-1], 3)
        token, _ = sample(logits.copy(), exponential_noise=np.array([[0.000001, 1, 1]], dtype=np.float32))
        self.assertEqual(token.item(), 0)

    def test_early_stop_preserves_strict_greater_and_idx_suffix(self):
        logits = np.array([3, 2, 1], dtype=np.float32)
        first = finish_nonstream_step([0], 1, logits, eos=2, step_index=0, prefix_length=1, early_stop_num=1)
        self.assertFalse(first.stopped)
        second = finish_nonstream_step(first.history, 0, logits, eos=2, step_index=1, prefix_length=1, early_stop_num=1)
        self.assertTrue(second.stopped)
        np.testing.assert_array_equal(second.history, [0, 1, 0])
        np.testing.assert_array_equal(second.official_suffix(), [0])
        zero = finish_nonstream_step([0], 1, logits, eos=2, step_index=0, prefix_length=1, early_stop_num=0)
        np.testing.assert_array_equal(zero.official_suffix(), [0, 1])


if __name__ == "__main__":
    unittest.main()
