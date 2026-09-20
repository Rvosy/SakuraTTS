"""CPU checks for the run-only arena experiment and unmodified whole PCM."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/tools"))
from windows_acoustic_arena import KEY, output_checks, run_options


class FakeRunOptions:
    def __init__(self):
        self.entries = {}

    def add_run_config_entry(self, key, value):
        self.entries[key] = value


class AcousticArenaTests(unittest.TestCase):
    def test_shrink_policy_does_not_change_short_runs_after_long(self):
        ort = SimpleNamespace(RunOptions=FakeRunOptions)
        cases = ("short", "long", "short", "multi", "punctuation")
        self.assertEqual([run_options(ort, "off", case) for case in cases], [None] * 5)
        selected = [run_options(ort, "after-long", case) for case in cases]
        self.assertEqual([item is not None for item in selected], [False, True, False, False, False])
        self.assertEqual(selected[1].entries, {KEY: "gpu:0"})
        self.assertTrue(all(run_options(ort, "always", case).entries == {KEY: "gpu:0"} for case in cases))

    def test_complete_waveform_and_pcm_boundaries_are_checked(self):
        expected = np.asarray([[[0.2, -0.3, 0.1]]], dtype=np.float32)
        exact = output_checks(expected.copy(), expected, 32000)
        self.assertTrue(exact["waveform_bitwise_equal"] and exact["pcm_bitwise_equal"])
        self.assertEqual(exact["waveform_samples"], 3)
        self.assertEqual(exact["pcm_samples"], 9603)
        for index in (0, -1):
            changed = expected.copy()
            changed[0, 0, index] += np.float32(.01)
            check = output_checks(changed, expected, 32000)
            self.assertFalse(check["waveform_bitwise_equal"] or check["pcm_bitwise_equal"])


if __name__ == "__main__":
    unittest.main()
