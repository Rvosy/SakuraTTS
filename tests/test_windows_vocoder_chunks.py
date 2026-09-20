"""Candidate gates must not weaken the existing acoustic engineering screen."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/tools"))
from windows_vocoder_chunks import check_output


class VocoderChunkGateTests(unittest.TestCase):
    def test_small_absolute_error_does_not_bypass_amplitude_gate(self):
        expected = np.full((1, 1, 200000), .05, np.float32)
        actual = expected.copy()
        actual[0, 0, 100000] = .054
        checks = check_output(actual, expected, [100000])
        self.assertFalse(checks["engineering_metrics"]["checks"]["amplitude"])
        self.assertTrue(checks["seams"][0]["passed"])
        self.assertFalse(checks["tight_engineering_passed"])

    def test_local_seam_error_cannot_hide_in_whole_waveform_rmse(self):
        expected = np.full((1, 1, 200000), .5, np.float32)
        actual = expected.copy()
        actual[0, 0, 99900:100100] += .004
        checks = check_output(actual, expected, [100000])
        self.assertLess(checks["engineering_metrics"]["rmse"], .0005)
        self.assertFalse(checks["seams"][0]["passed"])
        self.assertFalse(checks["tight_engineering_passed"])


if __name__ == "__main__":
    unittest.main()
