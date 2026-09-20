"""Exercise the official stopping protocol through the real generation loop."""

from pathlib import Path
import importlib.util
import io
import itertools
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts._internal.generation import SynthesisCancelled, generate_semantic


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

    def generate(self, *, early_stop_num=-1, eos_step=None, **options):
        return generate_semantic(
            ScriptedGPT(eos_step), np.asarray([[0]], dtype=np.int64), self.prompt,
            np.zeros((1, 1, 1024), dtype=np.float32), eos=3,
            top_k=1, repetition_penalty=1.0, early_stop_num=early_stop_num,
            random_draw=lambda step, shape: np.ones(shape, dtype=np.float32),
            **options,
        )

    def test_plain_progress_uses_real_steps_and_rate_limited_lines(self):
        with patch("sakuratts._internal.generation.sys.stderr", io.StringIO()), \
                patch("sakuratts._internal.generation.time.perf_counter", side_effect=itertools.count(0, .25)), \
                self.assertLogs("sakuratts.inference", level="INFO") as logs:
            result = self.generate(eos_step=11)
        lines = [line for line in logs.output if " it · " in line and "EOS" not in line]
        self.assertGreater(len(lines), 0)
        self.assertLess(len(lines), len(result.sampled_tokens))
        self.assertIn("预测语义Token", logs.output[0])
        self.assertIn("12 it", logs.output[-2])
        self.assertIn("it/s", logs.output[-2])
        self.assertIn("T2S Decoding EOS [2 -> 14]", logs.output[-1])
        self.assertNotIn("/1500", "\n".join(logs.output))

    def test_eos_lengths_use_current_prefix_and_include_last_sampling_iteration(self):
        self.prompt = np.zeros((1, 157), dtype=np.int64)
        with self.assertLogs("sakuratts.inference", level="INFO") as logs:
            result = self.generate(eos_step=252)
        self.assertIn("T2S Decoding EOS [157 -> 410]", logs.output[-1])
        self.assertEqual(result.stop.history.size, 409)
        self.assertEqual(result.semantic.shape[-1], 252)

    @unittest.skipUnless(importlib.util.find_spec("tqdm"), "Terminal progress uses the server extra")
    def test_real_terminal_renderer_clears_live_line_and_logs_final_count(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        terminal = Terminal()
        with patch("sakuratts._internal.generation.sys.stderr", terminal), \
                self.assertLogs("sakuratts.inference", level="INFO") as logs:
            self.generate(eos_step=11)
        self.assertIn("12 it", logs.output[-2])
        self.assertIn("T2S Decoding EOS [2 -> 14]", logs.output[-1])
        self.assertIn("it/s", terminal.getvalue())
        self.assertIn("\r", terminal.getvalue())
        self.assertNotIn("1500", terminal.getvalue())
        self.assertNotIn("%", terminal.getvalue())

    @unittest.skipUnless(importlib.util.find_spec("tqdm"), "Terminal progress uses the server extra")
    def test_terminal_bar_counts_eos_and_closes_on_completion_or_failure(self):
        for outcome in ("completed", "cancelled", "failed"):
            with self.subTest(outcome=outcome), \
                    patch("sakuratts._internal.generation.sys.stderr.isatty", return_value=True), \
                    patch("tqdm.tqdm") as bar, \
                    self.assertLogs("sakuratts.inference", level="DEBUG") as logs:
                if outcome == "completed":
                    result = self.generate(eos_step=11)
                    self.assertEqual(bar.return_value.update.call_count, len(result.sampled_tokens))
                elif outcome == "cancelled":
                    with self.assertRaises(SynthesisCancelled):
                        self.generate(cancel_requested=lambda: True)
                    self.assertIn("取消", logs.output[-1])
                    self.assertNotIn("语义预测结束", logs.output[-1])
                    self.assertNotIn("T2S Decoding", "\n".join(logs.output))
                else:
                    with patch.object(ScriptedGPT, "prefill", side_effect=RuntimeError("GPU failure")), \
                            self.assertRaisesRegex(RuntimeError, "GPU failure"):
                        self.generate()
                    self.assertIn("失败", logs.output[-1])
                    self.assertNotIn("T2S Decoding", "\n".join(logs.output))
                bar.return_value.close.assert_called_once()

    def test_iteration_limit_keeps_history_but_slices_off_first_generated_token(self):
        for threshold in (-1, 2700):
            with self.subTest(early_stop_num=threshold):
                with self.assertLogs("sakuratts.inference", level="INFO") as logs:
                    result = self.generate(early_stop_num=threshold)
                self.assertIn("T2S Decoding STOP [2 -> 1502]", logs.output[-1])
                self.assertIn("达到长度上限", logs.output[-1])
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
        with self.assertLogs("sakuratts.inference", level="INFO") as logs:
            result = self.generate(early_stop_num=3)
        self.assertIn("T2S Decoding STOP [2 -> 6]", logs.output[-1])
        self.assertIn("达到长度上限", logs.output[-1])
        self.assertNotIn("EOS", logs.output[-1])
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
                with self.assertLogs("sakuratts.inference", level="INFO") as logs:
                    result = self.generate(early_stop_num=eos_step, eos_step=eos_step)
                self.assertIn(f"T2S Decoding EOS [2 -> {eos_step + 3}]", logs.output[-1])
                self.assertIn("达到长度上限", logs.output[-1])
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
