"""Comparison failures must remain failures and mismatched inputs must be rejected."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/tools"))
import cpu_amd_benchmark as benchmark


class ComparisonHarnessTests(unittest.TestCase):
    def test_reference_identity_rejects_different_audio_and_checkpoint(self):
        reference = {"audio_sha256": "audio", "text": "reference", "language": "ja",
                     "checkpoints": {"gpt": {"sha256": "gpt"}, "sovits": {"sha256": "sovits"}}}
        identity = {"audio_sha256": "audio", "reference_text": "reference", "reference_language": "ja",
                    "gpt_checkpoint_sha256": "gpt", "sovits_checkpoint_sha256": "sovits"}
        benchmark.validate_identity(identity, reference)
        for key in ("audio_sha256", "gpt_checkpoint_sha256", "reference_text", "reference_language"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                benchmark.validate_identity(dict(identity, **{key: "different"}), reference)

    def test_empty_or_partial_pcm_is_not_a_successful_request(self):
        for data in (b"", b"\x00", b"\x00\x01\x02"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                benchmark.pcm_stats(data, 32000)
        self.assertEqual(benchmark.pcm_stats(b"\x00\x01" * 32000, 32000)["audio_seconds"], 1.)

    def test_genie_collects_public_pcm_in_arrival_order_without_playback(self):
        class Genie:
            @staticmethod
            async def tts_async(name, text, **kwargs):
                self.assertEqual(kwargs, {"play": False, "split_sentence": False})
                yield b""
                yield b"\x01\x00"
                yield b"\x02\x00"
        with patch.dict(os.environ):
            benchmark.configure_offline()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            data, arrivals = asyncio.run(benchmark.genie_request(Genie, "test"))
        self.assertEqual(data, b"\x01\x00\x02\x00")
        self.assertEqual(len(arrivals), 2)
        self.assertLessEqual(arrivals[0], arrivals[1])

    def test_unsampled_memory_is_unavailable(self):
        values = benchmark.request_memory([], 0, 1)
        self.assertIsNone(values["peak_sampled_tree_rss_bytes"])
        self.assertIsNone(values["peak_sampled_tree_private_bytes"])
        self.assertIsNone(values["sampled_tree_cpu_s"])

    def test_subprocess_failure_preserves_exit_code_and_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = benchmark.run_process([sys.executable, "-B", "-c",
                "import sys,time; print('explicit failure', flush=True); time.sleep(0.1); sys.exit(3)"],
                output, interval=0.01, timeout=10)
            self.assertEqual(result["exit_code"], 3)
            self.assertFalse(result["timed_out"])
            self.assertIn("explicit failure", (output / "worker.log").read_text())
            self.assertIsInstance(json.loads((output / "memory-samples.json").read_text()), list)

    def test_subprocess_timeout_remains_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = benchmark.run_process([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                Path(temporary), interval=0.01, timeout=0.1)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
