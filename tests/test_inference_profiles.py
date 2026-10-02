"""Presets reach the selected backend and survive service lifecycle boundaries."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from sakuratts import Engine, Model
from sakuratts.engine import Inference
from sakuratts._internal.inference_process import ProcessInference


class InferenceProfileTests(unittest.TestCase):
    def model(self, backend="cpu"):
        return Model(Path("model.json"), {"name": "test", "languages": ["ja"],
            "backend": {"preferred": backend}})

    def test_precision_profile_rejects_wrong_package_and_closes_runtime(self):
        runtime = Mock(acoustic_precision="fp32")
        with patch("sakuratts.backends.create_runtime", return_value=runtime):
            with self.assertRaisesRegex(ValueError, "requires a fp16 acoustic package"):
                Engine.load(self.model("cuda"), profile="fp16")
        runtime.close.assert_called_once()

    def test_profile_mismatch_retains_its_cause_when_cleanup_also_fails(self):
        runtime = Mock(acoustic_precision="fp32")
        runtime.close.side_effect = RuntimeError("frontend process cleanup failed")
        with patch("sakuratts.backends.create_runtime", return_value=runtime):
            with self.assertRaisesRegex(ValueError, "requires a fp16 acoustic package") as failure:
                Engine.load(self.model("cuda"), profile="fp16")
        self.assertIn("frontend process cleanup failed", failure.exception.__notes__[0])

    def test_invalid_service_profile_fails_before_conversion_or_worker_start(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "tts.json"
            config.write_text(json.dumps({"sakuratts": {"backend": "cpu", "profile": "fp32",
                "gpt_checkpoint": "original.ckpt", "sovits_checkpoint": "original.pth"}}), encoding="utf-8")
            with patch.object(Inference, "_convert_initial") as convert:
                for implementation in (Inference, ProcessInference):
                    with self.subTest(implementation=implementation.__name__), \
                            self.assertRaisesRegex(ValueError, "not supported"):
                        implementation(tts_config=config)
                convert.assert_not_called()

    def test_profile_reaches_worker_and_survives_sleep_wake(self):
        fixture = Path(__file__).parent / "fixtures" / "inference_worker.py"
        proxy = ProcessInference("fixture-model.json", backend="cpu", experimental={"policy": "staged"},
                                 startup_timeout=10)
        with patch.object(proxy, "_command", return_value=[sys.executable, str(fixture)]):
            try:
                pids = []
                for _ in range(2):
                    proxy.wake()
                    result = proxy.tts({"chunks": 1})
                    self.assertEqual(result.report["profile"], "int8")
                    self.assertEqual(result.report["backend"], "cpu")
                    pids.append(result.report["pid"])
                    proxy.sleep()
                    self.assertFalse(proxy.alive)
                self.assertNotEqual(*pids)
            finally:
                proxy.close()

if __name__ == "__main__":
    unittest.main()
